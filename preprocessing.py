import os
import json
import pickle

from tqdm import tqdm
import numba as nb
import numpy as np
import matplotlib.pyplot as plt 

import awkward as ak
import pyarrow as pa
import vector
import fastjet as fj
import torch
from tensordict import TensorDict

from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
from utils.histogram import TorchHist2D

vector.register_awkward()

pth_bins = (
    "11", "15", "20", "25", "35", "45", "55", "infty",
)
jet_columns = (
    "pt", "eta", "phi", "nef",
    "ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0",
    "leading_constit_pt", "leading_constit_eta", "leading_constit_phi",
    "subleading_constit_pt", "subleading_constit_eta", "subleading_constit_phi",
    "sd_pt", "sd_eta", "sd_phi", "sd_dR", "sd_symmetry",
    "sd_ch_ang_k1_b0.5", "sd_ch_ang_k1_b1", "sd_ch_ang_k1_b2", "sd_ch_ang_k2_b0",
)
jet_r = 0.4

@nb.jit
def match_sd(builder, sd_jet_constits, jet_constits):
    for sd_constits, constits in zip(sd_jet_constits, jet_constits):
        builder.begin_list()
        sd_constit_idx = list(range(len(sd_constits)))
        constit_idx = list(range(len(constits)))
        for sd_con in sd_constit_idx:
            is_matched = False
            for con in constit_idx:
                if sd_constits[sd_con].isclose(constits[con]):
                    is_matched = True
                    builder.append(con)
                    constit_idx.remove(con)
                    break
        builder.end_list()
    return builder

def get_softdrop_groomed_jets(constituents, jet_def, **kwargs):
    sd_clus_seq = fj.ClusterSequence(constituents, jet_def)
    sd_jet_data = sd_clus_seq.exclusive_jets_softdrop_grooming(**kwargs)
    sd_constit_data = sd_jet_data.constituents

    sd_jets = ak.zip(dict(
        pt = ak.enforce_type(sd_jet_data.ptsoftdrop, "float32"),
        eta = ak.enforce_type(sd_jet_data.etasoftdrop, "float32"),
        phi = ak.enforce_type(sd_jet_data.phisoftdrop, "float32"),
        e = ak.enforce_type(sd_jet_data.Esoftdrop, "float32"),
        m = ak.enforce_type(sd_jet_data.msoftdrop, "float32"),
        pz = ak.enforce_type(sd_jet_data.pzsoftdrop, "float32"),
        dR = ak.enforce_type(sd_jet_data.deltaRsoftdrop, "float32"),
        symmetry = ak.enforce_type(sd_jet_data.symmetrysoftdrop, "float32"),
        nconstituents = ak.enforce_type(ak.count(sd_constit_data.E, axis=1), "uint8")
    ), with_name="Momentum4D")
    
    sd_constit_vecs = ak.zip(dict(
        px = sd_constit_data.px,
        py = sd_constit_data.py,
        pz = sd_constit_data.pz,
        e  = sd_constit_data.E,
    ), with_name="Momentum4D")

    sd_constit_sorted_idx = ak.argsort(
        sd_constit_vecs.pt, axis=-1, ascending=False,
    )
    sd_constit_vecs = sd_constit_vecs[sd_constit_sorted_idx]
    sd_constit_indices = match_sd(ak.ArrayBuilder(), sd_constit_vecs, constituents).snapshot()
    sd_constituents = constituents[sd_constit_indices]

    sd_jets["ncharged"] = ak.enforce_type(ak.count_nonzero(sd_constituents.charge, axis=-1), "uint8")

    return sd_jets, sd_constituents

def to_jet_and_consitit_vectors(arr):
    jets = ak.zip(dict(
        pt = ak.enforce_type(arr.pt, "float32"), 
        eta = ak.enforce_type(arr.eta, "float32"), 
        phi = ak.enforce_type(arr.phi, "float32"), 
        e = ak.enforce_type(arr.e, "float32"),
        weight = ak.enforce_type(arr.weight, "float32"),
        ncharged = ak.enforce_type(arr.ncharged, "uint8"),
        nconstituents = ak.enforce_type(arr.nconstituents, "uint8"),
    ), with_name="Momentum4D")

    constituents = ak.zip({
        key : ak.enforce_type(
            arr[f"constit_{key}"], 
            "var*float32" if key != "charge" else "var*int8",
        ) 
        for key in ("pt", "eta", "phi", "e", "charge")
    }, with_name="Momentum4D")

    return jets, constituents

def calculate_angularities(jets, constituents):
    is_constit_charged = ak.fill_none(
        ak.mask(ak.ones_like(constituents.pt), constituents.charge != 0), 0
    )
    
    factors = {}
    factors["k1"] = constituents.pt/jets.pt 
    factors["k2"] = constituents.pt2/jets.pt2 
    factors["b2"] = constituents.deltaR2(jets)/(jet_r*jet_r)
    factors["b1"] = np.sqrt(factors["b2"])
    factors["b0.5"] = np.sqrt(factors["b1"])
    factors["b0"] = 1
    
    jets["nef"] =  ak.enforce_type(ak.nansum(
        ak.fill_none(ak.mask(factors["k1"], constituents.charge == 0), 0),
        axis = -1,
    ), "float32")

    for kappa, beta in ((1, 0), (1, 0.5), (1, 1), (1, 2), (2, 0)):
        factors[f"k{kappa}_b{beta}"] = is_constit_charged * factors[f"k{kappa}"]*factors[f"b{beta}"]
        jets[f"ch_ang_k{kappa}_b{beta}"] = ak.enforce_type(
            ak.nansum(factors[f"k{kappa}_b{beta}"], axis=-1), 
            "float32",
        )
    return jets

def process_table(table, **extra_fields):
    ak_array = ak.from_arrow(
        table, generate_bitmasks=True,
    )
    jets, constituents = to_jet_and_consitit_vectors(ak_array)
    sd_jets, sd_constituents = get_softdrop_groomed_jets(
        constituents, 
        fj.JetDefinition(fj.antikt_algorithm, jet_r, fj.E_scheme), 
        symmetry_cut=0.2, 
        R0=jet_r,
    )

    jets = calculate_angularities(jets, constituents)
    sd_jets = calculate_angularities(sd_jets, sd_constituents)
    
    for coord in ("pt", "eta", "phi"):
        jets[f"leading_constit_{coord}"] = getattr(constituents, coord)[:, 0]
        jets[f"subleading_constit_{coord}"] = getattr(constituents, coord)[:, 1]

    out_dict = {key : pa.array(getattr(jets, key)) for key in jets.fields}
    for key in sd_jets.fields:
        #print(key)
        out_dict[f"sd_{key}"] = pa.array(getattr(sd_jets, key))
    for key in constituents.fields:
        out_dict[f"constit_{key}"] = pa.array(getattr(constituents, key))
    for key in sd_constituents.fields:
        out_dict[f"sd_constit_{key}"] = pa.array(getattr(sd_constituents, key))

    print("------> Adding extra fields:", extra_fields)
    for key, val in extra_fields.items():
        broadcasted_arr, _ = ak.broadcast_arrays(val, jets.pt)
        out_dict[key] = pa.array(broadcasted_arr)

    return pa.RecordBatch.from_pydict(out_dict)

def preprocess_data(source_dir, file_name, **kwargs):
    #input_path = os.path.join(source_dir, "data.arrow")
    input_path = os.path.join(source_dir, file_name)
    buffer = pa.memory_map(input_path, "rb")
    output_rb = process_table(
        pa.ipc.open_file(buffer).read_all(), **kwargs
        
    )

    output_path = os.path.join(source_dir, f"preproc_{file_name}")
    with pa.OSFile(output_path, "wb") as sink:
        with pa.ipc.new_file(sink, output_rb.schema) as writer:
            writer.write_batch(output_rb)
    
    #return output_path

def preprocess_embedding_file(source_dir, sysvar, finput):
    root_dir = os.path.join(source_dir, "embedding", str(sysvar))
    extra_fields = {}
    extra_fields["is_data"] = False
    extra_fields["is_matched"] = 1 if "matches" in finput else 0
    outfile = os.path.join(root_dir, finput)
    print("Writing to file:", outfile)
    sink = pa.OSFile(outfile, "wb")
    writer = None
    njets = 0
    nbytes = 0
    for ipth, (pth_low, pth_high) in enumerate(zip(pth_bins[:-1], pth_bins[1:])):
        extra_fields["pth_bin"] = ipth
        infile = os.path.join(root_dir, f"ptHat{pth_low}to{pth_high}", finput)
        print("> Reading embedding file from:", infile)
        buffer = pa.memory_map(infile, "rb")
        output_rb = process_table(
            pa.ipc.open_file(buffer).read_all(),
            **extra_fields,
        )
        writer = writer or pa.ipc.new_file(sink, output_rb.schema)
        writer.write_batch(output_rb)
        njets += len(output_rb)
        nbytes += output_rb.nbytes
        print(f"---> Processed {njets} jets, wrote {nbytes/(1024*1024):.2f} mb to file...")
    if writer is not None:
        writer.close()

def preprocess_embedding(source_dir, sysvar):
    for infile in ("gen-matches", "reco-matches", "misses", "fakes"):
        preprocess_embedding_file(source_dir, sysvar, f"{infile}.arrow")

def to_tensordict(
    data_like, sim_like,
    columns = None,
    prefix = None, 
    max_chunksize = None,
):
    table = pa.concat_tables((data_like, sim_like))
    target = torch.concatenate((
        torch.ones(len(data_like), dtype=torch.float32), 
        torch.zeros(len(sim_like), dtype=torch.float32),
    ))

    columns = columns or table.column_names
    tdict = TensorDict(
        dict(
            input = torch.zeros((len(columns),), dtype = torch.float32), 
            target = torch.zeros((), dtype = torch.float32), 
            weight = torch.ones((), dtype = torch.float32),
            is_data = torch.zeros((), dtype = torch.bool), 
            is_matched = torch.zeros((), dtype = torch.int64),
            pth_bin = torch.zeros((), dtype = torch.int64),
        ), 
        batch_size=[],
    ).expand(len(table)).memmap_like(prefix=prefix)

    _input = table.select(columns)
    _target = torch.as_tensor(target, dtype=torch.float32)
    _weight = torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)
    _is_data = torch.as_tensor(table["is_data"].to_numpy(), dtype=torch.bool)
    _is_matched = torch.as_tensor(table["is_matched"].to_numpy(), dtype=torch.int64)
    _pth_bin = torch.as_tensor(table["pth_bin"].to_numpy(), dtype=torch.int64)
    
    pos = 0
    pbar = tqdm(total=len(_input))
    for chunk in _input.to_batches(max_chunksize=max_chunksize):
        chunk_size = len(chunk)
        _tdict = TensorDict(
            dict(
                input = torch.as_tensor(list(zip(*chunk.to_pydict().values())), dtype=torch.float32), 
                target = _target[pos : pos + chunk_size],
                weight = _weight[pos : pos + chunk_size],
                is_data = _is_data[pos : pos + chunk_size], 
                is_matched = _is_matched[pos : pos + chunk_size],
                pth_bin = _pth_bin[pos : pos + chunk_size], 
            ),
            batch_size=(chunk_size,),
            device="cpu",
        )

        tdict[pos : pos + chunk_size] = _tdict
        pos += chunk_size
        pbar.update(chunk_size)

    return tdict

def replace_table_column(table, name, array, new_name=None, **kwargs):
    col_index = table.schema.get_field_index(name)
    col_name = new_name or name
    column = pa.array(array, **kwargs)
    return table.set_column(col_index, col_name, column)

def make_datasets_for_unfolding(source_dir, sysvar):
    buffers = []

    if sysvar == SysVar.UNFOLDING_PRIOR: 
        root_dir = os.path.join(source_dir, "embedding", str(SysVar.NONE))
        buffers.append(pa.memory_map(os.path.join(root_dir, "reco-matches.arrow")))
        data_match_table = pa.ipc.open_file(buffers[-1]).read_all()
        buffers.append(pa.memory_map(os.path.join(root_dir, "fakes.arrow")))
        data_fake_table = pa.ipc.open_file(buffers[-1]).read_all()
        data_table = pa.concat_tables((data_match_table, data_fake_table))
 
        is_data_col = np.full_like(data_table["is_data"].to_numpy(), True)
        is_matched_col = np.full_like(data_table["is_matched"].to_numpy(), -1)
        data_table = replace_table_column(data_table, "is_data", is_data_col) 
        data_table = replace_table_column(data_table, "is_matched", is_matched_col)

        omniseq_wts = np.load("outputs/omnisequential/omniseq-wts-iter2.npz")
        omniseq_wt_keys =list(omniseq_wts.keys())
        closure_wt_index = omniseq_wt_keys[-2]
        closure_wts = pa.array(omniseq_wts[closure_wt_index], type=pa.float32())
        data_table = replace_table_column(data_table, "weight", closure_wts)
        out_dir = os.path.join(source_dir, "embedding", str(sysvar), "tensordicts")
    else:
        buffers.append(pa.memory_map(os.path.join(source_dir, "preproc_data.arrow")))
        data_table = pa.ipc.open_file(buffers[-1]).read_all()
        root_dir = os.path.join(source_dir, "embedding", str(sysvar))
        out_dir = os.path.join(root_dir, "tensordicts")

    os.makedirs(out_dir, exist_ok=True)
    print("Tensordicts will be written to", out_dir)

    buffers.append(pa.memory_map(os.path.join(root_dir, "reco-matches.arrow")))
    reco_match_table = pa.ipc.open_file(buffers[-1]).read_all()
    buffers.append(pa.memory_map(os.path.join(root_dir, "fakes.arrow")))
    reco_fakes_table = pa.ipc.open_file(buffers[-1]).read_all()
    reco_table = pa.concat_tables((reco_match_table, reco_fakes_table))

    detlvl_td = to_tensordict(
        data_table, 
        reco_table,
        columns=jet_columns, 
        prefix=os.path.join(out_dir, "det_lvl"), 
        max_chunksize=100000,
    )
 
    buffers.append(pa.memory_map(os.path.join(root_dir, "gen-matches.arrow")))
    gen_match_table = pa.ipc.open_file(buffers[-1]).read_all()
    buffers.append(pa.memory_map(os.path.join(root_dir, "misses.arrow")))
    gen_misses_table = pa.ipc.open_file(buffers[-1]).read_all()
    gen_table = pa.concat_tables((gen_match_table, gen_misses_table))
    
    is_data_col = np.full_like(gen_table["is_data"].to_numpy(), True)
    is_matched_col = np.full_like(gen_table["is_matched"].to_numpy(), -1)
    gen_table_data_like = replace_table_column(gen_table, "is_data", is_data_col) 
    gen_table_data_like = replace_table_column(gen_table_data_like, "is_matched", is_matched_col)

    partlvl_td = to_tensordict(
        gen_table_data_like, 
        gen_table, 
        columns=jet_columns, 
        prefix=os.path.join(out_dir, "part_lvl"), 
        max_chunksize=100000,
    )
    
def main(source_dir, sysvar, redo_preprocessing=True):
    if (sysvar != SysVar.UNFOLDING_PRIOR) and redo_preprocessing:
        print("Preprocessing data...")
        preprocess_data(
            source_dir, 
            "data.arrow", 
            is_data = True, 
            is_matched = -1, 
            pth_bin = -1,
        )
        print("Preprocessing embedding...")
        preprocess_embedding(source_dir, sysvar)
    print("Making tensordict for ML datasets", str(sysvar), "...")
    make_datasets_for_unfolding(source_dir, sysvar)
    
if __name__ == "__main__":
    source_dir : str = "./datasets/STAR_pp200GeV_production_2012/clustered_jets"
    main(source_dir, SysVar.NONE, redo_preprocessing=False)

