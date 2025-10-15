import os
from typing import Any, Optional
from concurrent.futures import ProcessPoolExecutor, wait
import tqdm

import pyarrow as pa 
from pyarrow import compute as pc 

import torch
from tensordict import TensorDict 

def arrow_to_tensordict(
    table : pa.Table,
    target : Optional[torch.Tensor] = None,
    weight : Optional[torch.Tensor] = None,
    *,
    labels : dict[str, torch.Tensor] = {},
    dtype : torch.dtype = torch.float32,
    columns : Optional[list[str]] = None,
    prefix : Optional[str] = None,
    max_write_chunk : Optional[int] = None,
):
    columns = columns if columns is not None else table.column_names
    input_table = table.select(columns)

    data = TensorDict(
        dict(
            input = torch.zeros((len(columns),), dtype=dtype),
            target = torch.zeros((), dtype=dtype),
            weight = torch.ones((), dtype=dtype),
            **{key : torch.zeros((), dtype=label.dtype) for key, label in labels.items()},
        ),
        batch_size=[],
    ).expand(len(table)).memmap_like(prefix=prefix)

    rbatches = input_table.to_batches(max_chunksize=max_write_chunk)

    target = target if target is not None else torch.zeros(len(input_table))
    weight = weight if weight is not None else torch.ones(len(input_table))
   
    i = 0
    pbar = tqdm.tqdm(total=len(input_table))
    for rbatch in rbatches:
        batch_size = len(rbatch)
        pbar.update(batch_size)
        batch  = TensorDict(
            dict(
                input = torch.as_tensor(list(zip(*rbatch.to_pydict().values())), dtype=dtype),
                target = target[i : i + batch_size],
                weight = weight[i : i + batch_size],
                **{k : v[i:i+batch_size] for k, v in labels.items()},
            ),
            batch_size=(batch_size,),
            device="cpu",
        )

        data[i : i + batch_size] = batch
        i += batch_size
    return data

def add_constit_slice_column(jet_table, consit_col_name, new_col_name, start, stop=None):
    if stop is None:
        stop = start+1
    carr = pc.list_slice(jet_table[consit_col_name], start, stop=stop).combine_chunks().flatten()
    return jet_table.append_column(new_col_name, carr)

def add_extra_columns(table : pa.Table) -> pa.Table:
    table = add_constit_slice_column(table, "constit_pt", "leading_constit_pt", 0)
    table = add_constit_slice_column(table, "constit_eta", "leading_constit_eta", 0)
    table = add_constit_slice_column(table, "constit_phi", "leading_constit_phi", 0)

    table = add_constit_slice_column(table, "constit_pt", "subleading_constit_pt", 1)
    table = add_constit_slice_column(table, "constit_eta", "subleading_constit_eta", 1)
    table = add_constit_slice_column(table, "constit_phi", "subleading_constit_phi", 1)

    return table

def make_mmap_tensor_files(
    in_dir : str, 
    out_dir : str, 
    jet_columns : Optional[list[str]] = None,
    itask : int = 0,
    is_data : bool = False,
    is_matched : bool = True,
    pth_bin_id : int = 0,
):
    #sys.stdout = utils.StreamToLogger(logging.getLogger("STDOUT"), f"logs/task_{itask}.log" , logging.DEBUG, mode="w"),
    #sys.stderr = utils.StreamToLogger(logging.getLogger("STDERR"), f"logs/task_{itask}.err" , logging.ERROR, mode = "w"),

    print(f"reading from {in_dir}")
    buffer = pa.memory_map(in_dir, "rb")
    table  = pa.ipc.open_file(buffer).read_all()
    table = add_extra_columns(table)

    weight = torch.as_tensor(
        table["weight"].to_numpy(),
        dtype=torch.float32,
    )

    target = torch.ones(len(table), dtype=torch.float32) if is_data else torch.zeros(len(table), dtype=torch.float32)
    labels = dict(
        is_matched = torch.ones(len(table), dtype=torch.float32) if is_matched else torch.zeros(len(table), dtype=torch.float32),
        pth_bin = torch.full((len(table),), pth_bin_id, dtype = torch.float32),
    )

    print(f"writing to {out_dir}")
    arrow_to_tensordict(
        table,
        target=target,
        weight=weight,
        labels=labels,
        columns=jet_columns,
        prefix=out_dir,
        max_write_chunk=100000,
    )

def make_mmap_tensor(
    in_paths : list[str],
    targets : list[float | int],
    columns : Optional[list[str]] = None,
    *,
    prefix : Optional[str] = None, 
    per_table_metadata : dict[pa.Field, list[Any]] = {},
):

    buffers = [pa.memory_map(path, "rb") for path in in_paths]
    tables  = [add_extra_columns(pa.ipc.open_file(buffer).read_all()) for buffer in buffers]
    per_table_metadata[pa.field("target", pa.float64())] = targets
    per_table_metadata[pa.field("table_id", pa.int64())] = list(range(len(tables))) 
    for field, md_list in per_table_metadata.items():
        assert len(md_list) == len(tables)
        mds = [[md]*len(tbl) for md, tbl in zip(md_list, tables)]
        #print(field)
        tables = [tbl.append_column(field, [md]) for md, tbl in zip(mds, tables)]

    table = pa.concat_tables(tables)

    target = torch.as_tensor(
        table["target"].to_numpy(),
        dtype=torch.float32,
    )

    weight = torch.as_tensor(
        table["weight"].to_numpy(),
        dtype=torch.float32,
    )

    labels = dict(
        **{k.name : torch.as_tensor(table[k.name].to_numpy()) for k in per_table_metadata.keys()}
    )

    labels.pop("target")

    print(f"writing to {prefix}")
    return arrow_to_tensordict(
        table,
        target=target,
        weight=weight,
        labels=labels,
        columns=columns,
        prefix=prefix,
        max_write_chunk=100000,
    )

if __name__ == "__main__": 
    source_dir = "partitioned_datasets/nominal"
    pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    pth_bin_dirs = [f"{source_dir}/arrow_data/ptHat{pth_low}to{pth_high}" 
        for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])]
    
    jet_columns = [
        "pt", "eta", "phi", "nef",
        "ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0",
        "leading_constit_pt", "leading_constit_eta", "leading_constit_phi",
        "subleading_constit_pt", "subleading_constit_eta", "subleading_constit_phi",
        "hc_pt", "hc_eta", "hc_phi",
        "hc_ch_ang_k1_b0.5", "hc_ch_ang_k1_b1", "hc_ch_ang_k1_b2", "hc_ch_ang_k2_b0",
    ]

    detlvl_infile_list = ( 
            [f"{source_dir}/arrow_data/jets-conPtMin0.2.arrow"] +
            [f"{dir}/reco-matches.arrow" for dir in pth_bin_dirs] + 
            [f"{dir}/fakes.arrow" for dir in pth_bin_dirs]
    )
   
    detlvl_targets = [1.] + [0.]*len(pth_bin_dirs) + [0.]*len(pth_bin_dirs)

    detlvl_per_tbl_md = {}
    detlvl_per_tbl_md[pa.field("is_data", pa.bool_())] = (
            [True] + [False]*len(pth_bin_dirs) + [False]*len(pth_bin_dirs)
    )
    detlvl_per_tbl_md[pa.field("is_matched", pa.int64())] = (
            [-1] + [1]*len(pth_bin_dirs) + [0]*len(pth_bin_dirs)
    )
    detlvl_per_tbl_md[pa.field("pth_bin", pa.int64())] = (
            [-1] + list(range(len(pth_bin_dirs))) + list(range(len(pth_bin_dirs))) 
    )
    detlvl_output_path = f"{source_dir}/det_lvl/all"
    
    if not os.path.exists(detlvl_output_path):
        os.makedirs(detlvl_output_path)

    detlvl_mmap_tensor = make_mmap_tensor(
        detlvl_infile_list,
        detlvl_targets,
        columns=jet_columns,
        prefix=detlvl_output_path,
        per_table_metadata=detlvl_per_tbl_md,
    )

    print(detlvl_mmap_tensor)

    reco_match_infile_list = 2*[f"{dir}/reco-matches.arrow" for dir in pth_bin_dirs] 
    reco_match_targets = [1.]*len(pth_bin_dirs) + [0.]*len(pth_bin_dirs)

    reco_match_per_tbl_md = {}
    reco_match_per_tbl_md[pa.field("is_data", pa.bool_())] = (
            [True]*len(pth_bin_dirs) + [False]*len(pth_bin_dirs)
    )
    reco_match_per_tbl_md[pa.field("pth_bin", pa.int64())] = (
            list(range(len(pth_bin_dirs))) + list(range(len(pth_bin_dirs))) 
    )
    reco_match_output_path = f"{source_dir}/reco_match/all"
    
    if not os.path.exists(reco_match_output_path):
        os.makedirs(reco_match_output_path)

    reco_match_mmap_tensor = make_mmap_tensor(
        reco_match_infile_list,
        reco_match_targets,
        columns=jet_columns,
        prefix=reco_match_output_path,
        per_table_metadata=reco_match_per_tbl_md,
    )

    print(reco_match_mmap_tensor)


    partlvl_infile_list =  (
        [f"{dir}/gen-matches.arrow" for dir in pth_bin_dirs] +
        [f"{dir}/misses.arrow" for dir in pth_bin_dirs]+
        [f"{dir}/gen-matches.arrow" for dir in pth_bin_dirs] +
        [f"{dir}/misses.arrow" for dir in pth_bin_dirs]
    )
    
    partlvl_targets = (
            [1.]*len(pth_bin_dirs) + [1.]*len(pth_bin_dirs) + 
            [0.]*len(pth_bin_dirs) + [0.]*len(pth_bin_dirs)
    )
    partlvl_per_tbl_md = {}
    partlvl_per_tbl_md[pa.field("is_data", pa.bool_())] = (
        [True]*len(pth_bin_dirs) + [True]*len(pth_bin_dirs) + 
        [False]*len(pth_bin_dirs) + [False]*len(pth_bin_dirs)
    )
    partlvl_per_tbl_md[pa.field("is_matched", pa.int64())] = (
        [-1]*len(pth_bin_dirs) + [-1]*len(pth_bin_dirs) + 
        [1]*len(pth_bin_dirs) + [0]*len(pth_bin_dirs)
    )

    partlvl_per_tbl_md[pa.field("pth_bin", pa.int64())] = (
        list(range(len(pth_bin_dirs))) + list(range(len(pth_bin_dirs))) + 
        list(range(len(pth_bin_dirs))) + list(range(len(pth_bin_dirs))) 
    )

    partlvl_output_path = f"{source_dir}/part_lvl/all"

    if not os.path.exists(partlvl_output_path):
        os.makedirs(partlvl_output_path)

    partlvl_mmap_tensor = make_mmap_tensor(
        partlvl_infile_list,
        partlvl_targets,
        columns=jet_columns,
        prefix=partlvl_output_path,
        per_table_metadata=partlvl_per_tbl_md,
    )

    print(partlvl_mmap_tensor)
 
    gen_match_infile_list = 2*[f"{dir}/gen-matches.arrow" for dir in pth_bin_dirs] 
    gen_match_targets = [1.]*len(pth_bin_dirs) + [0.]*len(pth_bin_dirs)

    gen_match_per_tbl_md = {}
    gen_match_per_tbl_md[pa.field("is_data", pa.bool_())] = (
            [True]*len(pth_bin_dirs) + [False]*len(pth_bin_dirs)
    )
    gen_match_per_tbl_md[pa.field("pth_bin", pa.int64())] = (
            list(range(len(pth_bin_dirs))) + list(range(len(pth_bin_dirs))) 
    )
    gen_match_output_path = f"{source_dir}/gen_match/all"
    
    if not os.path.exists(gen_match_output_path):
        os.makedirs(gen_match_output_path)

    gen_match_mmap_tensor = make_mmap_tensor(
        gen_match_infile_list,
        gen_match_targets,
        columns=jet_columns,
        prefix=gen_match_output_path,
        per_table_metadata=gen_match_per_tbl_md,
    )

    print(gen_match_mmap_tensor)
  
    exit()

    with ProcessPoolExecutor() as executor:
        futures = []
        for itask, (in_dir, out_dir) in enumerate(zip(in_dirs, out_dirs)):
            futures.append(
                executor.submit(
                    make_mmap_tensor_files,
                    itask,
                    in_dir,
                    out_dir,
                    jet_columns,
                )
            )
        wait(futures)

    
    



    







