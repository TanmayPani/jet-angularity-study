from copy import deepcopy

import numpy as np
import numba as nb

import pyarrow as pa

import awkward as ak
import vector

import fastjet as fj
import pythia8mc as pythia

from utils.pdgid import charge

from cluster_data import (
    get_schema,
    process_jets,
    jets_to_rb_dict,
    inclusive_jets_sorted_by_pt,
)

vector.register_awkward()

@nb.jit
def charge_from_pdgid(builder, pid_arr):
    for pids in pid_arr:
        builder.begin_list()
        for pid in pids:
            builder.append(charge(pid))
        builder.end_list()
    return builder

def select_particles_for_clustering(prt, con_kt_min=0.2, con_abs_eta_max = 1.0):
    prt = prt[~ak.is_none(prt)]

    final_state_sel = prt.status > 0
    prt_kt_sel = prt.p.pt > con_kt_min
    prt_eta_sel = np.abs(prt.p.eta) < 1

    prt_sel = final_state_sel & prt_kt_sel & prt_eta_sel
    return ak.drop_none(ak.mask(prt, prt_sel), axis=1)
    
def make_cluster_sequence_input(candidates, make_highlevel=True, **kwargs):
    candidates = select_particles_for_clustering(candidates, **kwargs)
    candidates = candidates[ak.num(candidates, axis=1) > 2]
    if not make_highlevel:
        return candidates
    coordinates = ["px", "py", "pz", "e"]
    coordinate_keys = [("p", coo) for coo in coordinates]
    candidates = ak.with_field(
        candidates, 
        charge_from_pdgid(ak.ArrayBuilder(), candidates.id).snapshot(), 
        where =("charge"),
    )
    prt_fields = deepcopy(candidates.fields)
    prt_fields.remove("p")
    coordinates = coordinates + prt_fields
    keys = coordinate_keys + [f for f in prt_fields]

    return ak.zip(dict(zip(coordinates, [candidates[key] for key in keys])), with_name="Momentum4D")
    
def cluster_batch(candidates, jet_definition, weights = None, **kwargs):
    candidates = make_cluster_sequence_input(candidates)

    cluster_seq = fj.ClusterSequence(candidates, jet_definition)
    jets, constituents = inclusive_jets_sorted_by_pt(cluster_seq)
    jets, constituents = process_jets(
        jets, 
        constituents, 
        jet_pt_min=10.0, 
    )

    n_jets_sel = ak.num(jets, axis=1) > 0
    jets = jets[n_jets_sel]
    constituents = constituents[n_jets_sel]

    if weights is not None:
        jets["weight"], _ = ak.broadcast_arrays(weights, jets.px)
    else:
        jets["weight"] = ak.ones_like(jets.px)

    return jets_to_rb_dict(jets, constituents)

def generate(pythia_gen, n_events, batch_size=-1, error_mode="none"):
    batch_size = n_events if batch_size < 0 else batch_size
    n_events_to_generate = n_events
    i_batch = 0
    while n_events_to_generate > 0:
        batch_size = n_events_to_generate if batch_size > n_events_to_generate else batch_size
        n_events_to_generate -= batch_size
        i_batch += 1
        print(f"{i_batch} batches generated...")
        yield pythia_gen.nextBatch(batch_size, error_mode)


def setup(
    config_file : str | None = None, 
    num_threads : int | None = None, 
    phase_space_bias : int | None = None,
    pt_hat_range : tuple[float, float] | None = None, 
    seed : int | None = None,
): 
    if num_threads is None:
        _pythia = pythia.Pythia()
    else:
        _pythia = pythia.PythiaParallel()
        _pythia.readString(f"Parallelism:numThreads = {num_threads}")
        _pythia.readString("Parallelism:processAsync = off")

    _pythia.readString("Print:quiet = on")
    if config_file is not None:
        _pythia.readFile(config_file)

    if seed is not None:
        _pythia.readString("Random:setSeed = on")
        _pythia.readString(f"Random:seed = {seed}")
    
    _pythia.readString("Beams:idA = 2212")
    _pythia.readString("Beams:idB = 2212")
    _pythia.readString("Beams:eCM = 200.")
    _pythia.readString("HardQCD:all = on")
    
    if pt_hat_range is not None:
        pt_hat_min, pt_hat_max = pt_hat_range
        _pythia.readString(f"PhaseSpace:pTHatMin = {pt_hat_min}")
        _pythia.readString(f"PhaseSpace:pTHatMax = {pt_hat_max}")
    
    if phase_space_bias is not None:
        _pythia.readString("PhaseSpace:bias2Selection = on")
        _pythia.readString(f"PhaseSpace:bias2SelectionPow = {phase_space_bias}")
        _pythia.readString("PhaseSpace:bias2SelectionRef = 11.")

    _pythia.init()
    return _pythia

def worker(
    islot, 
    jet_definition, 
    n_events, 
    batch_size, 
    output_file, 
    seed=None, 
    **kwargs,
):
    if islot is not None:
        output_file = f"{output_file}_{islot}.arrow"
    else:
        output_file = f"{output_file}.arrow"

    sink = pa.OSFile(output_file, "wb")
    writer = pa.ipc.RecordBatchFileWriter(sink, get_schema())

    _pythia = setup(
        config_file = "./runtime-files/pythia8_detroit_tune.txt", 
        num_threads = None,
        #phase_space_bias = 4,
        pt_hat_range = (11.0, -1), 
        seed = seed,
    )

    for batch in generate(_pythia, n_events, batch_size=batch_size):
        pa_record_batch = cluster_batch(
            batch.prt, 
            jet_definition, 
        )
        writer.write(pa_record_batch)

    writer.close()
    get_schema.cache_clear()


if __name__ == "__main__": 
    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.E_scheme)
    worker(
        None,
        jet_def, 
        1000000, 
        100000, 
        "./datasets/STAR_pp200GeV_production_2012/Pythia8_pp200GeV_r0.4",
    )




