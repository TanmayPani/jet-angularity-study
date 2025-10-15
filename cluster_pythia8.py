from copy import deepcopy

import numpy as np
import numba as nb

import pyarrow as pa

import awkward as ak
import vector

import fastjet as fj
import pythia8mc as pythia

from utils import clustering
from utils.pdgid import charge

vector.register_awkward()

@nb.jit
def charge_from_pdgid(builder, pid_arr):
    for pids in pid_arr:
        builder.begin_list()
        for pid in pids:
            builder.append(charge(pid))
        builder.end_list()
    return builder

def select_particles_for_clustering(events, con_kt_min=0.2, con_abs_eta_max = 1.0):

    invalids_ = ak.is_none(events.prt)
    #n_events_in_ = ak.num(events, axis=0)
    events = events[~invalids_]
    #print(f"{ak.sum(invalids_)} rejected out of {n_events_in_} events")

    final_state_sel_ = events.prt.status > 0
    prt_kt_sel_ = events.prt.p.pt > con_kt_min
    prt_eta_sel_ = np.abs(events.prt.p.eta) < 1

    prt_sel_ = final_state_sel_ & prt_kt_sel_ & prt_eta_sel_
    events["prt"] = ak.drop_none(ak.mask(events.prt, prt_sel_), axis=1)

    n_particles_sel_ = ak.num(events.prt, axis=1) > 0
    #n_events_in_ = ak.num(events, axis=0)
    #print(f"{ak.sum(~n_particles_sel_)} rejected out of {n_events_in_} events")
    events = events[n_particles_sel_]

    return events

def make_cluster_sequence_input(events, make_highlevel=True, **kwargs):
    events = select_particles_for_clustering(events, **kwargs)
    if not make_highlevel:
        return events
    coordinates_ = ["px", "py", "pz", "e"]
    coordinate_keys_ = [("prt", "p", coo) for coo in coordinates_]
    #coordinates_.append("charge")
    #coordinate_keys_.append(tuple(("prt", "charge")))
    events = ak.with_field(events, charge_from_pdgid(ak.ArrayBuilder(), events.prt.id).snapshot(), where =("prt", "charge"))
    prt_fields_ = deepcopy(events.prt.fields)
    prt_fields_.remove("p")
    coordinates_ = coordinates_ + prt_fields_
    keys_ = coordinate_keys_ + [("prt", f) for f in prt_fields_]

    candidates_ = ak.zip(dict(zip(coordinates_, [events[key] for key in keys_])), with_name="Momentum4D")

    return candidates_
    
def setup_generator(seed = 0) -> pythia.Pythia:
    pythia_ = pythia.Pythia()

    pythia_.readFile("runtime_files/pythia8_detroit_tune.txt")
    pythia_.readString("Beams:idA = 2212")
    pythia_.readString("Beams:idB = 2212")
    pythia_.readString("Beams:eCM = 200.")
    pythia_.readString("HardQCD:all = on")
    pythia_.readString("PhaseSpace:pTHatMin = 11.0")


    pythia_.readString("Random:setSeed = on")
    pythia_.readString(f"Random:seed = {seed}")

    return pythia_

def generate(n_events, batch_size=-1, error_mode="none", seed=0):
    pythia_ = setup_generator(seed=0)
    pythia_.init()
    batch_size = n_events if batch_size < 0 else batch_size
    n_events_to_generate_ = n_events
    i_batch = 0
    while n_events_to_generate_ > 0:
        batch_size = n_events_to_generate_ if batch_size > n_events_to_generate_ else batch_size
        n_events_to_generate_ -= batch_size
        i_batch += 1
        print(f"{i_batch} batches generated...")
        yield pythia_.nextBatch(batch_size, error_mode)

def cluster_batch(events, jet_definition, con_kt_min=0.2, hc_kt_min=2.0, cs_pt_min=2.0, do_hc_mode=True):
    events_ = make_cluster_sequence_input(events, con_kt_min=con_kt_min)
    cluster_seq_ = fj.ClusterSequence(events_, jet_definition)
    jets_, constituents_ = clustering.inclusive_jets_sorted_by_pt(cluster_seq_, min_pt=cs_pt_min)
    jets_, constituents_ = clustering.process_jets(
        jets_, constituents_, jet_pt_min=10.0, do_hc_mode=do_hc_mode, hc_kt_min=hc_kt_min
    )

    try:
        jets_["weight"], _ = ak.broadcast_arrays(events.info.weights, jets_.px)
    except ValueError:
        jets_["weight"] = ak.ones_like(jets_.px)

    return clustering.jets_to_rb_dict(jets_, constituents_, do_hc_mode=do_hc_mode)

def worker(n_events, batch_size, jet_definition, output_file, slot_id=None, seed=0, **kwargs):
    if slot_id is not None:
        output_file = f"{output_file}_{slot_id}.arrow"
    else:
        output_file = f"{output_file}.arrow"

    sink = pa.OSFile(output_file, "wb")
    writer = pa.ipc.RecordBatchFileWriter

    for batch in generate(n_events, batch_size=batch_size, seed=seed):
        pa_record_batch_ = cluster_batch(batch, jet_definition)
        if isinstance(writer, type):
            writer = writer(sink, pa_record_batch_.schema)

        writer.write(pa_record_batch_)

    if isinstance(writer, pa.ipc.RecordBatchFileWriter):
        writer.close()



if __name__ == "__main__": 

    jet_def = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.BIpt2_scheme)

    worker(1000000, 10000, jet_def, "outputs/pythia8")



    #print(select_particles_for_clustering(arr)[0:2].prt.fields)
    #print(select_particles_for_clustering(arr)[0:2].prt.p.fields)



