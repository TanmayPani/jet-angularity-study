# import glob
from enum import Enum

import uproot
import pyarrow as pa

import fastjet as fj

import vector
import awkward as ak

import numpy as np
import numba as nb

# import cupy as cpu
from utils import clustering
# import sys

# from concurrent.futures import ProcessPoolExecutor, wait
# import logging

vector.register_awkward()

class SysVar(Enum):
    NONE=0
    TOWER_ET_CORRECTION=1
    TRACK_EFFICIENCY=2

def apply_hadronic_correction_sys_var(events, hadr_corr_frac=0.5):
    tower_dE = events["towers._RawE"] - events["towers._E"]
    events["towers._E"] = events["towers._E"] - hadr_corr_frac*tower_dE
    mass_array = ak.full_like(events["towers._E"], 0.13957)
    tower_p2 = events["towers._E"]**2 - mass_array**2
    tower_p2 = ak.fill_none(ak.mask(tower_p2, tower_p2 > 0), value=0)
    tower_p = np.sqrt(tower_p2)
    events["towers._Pt"] = tower_p/np.cosh(events["towers._Eta"])
    return events

@nb.jit
def apply_flat_track_pt_factors(builder, event_track_pt, flat_rel_factors):
    i_trk = 0 
    for track_pt in event_track_pt:
        builder.begin_list()
        for pt in track_pt:
            #builder.append(pt + pt*flat_rel_factors[i_trk])
            if flat_rel_factors[i_trk] > 0.04:
                builder.append(True)
            else:
                builder.append(False)
            i_trk += 1
        builder.end_list()
    return builder

def get_tracking_efficiency_sys_var_mask(events):
    n_tot_trk = ak.sum(ak.count(events["tracks._Pt"], axis=0))
    #flat_factors = np.random.default_rng().uniform(-0.04, 0.04, n_tot_trk)
    flat_factors = np.random.default_rng().random(n_tot_trk)
    return apply_flat_track_pt_factors(ak.ArrayBuilder(), events["tracks._Pt"], flat_factors).snapshot()

def process_events(events, con_kt_min=None, sys_var=SysVar.NONE):
    coordinates = ["pt", "eta", "phi", "e", "charge"]
    branchNames = ["Pt", "Eta", "Phi", "E", "Charge"]

    tracks = ak.zip(
        dict(
            zip(
                coordinates,
                [events[f"tracks._{branch}"] for branch in branchNames],
            )
        ),
        with_name="Momentum4D",
    )

    if sys_var == SysVar.TRACK_EFFICIENCY:
        track_mask = get_tracking_efficiency_sys_var_mask(events)
        tracks = tracks[track_mask]

    if sys_var == SysVar.TOWER_ET_CORRECTION:
        events = apply_hadronic_correction_sys_var(events)

    towers = ak.zip(
        dict(
            zip(
                coordinates,
                [events[f"towers._{branch}"] for branch in branchNames],
            )
        ),
        with_name="Momentum4D",
    )

    if con_kt_min is not None:
        track_mask = tracks.pt > con_kt_min
        tower_mask = towers.et > con_kt_min
        # track_mask[0:3].show()
        # tracks[0:3].show()
        tracks = ak.drop_none(ak.mask(tracks, track_mask), axis=-1)
        # tracks[0:3].show()
        towers = ak.drop_none(ak.mask(towers, tower_mask), axis=-1)

    isVzGood = np.abs(events._pVtx_Z) < 30.0
    isMaxTrackPtOk = ak.fill_none(ak.max(tracks.pt, axis=-1), value=0) < 30.0
    isMaxTowEtOk = ak.fill_none(ak.max(towers.et, axis=-1), value=0) < 30.0
    isHT2Like = clustering.is_event_ht2(ak.ArrayBuilder(), events).snapshot()
    eventFilter = isVzGood & isMaxTrackPtOk & isMaxTowEtOk & isHT2Like

    candidates = ak.concatenate([tracks, towers], axis=-1)
    print(f"---Got {len(events)} ({len(candidates)}) events...")
    candidates = ak.drop_none(ak.mask(candidates, eventFilter))
    print(f"---Left with {len(candidates)} after cuts...")

    #if sys_var == SysVar.NONE:
    return candidates

    


def cluster_batch(
    events,
    jet_definition,
    con_kt_min=None,
    cs_pt_min=2.0,
    batch_weight=1.0,
    do_hc_mode=False,
    hc_kt_min=2.0,
    sys_var_type=SysVar.NONE
):
    candidates = process_events(events, con_kt_min=con_kt_min, sys_var=sys_var_type)
    clusterSeq = fj.ClusterSequence(candidates, jet_definition)
    jets, constituents = clustering.inclusive_jets_sorted_by_pt(clusterSeq, min_pt=cs_pt_min)
    jets, constituents = clustering.process_jets(
        jets, constituents, jet_pt_min=10.0, do_hc_mode=do_hc_mode, hc_kt_min=hc_kt_min
    )
    jets["weight"], _ = ak.broadcast_arrays(batch_weight, jets.pt)
    return clustering.jets_to_rb_dict(jets, constituents, do_hc_mode=do_hc_mode)


def worker(
    jet_definition,
    data_files,
    output_file_name,
    con_kt_min=None,
    cs_pt_min=2.0,
    nfiles_per_batch=100,
    nbatches=-1,
    max_events_per_batch=-1,
    do_hc_mode=False,
    hc_kt_min=2.0,
    sys_var_type = SysVar.NONE
):
    sink = pa.OSFile(output_file_name, "wb")
    writer = None

    nFiles = len(data_files)
    nBatches = nFiles // nfiles_per_batch + 1
    batchNumber = 0
    for startFile in range(0, len(data_files), nfiles_per_batch):
        batchNumber += 1
        endFile = startFile + nfiles_per_batch
        events = uproot.concatenate(data_files[startFile:endFile])
        if max_events_per_batch >= 0:
            events = events[:max_events_per_batch]
        print(
            f"Processing batch [{batchNumber}/{nBatches}], file # {startFile} to {endFile}, with {len(events)} events"
        )

        record_batch = cluster_batch(
            events,
            jet_definition,
            cs_pt_min=cs_pt_min,
            con_kt_min=con_kt_min,
            do_hc_mode=do_hc_mode,
            hc_kt_min=hc_kt_min,
            sys_var_type=sys_var_type
        )

        if writer is None:
            writer = pa.ipc.new_file(sink, record_batch.schema)
        else:
            writer.write(record_batch)

        if batchNumber == nbatches:
            break
    if writer:
        writer.close()


if __name__ == "__main__":
    do_test = False
    con_kt_min = 0.2
    do_hc_mode = True if con_kt_min < 0.21 else False
    print(do_hc_mode)
    
    #sys_var_type = SysVar.TOWER_ET_CORRECTION
    sys_var_type = SysVar.TRACK_EFFICIENCY

    sys_var_mod = ""
    if sys_var_type == SysVar.TOWER_ET_CORRECTION:
        sys_var_mod = "_wTowerHadrCorrSys"
    if sys_var_type == SysVar.TRACK_EFFICIENCY:
        sys_var_mod = "_wTrackPtSys"

    badRunsList = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime_files/runLists/pp200_production_2012_BAD_Issac.list"
    dataFolderPath = (
        "/run/media/tanmaypani/Samsung-1tb/data/pp200_production_2012/2024-03-12/Events"
    )
    # runListFile = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_goodRuns.list"
    dataFileList, good_runs = clustering.read_input_files(
        dataFolderPath, badRunsList, run_number_token_id=0
    )

    jetDef = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.BIpt2_scheme)
    outputFileName = (
        "outputs/jets-test.arrow"
        if do_test
        else f"outputs/jets_conPtMin{con_kt_min}{sys_var_mod}.arrow"
    )
    nbatches = 1 if do_test else -1
    nfiles_per_batch = 1 if do_test else 150
    max_events_per_batch = 10 if do_test else -1

    worker(
        jetDef,
        dataFileList,
        outputFileName,
        con_kt_min=con_kt_min,
        nfiles_per_batch=nfiles_per_batch,
        nbatches=nbatches,
        max_events_per_batch=max_events_per_batch,
        do_hc_mode=do_hc_mode,
        sys_var_type=sys_var_type
    )

    print("Done!")
