# import glob
import uproot
import pyarrow as pa

import fastjet as fj

import vector
import awkward as ak

import numpy as np
# import numba.cuda

# import cupy as cpu
import utils
# import sys

# from concurrent.futures import ProcessPoolExecutor, wait
# import logging

vector.register_awkward()


def process_events(events, con_kt_min=None):
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
    isHT2Like = utils.is_event_ht2(ak.ArrayBuilder(), events).snapshot()
    eventFilter = isVzGood & isMaxTrackPtOk & isMaxTowEtOk & isHT2Like

    candidates = ak.concatenate([tracks, towers], axis=-1)
    print(f"---Got {len(events)} ({len(candidates)}) events...")
    candidates = ak.drop_none(ak.mask(candidates, eventFilter))
    print(f"---Left with {len(candidates)} after cuts...")

    return candidates


def cluster_batch(
    events,
    jet_definition,
    con_kt_min=None,
    cs_pt_min=2.0,
    batch_weight=1.0,
    do_hc_mode=False,
    hc_kt_min=2.0,
):
    candidates = process_events(events, con_kt_min=con_kt_min)
    clusterSeq = fj.ClusterSequence(candidates, jet_definition)
    jets, constituents = utils.inclusive_jets_sorted_by_pt(clusterSeq, min_pt=cs_pt_min)
    jets, constituents = utils.process_jets(
        jets, constituents, jet_pt_min=10.0, do_hc_mode=do_hc_mode, hc_kt_min=hc_kt_min
    )
    jets["weight"], _ = ak.broadcast_arrays(batch_weight, jets.pt)
    return utils.jets_to_rb_dict(jets, constituents, do_hc_mode=do_hc_mode)


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
    badRunsList = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_BAD_Issac.list"
    dataFolderPath = (
        "/run/media/tanmaypani/Samsung-1tb/data/pp200_production_2012/2024-03-12/Events"
    )
    # runListFile = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_goodRuns.list"
    dataFileList, good_runs = utils.read_input_files(
        dataFolderPath, badRunsList, run_number_token_id=0
    )

    jetDef = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.BIpt2_scheme)
    outputFileName = (
        "outputs/jets-test.arrow"
        if do_test
        else f"outputs/jets-conPtMin{con_kt_min}.arrow"
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
    )

    print("Done!")
