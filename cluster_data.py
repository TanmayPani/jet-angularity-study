import sys
import logging
from pathlib import Path
from enum import Enum
from functools import cache

import numpy as np
import numba as nb

import pyarrow as pa
import awkward as ak
import uproot
import vector
import fastjet as fj

from systematics import (
    SysVar,
    apply_hadronic_correction_sys_var,
    # apply_flat_track_pt_factors,
    get_tracking_efficiency_sys_var_mask,
)

vector.register_awkward()

bad_run_list: str = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_BAD_Issac.list"
data_folder_path: str = (
    "/run/media/tanmaypani/Samsung-1tb/data/pp200_production_2012/2024-03-12/Events"
)
output_path_prefix: str = "/home/tanmaypani/star-workspace/jet-angularity-study/datasets/STAR_pp200GeV_production_2012/clustered_jets"

jet_col_fields = {
    "weight": pa.field("weight", pa.float32()),
    "pt": pa.field("pt", pa.float32()),
    "eta": pa.field("eta", pa.float32()),
    "phi": pa.field("phi", pa.float32()),
    "e": pa.field("e", pa.float32()),
    "ncharged": pa.field("ncharged", pa.uint8()),
    "nconstituents": pa.field("nconstituents", pa.uint8()),
}
constit_col_fields = {
    "pt": pa.field("constit_pt", pa.list_(pa.float32())),
    "eta": pa.field("constit_eta", pa.list_(pa.float32())),
    "phi": pa.field("constit_phi", pa.list_(pa.float32())),
    "e": pa.field("constit_e", pa.list_(pa.float32())),
    "charge": pa.field("constit_charge", pa.list_(pa.int8())),
}


@cache
def get_schema(metadata=None):
    jet_fields = list(jet_col_fields.values())
    constit_fields = list(constit_col_fields.values())
    schema = pa.schema(jet_fields + constit_fields, metadata=metadata)
    print("Pyarrow schema for writing:", schema)
    return schema


def jets_to_rb_dict(
    jets: ak.Array,
    constituents: ak.Array,
) -> pa.RecordBatch:
    rec_batch_dict = {}
    for col_name, col_field in jet_col_fields.items():
        arr = getattr(jets, col_name)
        rec_batch_dict[col_field.name] = pa.array(
            ak.flatten(arr, axis=1),
            col_field.type,
        )
    for col_name, col_field in constit_col_fields.items():
        arr = getattr(constituents, col_name)
        # print("constit", col_name, col_field, arr)
        rec_batch_dict[col_field.name] = pa.array(
            ak.flatten(arr, axis=1),
            col_field.type,
        )
    return pa.RecordBatch.from_pydict(rec_batch_dict, schema=get_schema())


def inclusive_jets_sorted_by_pt(
    cluster_sequence: fj.ClusterSequence,
    min_pt: float = 2.0,
) -> tuple[ak.Array, ak.Array]:
    jets = cluster_sequence.inclusive_jets(min_pt=min_pt)
    sortedIndex = ak.argsort(jets.pt, axis=-1, ascending=False)
    jets = jets[sortedIndex]
    constituents = cluster_sequence.constituents(min_pt=min_pt)[sortedIndex]
    sortedConstitIndex = ak.argsort(constituents.pt, axis=-1, ascending=False)
    constituents = constituents[sortedConstitIndex]
    print(f"------Clustered {ak.count(jets)} jets...")
    return jets, constituents


def process_jets(
    jets,
    constituents,
    jet_pt_min=5.0,
    jet_abs_rap_max=0.6,
):
    jetPtCut = jets.pt > jet_pt_min  # & (jets.pt < jet_pt_max)
    jetEtaCut = np.abs(jets.eta) < jet_abs_rap_max
    jets["ncharged"] = ak.count_nonzero(constituents.charge, axis=-1)
    jets["nconstituents"] = ak.count(constituents, axis=-1)
    jetNChargedCut = jets.ncharged > 1

    jetCut = jetPtCut & jetEtaCut & jetNChargedCut
    jets = ak.drop_none(ak.mask(jets, jetCut), axis=1)
    constituents = ak.drop_none(ak.mask(constituents, jetCut), axis=1)

    print(f"------After cuts, {ak.count(jets.pt)} jets left...")

    return jets, constituents


@nb.jit
def is_data_event_ht2(builder, events):
    triggerSetHT2 = set([370521, 370522, 370531, 370980])
    for triggers in events._Triggers:
        hasHT2 = False
        for trigger in triggers:
            if trigger in triggerSetHT2:
                # print(triggerId, hasHT2)
                hasHT2 = True
                break
        builder.append(hasHT2)
    return builder


def process_events(
    events,
    con_kt_min=None,
    is_embedding=False,
    sys_var=None,
    iseed=None,
):
    tracks = ak.zip(
        dict(
            zip(
                ("pt", "eta", "phi", "e", "charge"),
                [
                    events[f"tracks._{branch}"]
                    for branch in ("Pt", "Eta", "Phi", "E", "Charge")
                ],
            )
        ),
        with_name="Momentum4D",
    )

    if sys_var is not None:
        match sys_var:
            case SysVar.TRACK_EFFICIENCY:
                seed = np.random.default_rng(iseed).integers(sys.maxsize)
                tracks = tracks[get_tracking_efficiency_sys_var_mask(events, seed)]
            case SysVar.TOWER_ET_CORRECTION:
                events = apply_hadronic_correction_sys_var(events)
            case _:
                pass

    towers = ak.zip(
        dict(
            zip(
                ("pt", "eta", "phi", "e", "charge"),
                [
                    events[f"towers._{branch}"]
                    for branch in ("Pt", "Eta", "Phi", "E", "Charge")
                ],
            )
        ),
        with_name="Momentum4D",
    )

    if con_kt_min is not None:
        tracks = ak.drop_none(ak.mask(tracks, tracks.pt > con_kt_min), axis=-1)
        towers = ak.drop_none(ak.mask(towers, towers.et > con_kt_min), axis=-1)

    isVzGood = np.abs(events._pVtx_Z) < 30.0
    isMaxTrackPtOk = ak.fill_none(ak.max(tracks.pt, axis=-1), value=0) < 30.0
    maxTowEts = ak.fill_none(ak.max(towers.et, axis=-1), value=0)
    isMaxTowEtOk = maxTowEts < 30.0

    isHT2Like = (
        is_data_event_ht2(ak.ArrayBuilder(), events).snapshot()
        if not is_embedding
        else maxTowEts > 4.0
    )

    eventFilter = isVzGood & isMaxTrackPtOk & isMaxTowEtOk & isHT2Like

    return ak.concatenate([tracks, towers], axis=-1), eventFilter


def cluster_batch(
    events,
    jet_definition,
    con_kt_min=None,
    cs_pt_min=2.0,
    batch_weight=1.0,
):
    candidates, eventFilter = process_events(events, con_kt_min=con_kt_min)
    # candidates = ak.concatenate([tracks, towers], axis=-1)
    print(f"---Got {len(events)} ({len(candidates)}) events...")
    candidates = ak.drop_none(ak.mask(candidates, eventFilter))
    print(f"---Left with {len(candidates)} after cuts...")

    clusterSeq = fj.ClusterSequence(candidates, jet_definition)
    jets, constituents = inclusive_jets_sorted_by_pt(clusterSeq, min_pt=cs_pt_min)

    jets, constituents = process_jets(
        jets,
        constituents,
        jet_pt_min=10.0,
    )
    jets["weight"], _ = ak.broadcast_arrays(batch_weight, jets.pt)
    return jets_to_rb_dict(
        jets,
        constituents,
    )


def worker(
    jet_definition,
    data_files,
    output_file_name,
    con_kt_min=None,
    cs_pt_min=2.0,
    nfiles_per_batch=100,
    nbatches=-1,
    max_events_per_batch=-1,
):
    sink = pa.OSFile(output_file_name, "wb")
    writer = pa.ipc.new_file(sink, get_schema())

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
        )

        writer.write(record_batch)

        if batchNumber == nbatches:
            break
    writer.close()
    get_schema.cache_clear()


def read_input_files(
    folder_path,
    bad_run_list,
    glob_expr="*.tree.root",
    tree_name="Events",
    run_number_token_id=0,
):
    print(f"Reading input data files from {folder_path}")
    with open(bad_run_list, "r") as bad_run_stream:
        bad_runs = set(bad_run_stream.read().splitlines())
    print(f'{len(bad_runs)} runs numbers set as "bad" from file {bad_run_list}')

    good_run_list = []
    bad_run_files = []
    good_run_files = []
    for filePath in Path(folder_path).rglob(glob_expr):
        runNumber = str(filePath.stem).split("_")[run_number_token_id]
        if runNumber in bad_runs:
            bad_run_files.append(f"{str(filePath)}:{tree_name}")
        else:
            good_run_list.append(int(runNumber))
            good_run_files.append(f"{str(filePath)}:{tree_name}")
    good_runs = set(good_run_list)
    print(
        f"Read {len(good_run_files)} files for {len(good_runs)} good runs, {len(bad_run_files)} files for {len(bad_runs)} bad runs"
    )
    return good_run_files, bad_run_files


if __name__ == "__main__":
    do_test = False
    con_kt_min = 0.2

    data_file_list, good_runs = read_input_files(
        data_folder_path, bad_run_list, run_number_token_id=0
    )

    jetDef = fj.JetDefinition(
        fj.antikt_algorithm,
        0.4,
        fj.E_scheme,
    )
    output_file_name = (
        f"{output_path_prefix}/test.arrow"
        if do_test
        else f"{output_path_prefix}/data.arrow"
    )
    nbatches = 1 if do_test else -1
    nfiles_per_batch = 1 if do_test else 150
    max_events_per_batch = 10 if do_test else -1

    worker(
        jetDef,
        data_file_list,
        output_file_name,
        con_kt_min=con_kt_min,
        nfiles_per_batch=nfiles_per_batch,
        nbatches=nbatches,
        max_events_per_batch=max_events_per_batch,
    )

    print("Done!")
