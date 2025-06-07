import pyarrow as pa
import awkward as ak
import logging
from pathlib import Path
import numba as nb
import numpy as np
from typing_extensions import Tuple, Optional
import fastjet as fj


def inclusive_jets_sorted_by_pt(
    cluster_sequence: fj.ClusterSequence, min_pt: float = 2.0
) -> Tuple[ak.Array, ak.Array]:
    jets = cluster_sequence.inclusive_jets(min_pt=min_pt)
    sortedIndex = ak.argsort(jets.pt, axis=-1, ascending=False)
    jets = jets[sortedIndex]
    constituents = cluster_sequence.constituents(min_pt=min_pt)[sortedIndex]
    sortedConstitIndex = ak.argsort(constituents.pt, axis=-1, ascending=False) 
    constituents = constituents[sortedConstitIndex]
    print(f"------Clustered {ak.count(jets)} jets...")
    return jets, constituents


def calculate_angularity(kappa, beta, jets, constituents, charged_only=True, prefix=""):
    pt_k = (
        constituents.pt_fraction
        if kappa == 1
        else constituents.pt2_fraction
        if kappa == 2
        else (constituents.pt_fraction) ** kappa
    )
    dr_b = (
        constituents.dr
        if beta == 1
        else constituents.dr2
        if beta == 2
        else (constituents.dr) ** beta
    )

    pt_k_dr_b = (
        constituents.charge_factor * pt_k * dr_b if charged_only else pt_k * dr_b
    )

    new_col_base = f"ang_k{kappa}_b{beta}"
    new_col_name = f"{prefix}ch_{new_col_base}" if charged_only else new_col_base

    jets[new_col_name] = ak.nansum(pt_k_dr_b, axis=-1)
    return jets


def get_hard_core(constituents, hc_kt_min=2.0):
    hc_mask = ((constituents.charge != 0) & (constituents.pt > hc_kt_min)) | (
        (constituents.charge == 0) & (constituents.et > hc_kt_min)
    )
    hc_constituents = ak.drop_none(ak.mask(constituents, hc_mask), axis=-1)

    hc_jets = ak.sum(hc_constituents, axis=-1)
    return hc_jets, hc_constituents


def process_jets(
    jets,
    constituents,
    jet_pt_min=5.0,
    jet_abs_rap_max=0.6,
    do_hc_mode=False,
    hc_kt_min=2.0,
):
    jetPtCut = jets.pt > jet_pt_min  # & (jets.pt < jet_pt_max)
    jetEtaCut = np.abs(jets.eta) < jet_abs_rap_max
    jets["ncharged"] = ak.count_nonzero(constituents.charge, axis=-1)

    jetNChargedCut = jets.ncharged > 1

    jetCut = jetPtCut & jetEtaCut & jetNChargedCut
    jets = ak.drop_none(ak.mask(jets, jetCut), axis=1)
    constituents = ak.drop_none(ak.mask(constituents, jetCut), axis=1)
    print(f"------After cuts, {ak.count(jets.pt)} jets left...")

    constituents["dr2"] = constituents.deltaR2(jets)
    constituents["dr"] = np.sqrt(constituents.dr2)
    constituents["pt2_fraction"] = constituents.pt2 / jets.pt2
    constituents["pt_fraction"] = np.sqrt(constituents.pt2_fraction)
    constituents["charge_factor"] = ak.fill_none(
        ak.mask(ak.ones_like(constituents.pt), constituents.charge != 0), 0
    )
    constituents["neutral_factor"] = ak.fill_none(
        ak.mask(ak.ones_like(constituents.pt), constituents.charge == 0), 0
    )

    jets["nef"] = ak.nansum(
        constituents.neutral_factor * constituents.pt_fraction, axis=-1
    )

    jets = calculate_angularity(1, 0, jets, constituents)
    jets = calculate_angularity(1, 0.5, jets, constituents)
    jets = calculate_angularity(1, 1, jets, constituents)
    jets = calculate_angularity(1, 2, jets, constituents)
    jets = calculate_angularity(2, 0, jets, constituents)

    if do_hc_mode:
        hc_jets, hc_constituents = get_hard_core(constituents, hc_kt_min=hc_kt_min)
        # hc_jets, hc_constituents = process_jets(
        #    hc_jets,
        #    hc_constituents,
        #    jet_pt_min=0.0,
        #    jet_abs_rap_max=jet_abs_rap_max,
        #    do_hc_mode=False,
        # )
        jets["hc_pt"] = hc_jets.pt
        jets["hc_eta"] = hc_jets.eta
        jets["hc_phi"] = hc_jets.phi
        jets["hc_e"] = hc_jets.e
        for kappa, beta in [(1, 0), (1, 0.5), (1, 1), (1, 2), (2, 0)]:
            hc_jets = calculate_angularity(kappa, beta, hc_jets, hc_constituents)
            jets[f"hc_ch_ang_k{kappa}_b{beta}"] = hc_jets[f"ch_ang_k{kappa}_b{beta}"]

    # print("process jets done")
    return jets, constituents


@nb.jit
def is_event_ht2(builder, events, is_embedding=False):
    triggerSetHT2 = set([370521, 370522, 370531, 370980])
    if not is_embedding:
        for triggers in events._Triggers:
            hasHT2 = False
            for trigger in triggers:
                if trigger in triggerSetHT2:
                    # print(triggerId, hasHT2)
                    hasHT2 = True
                    break
            # print(hasHT2)
            builder.append(hasHT2)
    return builder


def ak_to_pa_array(col, type, axis=None):
    return


def jets_to_rb_dict(
    jets: ak.Array, constituents: ak.Array, prefix: str = "", do_hc_mode: bool = False
) -> pa.RecordBatch:
    #print(jets.fields)
    cols = [
        ("weight", jets.weight, pa.float32()),
        ("pt", jets.pt, pa.float32()),
        ("eta", jets.eta, pa.float32()),
        ("phi", jets.phi, pa.float32()),
        ("nef", jets.nef, pa.float32()),
        ("ncharged", jets.ncharged, pa.uint8()),
        ("ch_ang_k1_b0", jets.ch_ang_k1_b0, pa.float32()),
        ("ch_ang_k1_b0.5", jets["ch_ang_k1_b0.5"], pa.float32()),
        ("ch_ang_k1_b1", jets.ch_ang_k1_b1, pa.float32()),
        ("ch_ang_k1_b2", jets.ch_ang_k1_b2, pa.float32()),
        ("ch_ang_k2_b0", jets.ch_ang_k2_b0, pa.float32()),
        ("constit_pt", constituents.pt, pa.list_(pa.float32())),
        ("constit_eta", constituents.eta, pa.list_(pa.float32())),
        ("constit_phi", constituents.phi, pa.list_(pa.float32())),
        ("constit_charge", constituents.charge, pa.list_(pa.int8())),
    ]

    if do_hc_mode:
        cols.extend(
            [
                ("hc_pt", jets.hc_pt, pa.float32()),
                ("hc_eta", jets.hc_eta, pa.float32()),
                ("hc_phi", jets.hc_phi, pa.float32()),
                ("hc_ch_ang_k1_b0", jets.hc_ch_ang_k1_b0, pa.float32()),
                ("hc_ch_ang_k1_b0.5", jets["hc_ch_ang_k1_b0.5"], pa.float32()),
                ("hc_ch_ang_k1_b1", jets.hc_ch_ang_k1_b1, pa.float32()),
                ("hc_ch_ang_k1_b2", jets.hc_ch_ang_k1_b2, pa.float32()),
                ("hc_ch_ang_k2_b0", jets.hc_ch_ang_k2_b0, pa.float32()),
            ]
        )

    rec_batch_dict = {}
    for col_name, ak_col, col_type in cols:
        rec_batch_dict[f"{prefix}{col_name}"] = pa.array(
            ak.flatten(ak_col, axis=1), col_type
        )
    return pa.RecordBatch.from_pydict(rec_batch_dict)


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


class StreamToLogger(object):
    def __init__(self, logger, file_name, log_level=logging.DEBUG, mode="a"):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

        self.logger.setLevel(self.log_level)
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.file_handler = logging.FileHandler(file_name, mode=mode)
        self.file_handler.setFormatter(self.formatter)

        self.logger.addHandler(self.file_handler)
        self.logger.info(
            "####################################################################"
        )
        self.logger.info(f"Creating logger for slot at {file_name}")

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
