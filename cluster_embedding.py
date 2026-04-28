import os
import sys
from datetime import datetime
from concurrent.futures import (
    ProcessPoolExecutor,
    as_completed,
)
import logging
from typing import Optional

import numba as nb

import uproot
import vector
import fastjet as fj
import awkward as ak
import pyarrow as pa

from cluster_data import (
    process_events,
    process_jets,
    inclusive_jets_sorted_by_pt,
    jets_to_rb_dict,
    read_input_files,
    get_schema,
)
from systematics import SysVar

bad_run_list: str = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_BAD_Issac.list"
data_folder_path: str = "/run/media/tanmaypani/Samsung-1tb/data/Pythia6Embedding_pp200_production_2012_P12id_SL12d_20235003_MuToTree20250123"
output_path_prefix: str = "/home/tanmaypani/star-workspace/jet-angularity-study/datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding"

pt_hat_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
pt_hat_low_edges = [11, 15, 20, 25, 35, 45, 55]
weights = [
    0.0023021,
    0.000342608,
    4.56842e-05,
    9.71569e-06,
    4.69593e-07,
    2.69062e-08,
    1.43197e-09,
]
nevents = [
    17233020.0,
    16422119.0,
    3547865.0,
    2415179.0,
    2525739.0,
    1203188.0,
    1264931.0,
]


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


vector.register_awkward()


@nb.jit
def match_gen_to_reco(builder, gen_jet_events, reco_jet_events, max_dr=0.4):
    for gen_jets, reco_jets in zip(gen_jet_events, reco_jet_events):
        builder.begin_list()
        reco_idx = list(range(len(reco_jets)))
        gen_idx = list(range(len(gen_jets)))
        for igen in gen_idx:
            is_matched = False
            for ireco in reco_idx:
                dR = gen_jets[igen].deltaR(reco_jets[ireco])
                if dR < max_dr:
                    is_matched = True
                    builder.begin_record()
                    builder.field("gen").append(igen)
                    builder.field("reco").append(ireco)
                    builder.field("dr").append(dR)
                    builder.end_record()
                    reco_idx.remove(ireco)
                    break

            if not is_matched:
                builder.begin_record()
                builder.field("gen").append(igen)
                builder.field("reco").append(None)
                builder.field("dr").append(None)
                builder.end_record()

        for ifake in reco_idx:
            builder.begin_record()
            builder.field("gen").append(None)
            builder.field("reco").append(ifake)
            builder.field("dr").append(None)
            builder.end_record()
        builder.end_list()
    return builder


def process_particle_lvl(
    events: ak.Array,
    jet_definition: fj.JetDefinition,
    cs_min_pt: float = 2.0,
    max_jet_pt: float = 1000.0,
    batch_weight: float = 1.0,
):
    genCandidates = ak.zip(
        dict(
            zip(
                ("pt", "eta", "phi", "e", "charge"),
                [
                    events[f"genTracks._{branch}"]
                    for branch in ("Pt", "Eta", "Phi", "E", "Charge")
                ],
            )
        ),
        with_name="Momentum4D",
    )

    isVzGood = abs(events._pVtx_Z) < 30.0
    isMaxKtOk = ak.max(genCandidates.pt, axis=1) < 30.0

    genEventFilter = isMaxKtOk & isVzGood
    events = ak.drop_none(ak.mask(events, genEventFilter))

    genCandidates = ak.drop_none(
        ak.mask(
            genCandidates,
            genEventFilter,
        )
    )

    print("---Clustering gen jets...")
    genClusterSeq = fj.ClusterSequence(genCandidates, jet_definition)
    genJets, genConstituents = inclusive_jets_sorted_by_pt(
        genClusterSeq, min_pt=cs_min_pt
    )

    genJetPtMaxCut = ak.max(genJets.pt, axis=1) < max_jet_pt
    events = ak.drop_none(ak.mask(events, genJetPtMaxCut))

    genJets = ak.drop_none(ak.mask(genJets, genJetPtMaxCut))
    genConstituents = ak.drop_none(ak.mask(genConstituents, genJetPtMaxCut))
    genJets, genConstituents = process_jets(
        genJets,
        genConstituents,
        jet_pt_min=5.0,
    )
    genJets["weight"], _ = ak.broadcast_arrays(batch_weight, genJets.pt)
    return genJets, genConstituents, events


def cluster_batch(
    events: ak.Array,
    jet_definition: fj.JetDefinition,
    min_pt: float = 2.0,
    max_jet_pt: float = 1000.0,
    is_good_run: bool = True,
    batch_weight: float = 1.0,
    con_kt_min: Optional[float] = None,
    sys_var_type: SysVar = SysVar.NONE,
    iseed: Optional[int] = None,
) -> (
    tuple[(pa.RecordBatch, pa.RecordBatch, pa.RecordBatch, pa.RecordBatch)]
    | pa.RecordBatch
):
    print(f"---Got {len(events)} events")
    genJets, genConstituents, events = process_particle_lvl(
        events,
        jet_definition,
        min_pt,
        max_jet_pt,
        batch_weight,
    )
    print(
        f"---{len(events)} events left after event cuts and pt_jet < {max_jet_pt} GeV/c cut at gen level..."
    )

    if not is_good_run:
        miss_record_batch = jets_to_rb_dict(
            genJets,
            genConstituents,
        )
        print(f"---Bad run, added {len(miss_record_batch)} gen-jets to missed jets...")
        return miss_record_batch

    recoCandidates, recoEventFilter = process_events(
        events,
        is_embedding=True,
        sys_var=sys_var_type,
        iseed=iseed,
    )

    print("---Clustering reco jets...")
    recoClusterSeq = fj.ClusterSequence(recoCandidates, jet_definition)
    recoJets, recoConstituents = inclusive_jets_sorted_by_pt(
        recoClusterSeq, min_pt=min_pt
    )
    recoJets, recoConstituents = process_jets(
        recoJets,
        recoConstituents,
        jet_pt_min=10.0,
    )

    recoJetPtMaxCut = ak.max(recoJets.pt, axis=1) < max_jet_pt
    recoEventFilter = recoEventFilter & recoJetPtMaxCut

    recoJets = ak.drop_none(ak.mask(recoJets, recoEventFilter))
    recoConstituents = ak.drop_none(ak.mask(recoConstituents, recoEventFilter))
    recoJets["weight"], _ = ak.broadcast_arrays(batch_weight, recoJets.pt)
    print(f"---Reco lvl cut took nevents from {len(recoCandidates)} to {len(recoJets)}")

    missedGenJets = ak.drop_none(ak.mask(genJets, ~recoEventFilter))
    missedGenConstituents = ak.drop_none(ak.mask(genConstituents, ~recoEventFilter))
    genJets = ak.drop_none(ak.mask(genJets, recoEventFilter))
    genConstituents = ak.drop_none(ak.mask(genConstituents, recoEventFilter))

    matchIndices = match_gen_to_reco(ak.ArrayBuilder(), genJets, recoJets).snapshot()

    genMissSelection = ak.is_none(matchIndices.reco, axis=1)
    missIndices = ak.drop_none(ak.mask(matchIndices, genMissSelection)).gen
    genMisses = ak.concatenate([genJets[missIndices], missedGenJets], axis=0)
    genMissConstituents = ak.concatenate(
        [genConstituents[missIndices], missedGenConstituents], axis=0
    )

    recoFakeSelection = ak.is_none(matchIndices.gen, axis=1)
    fakeIndices = ak.drop_none(ak.mask(matchIndices, recoFakeSelection)).reco
    recoFakes = recoJets[fakeIndices]
    recoFakeConstituents = recoConstituents[fakeIndices]

    matchSelection = ak.is_none(matchIndices.dr, axis=1)
    goodMatchIndices = ak.drop_none(ak.mask(matchIndices, ~matchSelection))
    genMatches = genJets[goodMatchIndices.gen]
    genMatchConstituents = genConstituents[goodMatchIndices.gen]
    recoMatches = recoJets[goodMatchIndices.reco]
    recoMatchConstituents = recoConstituents[goodMatchIndices.reco]

    gen_match_record_batch = jets_to_rb_dict(
        genMatches,
        genMatchConstituents,
    )

    reco_match_record_batch = jets_to_rb_dict(
        recoMatches,
        recoMatchConstituents,
    )

    fake_record_batch = jets_to_rb_dict(
        recoFakes,
        recoFakeConstituents,
    )

    miss_record_batch = jets_to_rb_dict(
        genMisses,
        genMissConstituents,
    )

    print(
        "---Number of gen-matches, reco-matches, misses, fakes:",
        len(gen_match_record_batch),
        len(reco_match_record_batch),
        len(miss_record_batch),
        len(fake_record_batch),
    )

    return (
        gen_match_record_batch,
        reco_match_record_batch,
        miss_record_batch,
        fake_record_batch,
    )


def worker(
    slotId,
    goodRunFiles,
    badRunFiles,
    output_prefix,
    sample_weight=1.0,
    max_jet_pt=1000.0,
    con_kt_min=None,
    do_test=False,
    sys_var_type=SysVar.NONE,
):
    if not do_test:
        sys.stdout = StreamToLogger(
            logging.getLogger("STDOUT"), f"logs/{slotId}.log", logging.DEBUG, mode="w"
        )

        sys.stderr = StreamToLogger(
            logging.getLogger("STDERR"),
            f"logs/{slotId}.log",
            logging.ERROR,
            mode="w",
        )

    jetDefinition = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.E_scheme)
    output_dir = os.path.join(output_prefix, slotId)
    print(f"Will write outputs to path: {output_dir}")
    print("Checking path...")
    if not os.path.exists(output_dir):
        print(f"Creating folder: {output_dir}...")
        os.makedirs(output_dir)
        print("Done!")

    nFiles = len(goodRunFiles) + len(badRunFiles)
    nFilesPerBatch = 10 if do_test else 100
    nBatches = nFiles // nFilesPerBatch + 1
    maxNBatches = 10 if do_test else -1
    maxEventsPerBatch = -1
    print(f"Will process {nFiles} files for slot: {slotId} in {nBatches} batches")

    genMatchSink = pa.OSFile(f"{output_dir}/gen-matches.arrow", "wb")
    recoMatchSink = pa.OSFile(f"{output_dir}/reco-matches.arrow", "wb")
    missSink = pa.OSFile(f"{output_dir}/misses.arrow", "wb")
    fakeSink = pa.OSFile(f"{output_dir}/fakes.arrow", "wb")

    genMatchWriter = pa.ipc.new_file(genMatchSink, get_schema())
    recoMatchWriter = pa.ipc.new_file(recoMatchSink, get_schema())
    missWriter = pa.ipc.new_file(missSink, get_schema())
    fakeWriter = pa.ipc.new_file(fakeSink, get_schema())

    print(f"Writing matched jets at gen level to: {output_dir}/gen-matches.arrow")
    print(f"Writing matched jets at reco level to: {output_dir}/reco-matches.arrow")
    print(f"Writing missed jets at gen level to: {output_dir}/misses.arrow")
    print(f"Writing fake jets at reco level to: {output_dir}/fakes.arrow")

    iBatch = 0
    for startFile in range(0, len(goodRunFiles), nFilesPerBatch):
        iBatch += 1
        endFile = startFile + nFilesPerBatch
        events = uproot.concatenate(goodRunFiles[startFile:endFile])
        if maxEventsPerBatch >= 0:
            events = events[:maxEventsPerBatch]
        print(
            f"Processing batch [{iBatch}/{nBatches}], file # {startFile} to {endFile}, with {len(events)} events"
        )

        (
            gen_match_record_batch,
            reco_match_record_batch,
            miss_record_batch,
            fake_record_batch,
        ) = cluster_batch(
            events,
            jetDefinition,
            batch_weight=sample_weight,
            max_jet_pt=max_jet_pt,
            sys_var_type=sys_var_type,
            iseed=iBatch,
        )
        print(
            f"Clustered batch [{iBatch}/{nBatches}], file # {startFile} to {endFile}, with {len(events)} events"
        )
        genMatchWriter.write(gen_match_record_batch)
        recoMatchWriter.write(reco_match_record_batch)
        missWriter.write(miss_record_batch)
        fakeWriter.write(fake_record_batch)
        print(
            f"Wrote batch [{iBatch}/{nBatches}], file # {startFile} to {endFile}, with {len(events)} events"
        )
        if iBatch == 2 * maxNBatches - 1:
            break
        # return False
    print(f"Starting on the bad runs for slot {slotId}")
    for startFile in range(0, len(badRunFiles), nFilesPerBatch):
        iBatch += 1
        endFile = startFile + nFilesPerBatch
        events = uproot.concatenate(badRunFiles[startFile:endFile])
        if maxEventsPerBatch >= 0:
            events = events[:maxEventsPerBatch]
        print(
            f"Processing batch [{iBatch}/{nBatches}], file # {startFile} to {endFile}, with {len(events)} events"
        )

        miss_record_batch = cluster_batch(
            events,
            jetDefinition,
            is_good_run=False,
            batch_weight=sample_weight,
            max_jet_pt=max_jet_pt,
            sys_var_type=sys_var_type,
        )

        missWriter.write(miss_record_batch)
        if iBatch == 2 * maxNBatches:
            break

    genMatchWriter.close()
    recoMatchWriter.close()
    missWriter.close()
    fakeWriter.close()

    get_schema.cache_clear()

    print("Done")

    return slotId


if __name__ == "__main__":
    do_test = False

    # sys_var_type = SysVar.NONE
    # sys_var_type = SysVar.TOWER_ET_CORRECTION
    sys_var_type = SysVar.TRACK_EFFICIENCY

    ptHatBinsToRun = [6] if do_test else list(range(0, 7))
    # ptHatBinsToRun = [6]
    output_prefix = os.path.join(
        output_path_prefix,
        "test" if do_test else str(sys_var_type),
    )
    with ProcessPoolExecutor() as executor:
        futures = []
        for ipth in ptHatBinsToRun:
            goodRunFiles, badRunFiles = read_input_files(
                data_folder_path,
                bad_run_list,
                glob_expr=f"*{pt_hat_bins[ipth]}_{pt_hat_bins[ipth + 1]}_*/tree/*.tree.root",
                run_number_token_id=3,
            )

            futures.append(
                executor.submit(
                    worker,
                    f"ptHat{pt_hat_bins[ipth]}to{pt_hat_bins[ipth + 1]}",
                    goodRunFiles,
                    badRunFiles,
                    output_prefix=output_prefix,
                    sample_weight=weights[ipth] / nevents[ipth],
                    max_jet_pt=3.0 * pt_hat_low_edges[ipth],
                    do_test=do_test,
                    sys_var_type=sys_var_type,
                )
            )
        # wait(futures)
        for future in as_completed(futures):
            print("Finished slot for:", future.result())
    print("Done!")
