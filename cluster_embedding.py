from typing_extensions import Optional, Tuple
import uproot
import pyarrow as pa

import fastjet as fj

import vector
import awkward as ak

import numba as nb

from utils import clustering

import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, wait
import logging

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


def process_detector_lvl(events):
    coordinates = ["pt", "eta", "phi", "e", "charge"]
    branchNames = ["Pt", "Eta", "Phi", "E", "Charge"]

    recoTracks = ak.zip(
        dict(
            zip(
                coordinates,
                [events[f"tracks._{branch}"] for branch in branchNames],
            )
        ),
        with_name="Momentum4D",
    )
    recoTowers = ak.zip(
        dict(
            zip(
                coordinates,
                [events[f"towers._{branch}"] for branch in branchNames],
            )
        ),
        with_name="Momentum4D",
    )

    isMaxTrackPtOk = ak.max(recoTracks.pt, axis=1) < 30.0
    maxTowEts = ak.max(recoTowers.et, axis=1)
    isHT2Like = maxTowEts > 4.0
    isMaxTowEtOk = maxTowEts < 30.0

    eventFilter = isMaxTrackPtOk & isMaxTowEtOk & isHT2Like
    # print(recoTracks.type)
    return ak.concatenate([recoTracks, recoTowers], axis=1), eventFilter


def process_particle_lvl(events):
    coordinates = ["pt", "eta", "phi", "e", "charge"]
    branchNames = ["Pt", "Eta", "Phi", "E", "Charge"]

    genParticles = ak.zip(
        dict(
            zip(
                coordinates,
                [events[f"genTracks._{branch}"] for branch in branchNames],
            )
        ),
        with_name="Momentum4D",
    )

    isVzGood = abs(events._pVtx_Z) < 30.0
    isMaxKtOk = ak.max(genParticles.pt, axis=1) < 30.0
    return genParticles, isMaxKtOk & isVzGood


def cluster_batch(
    events: ak.Array,
    jet_definition: fj.JetDefinition,
    min_pt: float = 2.0,
    max_jet_pt: float = 1000.0,
    is_good_run_batch: bool = True,
    batch_weight: float = 1.0,
    con_kt_min: Optional[float] = None,
    do_hc_mode: bool = False,
    hc_kt_min: float = 2.0,
) -> Tuple[(pa.RecordBatch, pa.RecordBatch, pa.RecordBatch, pa.RecordBatch)]:
    genCandidates, genEventFilter = process_particle_lvl(events)
    print(f"---Got {len(events)} events")
    events = ak.drop_none(ak.mask(events, genEventFilter))
    print(f"---{len(events)} events left after gen and event level cuts")
    genCandidates = ak.drop_none(
        ak.mask(
            genCandidates,
            genEventFilter,
        )
    )

    print("---Clustering gen jets...")
    genClusterSeq = fj.ClusterSequence(genCandidates, jet_definition)
    genJets, genConstituents = clustering.inclusive_jets_sorted_by_pt(
        genClusterSeq, min_pt=min_pt
    )

    genJetPtMaxCut = ak.max(genJets.pt, axis=1) < max_jet_pt

    events = ak.drop_none(ak.mask(events, genJetPtMaxCut))
    genJets = ak.drop_none(ak.mask(genJets, genJetPtMaxCut))
    genConstituents = ak.drop_none(ak.mask(genConstituents, genJetPtMaxCut))
    print(
        f"------{len(events)} events left after applying {max_jet_pt} GeV cut on max gen event jet pt..."
    )

    genJets, genConstituents = clustering.process_jets(
        genJets,
        genConstituents,
        jet_pt_min=5.0,
        do_hc_mode=do_hc_mode,
        hc_kt_min=hc_kt_min,
    )
    genJets["weight"], _ = ak.broadcast_arrays(batch_weight, genJets.pt)

    if is_good_run_batch:
        recoCandidates, recoEventFilter = process_detector_lvl(events)

        print("---Clustering reco jets...")
        recoClusterSeq = fj.ClusterSequence(recoCandidates, jet_definition)
        recoJets, recoConstituents = clustering.inclusive_jets_sorted_by_pt(
            recoClusterSeq, min_pt=min_pt
        )
        recoJets, recoConstituents = clustering.process_jets(
            recoJets,
            recoConstituents,
            jet_pt_min=10.0,
            do_hc_mode=do_hc_mode,
            hc_kt_min=hc_kt_min,
        )

        recoJetPtMaxCut = ak.max(recoJets.pt, axis=1) < max_jet_pt
        recoEventFilter = recoEventFilter & recoJetPtMaxCut

        recoJets = ak.drop_none(ak.mask(recoJets, recoEventFilter))
        recoConstituents = ak.drop_none(ak.mask(recoConstituents, recoEventFilter))
        print(
            f"---Reco lvl cut took nevents from {len(recoCandidates)} to {len(recoJets)}"
        )

        missedGenJets = ak.drop_none(ak.mask(genJets, ~recoEventFilter))
        missedGenConstituents = ak.drop_none(ak.mask(genConstituents, ~recoEventFilter))
        genJets = ak.drop_none(ak.mask(genJets, recoEventFilter))
        genConstituents = ak.drop_none(ak.mask(genConstituents, recoEventFilter))

        recoJets["weight"], _ = ak.broadcast_arrays(batch_weight, recoJets.pt)

        matchIndices = match_gen_to_reco(
            ak.ArrayBuilder(), genJets, recoJets
        ).snapshot()

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

        gen_match_record_batch = clustering.jets_to_rb_dict(genMatches, genMatchConstituents, do_hc_mode=do_hc_mode)

        reco_match_record_batch = clustering.jets_to_rb_dict(
            recoMatches, recoMatchConstituents, do_hc_mode=do_hc_mode
        )

        fake_record_batch = clustering.jets_to_rb_dict(recoFakes, recoFakeConstituents, do_hc_mode=do_hc_mode)

    else:
        gen_match_record_batch = []
        reco_match_record_batch = []
        fake_record_batch = []
        genMisses = genJets
        genMissConstituents = genConstituents

    miss_record_batch = clustering.jets_to_rb_dict(genMisses, genMissConstituents, do_hc_mode=do_hc_mode)

    print(
        f"---Number of gen-matches, reco-matches, misses, fakes: {len(gen_match_record_batch)}, {len(reco_match_record_batch)}, {len(miss_record_batch)}, {len(fake_record_batch)}"
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
    sample_weight=1.0,
    max_jet_pt=1000.0,
    con_kt_min=None,
    do_hc_mode=False,
    hc_kt_min=2.0,
    do_test=False,
):
    if not do_test:
        sys.stdout = clustering.StreamToLogger(
            logging.getLogger("STDOUT"), f"logs/{slotId}.log", logging.DEBUG, mode="w"
        )

        sys.stderr = clustering.StreamToLogger(
            logging.getLogger("STDERR"), f"logs/{slotId}.log", logging.ERROR
        )

    jetDefinition = fj.JetDefinition(fj.antikt_algorithm, 0.4, fj.BIpt2_scheme)
    # device = "cpu"
    date_time_stamp = datetime.now().strftime("%d%b%y-%H%M") if not do_test else "embedding-test"
    outputPathStr = (
        f"/home/tanmaypani/star-workspace/jet-angularity-study/outputs/{date_time_stamp}/{slotId}"
    )
    print(f"Will write outputs to path: {outputPathStr}")
    print("Checking path...")
    if not os.path.exists(outputPathStr):
        print(f"Creating folder: {outputPathStr}...")
        os.makedirs(outputPathStr)
        print("Done!")

    nFiles = len(goodRunFiles) + len(badRunFiles)
    nFilesPerBatch = 1 if do_test else 100
    nBatches = nFiles // nFilesPerBatch + 1
    maxNBatches = 1 if do_test else -1
    maxEventsPerBatch = -1 if do_test else -1
    print(f"Will process {nFiles} files for slot: {slotId} in {nBatches} batches")

    genMatchSink = pa.OSFile(f"{outputPathStr}/gen-matches.arrow", "wb")
    recoMatchSink = pa.OSFile(f"{outputPathStr}/reco-matches.arrow", "wb")
    missSink = pa.OSFile(f"{outputPathStr}/misses.arrow", "wb")
    fakeSink = pa.OSFile(f"{outputPathStr}/fakes.arrow", "wb")

    print(f"Writing matched jets at gen level to: {outputPathStr}/gen-matches.arrow")
    print(f"Writing matched jets at reco level to: {outputPathStr}/reco-matches.arrow")
    print(f"Writing missed jets at gen level to: {outputPathStr}/misses.arrow")
    print(f"Writing fake jets at reco level to: {outputPathStr}/fakes.arrow")

    genMatchWriter = None
    recoMatchWriter = None
    missWriter = None
    fakeWriter = None

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
            do_hc_mode=do_hc_mode,
            hc_kt_min=hc_kt_min,
        )
        if genMatchWriter is None:
            genMatchWriter = pa.ipc.new_file(
                genMatchSink, gen_match_record_batch.schema
            )
        if recoMatchWriter is None:
            recoMatchWriter = pa.ipc.new_file(
                recoMatchSink, reco_match_record_batch.schema
            )
        if missWriter is None:
            missWriter = pa.ipc.new_file(missSink, miss_record_batch.schema)
        if fakeWriter is None:
            fakeWriter = pa.ipc.new_file(fakeSink, fake_record_batch.schema)

        genMatchWriter.write(gen_match_record_batch)
        recoMatchWriter.write(reco_match_record_batch)
        missWriter.write(miss_record_batch)
        fakeWriter.write(fake_record_batch)

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

        _, _, miss_record_batch, _ = cluster_batch(
            events,
            jetDefinition,
            is_good_run_batch=False,
            batch_weight=sample_weight,
            max_jet_pt=max_jet_pt,
            do_hc_mode=do_hc_mode,
            hc_kt_min=hc_kt_min,
        )
        if missWriter is None:
            missWriter = pa.ipc.new_file(missSink, miss_record_batch.schema)

        missWriter.write(miss_record_batch)
        if iBatch == 2 * maxNBatches:
            break
    if genMatchWriter:
        genMatchWriter.close()
    if recoMatchWriter:
        recoMatchWriter.close()
    if missWriter:
        missWriter.close()
    if fakeWriter:
        fakeWriter.close()

    print("Done")

    return True


if __name__ == "__main__":
    do_test = False

    badRunList = "/home/tanmaypani/star-workspace/jet-angularity-study/runtime-files/runLists/pp200_production_2012_BAD_Issac.list"

    ptHatBins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    ptHatLowEdges = [11, 15, 20, 25, 35, 45, 55]
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

    dataFolderPath = "/run/media/tanmaypani/Samsung-1tb/data/Pythia6Embedding_pp200_production_2012_P12id_SL12d_20235003_MuToTree20250123"

    ptHatBinsToRun = [6] if do_test else list(range(0, 7))
    with ProcessPoolExecutor() as executor:
        futures = []
        for ipth in ptHatBinsToRun:
            goodRunFiles, badRunFiles = clustering.read_input_files(
                dataFolderPath,
                badRunList,
                glob_expr=f"*{ptHatBins[ipth]}_{ptHatBins[ipth + 1]}_*/tree/*.tree.root",
                run_number_token_id=3,
            )

            futures.append(
                executor.submit(
                    worker,
                    f"ptHat{ptHatBins[ipth]}to{ptHatBins[ipth + 1]}",
                    goodRunFiles,
                    badRunFiles,
                    sample_weight=weights[ipth] / nevents[ipth],
                    max_jet_pt=3.0 * ptHatLowEdges[ipth],
                    do_test=do_test,
                    do_hc_mode=True,
                )
            )
        wait(futures)
    print("Done!")
