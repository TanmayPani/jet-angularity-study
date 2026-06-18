"""CPU-only self-tests for the Stage-1 `core/` groundwork (paths, features, density_ratio,
reweight). Purely additive: reads existing on-disk arrows, trains only tiny CPU classifiers,
writes nothing to the real tree (the reweight wiring test capture-patches the writer).

Run:  uv run --no-sync python test/test_core.py
Safe to run alongside the inflight GPU unfolding (everything here is forced to CPU).
"""

import os
import sys

# Make the repo root importable when run as a plain script (`python test/test_core.py`).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pyarrow as pa

import preprocessing
from config import Config, load_config
from preprocessing import to_tensordict
from systematics import SysVar

from core import paths, reweight
from core.features import FEATURE_SPECS, spec_for
from core.density_ratio import ClassifierEstimator, density_ratio

CFG = load_config("runtime-files/config.json")
ANG = "datasets/STAR_pp200GeV_production_2012/features/angularities"
BIN = "datasets/STAR_pp200GeV_production_2012/features/bin_counts"


def _rd(p):
    return pa.ipc.open_file(pa.memory_map(p, "rb")).read_all()


def test_paths():
    # SysVar stringifies to its on-disk dir name; produced paths equal the real layout.
    assert str(SysVar.NONE) == "nominal"
    emb = paths.embedding_dir(CFG, SysVar.NONE, mode="angularities")
    assert emb == CFG.dataset_root / "features" / "angularities" / "embedding" / "nominal", emb
    assert paths.data_arrow(CFG, "angularities").exists()
    assert paths.embedding_arrow(CFG, SysVar.NONE, "reco-matches", mode="angularities").exists()
    assert paths.det_lvl_dir(CFG, SysVar.NONE, mode="bin_counts").exists()
    assert paths.part_lvl_dir(CFG, SysVar.NONE, mode="bin_counts").exists()
    # default mode falls back to cfg.feature_mode (bin_counts in the current config).
    assert paths.features_root(CFG) == CFG.dataset_root / "features" / CFG["feature_mode"]
    # all four embedding arrows resolve and exist for the nominal angularities tree.
    for p in paths.embedding_arrows(CFG, SysVar.NONE, mode="angularities").values():
        assert p.exists(), p
    print("  test_paths: OK")


def test_features():
    assert set(FEATURE_SPECS) == set(preprocessing.FEATURE_MODES)
    ang = spec_for("angularities")
    assert ang.columns == tuple(preprocessing.jet_columns)
    assert ang.per_jet_shape == (len(preprocessing.jet_columns),)
    assert ang.architecture == "mlp" and ang.bin_block == "none"
    bc = spec_for("bin_counts")
    assert bc.per_jet_shape == (2, preprocessing.N_PT, preprocessing.N_DR)
    assert bc.is_image and bc.architecture == "cnn" and bc.input_dtype == "uint8"
    comb = spec_for("combined")
    assert comb.num_features == len(preprocessing.jet_columns) + preprocessing.N_BINS

    # The spec's per_jet_shape matches what to_tensordict actually assembles (tiny slice).
    pos = _rd(ANG + "/data.arrow").slice(0, 64)
    neg = _rd(ANG + "/embedding/nominal/reco-matches.arrow").slice(0, 64)
    td = to_tensordict(pos, neg, columns=list(ang.columns), prefix="/tmp/core_td_ang",
                       max_chunksize=64, feature_mode="angularities")
    assert tuple(td["input"].shape[1:]) == ang.per_jet_shape, td["input"].shape
    posb = _rd(BIN + "/data.arrow").slice(0, 64)
    negb = _rd(BIN + "/embedding/nominal/reco-matches.arrow").slice(0, 64)
    tdb = to_tensordict(posb, negb, columns=None, prefix="/tmp/core_td_bin",
                        max_chunksize=64, feature_mode="bin_counts")
    assert tuple(tdb["input"].shape[1:]) == bc.per_jet_shape, tdb["input"].shape
    print("  test_features: OK")


def _small_cfg():
    return Config({**CFG, "num_data_subsample": 4000, "batch_size": 1000})


def test_density_ratio_mlp():
    pos = _rd(ANG + "/data.arrow").slice(0, 6000)
    neg = _rd(ANG + "/embedding/nominal/reco-matches.arrow").slice(0, 6000)
    est = ClassifierEstimator(_small_cfg(), num_replicas=1, num_epochs=2, device="cpu",
                              cache_dir="/tmp/core_dr_mlp")
    r = density_ratio(pos, neg, spec=spec_for("angularities"), estimator=est, label="t_mlp")
    assert r.shape == (len(neg),) and np.isfinite(r).all() and (r > 0).all()
    assert 0.05 < float(r.mean()) < 20.0
    print(f"  test_density_ratio_mlp: OK (mean={r.mean():.3f}, max={r.max():.2f})")


def test_density_ratio_cnn():
    pos = _rd(BIN + "/data.arrow").slice(0, 4000)
    neg = _rd(BIN + "/embedding/nominal/reco-matches.arrow").slice(0, 4000)
    est = ClassifierEstimator(_small_cfg(), num_replicas=1, num_epochs=2, device="cpu",
                              cache_dir="/tmp/core_dr_cnn")
    r = density_ratio(pos, neg, spec=spec_for("bin_counts"), estimator=est, label="t_cnn")
    assert r.shape == (len(neg),) and np.isfinite(r).all() and (r > 0).all()
    print(f"  test_density_ratio_cnn: OK (mean={r.mean():.3f}, max={r.max():.2f})")


def test_reweight_wiring():
    """run_data_reco wiring/alignment with a stub estimator (unity) and a capture-patched
    writer — validates table loads + matched-pair alignment without training or writing."""
    spec = spec_for("angularities")

    class StubEstimator:
        calls = []

        def fit_predict(self, pos, neg, predict_table=None, *, spec, label="", **ov):
            pt = predict_table if predict_table is not None else neg
            StubEstimator.calls.append((label, len(pos), len(neg), len(pt)))
            return np.ones(len(pt), dtype=np.float32)

    captured = {}
    orig_write = reweight.write_four
    reweight.write_four = lambda out_dir, gm, rm, mi, fk: captured.update(
        out_dir=str(out_dir), counts=dict(gm=len(gm), rm=len(rm), mi=len(mi), fk=len(fk))
    )
    try:
        out = reweight.run_data_reco(CFG, spec=spec, estimator=StubEstimator())
    finally:
        reweight.write_four = orig_write

    calls = StubEstimator.calls
    assert calls[0][0] == "reco" and calls[1][0] == "gen_pullback"
    # reco step predicts on reco-matches+fakes; pull-back predicts on the full gen set.
    assert calls[1][3] == captured["counts"]["gm"] + captured["counts"]["mi"]
    assert captured["counts"]["gm"] == captured["counts"]["rm"]  # matched-pair alignment
    assert str(SysVar.UNFOLDING_PRIOR_LIKE_DATA) in captured["out_dir"]
    print(f"  test_reweight_wiring: OK (out={captured['out_dir'].split('/')[-1]}, "
          f"counts={captured['counts']})")


if __name__ == "__main__":
    for name in ("test_paths", "test_features", "test_density_ratio_mlp",
                 "test_density_ratio_cnn", "test_reweight_wiring"):
        print(f"{name} ...")
        globals()[name]()
    print("ALL CORE SELF-TESTS PASSED")
