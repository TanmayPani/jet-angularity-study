"""Estimator-agnostic four-arrow reweighting orchestration.

Stage-1 groundwork (#2). One job — "reweight the Pythia6 embedding toward a target and bake
the result into the four arrows `{gen-matches, reco-matches, misses, fakes}.arrow`" — is
currently implemented four times (`omnisequential`, `reverse_omnisequential`,
`make_alt_embedding`, `reweight_embedding`). This module is the single orchestration: it
composes `core.paths` (where), `core.features` (the FeatureSpec), and `core.density_ratio`
(the chosen pluggable estimator), and owns the load -> splice -> write of the four arrows.

The estimator is injected, so classifier / GP-marginal / BDT are a one-arg swap. This is the
clean target that `reweight_embedding.py`'s CLI is rewired onto in Stage 2; it is not used by
the existing pipeline yet.
"""

from __future__ import annotations

import os

import numpy as np
import pyarrow as pa

from preprocessing import replace_table_column
from systematics import SysVar

from core import paths as _paths
from core.density_ratio import density_ratio


# --- small self-contained arrow IO (keeps `core` independent of the legacy modules) ------
def _read_arrow(path):
    return pa.ipc.open_file(pa.memory_map(str(path), "rb")).read_all()


def _write_arrow(table, path):
    with pa.OSFile(str(path), "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            for batch in table.to_batches():
                writer.write_batch(batch)


def _scaled(table, ratio):
    """Return `table` with its `weight` column multiplied by per-row `ratio`."""
    w = table["weight"].to_numpy().astype(np.float32)
    return replace_table_column(table, "weight", w * np.asarray(ratio, dtype=np.float32))


def load_nominal_arrows(cfg, spec):
    """The four nominal Pythia6 embedding arrows for a feature mode, as `{name: Table}`."""
    return {
        name: _read_arrow(path)
        for name, path in _paths.embedding_arrows(cfg, SysVar.NONE, mode=spec.name).items()
    }


def write_four(out_dir, gm, rm, mi, fk):
    os.makedirs(out_dir, exist_ok=True)
    for name, t in (("gen-matches", gm), ("reco-matches", rm), ("misses", mi), ("fakes", fk)):
        path = os.path.join(str(out_dir), f"{name}.arrow")
        _write_arrow(t, path)
        print(f"  wrote {path}  (n={len(t)}, sum_w={t['weight'].to_numpy().sum():.4g})")


def run_gen_prior(cfg, *, alt_gen_table, out_sysvar, spec, estimator, out_dir=None):
    """Reweight Pythia6 *gen* toward an alternate generator's gen jets.

    `alt_gen_table` is the prepared alt-generator gen table (pos class); the caller builds it
    (e.g. via the herwig7/pythia8 preparation) so this stays estimator/path-focused. `r` is
    applied to gen-matches/misses, the same `r` reused on reco-matches, fakes unchanged.
    """
    arrows = load_nominal_arrows(cfg, spec)
    gm, rm, mi, fk = (arrows["gen-matches"], arrows["reco-matches"], arrows["misses"], arrows["fakes"])
    n_gm = len(gm)
    assert len(rm) == n_gm, f"reco-matches ({len(rm)}) != gen-matches ({n_gm})"

    p6_gen = pa.concat_tables((gm, mi))  # [gen-matches ; misses]
    r_gen = density_ratio(alt_gen_table, p6_gen, spec=spec, estimator=estimator, label="gen_prior")
    r_gm, r_mi = r_gen[:n_gm], r_gen[n_gm:]

    out_dir = out_dir or _paths.embedding_dir(cfg, out_sysvar, mode=spec.name)
    write_four(out_dir, _scaled(gm, r_gm), _scaled(rm, r_gm), _scaled(mi, r_mi), fk)
    return out_dir


def run_data_reco(cfg, *, spec, estimator, out_dir=None):
    """Reweight Pythia6 *reco* toward real data (the LIKE_DATA closure prior): a 1-iteration
    OmniFold — reco density-ratio step + gen pull-back step (misses get the learned
    pull-back only)."""
    out_sysvar = SysVar.UNFOLDING_PRIOR_LIKE_DATA
    arrows = load_nominal_arrows(cfg, spec)
    gm, rm, mi, fk = (arrows["gen-matches"], arrows["reco-matches"], arrows["misses"], arrows["fakes"])
    n_gm, n_rm = len(gm), len(rm)
    assert n_rm == n_gm, f"reco-matches ({n_rm}) != gen-matches ({n_gm}); matched-pair order broken"

    data_table = _read_arrow(_paths.data_arrow(cfg, spec.name))
    p6_reco = pa.concat_tables((rm, fk))  # [reco-matches ; fakes]

    # Step 1: reco density ratio (data vs p6 reco).
    r_reco = density_ratio(data_table, p6_reco, spec=spec, estimator=estimator, label="reco")
    r_reco_match, r_reco_fake = r_reco[:n_rm], r_reco[n_rm:]

    # Step 2: gen pull-back (OmniFold step 2). Push matched-reco ratio onto the gen partner,
    # learn a smooth gen density ratio, predict on ALL gen (incl. misses). No weight-flatten:
    # pos/neg are the same gen jets, only the weights differ.
    w_gm = gm["weight"].to_numpy().astype(np.float32)
    gm_pushed = replace_table_column(gm, "weight", w_gm * r_reco_match)
    p6_gen_full = pa.concat_tables((gm, mi))
    r_gen = density_ratio(
        gm_pushed, gm, spec=spec, estimator=estimator, predict_table=p6_gen_full,
        label="gen_pullback", weight_flatten=False,
    )
    r_gm, r_mi = r_gen[:n_gm], r_gen[n_gm:]

    out_dir = out_dir or _paths.embedding_dir(cfg, out_sysvar, mode=spec.name)
    write_four(
        out_dir,
        replace_table_column(gm, "weight", w_gm * r_gm),
        _scaled(rm, r_reco_match),
        _scaled(mi, r_mi),
        _scaled(fk, r_reco_fake),
    )
    return out_dir
