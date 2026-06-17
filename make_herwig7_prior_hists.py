"""SUPERSEDED (kept for reference). This built closure-style unfolded/truth/ratio
histograms for the OLD HERWIG7 prior wiring (reweighted reco as pseudo-data).
HERWIG7/PYTHIA8 are now model-dependence variations that unfold the REAL data
with the reweighted embedding as the prior, producing a top-level alternate
unfolded snapshot directly via ``histograms.py`` (no truth/ratio). Use
``histograms.py`` for the new path; ``systematics.calculate_uncertainties``
reads HERWIG7/PYTHIA8 from the top-level snapshot as ``|nominal - alt|``.

Generate HERWIG7-prior histograms (unfolded / truth / ratio) so that
``systematics.py`` can incorporate them as a prior systematic.

Reuses the histogramming helpers defined inside ``histograms.py`` (a marimo
notebook). The notebook's ``@app.function`` decorations expose them as
plain callables on the module, and importing it triggers the ``app.setup``
block that binds ``feature_mode`` etc. We deliberately ignore the cell
graph and only call the helpers we need.

Outputs (matching the layout ``systematics.calculate_prior_uncertainty``
reads from)::

    outputs/histograms/unf_prior_herwig7/<feature_mode>/<var>/
        unfolded/<file>.pt
        truth/<file>.pt
        ratio/<file>.pt

The HERWIG7 multifold run stopped after iteration 3 (the 4th was in
flight), so we use the last completed iteration's gen-side weights.
"""

import json
import os

import numpy as np
import pyarrow as pa
import torch

import histograms as H
from systematics import SysVar, get_jet_pt_bins


SYS_VAR = SysVar.UNFOLDING_PRIOR_HERWIG7
DATASET_ROOT = "./datasets/STAR_pp200GeV_production_2012"


def _last_completed_iter(w_unfolding_npz):
    """The npz stores (gen, reco) pairs per iteration: arr_0/1 = prior,
    arr_{2k}/arr_{2k+1} = iteration k. Return the largest k present."""
    keys = sorted(w_unfolding_npz.files, key=lambda k: int(k.split("_")[1]))
    return (len(keys) // 2) - 1


def load_bins():
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as f:
        bins = json.load(f)
    bins["pt"] = get_jet_pt_bins(SYS_VAR)
    for ang in H.angularities:
        bins[f"sd_{ang}"] = bins[ang]
    bins["m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    bins["sd_m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    return bins


def load_gen_table():
    bufs = []
    root = os.path.join(DATASET_ROOT, "jets", "embedding", str(SysVar.NONE))
    bufs.append(pa.memory_map(os.path.join(root, "gen-matches.arrow")))
    gm = pa.ipc.open_file(bufs[-1]).read_all()
    bufs.append(pa.memory_map(os.path.join(root, "misses.arrow")))
    ms = pa.ipc.open_file(bufs[-1]).read_all()
    return pa.concat_tables((gm, ms)), bufs


def load_unfolded_weights(iteration):
    path = os.path.join(
        DATASET_ROOT, "features", H.feature_mode,
        "embedding", str(SYS_VAR), "w_unfolding.npz",
    )
    print(f"Reading unfolded weights from {path}")
    w = np.load(path)
    last = _last_completed_iter(w)
    use = min(iteration, last)
    if use != iteration:
        print(f"  requested iter {iteration} not available, using last completed iter {use}")
    return torch.as_tensor(w[f"arr_{2 * use}"], dtype=torch.float32), use


def load_truth_weights():
    bufs = []
    root = os.path.join(
        DATASET_ROOT, "features", H.feature_mode, "embedding", str(SYS_VAR),
    )
    print(f"Reading truth weights from {root}")
    bufs.append(pa.memory_map(os.path.join(root, "gen-matches.arrow")))
    gm = pa.ipc.open_file(bufs[-1]).read_all()
    bufs.append(pa.memory_map(os.path.join(root, "misses.arrow")))
    ms = pa.ipc.open_file(bufs[-1]).read_all()
    w = np.concatenate([gm["weight"].to_numpy(), ms["weight"].to_numpy()])
    return torch.as_tensor(w, dtype=torch.float32), bufs


def main():
    bins = load_bins()
    gen_table, _gen_bufs = load_gen_table()
    unf_weights, used_iter = load_unfolded_weights(iteration=5)
    truth_weights, _truth_bufs = load_truth_weights()

    out_prefix = os.path.join("outputs", "histograms", str(SYS_VAR), H.feature_mode)
    os.makedirs(out_prefix, exist_ok=True)
    print(f"Writing histograms under: {out_prefix}  (using iter {used_iter})")

    # Common 1D vars vs pt
    for var in H.common_vars[1:]:
        h_unf = H.histogram(gen_table, bins, ("pt", var), unf_weights)
        h_true = H.histogram(gen_table, bins, ("pt", var), truth_weights)
        H.save_hist_2d(
            h_unf,
            prefix=os.path.join(out_prefix, var),
            fname_prefix="hist",
            batched=True,
            true_hist=h_true,
        )
        del h_unf, h_true

    # Angularities: hist_ang, hist_sd_ang, ratio_incl_vs_sd
    for ang in H.angularities:
        ang_prefix = os.path.join(out_prefix, ang)
        h_unf = H.histogram(gen_table, bins, ("pt", ang, f"sd_{ang}"), unf_weights)
        h_true = H.histogram(gen_table, bins, ("pt", ang, f"sd_{ang}"), truth_weights)

        h_unf_incl = h_unf.project("pt", ang)
        h_unf_sd = h_unf.project("pt", f"sd_{ang}")
        h_true_incl = h_true.project("pt", ang)
        h_true_sd = h_true.project("pt", f"sd_{ang}")

        H.save_hist_2d(h_unf_incl, prefix=ang_prefix, fname_prefix="hist_ang",
                       batched=True, true_hist=h_true_incl)
        H.save_hist_2d(h_unf_sd, prefix=ang_prefix, fname_prefix="hist_sd_ang",
                       batched=True, true_hist=h_true_sd)
        H.save_ratio_2d(h_unf_sd, h_unf_incl, prefix=ang_prefix,
                        fname_prefix="ratio_incl_vs_sd", batched=True,
                        true_hists=(h_true_sd, h_true_incl))
        del h_unf, h_true, h_unf_incl, h_unf_sd, h_true_incl, h_true_sd

    # Profiles for each angularity vs each common_var/ang/sd_ang
    for ang in H.angularities:
        ang_prefix = os.path.join(out_prefix, ang)

        # prof_incl_vs_<x>  (ang as y, binned in (pt, common_vars..., sd_ang))
        bin_cols_incl = (*H.common_vars, f"sd_{ang}")
        p_unf = H.profile(gen_table, bins, bin_cols_incl, ang, unf_weights)
        p_true = H.profile(gen_table, bins, bin_cols_incl, ang, truth_weights)
        for x in bin_cols_incl[1:]:
            H.save_hist_2d(
                p_unf.project("pt", x),
                prefix=ang_prefix,
                fname_prefix=f"prof_incl_vs_{x}",
                batched=True,
                true_hist=p_true.project("pt", x),
            )
        del p_unf, p_true

        # prof_sd_vs_<x>  (sd_ang as y, binned in (pt, common_vars..., ang))
        bin_cols_sd = (*H.common_vars, ang)
        p_unf = H.profile(gen_table, bins, bin_cols_sd, f"sd_{ang}", unf_weights)
        p_true = H.profile(gen_table, bins, bin_cols_sd, f"sd_{ang}", truth_weights)
        for x in bin_cols_sd[1:]:
            H.save_hist_2d(
                p_unf.project("pt", x),
                prefix=ang_prefix,
                fname_prefix=f"prof_sd_vs_{x}",
                batched=True,
                true_hist=p_true.project("pt", x),
            )
        del p_unf, p_true

    print("Done.")


if __name__ == "__main__":
    main()
