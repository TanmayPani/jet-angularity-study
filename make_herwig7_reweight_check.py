"""Overlay 1D distributions of nominal Pythia6, HERWIG7-reweighted Pythia6,
and native HERWIG7 in jet-pT bins, with a ratio panel relative to
nominal Pythia6.

Sanity check on the GP-based reweighting that produces the HERWIG7
prior: does the reweighted-P6 actually move from P6 toward HERWIG7 at
gen level? Per-variable, per-pT-bin overlays.
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
import matplotlib.pyplot as plt

from systematics import SysVar, get_jet_pt_bins

DATASET_ROOT = Path("./datasets/STAR_pp200GeV_production_2012")
HERWIG7_FILE = Path(
    "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs/combined_HwJets_nEv500000.arrow"
)
OUT_DIR = Path("outputs/histograms/plots/angularities/herwig7_reweight_check")

PT_BINS = get_jet_pt_bins(SysVar.NONE)

VARS_AND_BINS = {
    "m":            np.linspace(0, 12, 25),
    "sd_m":         np.linspace(0, 12, 25),
    "sd_dR":        np.linspace(0, 0.45, 16),
    "sd_symmetry":  np.linspace(0.1, 0.5, 17),
    "ch_ang_k1_b0.5": np.linspace(0.0, 0.75, 26),
    "ch_ang_k1_b1":   np.linspace(0.0, 0.5, 26),
    "ch_ang_k1_b2":   np.linspace(0.0, 0.4, 26),
    "ch_ang_k2_b0":   np.linspace(0.0, 1.0, 26),
    "sd_ch_ang_k1_b0.5": np.linspace(0.0, 0.75, 26),
    "sd_ch_ang_k1_b1":   np.linspace(0.0, 0.5, 26),
    "sd_ch_ang_k1_b2":   np.linspace(0.0, 0.4, 26),
    "sd_ch_ang_k2_b0":   np.linspace(0.0, 1.0, 26),
}

VAR_LABEL = {
    "m": r"$M_{\rm jet}$ (GeV)",
    "sd_m": r"$M_{\rm g, jet}$ (GeV)",
    "sd_dR": r"$\Delta R_{\rm g, jet}$",
    "sd_symmetry": r"$z_{\rm g, jet}$",
    "ch_ang_k1_b0.5": r"$\lambda^{\kappa=1}_{\beta=0.5}$",
    "ch_ang_k1_b1":   r"$\lambda^{\kappa=1}_{\beta=1}$",
    "ch_ang_k1_b2":   r"$\lambda^{\kappa=1}_{\beta=2}$",
    "ch_ang_k2_b0":   r"$\lambda^{\kappa=2}_{\beta=0}$",
    "sd_ch_ang_k1_b0.5": r"$\lambda^{\kappa=1, SD}_{\beta=0.5}$",
    "sd_ch_ang_k1_b1":   r"$\lambda^{\kappa=1, SD}_{\beta=1}$",
    "sd_ch_ang_k1_b2":   r"$\lambda^{\kappa=1, SD}_{\beta=2}$",
    "sd_ch_ang_k2_b0":   r"$\lambda^{\kappa=2, SD}_{\beta=0}$",
}


def load_arrow(path):
    src = pa.memory_map(str(path), "r")
    return pa.ipc.open_file(src).read_all(), src


def load_gen_p6():
    """Nominal Pythia6 gen-level table (concat of gen-matches + misses)."""
    bufs = []
    root = DATASET_ROOT / "jets" / "embedding" / "nominal"
    bufs.append(pa.memory_map(str(root / "gen-matches.arrow"), "r"))
    gm = pa.ipc.open_file(bufs[-1]).read_all()
    bufs.append(pa.memory_map(str(root / "misses.arrow"), "r"))
    ms = pa.ipc.open_file(bufs[-1]).read_all()
    return pa.concat_tables((gm, ms)), bufs


def load_gen_p6_reweighted():
    """HERWIG7-reweighted Pythia6 gen table: same kinematics as nominal
    P6 gen, but with the GP-reweight baked into the `weight` column."""
    bufs = []
    root = DATASET_ROOT / "features" / "angularities" / "embedding" / "unf_prior_herwig7"
    bufs.append(pa.memory_map(str(root / "gen-matches.arrow"), "r"))
    gm = pa.ipc.open_file(bufs[-1]).read_all()
    bufs.append(pa.memory_map(str(root / "misses.arrow"), "r"))
    ms = pa.ipc.open_file(bufs[-1]).read_all()
    return pa.concat_tables((gm, ms)), bufs


def load_herwig7():
    src = pa.memory_map(str(HERWIG7_FILE), "r")
    return pa.ipc.open_file(src).read_all(), [src]


def hist_1d(table, var, bins, pt_lo, pt_hi):
    pt = table["pt"].to_numpy()
    mask = (pt >= pt_lo) & (pt < pt_hi)
    if var not in table.column_names:
        return None
    x = table[var].to_numpy()[mask]
    w = table["weight"].to_numpy()[mask]
    finite = np.isfinite(x) & np.isfinite(w)
    x = x[finite]; w = w[finite]
    counts, _ = np.histogram(x, bins=bins, weights=w)
    sumw2, _ = np.histogram(x, bins=bins, weights=w * w)
    integral = counts.sum()
    if integral > 0:
        counts = counts / integral
        sumw2 = sumw2 / (integral * integral)
    err = np.sqrt(np.clip(sumw2, 0, None))
    return counts, err


def plot_one_var(var, bins, p6, p6r, hw, out_path):
    n_jpt = len(PT_BINS) - 1
    fig, axes = plt.subplots(
        2, n_jpt, sharex="col",
        figsize=(5 * n_jpt, 6.5),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05, "wspace": 0.05},
    )
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    bin_widths = bins[1:] - bins[:-1]

    style = {
        "P6":          dict(color="C0", linestyle="-",  label="PYTHIA6 (nominal)"),
        "P6_reweight": dict(color="C1", linestyle="--", label=r"PYTHIA6 $\times$ HERWIG7 reweight"),
        "HW":          dict(color="C2", linestyle=":",  label="HERWIG7 (native)"),
    }

    for ij in range(n_jpt):
        pt_lo, pt_hi = PT_BINS[ij], PT_BINS[ij + 1]
        ax_top = axes[0, ij] if n_jpt > 1 else axes[0]
        ax_bot = axes[1, ij] if n_jpt > 1 else axes[1]

        hists = {}
        for name, tbl in (("P6", p6), ("P6_reweight", p6r), ("HW", hw)):
            res = hist_1d(tbl, var, bins, pt_lo, pt_hi)
            hists[name] = res

        ref_counts, ref_err = hists["P6"]
        for name, res in hists.items():
            counts, err = res
            ax_top.stairs(counts, bins, **style[name], linewidth=1.6)
            ax_top.errorbar(
                bin_centers, counts, xerr=bin_widths / 2, yerr=err,
                fmt="none", color=style[name]["color"], alpha=0.6,
            )

            if name == "P6":
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = counts / ref_counts
                ratio_err = ratio * np.sqrt(
                    (err / np.where(counts != 0, counts, np.nan)) ** 2
                    + (ref_err / np.where(ref_counts != 0, ref_counts, np.nan)) ** 2
                )
            ax_bot.errorbar(
                bin_centers, ratio, xerr=bin_widths / 2, yerr=ratio_err,
                fmt="o", markersize=3, color=style[name]["color"],
            )

        ax_top.set_title(rf"${pt_lo:g} < p_{{T,jet}} < {pt_hi:g}$ GeV/$c$", fontsize="medium")
        ax_top.set_ylabel(r"$\frac{1}{N}\,dN/dx$" if ij == 0 else "")
        ax_bot.axhline(1.0, color="black", linewidth=0.7, linestyle="-")
        ax_bot.set_ylim(0.0, 2.0)
        ax_bot.set_xlabel(VAR_LABEL.get(var, var))
        ax_bot.set_ylabel("ratio / P6" if ij == 0 else "")
        if ij == n_jpt - 1:
            ax_top.legend(fontsize="small", frameon=False, loc="best")

    fig.suptitle(VAR_LABEL.get(var, var) + r" at gen level: PYTHIA6, PYTHIA6$\times$reweight, HERWIG7",
                 fontsize="large", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main():
    p6, _b1 = load_gen_p6()
    p6r, _b2 = load_gen_p6_reweighted()
    hw, _b3 = load_herwig7()
    print(f"P6 gen rows: {p6.num_rows}  P6-reweight rows: {p6r.num_rows}  HW rows: {hw.num_rows}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for var, bins in VARS_AND_BINS.items():
        plot_one_var(var, bins, p6, p6r, hw, OUT_DIR / f"{var}.pdf")


if __name__ == "__main__":
    main()
