"""Build a Pythia6-embedding variant whose per-jet weights are reweighted to match an
alternate gen-level generator (Herwig7, Pythia8, ...) via a classifier-based density
ratio. Detector simulation is not re-run; only the part-/det-level weight column changes.

CLI:
    python make_alt_embedding.py --generator herwig7
    python make_alt_embedding.py --generator pythia8

Outputs four arrows under
    ./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding/unf_prior_<gen>/
which `preprocessing.make_datasets_for_unfolding(sysvar=UNFOLDING_PRIOR_<GEN>, ...)`
picks up unchanged. Fakes pass through unmodified.
"""

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pyarrow as pa
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from torchstrap.callbacks import EarlyStopping
from torchstrap.optimizer import Adam
from torchstrap.stateless import StatelessModule
from torchstrap.utils.nn.archs import MLP

from dataset import TensorDictDataset, build_input_transform
from config import load_config
from multifold import (
    reweight_inference_loaders,
    train_test_multi_loaders,
)
from preprocessing import (
    N_BINS,
    process_table,
    pth_bins,
    replace_table_column,
    to_tensordict,
)
from systematics import SysVar


# Mirror multifold.get_sample_reweights' clamp_min / clamp_max defaults; tightened
# from the original [1e-3, 1e3] to [0.1, 10] for the alt-prior classifier. Two decades
# symmetric is more than enough range for Herwig7-vs-Pythia6 at RHIC; the wider window
# only ever served to inflate Σw² and collapse ESS.
ODDS_CLIP_LO = 0.1
ODDS_CLIP_HI = 10.0

# Accepted strings for build_input_transform (see dataset.build_input_transform,
# dataset.py:56-72). Kept local so make_alt_embedding doesn't re-export from dataset.
_VALID_TRANSFORMS = ("none", "z_norm", "log1p_z_norm")


def _odds_clipped(p, eps=None):
    """Convert sigmoid output `p = p(alt | x)` to clamped density-ratio odds
    `r = p / (1 - p)`. Mirrors the tail of multifold.get_sample_reweights
    (multifold.py:326-337); kept inline so the alt-prior path doesn't depend on
    multifold's internal helper layout.
    """
    preco = 1.0 - p
    if eps is None:
        eps = preco[preco > 0.0].min() * torch.finfo(preco.dtype).eps
    return (p / preco.add_(eps)).squeeze_(-1).clamp_(min=ODDS_CLIP_LO, max=ODDS_CLIP_HI)


# Columns actually produced by preprocessing.process_table for both p6 and herwig7
# (preprocessing.jet_columns lists nef + angularities that calculate_angularities
# would produce, but that call is currently disabled). Keep this list as the source
# of truth for the alt-prior classifier features so the schema check passes.
KINEMATIC_COLUMNS = (
    "pt",
    "eta",
    "phi",
    "m",
    "ncharged",
    "nconstituents",
    "leading_constit_pt",
    "leading_constit_eta",
    "leading_constit_phi",
    "subleading_constit_pt",
    "subleading_constit_eta",
    "subleading_constit_phi",
    "sd_pt",
    "sd_eta",
    "sd_phi",
    "sd_m",
    "sd_pz",
    "sd_dR",
    "sd_symmetry",
    "sd_ncharged",
    "sd_nconstituents",
)
ALT_FEATURE_MODES = ("kinematics", "bin_counts", "combined")
NUM_FEATURES = {
    "kinematics": len(KINEMATIC_COLUMNS),
    "bin_counts": N_BINS,
    "combined": len(KINEMATIC_COLUMNS) + N_BINS,
}


P6_EMBEDDING_DIR = (
    "./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding/"
    + str(SysVar.NONE)
)

HERWIG7_ROOT = (
    "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs"
)
PYTHIA8_ARROW = (
    "./datasets/STAR_pp200GeV_production_2012/preproc_Pythia8_pp200GeV_r0.4.arrow"
)

# Maps generator name → (output SysVar, cache subdir).
ALT_GENERATORS = {
    "herwig7": (SysVar.UNFOLDING_PRIOR_HERWIG7, "outputs/alt_embedding/herwig7"),
    "pythia8": (SysVar.UNFOLDING_PRIOR_PYTHIA8, "outputs/alt_embedding/pythia8"),
}


def _flatten_weights_per_bin(table, bin_col="pth_bin", weight_col="weight"):
    """Return per-jet weights normalised so each bin of `bin_col` contributes equally
    to a weighted loss: within bin `b`, `sum(out[bin==b]) == count(bin==b)` (i.e. the
    mean weight in every bin is 1.0). Equivalent to dividing each jet's weight by its
    bin's mean weight.

    Removes the multi-decade pT-hat MC weight spread from the BCE training loss; the
    classifier no longer over-fits low-pT-hat-dominated regions. Inference / post-hoc
    cross-section calibration must continue to use the ORIGINAL weight column.

    Bins with mean(w) <= 0 (no signal / degenerate) are left at their input values. A
    sentinel bin (e.g. pth_bin == -1 for Pythia8 unbinned) with uniform input weights
    collapses to all-ones, which is the desired behaviour for that case.
    """
    w = table[weight_col].to_numpy().astype(np.float32, copy=True)
    b = table[bin_col].to_numpy().astype(np.int64)
    out = w.copy()
    for ub in np.unique(b):
        sel = b == ub
        m = float(w[sel].mean())
        if m > 0.0:
            out[sel] = w[sel] / m
    return out


def histogram_reweight(alt_table, p6_table, *, var, bins):
    """Compute per-jet ratio r(pt) = w_alt(pt_bin) / w_p6(pt_bin) using histograms.

    Returns a `(len(p6_table),)` float32 array suitable for the existing splice-back.
    Within each bin the ratio is constant, so weights are guaranteed bounded — this is
    the standard fix for the classifier's weight degeneracy.
    """
    import thoda

    pt_p6 = np.asarray(p6_table[var].to_numpy(zero_copy_only=False), dtype=np.float64)
    pt_alt = np.asarray(alt_table[var].to_numpy(zero_copy_only=False), dtype=np.float64)
    w_p6 = np.asarray(
        p6_table["weight"].to_numpy(zero_copy_only=False), dtype=np.float64
    )
    w_alt = np.asarray(
        alt_table["weight"].to_numpy(zero_copy_only=False), dtype=np.float64
    )

    h_p6, _ = thoda.Histogram.create([pt_p6], bins=[bins], weights=w_p6)
    h_alt, _ = thoda.Histogram.create([pt_alt], bins=[bins], weights=w_alt)
    snap_p6 = h_p6.snapshot(density=False)
    snap_alt = h_alt.snapshot(density=False)

    # Thoda's snapshot values are length nbins+2 (underflow at [0], overflow at [-1]).
    # Take the in-range slice and compute ratio with a safe divide.
    vals_alt = np.asarray(snap_alt.values, dtype=np.float64)[1:-1]
    vals_p6 = np.asarray(snap_p6.values, dtype=np.float64)[1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_per_bin = np.where(vals_p6 > 0, vals_alt / vals_p6, 0.0)

    # Print per-bin ratios and warn on low-stats Herwig7 bins (raw counts, not weights).
    n_alt_raw, _ = np.histogram(pt_alt, bins=bins)
    n_p6_raw, _ = np.histogram(pt_p6, bins=bins)
    print(f"[alt-embedding] histogram reweight on '{var}', {len(bins) - 1} bins:")
    print(
        f"    {'bin':>5} {'edge_lo':>10} {'edge_hi':>10} {'n_alt':>10} {'n_p6':>10} {'ratio':>10}"
    )
    low_stats = []
    for i, (lo, hi, na, np_, r) in enumerate(
        zip(bins[:-1], bins[1:], n_alt_raw, n_p6_raw, ratio_per_bin)
    ):
        flag = " <-- low alt stats" if na < 10 else ""
        if na < 10:
            low_stats.append(i)
        print(
            f"    {i:>5d} {lo:>10.2f} {hi:>10.2f} {na:>10d} {np_:>10d} {r:>10.4f}{flag}"
        )
    if low_stats:
        print(
            f"[alt-embedding] WARNING: {len(low_stats)} bin(s) have < 10 herwig jets "
            f"(indices {low_stats}); ratios there are noisy or zero. Consider widening bins."
        )

    # np.digitize: 0 for x<bins[0], len(bins) for x>bins[-1], 1..len(bins)-1 otherwise.
    # Clamp out-of-range jets to the nearest in-range bin so they get a defined weight
    # (rather than a 0-weight underflow).
    bin_idx = np.clip(np.digitize(pt_p6, bins) - 1, 0, len(bins) - 2)
    return ratio_per_bin[bin_idx].astype(np.float32)


def _audit_feature_alignment(alt_table, p6_table, columns, max_samples=200_000, seed=0):
    """Compute distribution-overlap metrics between alt and p6 for each scalar column.

    Returns a list of (column, range_overlap, ks_dist) sorted by ks_dist desc, and
    prints a formatted table plus an auto-recommended `--drop-features` list.

    Why: the alt-prior classifier exploits any systematic difference in feature
    *convention* (jet-mass conventions, softdrop sentinels, particle-pT cutoffs) to
    over-confidently separate the generators — producing extreme per-jet ratios and
    collapsing the effective sample size. This audit surfaces the offenders.
    """
    try:
        from scipy.stats import ks_2samp

        def _ks(a, b):
            return float(ks_2samp(a, b).statistic)
    except ImportError:
        # numpy-only CDF-difference fallback: bins both into a common 1024-point grid
        # over the union range, computes empirical CDFs, returns max abs difference.
        def _ks(a, b):
            lo = float(min(a.min(), b.min()))
            hi = float(max(a.max(), b.max()))
            if hi <= lo:
                return 0.0
            edges = np.linspace(lo, hi, 1025)
            ca = np.searchsorted(np.sort(a), edges) / len(a)
            cb = np.searchsorted(np.sort(b), edges) / len(b)
            return float(np.max(np.abs(ca - cb)))

    rng = np.random.default_rng(seed)

    def _sample(table, col):
        x = np.asarray(
            table.column(col).to_numpy(zero_copy_only=False), dtype=np.float64
        )
        if len(x) > max_samples:
            x = x[rng.choice(len(x), max_samples, replace=False)]
        return x

    rows = []
    for col in columns:
        if col not in alt_table.column_names or col not in p6_table.column_names:
            continue
        a = _sample(alt_table, col)
        p = _sample(p6_table, col)
        a_lo, a_hi = float(a.min()), float(a.max())
        p_lo, p_hi = float(p.min()), float(p.max())
        union = max(a_hi, p_hi) - min(a_lo, p_lo)
        overlap_width = max(0.0, min(a_hi, p_hi) - max(a_lo, p_lo))
        range_overlap = overlap_width / union if union > 0 else 1.0
        ks = _ks(a, p)
        rows.append((col, range_overlap, ks))

    rows.sort(key=lambda r: -r[2])

    def _flag(ks):
        if ks > 0.5:
            return "***"
        if ks > 0.25:
            return "**"
        if ks > 0.1:
            return "*"
        return "ok"

    print(
        f"[alt-embedding] feature alignment audit (p6 vs alt, up to {max_samples} samples each):"
    )
    print(f"    {'column':<28}  {'range_overlap':>14}  {'ks_dist':>8}  flag")
    suspect = []
    for col, ro, ks in rows:
        f = _flag(ks)
        print(f"    {col:<28}  {ro:>14.4f}  {ks:>8.4f}  {f}")
        if f != "ok":
            suspect.append(col)

    if suspect:
        print(f"[alt-embedding] recommended: --drop-features {','.join(suspect)}")
    else:
        print("[alt-embedding] no suspect features flagged.")
    return rows, suspect


def _write_arrow(table_or_batch, path):
    """Write a Table or RecordBatch to an IPC file at `path`. Caller manages dirs."""
    if isinstance(table_or_batch, pa.Table):
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_file(sink, table_or_batch.schema) as writer:
                for batch in table_or_batch.to_batches():
                    writer.write_batch(batch)
    else:
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_file(sink, table_or_batch.schema) as writer:
                writer.write_batch(table_or_batch)


def _read_arrow(path):
    return pa.ipc.open_file(pa.memory_map(path, "rb")).read_all()


def prepare_herwig7(cache_dir):
    """Build a single arrow table of Herwig7 gen jets with the canonical part_lvl schema
    (weight, bin_index, bin_count, pth_bin, is_data, is_matched).

    The per-file `histogramScale = crossSection / sumWeights` from the sibling JSON is
    baked into the `weight` column, mirroring the per-pthat-bin xsec weighting that
    `cluster_embedding.py` applies for Pythia6.
    """
    cache_subdir = os.path.join(cache_dir, "preproc")
    os.makedirs(cache_subdir, exist_ok=True)

    # Map herwig7 pthat-folder names to indices in `preprocessing.pth_bins`.
    pth_idx = {}
    for i, (lo, hi) in enumerate(zip(pth_bins[:-1], pth_bins[1:])):
        hi_key = "150" if hi == "infty" else hi  # herwig7 caps at 150, p6 at infty
        pth_idx[f"ptHat{lo}-{hi_key}"] = i

    tables = []
    for folder in sorted(
        glob.glob(os.path.join(HERWIG7_ROOT, "HwJets_RHIC_ptHat*-*_nEv500000"))
    ):
        tag = os.path.basename(folder).split("_nEv")[0].replace("HwJets_RHIC_", "")
        if tag not in pth_idx:
            print(f"  [herwig7] skipping {folder} (no pthat-bin mapping for {tag!r})")
            continue
        ipth = pth_idx[tag]
        for arrow_path in sorted(glob.glob(os.path.join(folder, "processed_*.arrow"))):
            base = (
                os.path.basename(arrow_path)
                .replace("processed_", "")
                .replace(".arrow", "")
            )
            json_path = os.path.join(folder, "out", f"{base}.json")
            if not os.path.exists(json_path):
                print(f"  [herwig7] missing xsec JSON for {arrow_path}; skipping")
                continue
            with open(json_path) as f:
                meta = json.load(f)
            scale = float(meta["histogramScale"])

            cache_path = os.path.join(cache_subdir, f"preproc_{base}.arrow")
            if not os.path.exists(cache_path):
                raw = _read_arrow(arrow_path)
                # Bake histogramScale into the weight column before process_table consumes it.
                weight_col = np.full(len(raw), scale, dtype=np.float32)
                raw = replace_table_column(raw, "weight", weight_col)
                rb = process_table(
                    raw,
                    is_data=False,
                    is_matched=-1,
                    pth_bin=int(ipth),
                )
                _write_arrow(rb, cache_path)
                print(f"  [herwig7] processed {base}: rows={len(rb)} scale={scale:.4g}")
            tables.append(_read_arrow(cache_path))

    if not tables:
        raise RuntimeError(f"No herwig7 arrows found under {HERWIG7_ROOT}")
    return pa.concat_tables(tables)


def prepare_pythia8(cache_dir):
    """Pythia8 has a single unbinned arrow with weight=1 (no xsec normalization). The
    classifier-based ratio will compare unweighted-Pythia8 to xsec-weighted-Pythia6,
    which is a meaningful question (does the *shape* of Pythia8's gen-jet distribution
    differ from Pythia6's?) but is not a true xsec-weighted Pythia8 prior. Emit a
    loud warning here so the user can replace the weights with a real xsec lookup if
    needed.
    """
    print(
        "[pythia8] WARNING: weight column on disk is uniformly 1.0 — no pthat-xsec "
        "normalization. The trained classifier estimates the *unweighted shape ratio* "
        "Pythia8/Pythia6 rather than the xsec-weighted prior ratio. To fix, supply a "
        "per-file xsec table (analogous to herwig7's histogramScale)."
    )
    cache_path = os.path.join(cache_dir, "preproc_pythia8.arrow")
    if not os.path.exists(cache_path):
        os.makedirs(cache_dir, exist_ok=True)
        raw = _read_arrow(PYTHIA8_ARROW)
        rb = process_table(
            raw,
            is_data=False,
            is_matched=-1,
            pth_bin=-1,  # sentinel: pythia8 is not pthat-binned
        )
        _write_arrow(rb, cache_path)
        print(f"  [pythia8] processed: rows={len(rb)}")
    return _read_arrow(cache_path)


def load_p6_gen_table():
    """Load the Pythia6 part-level rows (gen-matches ∪ misses) from the default embedding.
    Row layout matches what `make_datasets_for_unfolding` produces downstream:
    [gen-matches ; misses]. We need to preserve that order so the per-row ratios can be
    spliced back into the four output arrows.
    """
    gm = _read_arrow(os.path.join(P6_EMBEDDING_DIR, "gen-matches.arrow"))
    mi = _read_arrow(os.path.join(P6_EMBEDDING_DIR, "misses.arrow"))
    return gm, mi, pa.concat_tables((gm, mi))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator",
        required=True,
        choices=sorted(ALT_GENERATORS.keys()),
        help="Which alternate generator's prior to bake into the embedding weights.",
    )
    parser.add_argument(
        "--config",
        default="runtime-files/config.json",
        help="Path to the runtime config (same one multifold.py reads).",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override config's num_epochs for the classifier (defaults to config).",
    )
    parser.add_argument(
        "--max-rows-per-class",
        type=int,
        default=None,
        help=(
            "Uniformly subsample each class (alt-gen, pythia6) down to this many rows "
            "before building the classifier TensorDict. The full Herwig7 sample is ~47M "
            "rows which blows up the vmap'd undersampling in train_test_multi_loaders "
            "when num_replicas > 1 — at num_replicas=1 the non-vmap path handles full "
            "samples comfortably. Default: 0 (no cap) when --num-replicas=1, else 2M. "
            "Pass 0 explicitly to disable the cap."
        ),
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help=(
            "Ensemble size for the alt-prior classifier. Default 1 (no ensembling) — "
            "decouples this script from cfg.num_replicas (used by the main unfolding "
            "multifold). At 1, the vmap-undersampling memory bottleneck vanishes and "
            "we can train on the FULL alt and p6 samples, which removes the extreme-r "
            "outliers caused by extrapolation into poorly-sampled regions. The "
            "r.std(dim=0) ensemble-spread diagnostic is meaningless at num_replicas=1 "
            "and will be skipped."
        ),
    )
    parser.add_argument(
        "--feature-mode",
        choices=ALT_FEATURE_MODES,
        default="kinematics",
        help=(
            "Classifier feature set for the alt-prior. 'kinematics' uses jet/softdrop "
            "scalars (incl. pT) — recommended for prior reweighting since substructure-"
            "only features cannot move the pT marginal. 'bin_counts' is the substructure-"
            "only mode (matches the unfolding's feature_mode but produces a flat pT "
            "reweighting). 'combined' concatenates both. Independent of cfg.feature_mode."
        ),
    )
    parser.add_argument(
        "--drop-features",
        default="",
        help=(
            "Comma-separated list of KINEMATIC_COLUMNS to exclude from the classifier "
            "(no-op for the histogram method). Use the feature-alignment audit's "
            "recommendation. Example: --drop-features m,sd_m,sd_pz"
        ),
    )
    parser.add_argument(
        "--method",
        choices=("histogram", "classifier"),
        default="histogram",
        help=(
            "Reweighting method. 'histogram' (default, recommended): bin both samples "
            "in --reweight-var, take per-bin ratio, apply per jet. Bounded weights, "
            "no NN training. 'classifier': the NN-based density ratio (use "
            "--drop-features per the audit to mitigate weight degeneracy)."
        ),
    )
    parser.add_argument(
        "--reweight-var",
        default="pt",
        help="Column to histogram-reweight on (histogram method only). Default 'pt'.",
    )
    parser.add_argument(
        "--reweight-bins",
        default="5.0,65.0,31",
        help=(
            "Bin edges for the histogram method as 'lo,hi,n' (n bin edges -> n-1 bins) "
            "or a comma-separated explicit list of edges. Default '5.0,65.0,31' "
            "(2 GeV bins from 5 to 65 GeV)."
        ),
    )
    parser.add_argument(
        "--balance-class-sizes",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Undersample the larger class (typically Herwig7, ~47M rows) to match "
            "the smaller (Pythia6 gen-matches+misses, ~14.5M). Default 'auto' = on "
            "for --method classifier, off for histogram. Reason: TensorDictDataset "
            "class-rebalancing (is_categorical=True) sets sum_w_class = N_class, so "
            "with N_alt = 3.24 * N_p6 the BCE optimum has E_p6[r*] = 3.24 instead of "
            "1 — a systematic calibration offset on top of the disjointness bias. "
            "Skipped if it would discard > 50%% of the rare class."
        ),
    )
    parser.add_argument(
        "--weight-flatten",
        choices=("per-pth-bin", "winsorise", "off"),
        default="per-pth-bin",
        help=(
            "How to tame the per-jet MC weight column before it enters the BCE loss. "
            "'per-pth-bin' (default, recommended): scale weights so each pth_bin's "
            "mean is 1 — removes the multi-decade pT-hat cross-section spread that "
            "otherwise lets low-pT-hat dominate the loss and over-fit low pT. "
            "'winsorise': cap weights at the --weight-winsorise-q quantile. "
            "'off': raw weights (today's behaviour, kept for A/B regression)."
        ),
    )
    parser.add_argument(
        "--weight-winsorise-q",
        type=float,
        default=0.99,
        help=(
            "Quantile cap for --weight-flatten=winsorise (no-op otherwise). "
            "Default 0.99 — caps the top 1%% of weights at the 99th-percentile value."
        ),
    )
    args = parser.parse_args()

    dropped = tuple(c.strip() for c in args.drop_features.split(",") if c.strip())
    unknown = [c for c in dropped if c not in KINEMATIC_COLUMNS]
    if unknown:
        raise ValueError(
            f"--drop-features contains columns not in KINEMATIC_COLUMNS: {unknown}. "
            f"Valid: {KINEMATIC_COLUMNS}"
        )
    effective_kinematic_columns = tuple(
        c for c in KINEMATIC_COLUMNS if c not in dropped
    )
    if dropped:
        print(f"[alt-embedding] dropping features: {list(dropped)}")
        print(
            f"[alt-embedding] effective kinematic columns ({len(effective_kinematic_columns)}): {list(effective_kinematic_columns)}"
        )

    # Parse --reweight-bins. Accept either "lo,hi,n" (linspace) or an explicit edge list.
    rb_parts = [p.strip() for p in args.reweight_bins.split(",") if p.strip()]
    if (
        len(rb_parts) == 3
        and all(
            p.replace(".", "", 1).replace("-", "", 1).isdigit() for p in rb_parts[:2]
        )
        and rb_parts[2].isdigit()
    ):
        reweight_bins = np.linspace(
            float(rb_parts[0]), float(rb_parts[1]), int(rb_parts[2])
        )
    else:
        reweight_bins = np.asarray([float(p) for p in rb_parts], dtype=np.float64)
        if len(reweight_bins) < 2:
            raise ValueError(
                f"--reweight-bins must give >= 2 edges; got {args.reweight_bins!r}"
            )

    print(f"[alt-embedding] method={args.method}", end="")
    if args.method == "histogram":
        print(
            f", var={args.reweight_var}, bins=[{reweight_bins[0]:.2f}..{reweight_bins[-1]:.2f}]"
            f" ({len(reweight_bins) - 1} bins)"
        )
    else:
        print(
            f", feature_mode={args.feature_mode}, num_replicas={int(args.num_replicas)}, "
            f"dropped_features={list(dropped) if dropped else 'none'}"
        )

    cfg = load_config(args.config)

    feature_mode = args.feature_mode
    count_transform = cfg.get("count_transform", "z_norm")
    # `sd_dR` and `sd_symmetry` carry -1 as a "soft-drop grooming failed" sentinel
    # (see preprocessing.get_softdrop_groomed_jets). log1p(-1) = -inf, which after
    # Normalize gives inf logits and NaN BCE from epoch 1. Force z_norm whenever the
    # feature set includes kinematic scalars; bin_counts alone is non-negative so its
    # log1p_z_norm path stays untouched.
    if feature_mode in ("kinematics", "combined") and count_transform == "log1p_z_norm":
        print(
            f"[alt-embedding] overriding count_transform 'log1p_z_norm' -> 'z_norm' for "
            f"feature_mode={feature_mode!r}: kinematic columns contain -1 sentinels "
            f"(failed softdrop) and log1p(-1)=-inf would NaN the loss."
        )
        count_transform = "z_norm"
    num_replicas = int(args.num_replicas)
    if num_replicas < 1:
        raise ValueError(f"--num-replicas must be >= 1, got {num_replicas}")
    # Default --max-rows-per-class depends on num_replicas: full samples at 1, capped at 2M
    # for ensembled runs (where vmap'd index tensors blow up at full scale).
    if args.max_rows_per_class is None:
        max_rows_per_class = 0 if num_replicas == 1 else 2_000_000
    else:
        max_rows_per_class = args.max_rows_per_class
    batch_size = int(cfg["batch_size"])
    train_size = float(cfg["train_size"])
    num_epochs = int(
        args.num_epochs if args.num_epochs is not None else cfg["num_epochs"]
    )
    dataseed = int(cfg["dataseed"])
    modelseed = int(cfg["modelseed"])
    num_data_subsample = cfg.get("num_data_subsample", 1.0)

    if feature_mode not in ALT_FEATURE_MODES:
        raise ValueError(f"feature_mode {feature_mode!r} not in {ALT_FEATURE_MODES}")
    if count_transform not in _VALID_TRANSFORMS:
        raise ValueError(
            f"count_transform {count_transform!r} not in {_VALID_TRANSFORMS}"
        )

    out_sysvar, cache_dir = ALT_GENERATORS[args.generator]
    os.makedirs(cache_dir, exist_ok=True)
    out_dir = os.path.join(
        "./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding",
        str(out_sysvar),
    )
    os.makedirs(out_dir, exist_ok=True)
    print(
        f"[alt-embedding] generator={args.generator}  feature_mode={feature_mode}  "
        f"count_transform={count_transform}  num_replicas={num_replicas}"
    )
    print(f"[alt-embedding] cache={cache_dir}  output={out_dir}")

    # 1. Build the alt-gen table (cached) and load Pythia6 gen-level rows.
    if args.generator == "herwig7":
        alt_table = prepare_herwig7(cache_dir)
    else:
        alt_table = prepare_pythia8(cache_dir)
    gm_table, mi_table, p6_table = load_p6_gen_table()
    n_gm = len(gm_table)
    n_mi = len(mi_table)
    n_p6 = n_gm + n_mi
    print(
        f"[alt-embedding] alt_rows={len(alt_table)}  "
        f"p6_rows={n_p6} (gen-matches={n_gm}, misses={n_mi})"
    )

    # 1b. Feature alignment audit — informational; helps identify which columns are
    # systematically separating the two generators (different conventions, sentinels,
    # cutoffs). Always runs since it's cheap; the recommended `--drop-features` line
    # is what the user acts on if they pick the classifier path.
    _audit_feature_alignment(alt_table, p6_table, KINEMATIC_COLUMNS, seed=dataseed)

    # 2. Pre-normalize: equalize sum(w_alt) to sum(w_p6) so the classifier learns the
    # density ratio rather than a mixture-prior-corrupted variant. Do this BEFORE
    # subsampling so the equalization reflects the full population, not the sample.
    w_p6_full = p6_table["weight"].to_numpy().astype(np.float32)
    w_alt_full = alt_table["weight"].to_numpy().astype(np.float32)
    s_p6_full = float(w_p6_full.sum())
    s_alt_full = float(w_alt_full.sum())
    if s_alt_full <= 0:
        raise RuntimeError(f"sum(w_alt)={s_alt_full}; cannot equalize")
    eq_factor = s_p6_full / s_alt_full
    w_alt_full = w_alt_full * eq_factor
    alt_table = replace_table_column(alt_table, "weight", w_alt_full)
    print(
        f"[alt-embedding] sum(w_p6)={s_p6_full:.4g}  sum(w_alt)_raw={s_alt_full:.4g}  "
        f"-> equalized (alt scaled by {eq_factor:.4g})"
    )

    # 2.5. Method dispatch. Both branches must end with r (shape (K, n_p6) tensor, clipped
    # to [ODDS_CLIP_LO, ODDS_CLIP_HI]) and r_mean (np float32 array, shape (n_p6,)) so the
    # post-hoc calibration, diagnostics, and splice-back steps below work uniformly.
    p6_table_full = (
        p6_table  # preserved for inference / histogram path; never subsampled
    )

    if args.method == "histogram":
        r_mean = histogram_reweight(
            alt_table, p6_table_full, var=args.reweight_var, bins=reweight_bins
        )
        # Wrap in a 1-replica tensor so the existing quantile/clamp diagnostics work.
        r = (
            torch.from_numpy(r_mean)
            .unsqueeze(0)
            .clamp_(min=ODDS_CLIP_LO, max=ODDS_CLIP_HI)
        )

    elif args.method == "classifier":
        # 2b. Subsample each class so the classifier *training* dataset is tractable.
        # At --num-replicas 1 the vmap-undersample memory pressure vanishes and we use
        # the full samples (default max_rows_per_class=0). The cap still applies when
        # the user explicitly raises num_replicas > 1 (vmap'd randperm OOMs otherwise).
        alt_train_table = alt_table
        p6_train_table = p6_table
        rng_np = np.random.default_rng(dataseed)

        # 2b.0 Class-size balancing. TensorDictDataset(is_categorical=True) sets
        # sum_w_class = N_class via class_weights, so unbalanced row counts translate
        # directly into a BCE-optimum calibration offset E_p6[r*] = N_alt / N_p6.
        # Undersample the larger class to match, unless that would discard >50% of
        # the rare class (Pythia8 has ~3M rows; cutting p6 to 3M loses most of it).
        balance_on = args.balance_class_sizes == "on" or (
            args.balance_class_sizes == "auto"
        )
        if balance_on:
            target = min(len(alt_train_table), len(p6_train_table))
            larger_name, smaller_name = (
                ("alt", "p6")
                if len(alt_train_table) >= len(p6_train_table)
                else ("p6", "alt")
            )
            smaller_n = target
            larger_n = max(len(alt_train_table), len(p6_train_table))
            if smaller_n / larger_n < 0.5 and args.balance_class_sizes == "auto":
                print(
                    f"[alt-embedding] class-size balance: skipped (would discard "
                    f">50% of {larger_name}; {larger_name}={larger_n}, "
                    f"{smaller_name}={smaller_n}). Pass --balance-class-sizes on to "
                    f"force it anyway."
                )
            else:
                if len(alt_train_table) > target:
                    n_before = len(alt_train_table)
                    idx = rng_np.choice(n_before, target, replace=False)
                    idx.sort()
                    alt_train_table = alt_train_table.take(pa.array(idx))
                    # Rescale to preserve the equalized per-class total (s_p6_full).
                    scale = s_p6_full / float(
                        alt_train_table["weight"].to_numpy().sum()
                    )
                    alt_train_table = replace_table_column(
                        alt_train_table,
                        "weight",
                        alt_train_table["weight"].to_numpy().astype(np.float32) * scale,
                    )
                    print(
                        f"[alt-embedding] class-size balance: alt -> "
                        f"{len(alt_train_table)} rows (was {n_before})"
                    )
                if len(p6_train_table) > target:
                    n_before = len(p6_train_table)
                    idx = rng_np.choice(n_before, target, replace=False)
                    idx.sort()
                    p6_train_table = p6_train_table.take(pa.array(idx))
                    scale = s_p6_full / float(p6_train_table["weight"].to_numpy().sum())
                    p6_train_table = replace_table_column(
                        p6_train_table,
                        "weight",
                        p6_train_table["weight"].to_numpy().astype(np.float32) * scale,
                    )
                    print(
                        f"[alt-embedding] class-size balance: p6 -> "
                        f"{len(p6_train_table)} rows (was {n_before})"
                    )

        if max_rows_per_class > 0:
            if len(alt_train_table) > max_rows_per_class:
                idx = rng_np.choice(
                    len(alt_train_table), max_rows_per_class, replace=False
                )
                idx.sort()
                alt_train_table = alt_train_table.take(pa.array(idx))
                # Rescale weights so the subsample sums to the same total (preserves the
                # equalization with p6 done in step 2). Correct for uniform subsampling.
                scale = s_p6_full / float(alt_train_table["weight"].to_numpy().sum())
                alt_train_table = replace_table_column(
                    alt_train_table,
                    "weight",
                    alt_train_table["weight"].to_numpy().astype(np.float32) * scale,
                )
                print(
                    f"[alt-embedding] training subsample: alt -> {len(alt_train_table)} rows "
                    f"(was {len(w_alt_full)})"
                )
            if len(p6_train_table) > max_rows_per_class:
                idx = rng_np.choice(
                    len(p6_train_table), max_rows_per_class, replace=False
                )
                idx.sort()
                p6_train_table = p6_train_table.take(pa.array(idx))
                scale = s_p6_full / float(p6_train_table["weight"].to_numpy().sum())
                p6_train_table = replace_table_column(
                    p6_train_table,
                    "weight",
                    p6_train_table["weight"].to_numpy().astype(np.float32) * scale,
                )
                print(
                    f"[alt-embedding] training subsample: p6 -> {len(p6_train_table)} rows "
                    f"(was {len(w_p6_full)})"
                )

        # 2c. Tame the training-side weight column so the BCE loss is not dominated by
        # the multi-decade pT-hat cross-section spread. The TRAINING tables only —
        # `p6_table_full` keeps the original weight column so the post-hoc weighted-mean
        # calibration of r and the splice-back into the four output arrows still use
        # cross-section-correct weights. Without this step the classifier carves a sharp
        # boundary in the low-pT-hat-dominated region and produces saturated odds there.
        if args.weight_flatten == "per-pth-bin":
            alt_train_table = replace_table_column(
                alt_train_table, "weight", _flatten_weights_per_bin(alt_train_table)
            )
            p6_train_table = replace_table_column(
                p6_train_table, "weight", _flatten_weights_per_bin(p6_train_table)
            )
            print(
                "[alt-embedding] training weights flattened per pth_bin "
                "(within each bin, mean(w)=1; raw weights preserved on p6_table_full)"
            )
        elif args.weight_flatten == "winsorise":
            q = float(args.weight_winsorise_q)
            if not (0.0 < q < 1.0):
                raise ValueError(f"--weight-winsorise-q must be in (0, 1), got {q}")
            w_alt = alt_train_table["weight"].to_numpy().astype(np.float32)
            cap_alt = float(np.quantile(w_alt, q))
            alt_train_table = replace_table_column(
                alt_train_table, "weight", np.minimum(w_alt, cap_alt)
            )
            w_p6 = p6_train_table["weight"].to_numpy().astype(np.float32)
            cap_p6 = float(np.quantile(w_p6, q))
            p6_train_table = replace_table_column(
                p6_train_table, "weight", np.minimum(w_p6, cap_p6)
            )
            print(
                f"[alt-embedding] training weights winsorised at q={q} "
                f"(alt cap={cap_alt:.4g}, p6 cap={cap_p6:.4g})"
            )
        else:  # "off"
            print(
                "[alt-embedding] training weights NOT flattened (--weight-flatten=off); "
                "expect low-pT loss dominance and degraded ESS."
            )

        # 3. Align schemas before concatenation. `to_tensordict` calls `pa.concat_tables`
        # internally which requires matching schemas (column set, dtype, order). Trim every
        # table (training + inference) to the same common column set and cast to one schema.
        common_cols = ["weight", "is_data", "is_matched", "pth_bin"]
        if feature_mode in ("bin_counts", "combined"):
            common_cols += ["bin_index", "bin_count"]
        if feature_mode in ("kinematics", "combined"):
            common_cols += list(effective_kinematic_columns)

        missing_alt = [c for c in common_cols if c not in alt_train_table.column_names]
        missing_p6 = [c for c in common_cols if c not in p6_train_table.column_names]
        if missing_alt or missing_p6:
            raise RuntimeError(
                f"schema mismatch: alt missing {missing_alt}, p6 missing {missing_p6}"
            )
        alt_train_table = alt_train_table.select(common_cols)
        p6_train_table = p6_train_table.select(common_cols)
        p6_table_full = p6_table_full.select(common_cols)
        # Promote dtypes to a common schema (e.g. is_matched may be int32 vs int64 across paths).
        p6_train_table = p6_train_table.cast(alt_train_table.schema, safe=False)
        p6_table_full = p6_table_full.cast(alt_train_table.schema, safe=False)

        classifier_td_dir = os.path.join(cache_dir, feature_mode, "classifier_td")
        if os.path.exists(classifier_td_dir):
            shutil.rmtree(classifier_td_dir)
        classifier_td = to_tensordict(
            alt_train_table,
            p6_train_table,
            columns=effective_kinematic_columns,
            prefix=classifier_td_dir,
            max_chunksize=100000,
            feature_mode=feature_mode,
        )
        print(f"[alt-embedding] classifier TD rows: {len(classifier_td)}")

        # 4. Train the binary classifier ensemble.
        torch.manual_seed(modelseed)
        rng = torch.Generator().manual_seed(dataseed)

        classifier_ds = TensorDictDataset(
            classifier_td,
            is_categorical=True,
            num_replicas=num_replicas,
        )

        stratify_alt = args.generator != "pythia8"
        train_loader, valid_loader = train_test_multi_loaders(
            classifier_ds,
            train_size=train_size,
            undersample_size=num_data_subsample,
            batch_size=batch_size,
            num_replicas=num_replicas,
            generator=rng,
            stratifys=(stratify_alt, True),
        )

        num_features = {
            "kinematics": len(effective_kinematic_columns),
            "bin_counts": N_BINS,
            "combined": len(effective_kinematic_columns) + N_BINS,
        }[feature_mode]
        assert classifier_ds.input.shape[-1] == num_features, (
            f"classifier TD input dim {classifier_ds.input.shape[-1]} != expected "
            f"{num_features} for feature_mode={feature_mode!r}"
        )
        device = cfg.device
        layer_sizes = cfg.layer_sizes(num_features)
        optimizer_kwargs = cfg.optimizer_kwargs

        classifier_ensemble, _, classifier_state = StatelessModule.init(
            MLP,
            Adam,
            model_init_kwargs={
                "layer_sizes": layer_sizes,
                "input_transform": build_input_transform(
                    count_transform,
                    classifier_ds.input,
                    device,
                ),
                "dropout_prob": cfg.dropout_prob,
            },
            num_replicas=num_replicas,
            device=device,
            init_randomness="different",
            optimizer_kwargs=optimizer_kwargs,
        )

        print(
            f"[alt-embedding] training classifier on device={device}, num_epochs={num_epochs}"
        )
        history = classifier_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits,
            classifier_state,
            train_loader,
            valid_iterator=valid_loader,
            num_epochs=num_epochs,
            callbacks=[("early_stopping", EarlyStopping(patience=cfg.early_stopping_patience))],
            randomness="different",
        )

        state_dir = os.path.join(cache_dir, feature_mode, "classifier_state")
        os.makedirs(state_dir, exist_ok=True)
        classifier_state.to_file(state_dir, "state_dict.pt")
        history_path = os.path.join(state_dir, "history.json")
        history.to_file(history_path)
        print(f"[alt-embedding] saved training history -> {history_path}")

        # Print convergence summary. If final valid_loss is at or above ln(2) ~= 0.6931 the
        # classifier produced ~p=0.5 on average (no class separation) regardless of what
        # train_loss did — flag that loudly so the user doesn't waste time on downstream
        # diagnostics in a no-signal run.
        train_losses = list(history.get("train_loss", []))
        valid_losses = list(history.get("valid_loss", []))
        if train_losses and valid_losses:
            epochs_run = len(train_losses)
            tl = [float(np.asarray(v).mean()) for v in train_losses]
            vl = [float(np.asarray(v).mean()) for v in valid_losses]
            print(
                f"[alt-embedding] loss curve over {epochs_run} epoch(s):\n"
                f"    train_loss: first={tl[0]:.4f}  min={min(tl):.4f}  last={tl[-1]:.4f}\n"
                f"    valid_loss: first={vl[0]:.4f}  min={min(vl):.4f}  last={vl[-1]:.4f}\n"
                f"    entropy floor ln(2) = 0.6931"
            )
            if vl[-1] > 0.69:
                print(
                    "[alt-embedding] WARNING: final valid_loss >= ln(2). The classifier did "
                    "NOT learn a separating discriminator — ratios will collapse to ~1 and the "
                    "reweighting will be a no-op. Try a different --feature-mode."
                )

        # 5. Predict on the FULL Pythia6 gen sample (not the subsampled training half!) so
        # the per-jet ratios align row-by-row with gm_table / mi_table for the splice-back
        # in step 8. Build a small inference TensorDict whose neg half (target=0) is the
        # full p6 table; the pos half is a 1-row dummy required only by to_tensordict's
        # data_like / sim_like API and discarded by the neg-only inference loader.
        inference_td_dir = os.path.join(cache_dir, feature_mode, "inference_td")
        if os.path.exists(inference_td_dir):
            shutil.rmtree(inference_td_dir)
        inference_td = to_tensordict(
            p6_table_full.slice(0, 1),
            p6_table_full,
            columns=effective_kinematic_columns,
            prefix=inference_td_dir,
            max_chunksize=100000,
            feature_mode=feature_mode,
        )
        inference_ds = TensorDictDataset(
            inference_td,
            is_categorical=True,
            num_replicas=num_replicas,
        )
        p6_inference_loader = reweight_inference_loaders(
            inference_ds,
            batch_size=batch_size * 5,
            num_replicas=num_replicas,
            has_unmatched=False,
        )
        p6_preds = classifier_ensemble.predict(
            classifier_state, p6_inference_loader, non_linearity=sigmoid
        )
        classifier_state.reset_status()

        # _odds_clipped returns shape (num_replicas, n_p6) clamped to [ODDS_CLIP_LO, HI].
        # Neg-only loader iterates TD rows 1..n_p6 in sequential order via TensorBatchSampler;
        # those map row-for-row to p6_table_full = [gen-matches ; misses].
        r = _odds_clipped(p6_preds)
        assert r.shape == (num_replicas, n_p6), (
            f"expected ratio shape ({num_replicas}, {n_p6}), got {tuple(r.shape)}"
        )

        # 6. Collapse replicas: geometric mean is the natural average for a density ratio
        # whose log behaves like a Gaussian random variable under classifier noise.
        # Arithmetic mean is dominated by the high-r tail of a single noisy replica and
        # systematically inflates ESS-killing outliers; geometric mean stays well-behaved.
        r_mean = (
            torch.exp(torch.log(r.clamp_min(1e-12)).mean(dim=0))
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    # 6b. Post-hoc calibration. At the BCE optimum, E_p6[r] = sum_w_alt / sum_w_p6,
    # which after the equalization in step 2 is 1. In practice, if alt and p6 occupy
    # nearly-disjoint regions of feature space (e.g. different jet-mass conventions,
    # different track-pT cutoffs), the classifier learns a sharp boundary and outputs
    # sigmoid ~= 0 for every p6 jet — giving systematically tiny odds. The per-pth_bin
    # *shape* is still correct, but the overall scale is biased. Renormalize so the
    # w_p6-weighted mean is 1, which preserves total cross-section (we're reweighting
    # events, not adding them) and the per-pT shape (the physical signal).
    pth_p6 = p6_table_full["pth_bin"].to_numpy().astype(np.int64)
    w_p6 = p6_table_full["weight"].to_numpy().astype(np.float64)
    r_calib = float((r_mean.astype(np.float64) * w_p6).sum() / w_p6.sum())
    if np.isfinite(r_calib) and r_calib > 0:
        print(
            f"[alt-embedding] post-hoc calibration: raw w_p6-weighted mean(r) = {r_calib:.4g}"
        )
        if r_calib < 0.5 or r_calib > 2.0:
            print(
                "[alt-embedding] WARNING: calibration factor is far from 1 — the classifier "
                "is producing systematically biased odds, likely because the two generators "
                "occupy nearly-disjoint regions of the feature space. Inspect KINEMATIC_COLUMNS "
                "for features that differ in convention/cutoff (jet mass 'm', softdrop "
                "sentinels, particle-pT thresholds). The per-pth_bin SHAPE is preserved by "
                "the renormalization below; the absolute scale was an artifact."
            )
        r_mean = (r_mean.astype(np.float64) / r_calib).astype(np.float32)
    else:
        print(
            f"[alt-embedding] WARNING: skipping renormalization, raw w_p6-weighted mean(r) "
            f"= {r_calib} is non-positive/non-finite. Output weights will be unusable."
        )

    # 6c. ESS of the reweighted P6 sample, computed against the ORIGINAL w_p6 (the
    # cross-section weight, not any flattened training-side variant). This is the metric
    # the alt-prior iteration loop is driven by; baking it in removes the need to compute
    # it manually in a notebook.
    w_p6_orig = w_p6  # already loaded above from p6_table_full
    w_new_p6 = w_p6_orig * r_mean.astype(np.float64)
    ess_pre = float((w_p6_orig.sum() ** 2) / (w_p6_orig**2).sum())
    ess_post = float((w_new_p6.sum() ** 2) / (w_new_p6**2).sum())
    print(
        f"[alt-embedding] ESS / n_p6:  before = {ess_pre / n_p6:.3f}  "
        f"after = {ess_post / n_p6:.3f}  (target >= 0.50)"
    )

    r_gm = r_mean[:n_gm]
    r_mi = r_mean[n_gm:]

    # 7. Diagnostics — what does the ratio look like?
    # `torch.quantile` is limited to ~2**24 elements (int32 kthvalue indices); r is
    # num_replicas * n_p6 which exceeds that for realistic inputs, so use numpy.
    n_at_lo = int((r <= ODDS_CLIP_LO * 1.001).sum().item())
    n_at_hi = int((r >= ODDS_CLIP_HI * 0.999).sum().item())
    r_flat_np = r.detach().cpu().numpy().ravel()
    qs = np.quantile(r_flat_np, [0.01, 0.1, 0.5, 0.9, 0.99])
    spread_line = ""
    if num_replicas > 1:
        # Across-replica spread distinguishes "no per-jet signal" (mean~1, std~0) from
        # "signal averaged away" (mean~1, std large). Meaningless at num_replicas==1.
        r_std_per_jet_mean = float(r.std(dim=0).mean().item())
        spread_line = (
            f"\n    across-replica std (per-jet, averaged) = {r_std_per_jet_mean:.4g}"
        )
    else:
        spread_line = "\n    (single-replica run, ensemble-spread diagnostic skipped)"
    print(
        f"[alt-embedding] ratio diagnostics (over all replicas, {r.numel()} jets):\n"
        f"    min={r.min().item():.4g}  max={r.max().item():.4g}  "
        f"mean={r.mean().item():.4g}\n"
        f"    quantiles [1%,10%,50%,90%,99%] = "
        f"{[f'{q:.4g}' for q in qs]}"
        f"{spread_line}\n"
        f"    fraction clamped at lo={n_at_lo / r.numel():.2%}  "
        f"hi={n_at_hi / r.numel():.2%}"
    )

    # Per-pth_bin reweighting (using the post-hoc renormalized r_mean). If the
    # unweighted mean is ~1 in every pth_bin AND the renorm factor (above) was ~1, the
    # reweighting won't move the pT marginal. If the renorm factor was far from 1 but
    # the per-pth_bin shape varies, the *shape* signal is real and useful.
    print("[alt-embedding] per-pth_bin r_mean (post-renorm):")
    print(f"    {'pth_bin':>8}  {'n_jets':>10}  {'<r> unwt':>10}  {'<r> w_p6':>10}")
    for ipth in sorted(np.unique(pth_p6).tolist()):
        sel = pth_p6 == ipth
        n_sel = int(sel.sum())
        if n_sel == 0:
            continue
        r_bin = r_mean[sel]
        w_bin = w_p6[sel]
        r_unwt = float(r_bin.mean())
        r_wt = (
            float((r_bin * w_bin).sum() / w_bin.sum())
            if w_bin.sum() > 0
            else float("nan")
        )
        print(f"    {ipth:>8d}  {n_sel:>10d}  {r_unwt:>10.4f}  {r_wt:>10.4f}")

    # 8. Write the four reweighted arrows. Fakes pass through unchanged.
    rm_table = _read_arrow(os.path.join(P6_EMBEDDING_DIR, "reco-matches.arrow"))
    fk_table = _read_arrow(os.path.join(P6_EMBEDDING_DIR, "fakes.arrow"))
    assert len(rm_table) == n_gm, (
        f"reco-matches ({len(rm_table)}) and gen-matches ({n_gm}) row counts disagree — "
        f"matching pairing assumption is broken"
    )

    w_gm_old = gm_table["weight"].to_numpy().astype(np.float32)
    w_rm_old = rm_table["weight"].to_numpy().astype(np.float32)
    w_mi_old = mi_table["weight"].to_numpy().astype(np.float32)
    w_fk_old = fk_table["weight"].to_numpy().astype(np.float32)

    w_gm_new = w_gm_old * r_gm
    w_rm_new = w_rm_old * r_gm  # same ratio: matched pair shares the gen jet
    w_mi_new = w_mi_old * r_mi

    gm_out = replace_table_column(gm_table, "weight", w_gm_new)
    rm_out = replace_table_column(rm_table, "weight", w_rm_new)
    mi_out = replace_table_column(mi_table, "weight", w_mi_new)
    fk_out = fk_table  # fakes unchanged

    print(
        f"[alt-embedding] weight totals  gen-matches: {w_gm_old.sum():.4g} -> {w_gm_new.sum():.4g}\n"
        f"                              reco-matches: {w_rm_old.sum():.4g} -> {w_rm_new.sum():.4g}\n"
        f"                              misses:       {w_mi_old.sum():.4g} -> {w_mi_new.sum():.4g}\n"
        f"                              fakes:        {w_fk_old.sum():.4g} (unchanged)"
    )

    for name, t in (
        ("gen-matches", gm_out),
        ("reco-matches", rm_out),
        ("misses", mi_out),
        ("fakes", fk_out),
    ):
        path = os.path.join(out_dir, f"{name}.arrow")
        _write_arrow(t, path)
        print(f"  wrote {path}")

    # Downstream unfolding uses cfg['feature_mode'] (the main pipeline's choice),
    # which is independent of the alt-prior reweighter's --method/--feature-mode used here.
    cfg_fm = cfg.get("feature_mode", "bin_counts")
    print(
        "[alt-embedding] done.\n"
        f"  Next:        preprocessing.make_datasets_for_unfolding(source_dir=..., "
        f"sysvar=SysVar.{out_sysvar.name}, feature_mode={cfg_fm!r})\n"
        f"  Production:  python make_alt_embedding.py --generator {args.generator}\n"
        f"  Experiment:  python make_alt_embedding.py --generator {args.generator} "
        f"--method classifier --drop-features <from audit table above>"
    )


if __name__ == "__main__":
    main()
