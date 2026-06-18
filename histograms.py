import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import os
    import json

    import numpy as np
    import pyarrow as pa
    import torch

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
    from thoda import Profile, Histogram, Snapshot, bayesian_blocks

    common_vars = (
        "pt",
        "m",
        "sd_m",
        "sd_dR",
        "sd_symmetry",
    )

    angularities = (
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
    )

    from config import load_config

    _cfg_setup = load_config()
    feature_mode = _cfg_setup["feature_mode"]
    # NOTE: must NOT be underscore-prefixed — marimo treats leading-underscore
    # names (even in app.setup) as cell-private, so consumer cells can't see it.
    dataset_root = str(_cfg_setup.dataset_root)
    # Patch G: how the per-replica ensemble axis is collapsed to a central value
    # ("mean" | "median" | "trimmed_mean"). median/trimmed are robust to a single
    # blown-up replica. Default "mean" preserves old behavior for stale snapshots.
    replica_reduce = _cfg_setup.get("replica_reduce", "mean")
    replica_trim_frac = float(_cfg_setup.get("replica_trim_frac", 0.1))

    # Target variation, read from runtime-files/config.json (cfg.sys_var resolves
    # the string value to a SysVar, defaulting to SysVar.NONE = "nominal"). Set
    # "sys_var" in config.json to loop histograms.py over systematic variations.
    # sys_var = SysVar.NONE  # old: hardcoded nominal
    sys_var = _cfg_setup.sys_var


@app.function
def histogram(table, bins, cols, weights):
    hist, _ = Histogram.create(
        [table[col].to_numpy() for col in cols],
        bins=[bins[key] for key in cols],
        weights=weights,
        axis_names=cols,
    )
    return hist


@app.function
def profile(table, bins, cols, xcol, weights):
    prof, _ = Profile.create(
        [table[col].to_numpy() for col in cols],
        bins=[bins[key] for key in cols],
        y=table[xcol].to_numpy(),
        weights=weights,
        axis_names=cols,
    )
    return prof


@app.function
def perpt_masks(table, jpt_bins):
    """Boolean row masks selecting [jpt_bins[i], jpt_bins[i+1]) per jet-pT slice."""
    pt = np.asarray(table["pt"].to_numpy())
    return [(pt >= jpt_bins[i]) & (pt < jpt_bins[i + 1]) for i in range(len(jpt_bins) - 1)]


@app.function
def compute_perpt_bins(
    table,
    flat_bins,
    jpt_bins,
    bb_cols,
    uniform_cols,
    weights=None,
    p0=0.02,
    undersample=30000,
    seed=0,
):
    """Per-(column, jet-pT slice) bin edges. CPU-only.

    ``bb_cols`` are recomputed with ``thoda.bayesian_blocks`` WITHIN each jet-pT
    slice (the range is taken from the existing global flat binning so the
    x-extent is unchanged). ``uniform_cols`` simply replicate the flat global
    edges for every slice (e.g. m / sd_m are already clean uniform bins).
    ``weights`` should be the per-jet replica-mean unfolded weights so the
    blocks follow the nominal unfolded yield. Returns ``{col: {jpt_idx: edges}}``.
    """
    pt = np.asarray(table["pt"].to_numpy())
    if weights is None:
        w = None
    else:
        w = weights.numpy() if torch.is_tensor(weights) else np.asarray(weights)

    out = {}
    for _c in uniform_cols:
        _edges = [float(_e) for _e in np.asarray(flat_bins[_c], dtype=float)]
        out[_c] = {i: _edges for i in range(len(jpt_bins) - 1)}

    for _c in bb_cols:
        rng = (float(flat_bins[_c][0]), float(flat_bins[_c][-1]))
        vals = np.asarray(table[_c].to_numpy())
        out[_c] = {}
        for i in range(len(jpt_bins) - 1):
            _m = (pt >= jpt_bins[i]) & (pt < jpt_bins[i + 1])
            _n = int(_m.sum())
            _us = min(undersample, _n) if _n > 0 else None
            _gen = torch.Generator().manual_seed(seed)
            _edges = bayesian_blocks(
                torch.as_tensor(vals[_m], dtype=torch.float64),
                weights=(torch.as_tensor(w[_m], dtype=torch.float64) if w is not None else None),
                p0=p0,
                ranges=rng,
                undersample=_us,
                generator=_gen,
                device="cpu",
            )
            out[_c][i] = _edges.tolist()
            print(f"  {_c} jpt{i}: {len(_edges)} edges (n={_n})")
    return out


@app.function
def histogram_perpt(table, bins_perpt, var, jpt_bins, weights=None, masks=None):
    """List of 1-D Histograms (one per jet-pT slice) with per-slice ``var`` edges."""
    if masks is None:
        masks = perpt_masks(table, jpt_bins)
    vals = table[var].to_numpy()
    h1ds = []
    for i in range(len(jpt_bins) - 1):
        _m = masks[i]
        _w = None if weights is None else weights[..., torch.as_tensor(_m)]
        _h, _ = Histogram.create(
            [vals[_m]], bins=[bins_perpt[var][i]], weights=_w, axis_names=(var,)
        )
        h1ds.append(_h)
    return h1ds


@app.function
def profile_perpt(table, bins_perpt, cols, ycol, jpt_bins, weights=None, masks=None):
    """Per jet-pT slice, build a multi-D Profile of ``ycol`` over ``cols`` (no pt
    axis) and project to a 1-D profile vs each col. Returns ``{col: [per-jpt 1d
    profiles]}``."""
    if masks is None:
        masks = perpt_masks(table, jpt_bins)
    col_vals = {c: table[c].to_numpy() for c in cols}
    yv = table[ycol].to_numpy()
    out = {c: [] for c in cols}
    for i in range(len(jpt_bins) - 1):
        _m = masks[i]
        _w = None if weights is None else weights[..., torch.as_tensor(_m)]
        _prof, _ = Profile.create(
            [col_vals[c][_m] for c in cols],
            bins=[bins_perpt[c][i] for c in cols],
            y=yv[_m],
            weights=_w,
            axis_names=cols,
        )
        for c in cols:
            out[c].append(_prof.project(c))
        del _prof
    return out


@app.function
def save_hist_1d_list(h1ds, prefix=".", fname_prefix="hist", batched=False, true_h1ds=None):
    """``save_hist_2d`` for an already-per-jet-pT list (skips unbind/[1:-1])."""
    if true_h1ds is None:
        true_h1ds = (None,) * len(h1ds)
    for ijpt, (h1d, true_h1d) in enumerate(zip(h1ds, true_h1ds)):
        fname = f"{fname_prefix}_jpt{ijpt}"
        h1d_snap = h1d.snapshot()
        if true_h1d is not None:
            save_snapshot(
                h1d_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )
            true_snap = true_h1d.snapshot()
            save_snapshot(
                true_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )
            rsnap = ratio_snapshot(h1d_snap, true_snap)
            save_snapshot(
                rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )
        else:
            save_snapshot(h1d_snap, prefix=prefix, fname=fname, batched=batched)


@app.function
def save_ratio_1d_list(
    num_h1ds,
    den_h1ds,
    prefix=".",
    fname_prefix="ratio",
    batched=False,
    true_num_h1ds=None,
    true_den_h1ds=None,
):
    """``save_ratio_2d`` for already-per-jet-pT lists (skips unbind/[1:-1])."""
    if true_num_h1ds is None:
        true_num_h1ds = (None,) * len(num_h1ds)
    if true_den_h1ds is None:
        true_den_h1ds = (None,) * len(den_h1ds)
    for ijpt, (num_h1d, den_h1d, true_num_h1d, true_den_h1d) in enumerate(
        zip(num_h1ds, den_h1ds, true_num_h1ds, true_den_h1ds)
    ):
        fname = f"{fname_prefix}_jpt{ijpt}"
        ratio_snap = ratio_snapshot(num_h1d.snapshot(), den_h1d.snapshot())
        if true_num_h1d is not None and true_den_h1d is not None:
            save_snapshot(
                ratio_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )
            true_ratio_snap = ratio_snapshot(true_num_h1d.snapshot(), true_den_h1d.snapshot())
            save_snapshot(
                true_ratio_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )
            ratio_rsnap = ratio_snapshot(ratio_snap, true_ratio_snap)
            save_snapshot(
                ratio_rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )
        else:
            save_snapshot(ratio_snap, prefix=prefix, fname=fname, batched=batched)


@app.function
def collapse_replicas(x, dim=0):
    """Collapse the per-replica ensemble axis to a central value, per the
    `replica_reduce` config knob (Patch G). "median"/"trimmed_mean" are robust to a
    single blown-up replica corrupting the central spectrum; "mean" is the old
    behavior. Only the central value uses this — the replica *spread* (std/var) keeps
    its plain definition (it is the uncertainty band)."""
    if x.dim() <= dim or x.shape[dim] == 1 or replica_reduce == "mean":
        return x.mean(dim)
    if replica_reduce == "median":
        return x.median(dim=dim).values
    if replica_reduce == "trimmed_mean":
        _k = int(x.shape[dim] * replica_trim_frac)
        if _k == 0:
            return x.mean(dim)
        _xs, _ = x.sort(dim=dim)
        _sl = [slice(None)] * x.dim()
        _sl[dim] = slice(_k, x.shape[dim] - _k)
        return _xs[tuple(_sl)].mean(dim)
    return x.mean(dim)


@app.function
def snapshot_state_dict(hsnap, batched=False):
    sdict = dict(
        bin_center=hsnap.bin_centers[1:-1],
        half_bin_width=hsnap.bin_widths[1:-1] / 2.0,
        bin_count=hsnap.values[1:-1] if not batched else collapse_replicas(hsnap.values[:, 1:-1]),
        bin_count_err=hsnap.variances[1:-1].sqrt()
        if not batched
        else collapse_replicas(hsnap.variances[:, 1:-1].sqrt()),
    )

    if batched:
        sdict["bin_count_std"] = hsnap.values[:, 1:-1].std(0)
    return sdict


@app.function
def closure_state_dict(hsnap, batched=False):
    """Like ``snapshot_state_dict`` but, for batched snapshots, the displayed
    error (``bin_count_std`` — the field ``plot_data_points`` uses as ``yerr``)
    is the quadrature sum of the per-bin statistical uncertainty
    (``bin_count_err`` = sqrt(variance)) and the per-replica ensemble spread.
    Closure error bars therefore include statistics. Non-batched snapshots (e.g.
    the truth side) are returned unchanged: statistical uncertainty only.
    """
    sdict = snapshot_state_dict(hsnap, batched=batched)
    if batched:
        stat = sdict["bin_count_err"]
        spread = sdict["bin_count_std"]
        sdict["bin_count_std"] = (stat.square() + spread.square()).sqrt()
    return sdict


@app.function
def save_snapshot(hsnap, prefix=".", fname="hist", batched=False):
    os.makedirs(prefix, exist_ok=True)
    sdict = snapshot_state_dict(hsnap, batched=batched)
    torch.save(sdict, os.path.join(prefix, f"{fname}.pt"))


@app.function
def ratio_snapshot(hn_snap, hd_snap):
    bin_centers = hn_snap.bin_centers
    bin_widths = hn_snap.bin_widths
    ratio = hn_snap.values / hd_snap.values
    rel_err2_num = hn_snap.variances / (hn_snap.values) ** 2
    rel_err2_den = hd_snap.variances / (hd_snap.values) ** 2
    ratio_err2 = ratio**2 * (rel_err2_den + rel_err2_num)

    return Snapshot(bin_centers, bin_widths, ratio, ratio_err2)


@app.function
def save_ratio_2d(
    hist_num, hist_den, prefix=".", fname_prefix="ratio", batched=False, true_hists=None
):
    num_h1ds = hist_num.unbind("pt")[1:-1]
    den_h1ds = hist_den.unbind("pt")[1:-1]
    if true_hists is not None:
        true_num_h1ds = true_hists[0].unbind("pt")[1:-1]
        true_den_h1ds = true_hists[1].unbind("pt")[1:-1]
    else:
        true_num_h1ds = (None,) * len(num_h1ds)
        true_den_h1ds = (None,) * len(den_h1ds)

    for ijpt, (num_h1d, den_h1d, true_num_h1d, true_den_h1d) in enumerate(
        zip(num_h1ds, den_h1ds, true_num_h1ds, true_den_h1ds)
    ):
        fname = f"{fname_prefix}_jpt{ijpt}"
        ratio_snap = ratio_snapshot(num_h1d.snapshot(), den_h1d.snapshot())
        if true_num_h1d is not None and true_den_h1d is not None:
            save_snapshot(
                ratio_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )
            true_ratio_snap = ratio_snapshot(true_num_h1d.snapshot(), true_den_h1d.snapshot())
            save_snapshot(
                true_ratio_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )
            ratio_rsnap = ratio_snapshot(ratio_snap, true_ratio_snap)
            save_snapshot(
                ratio_rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )

        else:
            save_snapshot(ratio_snap, prefix=prefix, fname=fname, batched=batched)


@app.function
def save_hist_2d(hist, prefix=".", fname_prefix="hist", batched=False, true_hist=None):
    h1ds = hist.unbind("pt")[1:-1]
    if true_hist is not None:
        true_h1ds = true_hist.unbind("pt")[1:-1]
    else:
        true_h1ds = (None,) * len(h1ds)

    for ijpt, (h1d, true_h1d) in enumerate(zip(h1ds, true_h1ds)):
        # print(h1d.shape)
        fname = f"{fname_prefix}_jpt{ijpt}"
        h1d_snap = h1d.snapshot()
        if true_h1d is not None:
            # print(true_h1d.shape)
            save_snapshot(
                h1d_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )

            true_h1d_snap = true_h1d.snapshot()
            save_snapshot(
                true_h1d_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )

            rsnap = ratio_snapshot(h1d_snap, true_h1d_snap)
            save_snapshot(
                rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )
        else:
            save_snapshot(h1d_snap, prefix=prefix, fname=fname, batched=batched)


@app.cell
def _():
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as file:
        bins = json.load(file)
    bins["pt"] = get_jet_pt_bins(sys_var)
    for _ang in angularities:
        bins[f"sd_{_ang}"] = bins[_ang]

    bins["m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    bins["sd_m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    return (bins,)


@app.cell
def _(unf_weights):
    # Read the PREPROCESSED embedding (features/<mode>/embedding/<sysvar>/), not
    # the raw clustering output (jets/embedding/): only the preprocessed arrows
    # carry the process_table-computed angularity columns (ch_ang_*), and their
    # gen row order matches w_unfolding. The nominal jets/ arrows happen to carry
    # angularities so the old path worked for NONE, but detector-systematic
    # variations (e.g. tower_et_corr_sys) were clustered without them.
    # NB: these arrows are deleted after multifold to save space — re-run
    # preprocessing.py for the target sys_var to regenerate them.
    # _source_dir = dataset_root + "/jets/embedding"   # old (raw, no angularities)

    # bin_counts split: the softdrop observables (sd_m, sd_dR, sd_symmetry, ...) live
    # ONLY in the angularities arrows. The bin_counts run supplies the WEIGHTS
    # (w_unfolding.npz, read in cell TqIu); its arrows carry the (pT,dR) bin columns
    # but no softdrop scalars. The jets are the same set in the same row order across
    # feature modes, so the bin_counts weights apply row-for-row to the angularities
    # observable tables (asserted below). Source observables from angularities here;
    # `feature_mode` (the run mode, e.g. bin_counts) still labels the output dirs.
    obs_feature_mode = "angularities"
    # old (broke in bin_counts mode — those arrows have no sd_* columns):
    # _source_dir = dataset_root + "/features/" + feature_mode + "/embedding"
    _source_dir = dataset_root + "/features/" + obs_feature_mode + "/embedding"

    if sys_var in {SysVar.NONE, SysVar.TOWER_ET_CORRECTION, SysVar.TRACK_EFFICIENCY}:
        _input_root_dir = os.path.join(_source_dir, str(sys_var))
    else:
        _input_root_dir = os.path.join(_source_dir, str(SysVar.NONE))

    _buffers = []
    _buffers.append(pa.memory_map(os.path.join(_input_root_dir, "gen-matches.arrow")))
    _gen_match_table = pa.ipc.open_file(_buffers[-1]).read_all()
    _buffers.append(pa.memory_map(os.path.join(_input_root_dir, "misses.arrow")))
    _gen_misses_table = pa.ipc.open_file(_buffers[-1]).read_all()
    gen_table = pa.concat_tables((_gen_match_table, _gen_misses_table))

    # Row-alignment guard: the bin_counts w_unfolding weights (unf_weights, cell TqIu)
    # are applied to these angularities observables row-for-row, valid only if the
    # gen-side row counts match across feature modes.
    assert gen_table.num_rows == unf_weights.shape[1], (
        f"gen_table rows {gen_table.num_rows} != unf_weights cols {unf_weights.shape[1]}; "
        "angularities/bin_counts gen arrows are misaligned."
    )
    return (gen_table,)


@app.cell
def _(gen_table):
    mc_tables = []

    if sys_var == SysVar.NONE:
        _py6_weights = torch.as_tensor(gen_table["weight"].to_numpy(), dtype=torch.float32)

        mc_tables.append(("pythia6", gen_table, _py6_weights))

        _py8_dir = dataset_root
        _py8_file = "Pythia8_pp200GeV.arrow"

        _py8_path = os.path.join(_py8_dir, f"preproc_{_py8_file}")
        _py8_buffer = pa.memory_map(_py8_path, "rb")
        _py8_table = pa.ipc.open_file(_py8_buffer).read_all()
        _py8_weights = torch.as_tensor(_py8_table["weight"].to_numpy(), dtype=torch.float32)

        mc_tables.append(("pythia8", _py8_table, _py8_weights))

        _hw_dir = "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs"
        _hw_file = "combined_HwJets_nEv500000.arrow"

        _hw_path = os.path.join(_hw_dir, _hw_file)
        _hw_buffer = pa.memory_map(_hw_path, "rb")
        _hw_table = pa.ipc.open_file(_hw_buffer).read_all()
        _hw_weights = torch.as_tensor(_hw_table["weight"].to_numpy(), dtype=torch.float32)

        mc_tables.append(("herwig7", _hw_table, _hw_weights))
    return (mc_tables,)


@app.cell
def _():
    # Closure variant: LIKE_DATA unfolds reweighted reco pseudo-data, so it has
    # a meaningful gen-side "truth" (the baked prior gen weights) -> emit
    # unfolded/truth/ratio for the non-closure systematic.
    _closure_sysvars = {
        SysVar.UNFOLDING_PRIOR_LIKE_DATA,
    }
    # Model-prior variants: HERWIG7/PYTHIA8 now unfold the REAL data with their
    # reweighted embedding as the prior, so the result is an alternate unfolded
    # *data* spectrum with NO truth -> treat like a detector sysvar (top-level
    # snapshot, truth_weights=None) but still read their own w_unfolding.npz.
    _model_prior_sysvars = {
        SysVar.UNFOLDING_PRIOR_HERWIG7,
        SysVar.UNFOLDING_PRIOR_PYTHIA8,
    }
    # --- old: all three priors set truth_weights and saved unfolded/truth/ratio
    # _prior_sysvars = {LIKE_DATA, HERWIG7, PYTHIA8} ---
    _has_own_unfolding = (
        {
            SysVar.TOWER_ET_CORRECTION,
            SysVar.TRACK_EFFICIENCY,
        }
        | _closure_sysvars
        | _model_prior_sysvars
    )

    _unf_dir = os.path.join(
        dataset_root + "/features",
        feature_mode,
        "embedding",
        str(sys_var if sys_var in _has_own_unfolding else SysVar.NONE),
    )
    _unf_wts_filename = os.path.join(_unf_dir, "w_unfolding.npz")
    print(f"Getting unfolded weights from {_unf_wts_filename}")
    _unf_weights_dict = np.load(_unf_wts_filename)
    # Iteration choice is owned by systematics.get_unfolding_iter (keyed by
    # sys_var); the gen-side weights are the even-indexed arrays arr_{2*iter}.
    _iteration = get_unfolding_iter(sys_var, 5)
    unf_weights = torch.as_tensor(_unf_weights_dict[f"arr_{2 * _iteration}"], dtype=torch.float32)

    if sys_var in _closure_sysvars:
        # Truth weights live in the closure-variant gen-side arrows (baked by
        # omnisequential.py for LIKE_DATA). Concatenate [gen-matches | misses]
        # to match the row order of the gen_table used downstream.
        _prior_root = os.path.join(
            dataset_root + "/features",
            feature_mode,
            "embedding",
            str(sys_var),
        )
        print("Reading truth weights from arrows under:", _prior_root)
        _gm_buf = pa.memory_map(os.path.join(_prior_root, "gen-matches.arrow"), "rb")
        _mi_buf = pa.memory_map(os.path.join(_prior_root, "misses.arrow"), "rb")
        _gm_truth = pa.ipc.open_file(_gm_buf).read_all()
        _mi_truth = pa.ipc.open_file(_mi_buf).read_all()
        truth_weights = torch.as_tensor(
            np.concatenate([_gm_truth["weight"].to_numpy(), _mi_truth["weight"].to_numpy()]),
            dtype=torch.float32,
        )
    else:
        truth_weights = None
    return truth_weights, unf_weights


@app.cell
def _(bins, gen_table, mc_tables):
    # Per-(variable, jet-pT) adaptive binning. CPU-ONLY (the GPU is reserved for
    # other running jobs). Derived once from the NOMINAL unfolded gen
    # distribution and cached to runtime-files/bins_perpt.json so every sysvar /
    # MC reuses identical edges -> ratios and the systematics combination
    # (which reads bin edges back off each saved snapshot) line up.
    jpt_bins = get_jet_pt_bins(sys_var)

    _perpt_bin_file = "./runtime-files/bins_perpt.json"
    _bb_cols = (*angularities, "sd_dR", "sd_symmetry")  # recompute per pT slice
    _uniform_cols = ("m", "sd_m")  # already-clean uniform bins; replicate per slice
    _p0 = 0.02
    _undersample = 30000  # one-time CPU DP cost; capped for speed (no GPU)

    # Per-variable uniform x-bin counts for the PROFILES only. The distributions
    # keep the adaptive bins_perpt edges above; the profiles read prof_bins_perpt
    # (below) instead, so their x-axis is evenly spaced and easier to read /
    # compare across jet-pT panels. Range = each var's global flat-bins extent.
    _prof_nbins = {
        "m": 12,
        "sd_m": 12,
        "sd_dR": 10,
        "sd_symmetry": 8,
        "ch_ang_k1_b0.5": 10,
        "ch_ang_k1_b1": 10,
        "ch_ang_k1_b2": 10,
        "ch_ang_k2_b0": 10,
    }
    _prof_nbins_default = 10  # fallback for any profile col not listed above

    bins_perpt = {}
    try:
        with open(_perpt_bin_file, "rb") as _f:
            bins_perpt = {
                _c: {int(_k): _v for _k, _v in _d.items()} for _c, _d in json.load(_f).items()
            }
        print("Read per-pT bins from:", _perpt_bin_file)
    except Exception as _exc:
        print(_exc, "- computing per-pT bins (CPU, one-time)")
        bins_perpt = {}

    _needed = set(_bb_cols) | set(_uniform_cols)
    if not _needed.issubset(bins_perpt.keys()):
        # Bin UNWEIGHTED (raw per-slice jet density), matching the original
        # omnisequential binning convention. The unfolded cross-section weights
        # span orders of magnitude across pT-hat bins, so weighting here crushes
        # the effective sample size and yields too few Bayesian blocks (e.g. a
        # single bin at high jet-pT). Unweighted blocks adapt to where the jets
        # in each slice actually sit, so the high-pT panels get well-placed edges
        # with no empty high-lambda bins.
        bins_perpt = compute_perpt_bins(
            gen_table,
            bins,
            jpt_bins,
            bb_cols=_bb_cols,
            uniform_cols=_uniform_cols,
            weights=None,
            p0=_p0,
            undersample=_undersample,
        )
        # sd_<ang> shares its inclusive angularity's edges so ratio_incl_vs_sd
        # divides element-wise (mirrors the old bins[f"sd_{ang}"] = bins[ang]).
        for _ang in angularities:
            bins_perpt[f"sd_{_ang}"] = bins_perpt[_ang]
        with open(_perpt_bin_file, "w") as _f:
            json.dump(
                {_c: {str(_k): _v for _k, _v in _d.items()} for _c, _d in bins_perpt.items()},
                _f,
                indent=2,
            )
        print("Saved per-pT bins to:", _perpt_bin_file)
    else:
        for _ang in angularities:
            bins_perpt.setdefault(f"sd_{_ang}", bins_perpt[_ang])

    # Uniform x-bins for the PROFILES only (the distributions use bins_perpt).
    # linspace over each var's global flat range, replicated across every jet-pT
    # slice (the range is pT-independent). Deterministic from the shared `bins`
    # flat file + fixed counts, so every sysvar/MC lines up automatically -> no
    # JSON cache needed (unlike the sample-derived bins_perpt).
    _n_jpt = len(jpt_bins) - 1
    prof_bins_perpt = {}
    for _c in (*_uniform_cols, *_bb_cols):
        _lo, _hi = float(bins[_c][0]), float(bins[_c][-1])
        _n = _prof_nbins.get(_c, _prof_nbins_default)
        _edges = np.linspace(_lo, _hi, _n + 1).tolist()
        prof_bins_perpt[_c] = {i: _edges for i in range(_n_jpt)}
    for _ang in angularities:
        prof_bins_perpt.setdefault(f"sd_{_ang}", prof_bins_perpt[_ang])

    # Mask the (large) gen/MC arrow tables to each jet-pT slice ONCE and reuse
    # across every variable (avoid re-masking O(10^7) rows per variable).
    gen_masks = perpt_masks(gen_table, jpt_bins)
    mc_masks = {_name: perpt_masks(_tbl, jpt_bins) for _name, _tbl, _ in mc_tables}
    return bins_perpt, gen_masks, jpt_bins, mc_masks, prof_bins_perpt


@app.cell
def _(
    bins_perpt,
    gen_masks,
    gen_table,
    jpt_bins,
    mc_masks,
    mc_tables,
    truth_weights,
    unf_weights,
):
    prefix = os.path.join("outputs", "histograms", str(sys_var), feature_mode)

    print("Data histograms will be saved to:", prefix)

    print("Calculating histograms for:", common_vars[1:])

    for _var in common_vars[1:]:
        # --- old: one global 2-D (pt, var) histogram shared across pT panels ---
        # _hist = histogram(gen_table, bins, ("pt", _var), unf_weights)
        # _truth_hist = (
        #     histogram(gen_table, bins, ("pt", _var), truth_weights)
        #     if truth_weights is not None else None
        # )
        # save_hist_2d(_hist, prefix=_var_prefix, fname_prefix="hist",
        #              batched=True, true_hist=_truth_hist)
        # --- new: per-jet-pT adaptive binning (1-D hist per slice) ---
        _var_prefix = os.path.join(prefix, _var)
        _h1ds = histogram_perpt(
            gen_table, bins_perpt, _var, jpt_bins, weights=unf_weights, masks=gen_masks
        )
        _truth_h1ds = (
            histogram_perpt(
                gen_table,
                bins_perpt,
                _var,
                jpt_bins,
                weights=truth_weights,
                masks=gen_masks,
            )
            if truth_weights is not None
            else None
        )
        save_hist_1d_list(
            _h1ds,
            prefix=_var_prefix,
            fname_prefix="hist",
            batched=True,
            true_h1ds=_truth_h1ds,
        )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name, feature_mode)
            print(f"{_mc_name} histograms will be saved to:", _mc_prefix)

            _mc_h1ds = histogram_perpt(
                _mc_table,
                bins_perpt,
                _var,
                jpt_bins,
                weights=_mc_weights,
                masks=mc_masks[_mc_name],
            )
            save_hist_1d_list(
                _mc_h1ds,
                prefix=os.path.join(_mc_prefix, _var),
                fname_prefix="hist",
                batched=False,
            )
            save_ratio_1d_list(
                _mc_h1ds,
                _h1ds,
                prefix=os.path.join(_mc_prefix, _var),
                fname_prefix=f"ratio_data_vs_{_mc_name}",
                batched=True,
            )
            save_ratio_1d_list(
                # MC/Data now (was Data/MC: _h1ds, _mc_h1ds). Filename kept as
                # "data_vs" so old snapshots overwrite in place (renaming would
                # strand stale-binned files and re-break systematics.py).
                _mc_h1ds,
                _h1ds,
                prefix=os.path.join(_mc_prefix, _var),
                fname_prefix=f"ratio_hist_data_vs_{_mc_name}",
                batched=True,
            )
            del _mc_h1ds

        del _h1ds, _truth_h1ds
        torch.cuda.empty_cache()
    return (prefix,)


@app.cell
def _(
    bins_perpt,
    gen_masks,
    gen_table,
    jpt_bins,
    mc_masks,
    mc_tables,
    prefix,
    truth_weights,
    unf_weights,
):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)

        print("Calculating histograms for:", ("pt", _ang, f"sd_{_ang}"))

        # --- old: one global 3-D (pt, ang, sd_ang) hist, projected per pT ---
        # _hist = histogram(gen_table, bins, ("pt", _ang, f"sd_{_ang}"), unf_weights)
        # _hist_incl = _hist.project("pt", _ang); _hist_sd = _hist.project(...)
        # --- new: per-jet-pT adaptive 1-D hists (sd_<ang> shares incl edges) ---
        _incl_h1ds = histogram_perpt(
            gen_table, bins_perpt, _ang, jpt_bins, weights=unf_weights, masks=gen_masks
        )
        _sd_h1ds = histogram_perpt(
            gen_table,
            bins_perpt,
            f"sd_{_ang}",
            jpt_bins,
            weights=unf_weights,
            masks=gen_masks,
        )
        _truth_incl = (
            histogram_perpt(
                gen_table,
                bins_perpt,
                _ang,
                jpt_bins,
                weights=truth_weights,
                masks=gen_masks,
            )
            if truth_weights is not None
            else None
        )
        _truth_sd = (
            histogram_perpt(
                gen_table,
                bins_perpt,
                f"sd_{_ang}",
                jpt_bins,
                weights=truth_weights,
                masks=gen_masks,
            )
            if truth_weights is not None
            else None
        )

        save_hist_1d_list(
            _incl_h1ds,
            prefix=_ang_prefix,
            fname_prefix="hist_ang",
            batched=True,
            true_h1ds=_truth_incl,
        )
        save_hist_1d_list(
            _sd_h1ds,
            prefix=_ang_prefix,
            fname_prefix="hist_sd_ang",
            batched=True,
            true_h1ds=_truth_sd,
        )
        save_ratio_1d_list(
            _sd_h1ds,
            _incl_h1ds,
            prefix=_ang_prefix,
            fname_prefix="ratio_incl_vs_sd",
            batched=True,
            true_num_h1ds=_truth_sd,
            true_den_h1ds=_truth_incl,
        )

        del _truth_incl, _truth_sd

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name, feature_mode)

            _mc_incl = histogram_perpt(
                _mc_table,
                bins_perpt,
                _ang,
                jpt_bins,
                weights=_mc_weights,
                masks=mc_masks[_mc_name],
            )
            save_hist_1d_list(
                _mc_incl,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="hist_ang",
                batched=False,
            )
            save_ratio_1d_list(
                _mc_incl,
                _incl_h1ds,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_incl_vs_{_mc_name}",
                batched=True,
            )
            save_ratio_1d_list(
                # MC/Data now (was Data/MC: _incl_h1ds, _mc_incl).
                _mc_incl,
                _incl_h1ds,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_ang_data_vs_{_mc_name}",
                batched=True,
            )

            _mc_sd = histogram_perpt(
                _mc_table,
                bins_perpt,
                f"sd_{_ang}",
                jpt_bins,
                weights=_mc_weights,
                masks=mc_masks[_mc_name],
            )
            save_hist_1d_list(
                _mc_sd,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="hist_sd_ang",
                batched=False,
            )
            save_ratio_1d_list(
                _mc_sd,
                _sd_h1ds,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_sd_vs_{_mc_name}",
                batched=True,
            )
            save_ratio_1d_list(
                # MC/Data now (was Data/MC: _sd_h1ds, _mc_sd).
                _mc_sd,
                _sd_h1ds,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_sd_ang_data_vs_{_mc_name}",
                batched=True,
            )

            save_ratio_1d_list(
                _mc_sd,
                _mc_incl,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="ratio_incl_vs_sd",
            )
            del _mc_incl, _mc_sd

        del _sd_h1ds, _incl_h1ds
        torch.cuda.empty_cache()
    return


@app.cell
def _(
    gen_masks,
    gen_table,
    jpt_bins,
    mc_masks,
    mc_tables,
    prefix,
    prof_bins_perpt,
    truth_weights,
    unf_weights,
):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)
        # axes to profile over (drop pt: it is now the per-slice masking axis)
        _cols = (*common_vars[1:], f"sd_{_ang}")
        print("Calculating profile histograms for", _ang, "in bins of", ("pt", *_cols))

        # --- old: one global multi-D profile over (pt, *cols), projected per pT ---
        # _prof = profile(gen_table, bins, (*common_vars, f"sd_{_ang}"), _ang, unf_weights)
        # _data_projs[_var] = _prof.project("pt", _var)
        # --- new: per-jet-pT 1-D profiles vs each var (adaptive binning) ---
        _data_projs = profile_perpt(
            gen_table,
            # bins_perpt,  # old: adaptive bins shared with distributions
            prof_bins_perpt,  # uniform x-bins for profiles only
            _cols,
            _ang,
            jpt_bins,
            weights=unf_weights,
            masks=gen_masks,
        )
        _truth_projs = (
            profile_perpt(
                gen_table,
                # bins_perpt,  # old: adaptive bins shared with distributions
                prof_bins_perpt,  # uniform x-bins for profiles only
                _cols,
                _ang,
                jpt_bins,
                weights=truth_weights,
                masks=gen_masks,
            )
            if truth_weights is not None
            else None
        )

        for _var in _cols:
            save_hist_1d_list(
                _data_projs[_var],
                prefix=_ang_prefix,
                fname_prefix=f"prof_incl_vs_{_var}",
                batched=True,
                true_h1ds=_truth_projs[_var] if _truth_projs is not None else None,
            )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name, feature_mode)
            _mc_projs = profile_perpt(
                _mc_table,
                # bins_perpt,  # old: adaptive bins shared with distributions
                prof_bins_perpt,  # uniform x-bins for profiles only
                _cols,
                _ang,
                jpt_bins,
                weights=_mc_weights,
                masks=mc_masks[_mc_name],
            )
            for _var in _cols:
                save_hist_1d_list(
                    _mc_projs[_var],
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"prof_incl_vs_{_var}",
                    batched=False,
                )
                save_ratio_1d_list(
                    # MC/Data now (was Data/MC: _data_projs, _mc_projs).
                    _mc_projs[_var],
                    _data_projs[_var],
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"ratio_prof_incl_vs_{_var}_data_vs_{_mc_name}",
                    batched=True,
                )
            del _mc_projs
        del _data_projs, _truth_projs
        torch.cuda.empty_cache()
    return


@app.cell
def _(
    gen_masks,
    gen_table,
    jpt_bins,
    mc_masks,
    mc_tables,
    prefix,
    prof_bins_perpt,
    truth_weights,
    unf_weights,
):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)
        # axes to profile over (drop pt: it is now the per-slice masking axis)
        _cols = (*common_vars[1:], _ang)
        print(
            "Calculating profile histograms for",
            f"sd_{_ang}",
            "in bins of",
            ("pt", *_cols),
        )

        # --- old: one global multi-D profile over (pt, *cols), projected per pT ---
        # _prof = profile(gen_table, bins, (*common_vars, _ang), f"sd_{_ang}", unf_weights)
        # --- new: per-jet-pT 1-D profiles vs each var (adaptive binning) ---
        _data_projs = profile_perpt(
            gen_table,
            # bins_perpt,  # old: adaptive bins shared with distributions
            prof_bins_perpt,  # uniform x-bins for profiles only
            _cols,
            f"sd_{_ang}",
            jpt_bins,
            weights=unf_weights,
            masks=gen_masks,
        )
        _truth_projs = (
            profile_perpt(
                gen_table,
                # bins_perpt,  # old: adaptive bins shared with distributions
                prof_bins_perpt,  # uniform x-bins for profiles only
                _cols,
                f"sd_{_ang}",
                jpt_bins,
                weights=truth_weights,
                masks=gen_masks,
            )
            if truth_weights is not None
            else None
        )

        for _var in _cols:
            save_hist_1d_list(
                _data_projs[_var],
                prefix=_ang_prefix,
                fname_prefix=f"prof_sd_vs_{_var}",
                batched=True,
                true_h1ds=_truth_projs[_var] if _truth_projs is not None else None,
            )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name, feature_mode)
            _mc_projs = profile_perpt(
                _mc_table,
                # bins_perpt,  # old: adaptive bins shared with distributions
                prof_bins_perpt,  # uniform x-bins for profiles only
                _cols,
                f"sd_{_ang}",
                jpt_bins,
                weights=_mc_weights,
                masks=mc_masks[_mc_name],
            )
            for _var in _cols:
                save_hist_1d_list(
                    _mc_projs[_var],
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"prof_sd_vs_{_var}",
                    batched=False,
                )
                save_ratio_1d_list(
                    # MC/Data now (was Data/MC: _data_projs, _mc_projs).
                    _mc_projs[_var],
                    _data_projs[_var],
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"ratio_prof_sd_vs_{_var}_data_vs_{_mc_name}",
                    batched=True,
                )
            del _mc_projs
        del _data_projs, _truth_projs
        torch.cuda.empty_cache()
    return


@app.cell
def _():
    _closure_dir = "outputs/closure"
    A_gen_table = None
    A_reco_table = None
    closure_gen_weights = None
    closure_reco_weights = None
    if os.path.exists(os.path.join(_closure_dir, "manifest.json")):
        with open(os.path.join(_closure_dir, "manifest.json"), "r") as _f:
            _manifest = json.load(_f)
        _idx = np.load(os.path.join(_closure_dir, "indices.npz"))
        _w_unfolding = np.load(os.path.join(_closure_dir, "w_unfolding.npz"))

        _n_match = _manifest["n_match"]
        _A_m = _idx["A_m"]
        _A_mi = _idx["A_mi"]
        _A_f = _idx["A_f"]

        _src_dir = dataset_root + "/jets/embedding"
        _root_dir = os.path.join(_src_dir, str(SysVar.NONE))

        _bufs = []
        _bufs.append(pa.memory_map(os.path.join(_root_dir, "gen-matches.arrow")))
        _gm = pa.ipc.open_file(_bufs[-1]).read_all()
        _bufs.append(pa.memory_map(os.path.join(_root_dir, "misses.arrow")))
        _ms = pa.ipc.open_file(_bufs[-1]).read_all()
        _bufs.append(pa.memory_map(os.path.join(_root_dir, "reco-matches.arrow")))
        _rm = pa.ipc.open_file(_bufs[-1]).read_all()
        _bufs.append(pa.memory_map(os.path.join(_root_dir, "fakes.arrow")))
        _fk = pa.ipc.open_file(_bufs[-1]).read_all()

        _gen_full = pa.concat_tables((_gm, _ms))
        _reco_full = pa.concat_tables((_rm, _fk))

        _A_gen_rows = np.concatenate([_A_m, _n_match + _A_mi]).astype(np.int64)
        _A_reco_rows = np.concatenate([_A_m, _n_match + _A_f]).astype(np.int64)

        A_gen_table = _gen_full.take(_A_gen_rows)
        A_reco_table = _reco_full.take(_A_reco_rows)

        _n_iter = _manifest["num_iterations"]
        _gen_key = f"arr_{2 * _n_iter}"
        _reco_key = f"arr_{2 * _n_iter + 1}"
        closure_gen_weights = torch.as_tensor(_w_unfolding[_gen_key], dtype=torch.float32)
        closure_reco_weights = torch.as_tensor(_w_unfolding[_reco_key], dtype=torch.float32)

        print(
            f"Closure: A_gen {len(A_gen_table)} rows, "
            f"A_reco {len(A_reco_table)} rows; "
            f"gen weights {tuple(closure_gen_weights.shape)}, "
            f"reco weights {tuple(closure_reco_weights.shape)}"
        )
    else:
        print("No closure manifest at", _closure_dir, "- skipping closure plots")
    return A_gen_table, A_reco_table, closure_gen_weights, closure_reco_weights


@app.cell
def _(
    A_gen_table,
    A_reco_table,
    bins,
    closure_gen_weights,
    closure_reco_weights,
):
    _prefix = os.path.join("outputs", "histograms", "closure", feature_mode)
    for _var in (*common_vars[1:], *angularities):
        _gen_truth = histogram(A_gen_table, bins, ("pt", _var), None)
        _gen_unf = histogram(A_gen_table, bins, ("pt", _var), closure_gen_weights)
        _reco_unf = histogram(A_reco_table, bins, ("pt", _var), closure_reco_weights)
        _var_prefix = os.path.join(_prefix, _var)
        save_hist_2d(
            _gen_unf,
            prefix=_var_prefix,
            fname_prefix="hist_gen_closure",
            batched=True,
            true_hist=_gen_truth,
        )
        save_hist_2d(
            _reco_unf,
            prefix=_var_prefix,
            fname_prefix="hist_reco_closure",
            batched=True,
            true_hist=_gen_truth,
        )
        del _gen_truth, _gen_unf, _reco_unf
    torch.cuda.empty_cache()
    return


if __name__ == "__main__":
    app.run()
