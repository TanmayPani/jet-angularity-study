import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def imports():
    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import pyarrow as pa
    from pathlib import Path
    import marimo as mo

    # Match plot_physics / the poster: reset to matplotlib's default style so text and
    # ticks render black-on-white. (The live notebook's dark theme leaves text.color etc.
    # white, which is invisible on a white facecolor.)
    plt.style.use("default")
    return Path, mo, np, pa, plt, torch


@app.cell
def config(Path):
    # Generalized feature-mode comparison: NUMERATOR mode vs DENOMINATOR mode.
    # Both sides are unfolded-DATA results (sysvar "nominal") and are RE-PROJECTED from
    # their own per-replica weights (w_unfolding.npz) at a chosen OmniFold iteration, so
    # each mode can be stopped at its own runaway-safe iteration with its own rogue-replica
    # trim. The observables themselves are shared (same angularities binning across modes);
    # only the unfolding weights differ.
    #
    # This is the direct generalization of plot_ptd_comparison.py, where the special case
    # was MODE_NUM="angularities_noptd" (iter2, no trim) vs MODE_DEN="angularities" (iter5).
    HIST_ROOT = Path("outputs/histograms")

    MODE_NUM = "angularities_noptd"  # numerator feature mode
    MODE_DEN = "angularities"  # denominator feature mode
    ITER_NUM = 2  # OmniFold iteration to re-project the numerator at
    ITER_DEN = 2  # OmniFold iteration to re-project the denominator at
    DROP_NUM = 0  # drop this many runaway replicas (largest peak-bin) from the numerator
    DROP_DEN = 0  # ... and from the denominator
    SYSVAR = "nominal"  # unfolded-data result

    # Mode whose `angularities` arrows back the GEN-level observables used for re-projection.
    # All angularities-family modes (full / noptd / minimal) project the SAME observables
    # from the shared `angularities` arrows; the subset modes' arrows are symlinks into it,
    # so the gen-jet ordering matches the per-replica weight index for every such mode.
    OBS_MODE = "angularities"
    # Mode whose .pt snapshot supplies the bin EDGES + x positions (shared across modes,
    # baked from the common bins_perpt.json). Use the denominator's snapshot by default.
    EDGE_MODE = MODE_DEN

    JPT_BINS = (0, 1, 2, 3)
    CENTER_JPT = 2  # 20<pT<30 GeV/c -- the headline slice


    def _short(mode):
        # Compact slug used for the OUT_DIR path + the ratio-axis label:
        # "angularities" -> "full", "angularities_noptd" -> "noptd", etc.
        s = mode.replace("angularities_", "")
        return "full" if s == "angularities" else s


    # Legend label: the observable(s) each mode DROPS from the full angularity input set
    # (this is what physically distinguishes the modes). Falls back to the slug for any
    # mode not listed here.
    _DROPPED = {
        "angularities": "all features",
        "angularities_noptd": r"no $\lambda_{\beta=0}^{\kappa=2},\,\lambda_{\beta=0, \mathrm{g}}^{\kappa=2}$",
        "angularities_minimal": r"no $M,\,M_g,\,R_g,\,\lambda_{\beta=0}^{\kappa=2},\,\lambda_{\beta=0, \mathrm{g}}^{\kappa=2}$",
    }

    NUM_SLUG = _short(MODE_NUM)
    DEN_SLUG = _short(MODE_DEN)
    NUM_LABEL = _DROPPED.get(MODE_NUM, NUM_SLUG)  # legend text for the numerator
    DEN_LABEL = _DROPPED.get(MODE_DEN, DEN_SLUG)  # legend text for the denominator
    RATIO_LABEL = f"{NUM_SLUG}/{DEN_SLUG}"

    OUT_DIR = Path("outputs/hp2026_poster/featmode_comp") / f"{NUM_SLUG}_vs_{DEN_SLUG}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        CENTER_JPT,
        DEN_LABEL,
        DROP_DEN,
        DROP_NUM,
        EDGE_MODE,
        HIST_ROOT,
        ITER_DEN,
        ITER_NUM,
        JPT_BINS,
        MODE_DEN,
        MODE_NUM,
        NUM_LABEL,
        OBS_MODE,
        OUT_DIR,
        RATIO_LABEL,
        SYSVAR,
    )


@app.cell
def helpers(HIST_ROOT, SYSVAR, np, torch):
    def load_snap(mode, var, stem, jpt, sysvar=SYSVAR):
        """Load one histogram-snapshot dict (bin_center/half_bin_width/bin_count/bin_count_err[/std])."""
        return torch.load(HIST_ROOT / sysvar / mode / var / f"{stem}_jpt{jpt}.pt", mmap=True)


    def disp_err(snap):
        """The error the poster plots actually draw (plot_physics.plot_data_points): the
        across-replica ensemble spread if batched, else the statistical error. nan->0."""
        e = snap.get("bin_count_std", snap["bin_count_err"])
        return np.nan_to_num(e.numpy())

    return (load_snap,)


@app.cell
def manifest():
    # Poster distribution set: (var dir, snapshot stem, x-axis label).
    # ch_ang_k1_* are the headline angularities; ch_ang_k2_b0 is p_T^D itself; m/sd_* are
    # the jet-shape observables. (Shared across feature modes -- only the weights differ.)
    MANIFEST = [
        ("ch_ang_k1_b0.5", "hist_ang", r"$\lambda^{1}_{0.5}$ (incl)"),
        ("ch_ang_k1_b0.5", "hist_sd_ang", r"$\lambda^{1}_{0.5}$ (SD)"),
        ("ch_ang_k1_b1", "hist_ang", r"$\lambda^{1}_{1}$ (incl)"),
        ("ch_ang_k1_b1", "hist_sd_ang", r"$\lambda^{1}_{1}$ (SD)"),
        ("ch_ang_k1_b2", "hist_ang", r"$\lambda^{1}_{2}$ (incl)"),
        ("ch_ang_k1_b2", "hist_sd_ang", r"$\lambda^{1}_{2}$ (SD)"),
        ("ch_ang_k2_b0", "hist_ang", r"$p_T^D$ (incl)"),
        ("ch_ang_k2_b0", "hist_sd_ang", r"$p_T^D$ (SD)"),
        ("m", "hist", r"$M$ [GeV]"),
        ("sd_m", "hist", r"$M_{SD}$ [GeV]"),
        ("sd_dR", "hist", r"$\Delta R_{SD}$"),
        ("sd_symmetry", "hist", r"$z_g$"),
    ]

    # Poster PROFILE set: (var dir, x-axis var, x-axis label). Each profiles BOTH the
    # inclusive angularity (y=var) and its SoftDrop variant (y=sd_<var>) vs the x-var.
    PROF_MANIFEST = [
        ("ch_ang_k1_b0.5", "sd_dR", r"$R_g$"),
        ("ch_ang_k1_b1", "sd_dR", r"$R_g$"),
        ("ch_ang_k1_b2", "sd_dR", r"$R_g$"),
        ("ch_ang_k2_b0", "sd_symmetry", r"$z_g$"),
    ]

    PROF_YLABEL = {
        "ch_ang_k1_b0.5": r"$\langle\lambda^{1}_{0.5}\rangle$",
        "ch_ang_k1_b1": r"$\langle\lambda^{1}_{1}\rangle$",
        "ch_ang_k1_b2": r"$\langle\lambda^{1}_{2}\rangle$",
        "ch_ang_k2_b0": r"$\langle p_T^D\rangle$",
    }
    return MANIFEST, PROF_MANIFEST, PROF_YLABEL


@app.cell
def reproj_inputs(MANIFEST, OBS_MODE, Path, SYSVAR, np, pa):
    # --- Re-projection inputs ---
    # The saved .pt snapshots are the per-mode iteration-mean over all replicas. To compare
    # ANY two feature modes at ANY (per-mode) iteration -- and to trim runaway replicas -- we
    # re-derive each mode's distributions/profiles by re-binning the GEN-level observables
    # (gen-matches + misses) weighted by that mode's per-replica unfolded weights arr_{2*iter}
    # in its own w_unfolding.npz -- exactly as histograms.py does.
    _FEAT = Path("datasets/STAR_pp200GeV_production_2012/features")
    GEN_EMB = _FEAT / OBS_MODE / "embedding" / SYSVAR  # shared observables (same jet order)
    JPT_EDGES = (10.0, 15.0, 20.0, 30.0, 60.0)


    def wunf_path(mode):
        """Per-mode unfolding-weights archive (per-jet per-replica weights, arr_{2*iter})."""
        return _FEAT / mode / "embedding" / SYSVAR / "w_unfolding.npz"


    def obs_col(var, stem):
        """Gen observable column backing a (var dir, snapshot stem)."""
        if stem == "hist_sd_ang":
            return f"sd_{var}"
        return var  # hist_ang -> var ; hist (m / sd_*) -> var


    def _gencol(col):
        _a = []
        for _f in ("gen-matches.arrow", "misses.arrow"):
            _b = pa.memory_map(str(GEN_EMB / _f))
            _a.append(pa.ipc.open_file(_b).read_all()[col].to_numpy())
        return np.concatenate(_a).astype(np.float64)


    gen_pt = _gencol("pt")
    gen_obs = {obs_col(v, s): _gencol(obs_col(v, s)) for v, s, _ in MANIFEST}
    jpt_masks = [
        (gen_pt >= JPT_EDGES[i]) & (gen_pt < JPT_EDGES[i + 1]) for i in range(len(JPT_EDGES) - 1)
    ]
    return gen_obs, jpt_masks, obs_col, wunf_path


@app.cell
def reproj_funcs(
    EDGE_MODE,
    JPT_BINS,
    MANIFEST,
    PROF_MANIFEST,
    gen_obs,
    jpt_masks,
    load_snap,
    np,
    obs_col,
    wunf_path,
):
    def _edges_from_snapshot(var, stem, jpt):
        """Bin edges recovered from the EDGE_MODE snapshot so re-binning matches it exactly.
        Edges are baked from the shared bins_perpt.json, so they're identical across modes."""
        _s = load_snap(EDGE_MODE, var, stem, jpt)
        _bc = _s["bin_center"].numpy()
        _hw = _s["half_bin_width"].numpy()
        return np.append(_bc - _hw, _bc[-1] + _hw[-1])


    def load_mode_weights(mode, iteration):
        """Per-jet per-replica unfolded weights for `mode` at OmniFold `iteration` (arr_{2*iter})."""
        return np.load(wunf_path(mode))[f"arr_{2 * iteration}"]


    def perreplica_hist(weights2d, var, stem, jpt):
        """(R, nbins) per-replica unit-area histogram of the gen observable, weighted by weights2d."""
        _e = _edges_from_snapshot(var, stem, jpt)
        _m = jpt_masks[jpt]
        _v = gen_obs[obs_col(var, stem)][_m]
        _W = weights2d[:, _m]
        return np.stack(
            [np.histogram(_v, bins=_e, weights=_W[r], density=True)[0] for r in range(_W.shape[0])]
        )


    def mode_collapse(mode, iteration, *, drop_n=0, weights=None):
        """Collapse a mode's per-replica histograms to {(var,stem,jpt): (mean, std)} over all
        MANIFEST slices. `drop_n` > 0 drops the drop_n replicas with the largest peak-bin across
        all slices (runaway members). Returns (dict, kept_idx)."""
        _w = load_mode_weights(mode, iteration) if weights is None else weights
        _H = {(v, s, j): perreplica_hist(_w, v, s, j) for v, s, _ in MANIFEST for j in JPT_BINS}
        _R = _w.shape[0]
        if drop_n:
            _score = np.zeros(_R)
            for _Hk in _H.values():
                _score += _Hk.max(1)
            _keep = np.sort(np.argsort(_score)[: _R - drop_n])
        else:
            _keep = np.arange(_R)
        return {k: (Hk[_keep].mean(0), Hk[_keep].std(0)) for k, Hk in _H.items()}, _keep


    # ---------------------------------------------------------------------------
    # Profile re-projection: <y> vs x in x-bins, weighted by the per-replica unfolded
    # weights -- the profile analogue of perreplica_hist/mode_collapse, matching
    # histograms.profile_perpt (Profile.create: <y>_bin = sum(w*y)/sum(w)).
    # ---------------------------------------------------------------------------
    def prof_stem(x_var, kind):
        """Snapshot stem for a profile: kind in {'incl','sd'}."""
        return f"prof_{kind}_vs_{x_var}"


    def prof_yobs(var, kind):
        """Gen y-observable backing a profile: inclusive var, or its SoftDrop variant."""
        return var if kind == "incl" else f"sd_{var}"


    def perreplica_profile(weights2d, yobs, xobs, edges, jpt):
        """(R, nbins) per-replica weighted profile <yobs> vs xobs over `edges`. Empty
        x-bins yield nan (matplotlib skips them)."""
        _m = jpt_masks[jpt]
        _x = gen_obs[xobs][_m]
        _y = gen_obs[yobs][_m]
        _W = weights2d[:, _m]
        _rows = []
        for _r in range(_W.shape[0]):
            _wr = _W[_r]
            _sw, _ = np.histogram(_x, bins=edges, weights=_wr)
            _swy, _ = np.histogram(_x, bins=edges, weights=_wr * _y)
            with np.errstate(divide="ignore", invalid="ignore"):
                _rows.append(_swy / _sw)
        return np.stack(_rows)


    def mode_profile_collapse(mode, iteration, *, drop_n=0, weights=None):
        """Collapse a mode's per-replica PROFILES to {(var,x,kind,jpt): (mean, std)} over all
        PROF_MANIFEST slices (incl + sd). Edges recovered from the EDGE_MODE profile snapshot.
        Returns (dict, kept_idx)."""
        _w = load_mode_weights(mode, iteration) if weights is None else weights
        _H = {}
        for _var, _x, _ in PROF_MANIFEST:
            for _kind in ("incl", "sd"):
                _stem = prof_stem(_x, _kind)
                _y = prof_yobs(_var, _kind)
                for _j in JPT_BINS:
                    _e = _edges_from_snapshot(_var, _stem, _j)
                    _H[(_var, _x, _kind, _j)] = perreplica_profile(_w, _y, _x, _e, _j)
        _R = _w.shape[0]
        if drop_n:
            _score = np.zeros(_R)
            for _Hk in _H.values():
                _score += np.nan_to_num(_Hk).max(1)
            _keep = np.sort(np.argsort(_score)[: _R - drop_n])
        else:
            _keep = np.arange(_R)
        return {k: (np.nanmean(Hk[_keep], 0), np.nanstd(Hk[_keep], 0)) for k, Hk in _H.items()}, _keep

    return load_mode_weights, mode_collapse, mode_profile_collapse, prof_stem


@app.cell
def reproj_compute(
    DROP_DEN,
    DROP_NUM,
    ITER_DEN,
    ITER_NUM,
    MODE_DEN,
    MODE_NUM,
    load_mode_weights,
    mo,
    mode_collapse,
    mode_profile_collapse,
):
    # Heavy cell: re-projects BOTH modes' unfolded weights at their chosen iterations. Each
    # mode's weights are loaded once (arr_{2*iter}) and reused for the dist + profile collapse
    # to bound memory.
    _wn = load_mode_weights(MODE_NUM, ITER_NUM)
    num_dist, num_keep = mode_collapse(MODE_NUM, ITER_NUM, drop_n=DROP_NUM, weights=_wn)
    num_prof, _ = mode_profile_collapse(MODE_NUM, ITER_NUM, drop_n=DROP_NUM, weights=_wn)
    del _wn

    _wd = load_mode_weights(MODE_DEN, ITER_DEN)
    den_dist, den_keep = mode_collapse(MODE_DEN, ITER_DEN, drop_n=DROP_DEN, weights=_wd)
    den_prof, _ = mode_profile_collapse(MODE_DEN, ITER_DEN, drop_n=DROP_DEN, weights=_wd)
    del _wd

    mo.md(
        f"Re-projected. **num** `{MODE_NUM}` @ iter{ITER_NUM} (drop {DROP_NUM}, kept "
        f"`{num_keep.tolist()}`) vs **den** `{MODE_DEN}` @ iter{ITER_DEN} (drop {DROP_DEN}, "
        f"kept `{den_keep.tolist()}`): `{len(num_dist)}` dist + `{len(num_prof)}` profile slices."
    )
    return den_dist, den_prof, num_dist, num_prof


@app.cell
def compare_dist_fn(
    DEN_LABEL,
    EDGE_MODE,
    JPT_BINS,
    NUM_LABEL,
    RATIO_LABEL,
    load_snap,
    np,
    plt,
):
    def compare_grid_reproj(
        var, stem, num_results, den_results, *, xlabel=None, title=None, ratio_ylim=(0.5, 1.5)
    ):
        """2 rows (overlay + num/den ratio) x len(JPT_BINS) cols for one (var, stem).
        Both numerator and denominator come from precomputed {(var,stem,jpt): (mean, std)}
        re-projection dicts; x positions / bin widths come from the EDGE_MODE snapshot."""
        ncol = len(JPT_BINS)
        fig, axs = plt.subplots(
            2,
            ncol,
            figsize=(3.2 * ncol, 5.0),
            sharex="col",
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05, wspace=0.30),
            squeeze=False,
        )
        for _j, _jpt in enumerate(JPT_BINS):
            _ref = load_snap(EDGE_MODE, var, stem, _jpt)
            _x = _ref["bin_center"].numpy()
            _xw = _ref["half_bin_width"].numpy()
            _yden, _eden = den_results[(var, stem, _jpt)]
            _ynum, _enum = num_results[(var, stem, _jpt)]
            _ax, _axr = axs[0, _j], axs[1, _j]
            _ax.errorbar(
                _x, _yden, xerr=_xw, yerr=_eden, fmt="s", ms=3, lw=1, color="C0", label=DEN_LABEL
            )
            _ax.errorbar(
                _x,
                _ynum,
                xerr=_xw,
                yerr=_enum,
                fmt="o",
                ms=3,
                lw=1,
                color="C3",
                mfc="none",
                label=NUM_LABEL,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                _r = _ynum / _yden
                _re = np.abs(_r) * np.sqrt((_enum / _ynum) ** 2 + (_eden / _yden) ** 2)
            _axr.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color="C3", mfc="none")
            _axr.axhline(1.0, color="k", lw=0.8, ls="--")
            _axr.set_ylim(*ratio_ylim)
            _ax.set_title(f"jet $p_T$ bin {_jpt}", fontsize=9)
            if xlabel:
                _axr.set_xlabel(xlabel)
            if _j == 0:
                _ax.set_ylabel("unfolded (a.u.)")
                _axr.set_ylabel(RATIO_LABEL)
                _ax.legend(fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        fig.tight_layout()
        return fig

    return (compare_grid_reproj,)


@app.cell
def compare_prof_fn(
    DEN_LABEL,
    EDGE_MODE,
    JPT_BINS,
    NUM_LABEL,
    PROF_YLABEL,
    RATIO_LABEL,
    load_snap,
    np,
    plt,
    prof_stem,
):
    def compare_profile_reproj(
        var, x_var, kind, num_results, den_results, *, xlabel=None, title=None, ratio_ylim=(0.9, 1.1)
    ):
        """Profile analogue of compare_grid_reproj: <y> vs x_var (kind in {'incl','sd'}).
        Both num/den are re-projected dicts {(var,x,kind,jpt): (mean, std)}; x positions come
        from the EDGE_MODE profile snapshot."""
        _stem = prof_stem(x_var, kind)
        ncol = len(JPT_BINS)
        fig, axs = plt.subplots(
            2,
            ncol,
            figsize=(3.2 * ncol, 5.0),
            sharex="col",
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05, wspace=0.30),
            squeeze=False,
        )
        for _j, _jpt in enumerate(JPT_BINS):
            _ref = load_snap(EDGE_MODE, var, _stem, _jpt)
            _x = _ref["bin_center"].numpy()
            _xw = _ref["half_bin_width"].numpy()
            _yden, _eden = den_results[(var, x_var, kind, _jpt)]
            _ynum, _enum = num_results[(var, x_var, kind, _jpt)]
            _ax, _axr = axs[0, _j], axs[1, _j]
            _ax.errorbar(
                _x, _yden, xerr=_xw, yerr=_eden, fmt="s", ms=3, lw=1, color="C0", label=DEN_LABEL
            )
            _ax.errorbar(
                _x,
                _ynum,
                xerr=_xw,
                yerr=_enum,
                fmt="o",
                ms=3,
                lw=1,
                color="C3",
                mfc="none",
                label=NUM_LABEL,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                _r = _ynum / _yden
                _re = np.abs(_r) * np.sqrt((_enum / _ynum) ** 2 + (_eden / _yden) ** 2)
            _axr.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color="C3", mfc="none")
            _axr.axhline(1.0, color="k", lw=0.8, ls="--")
            _axr.set_ylim(*ratio_ylim)
            _ax.set_title(f"jet $p_T$ bin {_jpt}", fontsize=9)
            if xlabel:
                _axr.set_xlabel(xlabel)
            if _j == 0:
                _tag = "incl" if kind == "incl" else "SD"
                _ax.set_ylabel(f"{PROF_YLABEL[var]}  ({_tag})")
                _axr.set_ylabel(RATIO_LABEL)
                _ax.legend(fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        fig.tight_layout()
        return fig

    return (compare_profile_reproj,)


@app.cell
def demo(DEN_LABEL, NUM_LABEL, compare_grid_reproj, den_dist, num_dist):
    fig_demo = compare_grid_reproj(
        "ch_ang_k1_b1",
        "hist_ang",
        num_dist,
        den_dist,
        xlabel=r"$\lambda^{1}_{1}$ (inclusive)",
        title=rf"ch_ang_k1_b1 inclusive  —  {NUM_LABEL} vs {DEN_LABEL}",
    )
    fig_demo
    return


@app.cell
def driver_dist(
    DEN_LABEL,
    MANIFEST,
    NUM_LABEL,
    OUT_DIR,
    compare_grid_reproj,
    den_dist,
    mo,
    num_dist,
):
    # === Distribution comparisons: NUM mode vs DEN mode, each re-projected at its iteration.
    # Owns the canonical fig_cmp_<var>_<stem>.pdf under OUT_DIR.
    _panels = {}
    saved_paths = []
    for _var, _stem, _xl in MANIFEST:
        _fig = compare_grid_reproj(
            _var,
            _stem,
            num_dist,
            den_dist,
            xlabel=_xl,
            title=f"{_var}  [{_stem}]  --  {NUM_LABEL} vs {DEN_LABEL}",
        )
        _out = OUT_DIR / f"fig_cmp_{_var}_{_stem}.pdf"
        _fig.savefig(_out, bbox_inches="tight")
        saved_paths.append(str(_out))
        _panels[f"{_var} [{_stem}]"] = _fig
    mo.accordion(_panels)
    return


@app.cell
def driver_prof(
    DEN_LABEL,
    NUM_LABEL,
    OUT_DIR,
    PROF_MANIFEST,
    PROF_YLABEL,
    compare_profile_reproj,
    den_prof,
    mo,
    num_prof,
):
    # === Profile comparisons: NUM vs DEN, <y> vs x (incl + SD). One PDF per (observable, kind).
    prof_panels = {}
    prof_saved = []
    for _var, _x, _xl in PROF_MANIFEST:
        for _kind in ("incl", "sd"):
            _tag = "incl" if _kind == "incl" else "SD"
            _fig = compare_profile_reproj(
                _var,
                _x,
                _kind,
                num_prof,
                den_prof,
                xlabel=_xl,
                title=f"{PROF_YLABEL[_var]} ({_tag}) vs {_xl}  --  {NUM_LABEL} vs {DEN_LABEL}",
            )
            _out = OUT_DIR / f"fig_cmp_prof_{_var}_vs_{_x}_{_kind}.pdf"
            _fig.savefig(_out, bbox_inches="tight")
            prof_saved.append(str(_out))
            prof_panels[f"{PROF_YLABEL[_var]} ({_tag}) vs {_xl}"] = _fig
    mo.accordion(prof_panels)
    return


@app.cell
def headline_grid(
    CENTER_JPT,
    DEN_LABEL,
    EDGE_MODE,
    NUM_LABEL,
    OUT_DIR,
    PROF_YLABEL,
    RATIO_LABEL,
    den_prof,
    load_snap,
    np,
    num_prof,
    plt,
    prof_stem,
):
    # === HEADLINE profile grid: <lambda> vs R_g for the three kappa=1 angularities
    # (incl=blue, SD=red), at CENTER_JPT. Markers: filled square = DEN, open circle = NUM. ===
    def make_prof_grid_cmp(
        jpt, x_var="sd_dR", vars3=("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2")
    ):
        _n = len(vars3)
        fig, axs = plt.subplots(
            3,
            _n,
            figsize=(4.4 * _n, 6.6),
            sharex="col",
            gridspec_kw=dict(height_ratios=[3, 1, 1], hspace=0.07, wspace=0.27),
            squeeze=False,
        )
        for _k, _var in enumerate(vars3):
            _amain, _ari, _ars = axs[0, _k], axs[1, _k], axs[2, _k]
            for _kind, _color, _ar in (("incl", "C0", _ari), ("sd", "C3", _ars)):
                _stem = prof_stem(x_var, _kind)
                _ref = load_snap(EDGE_MODE, _var, _stem, jpt)
                _x, _xw = _ref["bin_center"].numpy(), _ref["half_bin_width"].numpy()
                _yden, _eden = den_prof[(_var, x_var, _kind, jpt)]
                _ynum, _enum = num_prof[(_var, x_var, _kind, jpt)]
                _amain.errorbar(
                    _x,
                    _yden,
                    xerr=_xw,
                    yerr=_eden,
                    fmt="s",
                    ms=4,
                    lw=1,
                    color=_color,
                    label=DEN_LABEL,
                )
                _amain.errorbar(
                    _x,
                    _ynum,
                    xerr=_xw,
                    yerr=_enum,
                    fmt="o",
                    ms=4,
                    lw=1,
                    color=_color,
                    mfc="none",
                    label=NUM_LABEL,
                )
                with np.errstate(divide="ignore", invalid="ignore"):
                    _r = _ynum / _yden
                    _re = np.abs(_r) * np.sqrt((_enum / _ynum) ** 2 + (_eden / _yden) ** 2)
                _ar.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color=_color, mfc="none")
                _ar.axhline(1.0, color="k", lw=0.8, ls="--")
                _ar.set_ylim(0.9, 1.1)
            _amain.set_title(PROF_YLABEL[_var], fontsize=13)
            _ars.set_xlabel(r"$R_g$")
            if _k == 0:
                _amain.set_ylabel(r"$\langle\lambda^{\kappa=1}_{\beta}\rangle$")
                _ari.set_ylabel(f"{RATIO_LABEL}\n(incl)", fontsize=9)
                _ars.set_ylabel(f"{RATIO_LABEL}\n(SD)", fontsize=9)
                _amain.legend(fontsize=7, frameon=False)
        fig.suptitle(
            rf"$\langle\lambda\rangle$ vs $R_g$, jet $p_T$ bin {jpt}  --  {NUM_LABEL} vs {DEN_LABEL}",
            fontsize=12,
        )
        fig.tight_layout()
        return fig


    fig_prof_grid_cmp = make_prof_grid_cmp(CENTER_JPT)
    fig_prof_grid_cmp.savefig(OUT_DIR / "fig_cmp_prof_grid.pdf", bbox_inches="tight")
    fig_prof_grid_cmp
    return


if __name__ == "__main__":
    app.run()
