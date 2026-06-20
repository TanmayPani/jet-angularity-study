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
    # Comparison config: no-p_T^D ("angularities_noptd") vs with-p_T^D ("angularities").
    # Snapshots are the unfolded-DATA result (sysvar "nominal"); binning is identical
    # across modes (same angularities observables), only the unfolding weights differ.
    HIST_ROOT = Path("outputs/histograms")
    OUT_DIR = Path("outputs/hp2026_poster/ptd_comparison")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    MODE_DEN = "angularities"  # denominator: with p_T^D
    MODE_NUM = "angularities_noptd"  # numerator:   without p_T^D
    SYSVAR = "nominal"  # unfolded-data result
    JPT_BINS = (0, 1, 2, 3)

    # First-class no-p_T^D iteration for this notebook. The no-ptd angularities
    # ensemble runs away by iter5 (a few replicas blow up); the LIKE_DATA closure
    # chi2 and the denoising study favour an EARLY stop. iter2 is the headline
    # no-ptd result here; the with-ptd reference stays at the poster iter5.
    NOPTD_ITER = 2
    CENTER_JPT = 2  # 20<pT<30 GeV/c -- the slice the colleague flagged
    return (
        CENTER_JPT,
        HIST_ROOT,
        JPT_BINS,
        MODE_DEN,
        MODE_NUM,
        NOPTD_ITER,
        OUT_DIR,
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


    def ratio_arrays(num, den):
        """no-ptd / with-ptd, with error propagation (cf. histograms.ratio_snapshot).
        Empty (zero-count) bins yield nan/inf, which matplotlib silently skips."""
        nb, na = num["bin_count"].numpy(), den["bin_count"].numpy()
        eb, ea = disp_err(num), disp_err(den)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = nb / na
            rerr = np.abs(ratio) * np.sqrt((eb / nb) ** 2 + (ea / na) ** 2)
        return ratio, rerr

    return disp_err, load_snap, ratio_arrays


@app.cell
def plot_helper(
    JPT_BINS,
    MODE_DEN,
    MODE_NUM,
    disp_err,
    load_snap,
    plt,
    ratio_arrays,
):
    def compare_grid(var, stem, *, xlabel=None, title=None, logy=False, ratio_ylim=(0.5, 1.5)):
        """2 rows (overlay + no/with ratio) x len(JPT_BINS) cols for one (var, stem)."""
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
            _den = load_snap(MODE_DEN, var, stem, _jpt)
            _num = load_snap(MODE_NUM, var, stem, _jpt)
            _x = _den["bin_center"].numpy()
            _xw = _den["half_bin_width"].numpy()
            _ax, _axr = axs[0, _j], axs[1, _j]
            _ax.errorbar(
                _x,
                _den["bin_count"].numpy(),
                xerr=_xw,
                yerr=disp_err(_den),
                fmt="s",
                ms=3,
                lw=1,
                color="C0",
                label=r"with $p_T^D$",
            )
            _ax.errorbar(
                _x,
                _num["bin_count"].numpy(),
                xerr=_xw,
                yerr=disp_err(_num),
                fmt="o",
                ms=3,
                lw=1,
                color="C3",
                mfc="none",
                label=r"no $p_T^D$",
            )
            if logy:
                _ax.set_yscale("log")
            _r, _re = ratio_arrays(_num, _den)
            _axr.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color="C3", mfc="none")
            _axr.axhline(1.0, color="k", lw=0.8, ls="--")
            _axr.set_ylim(*ratio_ylim)
            _ax.set_title(f"jet $p_T$ bin {_jpt}", fontsize=9)
            if xlabel:
                _axr.set_xlabel(xlabel)
            if _j == 0:
                _ax.set_ylabel("unfolded (a.u.)")
                _axr.set_ylabel("no/with")
                _ax.legend(fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        fig.tight_layout()
        return fig

    return (compare_grid,)


@app.cell
def demo(compare_grid):
    fig_demo = compare_grid(
        "ch_ang_k1_b1",
        "hist_ang",
        xlabel=r"$\lambda^{1}_{1}$ (inclusive)",
        title=r"ch_ang_k1_b1 inclusive  —  no-$p_T^D$ vs with-$p_T^D$ unfolding",
    )
    fig_demo
    return


@app.cell
def manifest():
    # Poster distribution set: (var dir, snapshot stem, x-axis label).
    # ch_ang_k1_* are the headline angularities; ch_ang_k2_b0 is p_T^D itself (cross-check,
    # since no-ptd does not unfold-inform it); m/sd_* are the jet-shape observables.
    MANIFEST = [
        ("ch_ang_k1_b0.5", "hist_ang", r"$\lambda^{1}_{0.5}$ (incl)"),
        ("ch_ang_k1_b0.5", "hist_sd_ang", r"$\lambda^{1}_{0.5}$ (SD)"),
        ("ch_ang_k1_b1", "hist_ang", r"$\lambda^{1}_{1}$ (incl)"),
        ("ch_ang_k1_b1", "hist_sd_ang", r"$\lambda^{1}_{1}$ (SD)"),
        ("ch_ang_k1_b2", "hist_ang", r"$\lambda^{1}_{2}$ (incl)"),
        ("ch_ang_k1_b2", "hist_sd_ang", r"$\lambda^{1}_{2}$ (SD)"),
        ("ch_ang_k2_b0", "hist_ang", r"$p_T^D$ (incl) [cross-check]"),
        ("ch_ang_k2_b0", "hist_sd_ang", r"$p_T^D$ (SD) [cross-check]"),
        ("m", "hist", r"$M$ [GeV]"),
        ("sd_m", "hist", r"$M_{SD}$ [GeV]"),
        ("sd_dR", "hist", r"$\Delta R_{SD}$"),
        ("sd_symmetry", "hist", r"$z_g$"),
    ]

    # Poster PROFILE set: (var dir, x-axis var, x-axis label). Each profiles BOTH the
    # inclusive angularity (y=var) and its SoftDrop variant (y=sd_<var>) vs the x-var.
    #   * kappa=1 angularities vs R_g (sd_dR)   -> poster fig_prof_grid / fig_prof_*
    #   * p_T^D (ch_ang_k2_b0)   vs z_g (sd_symmetry) -> poster fig_zg_* [cross-check]
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
def driver(MANIFEST, OUT_DIR, compare_grid_reproj, mo, noptd_iter2):
    # === FIRST-CLASS distribution comparisons: no-p_T^D (iter2 re-projection)
    # vs with-p_T^D (iter5 snapshot). Owns the canonical fig_cmp_<var>_<stem>.pdf.
    _panels = {}
    saved_paths = []
    for _var, _stem, _xl in MANIFEST:
        _fig = compare_grid_reproj(
            _var,
            _stem,
            noptd_iter2,
            num_label=rf"no $p_T^D$",
            xlabel=_xl,
            title=f"{_var}  [{_stem}]  --  no-$p_T^D$ vs with-$p_T^D$",
        )
        _out = OUT_DIR / f"fig_cmp_{_var}_{_stem}.pdf"
        _fig.savefig(_out, bbox_inches="tight")
        saved_paths.append(str(_out))
        _panels[f"{_var} [{_stem}]"] = _fig
    mo.accordion(_panels)

    # ---- old iter5-snapshot version (superseded by the iter2 re-projection) ----
    # _panels = {}
    # saved_paths = []
    # for _var, _stem, _xl in MANIFEST:
    #     _fig = compare_grid(
    #         _var,
    #         _stem,
    #         xlabel=_xl,
    #         title=f"{_var}  [{_stem}]  —  no-$p_T^D$ vs with-$p_T^D$",
    #     )
    #     _out = OUT_DIR / f"fig_cmp_{_var}_{_stem}.pdf"
    #     _fig.savefig(_out, bbox_inches="tight")
    #     saved_paths.append(str(_out))
    #     _panels[f"{_var} [{_stem}]"] = _fig
    # mo.accordion(_panels)
    return


@app.cell
def reproj_inputs(MANIFEST, Path, np, pa):
    # --- Re-projection inputs (for trying alternative iterations / replica subsets) ---
    # The saved .pt snapshots are the iter-5 mean over all 20 replicas. The no-ptd ensemble
    # has a few runaway replicas at iter 5, so we also re-derive the no-ptd distributions by
    # re-binning the gen-level observables (gen-matches + misses) weighted by the per-replica
    # unfolded weights arr_{2*iter} in w_unfolding.npz -- exactly as histograms.py does.
    _FEAT = Path("datasets/STAR_pp200GeV_production_2012/features")
    GEN_EMB = _FEAT / "angularities" / "embedding" / "nominal"  # shared observables
    NOPTD_WUNF = _FEAT / "angularities_noptd" / "embedding" / "nominal" / "w_unfolding.npz"
    JPT_EDGES = (10.0, 15.0, 20.0, 30.0, 60.0)


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
    return NOPTD_WUNF, gen_obs, jpt_masks, obs_col


@app.cell
def reproj_funcs(
    JPT_BINS,
    MANIFEST,
    MODE_DEN,
    NOPTD_WUNF,
    PROF_MANIFEST,
    gen_obs,
    jpt_masks,
    load_snap,
    np,
    obs_col,
):
    def _edges_from_snapshot(var, stem, jpt):
        """Bin edges recovered from the with-ptd snapshot so the re-binning matches it exactly."""
        _s = load_snap(MODE_DEN, var, stem, jpt)
        _bc = _s["bin_center"].numpy()
        _hw = _s["half_bin_width"].numpy()
        return np.append(_bc - _hw, _bc[-1] + _hw[-1])


    def perreplica_hist(weights2d, var, stem, jpt):
        """(R, nbins) per-replica unit-area histogram of the gen observable, weighted by weights2d."""
        _e = _edges_from_snapshot(var, stem, jpt)
        _m = jpt_masks[jpt]
        _v = gen_obs[obs_col(var, stem)][_m]
        _W = weights2d[:, _m]
        return np.stack(
            [np.histogram(_v, bins=_e, weights=_W[r], density=True)[0] for r in range(_W.shape[0])]
        )


    def load_noptd_weights(iteration):
        """Per-jet per-replica no-ptd unfolded weights at OmniFold `iteration` (arr_{2*iter})."""
        return np.load(NOPTD_WUNF)[f"arr_{2 * iteration}"]


    def noptd_collapse(iteration, *, drop_n=0, weights=None):
        """Collapse the no-ptd per-replica histograms to {(var,stem,jpt): (mean, std)} over all
        MANIFEST slices. `drop_n` > 0 drops the drop_n replicas with the largest peak-bin across
        all slices (the runaway members that inflate the iter-5 spread). Returns (dict, kept_idx)."""
        _w = load_noptd_weights(iteration) if weights is None else weights
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
    # Profile re-projection: <y> vs x in x-bins, weighted by the per-replica no-ptd
    # unfolded weights -- the profile analogue of perreplica_hist/noptd_collapse,
    # matching histograms.profile_perpt (Profile.create: <y>_bin = sum(w*y)/sum(w)).
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


    def noptd_profile_collapse(iteration, *, drop_n=0, weights=None):
        """Collapse no-ptd per-replica PROFILES to {(var,x,kind,jpt): (mean, std)} over all
        PROF_MANIFEST slices (incl + sd). Edges recovered from the with-ptd profile snapshot
        so binning matches exactly. Returns (dict, kept_idx)."""
        _w = load_noptd_weights(iteration) if weights is None else weights
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

    return (
        load_noptd_weights,
        noptd_collapse,
        noptd_profile_collapse,
        prof_stem,
    )


@app.cell
def reproj_compute(
    NOPTD_ITER,
    load_noptd_weights,
    mo,
    noptd_collapse,
    noptd_profile_collapse,
):
    # Heavy cell: re-projects the no-ptd unfolded weights at the candidate iterations.
    #   * iter5 with the 5 runaway replicas trimmed (alternative candidate, arr_10).
    #   * iter2 = the FIRST-CLASS no-ptd result -- distributions AND profiles, built
    #     from a single load of arr_4 (the iter2 gen weights) to bound memory.
    noptd_iter5_trim, trim_keep = noptd_collapse(5, drop_n=5)

    _w_it2 = load_noptd_weights(NOPTD_ITER)
    noptd_iter2, _ = noptd_collapse(NOPTD_ITER, weights=_w_it2)
    noptd_iter2_prof, _ = noptd_profile_collapse(NOPTD_ITER, weights=_w_it2)
    del _w_it2

    mo.md(
        f"Re-projected no-ptd candidates ready. First-class **iter{NOPTD_ITER}**: "
        f"`{len(noptd_iter2)}` dist slices + `{len(noptd_iter2_prof)}` profile slices. "
        f"iter5-trim kept replicas: `{trim_keep.tolist()}`"
    )
    return noptd_iter2, noptd_iter2_prof, noptd_iter5_trim


@app.cell
def compare_reproj_fn(JPT_BINS, MODE_DEN, disp_err, load_snap, np, plt):
    def compare_grid_reproj(var, stem, num_results, *, num_label, xlabel=None, title=None):
        """Like compare_grid, but the numerator (no-ptd) comes from a precomputed
        {(var,stem,jpt): (mean, std)} dict (a re-projected candidate) rather than the .pt snapshot.
        The denominator/reference is always the with-ptd iter-5 snapshot."""
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
            _den = load_snap(MODE_DEN, var, stem, _jpt)
            _x = _den["bin_center"].numpy()
            _xw = _den["half_bin_width"].numpy()
            _yden, _eden = _den["bin_count"].numpy(), disp_err(_den)
            _ynum, _enum = num_results[(var, stem, _jpt)]
            _ax, _axr = axs[0, _j], axs[1, _j]
            _ax.errorbar(
                _x,
                _yden,
                xerr=_xw,
                yerr=_eden,
                fmt="s",
                ms=3,
                lw=1,
                color="C0",
                label=r"with $p_T^D$",
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
                label=num_label,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                _r = _ynum / _yden
                _re = np.abs(_r) * np.sqrt((_enum / _ynum) ** 2 + (_eden / _yden) ** 2)
            _axr.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color="C3", mfc="none")
            _axr.axhline(1.0, color="k", lw=0.8, ls="--")
            _axr.set_ylim(0.5, 1.5)
            _ax.set_title(f"jet $p_T$ bin {_jpt}", fontsize=9)
            if xlabel:
                _axr.set_xlabel(xlabel)
            if _j == 0:
                _ax.set_ylabel("unfolded (a.u.)")
                _axr.set_ylabel("no/with")
                _ax.legend(fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        fig.tight_layout()
        return fig

    return (compare_grid_reproj,)


@app.cell
def plot_iter5_trim(compare_grid_reproj, noptd_iter5_trim):
    # Candidate 1: iter 5 with the 5 runaway replicas trimmed (option 1).
    fig_iter5_trim = compare_grid_reproj(
        "ch_ang_k1_b1",
        "hist_ang",
        noptd_iter5_trim,
        num_label=r"no $p_T^D$",
        xlabel=r"$\lambda^{1}_{1}$ (inclusive)",
        title=r"Option 1 -- iter5 rogue-trimmed no-$p_T^D$ vs with-$p_T^D$ (it5 ref)",
    )
    fig_iter5_trim
    return


@app.cell
def plot_iter2(compare_grid_reproj, noptd_iter2):
    # Candidate 2: iter 2, all 20 replicas (option 2 -- best agreement + lowest noise).
    fig_iter2 = compare_grid_reproj(
        "ch_ang_k1_b1",
        "hist_ang",
        noptd_iter2,
        num_label=r"no $p_T^D$ (it2)",
        xlabel=r"$\lambda^{1}_{1}$ (inclusive)",
        title=r"Option 2 -- iter2 no-$p_T^D$ vs with-$p_T^D$ (it5 ref)",
    )
    fig_iter2
    return


@app.cell
def _(
    JPT_BINS,
    MODE_DEN,
    PROF_YLABEL,
    disp_err,
    load_snap,
    np,
    plt,
    prof_stem,
):
    def compare_profile_reproj(var, x_var, kind, num_results, *, num_label, xlabel=None, title=None):
        """Profile analogue of compare_grid_reproj: <y> vs x_var (kind in {'incl','sd'}).
        Denominator = with-ptd iter5 profile snapshot; numerator = re-projected no-ptd dict
        {(var,x,kind,jpt): (mean, std)}. 2 rows (overlay + no/with ratio) x len(JPT_BINS) cols."""
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
            _den = load_snap(MODE_DEN, var, _stem, _jpt)
            _x = _den["bin_center"].numpy()
            _xw = _den["half_bin_width"].numpy()
            _yden, _eden = _den["bin_count"].numpy(), disp_err(_den)
            _ynum, _enum = num_results[(var, x_var, kind, _jpt)]
            _ax, _axr = axs[0, _j], axs[1, _j]
            _ax.errorbar(
                _x,
                _yden,
                xerr=_xw,
                yerr=_eden,
                fmt="s",
                ms=3,
                lw=1,
                color="C0",
                label=r"with $p_T^D$ (it5)",
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
                label=num_label,
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                _r = _ynum / _yden
                _re = np.abs(_r) * np.sqrt((_enum / _ynum) ** 2 + (_eden / _yden) ** 2)
            _axr.errorbar(_x, _r, xerr=_xw, yerr=_re, fmt="o", ms=3, lw=1, color="C3", mfc="none")
            _axr.axhline(1.0, color="k", lw=0.8, ls="--")
            _axr.set_ylim(0.9, 1.1)
            _ax.set_title(f"jet $p_T$ bin {_jpt}", fontsize=9)
            if xlabel:
                _axr.set_xlabel(xlabel)
            if _j == 0:
                _tag = "incl" if kind == "incl" else "SD"
                _ax.set_ylabel(f"{PROF_YLABEL[var]}  ({_tag})")
                _axr.set_ylabel("no/with")
                _ax.legend(fontsize=8, frameon=False)
        if title:
            fig.suptitle(title, y=0.99, fontsize=11)
        fig.tight_layout()
        return fig

    return (compare_profile_reproj,)


@app.cell
def _(
    OUT_DIR,
    PROF_MANIFEST,
    PROF_YLABEL,
    compare_profile_reproj,
    mo,
    noptd_iter2_prof,
):
    # === FIRST-CLASS profile comparisons: no-p_T^D (iter2) vs with-p_T^D (iter5) ===
    # Mirrors the poster fig_prof_{0.5,1,2} (<lambda> vs R_g) + fig_zg_* (<p_T^D> vs z_g),
    # incl + SD. Saves one PDF per (observable, kind).
    prof_panels = {}
    prof_saved = []
    for _var, _x, _xl in PROF_MANIFEST:
        for _kind in ("incl", "sd"):
            _tag = "incl" if _kind == "incl" else "SD"
            _fig = compare_profile_reproj(
                _var,
                _x,
                _kind,
                noptd_iter2_prof,
                num_label=rf"no $p_T^D$",
                xlabel=_xl,
                title=f"{PROF_YLABEL[_var]} ({_tag}) vs {_xl}  --  no-$p_T^D$",
            )
            _out = OUT_DIR / f"fig_cmp_prof_{_var}_vs_{_x}_{_kind}.pdf"
            _fig.savefig(_out, bbox_inches="tight")
            prof_saved.append(str(_out))
            prof_panels[f"{PROF_YLABEL[_var]} ({_tag}) vs {_xl}"] = _fig
    mo.accordion(prof_panels)
    return


@app.cell
def _(
    CENTER_JPT,
    MODE_DEN,
    OUT_DIR,
    PROF_YLABEL,
    disp_err,
    load_snap,
    noptd_iter2_prof,
    np,
    plt,
    prof_stem,
):
    # === HEADLINE profile grid: the direct no-/with-p_T^D analogue of the poster
    # fig_prof_grid the colleague flagged -- <lambda> vs R_g for the three kappa=1
    # angularities (incl=blue, SD=red), at CENTER_JPT (20<pT<30 GeV/c). Markers:
    # filled square = with-p_T^D (it5), open circle = no-p_T^D (it2). ===
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
                _den = load_snap(MODE_DEN, _var, _stem, jpt)
                _x, _xw = _den["bin_center"].numpy(), _den["half_bin_width"].numpy()
                _yden, _eden = _den["bin_count"].numpy(), disp_err(_den)
                _ynum, _enum = noptd_iter2_prof[(_var, x_var, _kind, jpt)]
                _lab = "incl" if _kind == "incl" else "SD"
                _amain.errorbar(
                    _x,
                    _yden,
                    xerr=_xw,
                    yerr=_eden,
                    fmt="s",
                    ms=4,
                    lw=1,
                    color=_color,
                    label=rf"with $p_T^D$",
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
                    label=rf"no $p_T^D$",
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
                _ari.set_ylabel("no/with\n(incl)", fontsize=9)
                _ars.set_ylabel("no/with\n(SD)", fontsize=9)
                _amain.legend(fontsize=7, frameon=False)
        fig.suptitle(
            rf"$\langle\lambda\rangle$ vs $R_g$, jet $p_T$ bin {jpt} (20$<p_T<$30 GeV)",
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
