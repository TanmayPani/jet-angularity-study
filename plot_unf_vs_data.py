import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")

with app.setup:
    # ---------------------------------------------------------------------------
    # plot_unf_vs_data.py  --  "is the unfolded result just the prior?" defense.
    #
    # The poster (fig_prof_grid.pdf) shows the UNFOLDED data <lambda>(Rg) sitting
    # essentially on top of the PYTHIA6 prediction (the unfolding prior), which
    # invites the prior-domination worry. This notebook puts the answer on one
    # page by drawing BOTH levels side by side, per kappa=1 angularity and jet-pT
    # slice:
    #
    #   * DETECTOR level  -- RAW data (reco, data.arrow, weight=1) vs PYTHIA6 reco
    #                        (reco-matches (+) fakes, cross-section weighted).
    #   * PARTICLE level  -- UNFOLDED data (gen jets weighted by w_unfolding) vs
    #                        PYTHIA6 gen (the prior, gen `weight` column).
    #
    # If data already matches p6 at RECO level (top ratio ~ 1) then unfolded ~ p6
    # at GEN level (bottom ratio ~ 1) is the faithful propagation of genuine
    # detector-level agreement through a near-diagonal response -- inherited
    # physics, NOT a prior the unfolding failed to leave. (The complementary proof
    # that the unfolder CAN leave the prior is the reweighted closure in
    # plot_closure.py.)
    #
    # Config-driven like histograms.py: observables come from the `angularities`
    # arrows (obs_feature_mode), the WEIGHTS + output dir are keyed on the run
    # `feature_mode` from config.json (currently angularities_noptd). Data is data
    # -- detector level is identical across feature modes (the noptd data.arrow is
    # a symlink to the angularities one).
    # ---------------------------------------------------------------------------
    import os
    import json

    import numpy as np
    import pyarrow as pa
    import torch
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
    from thoda import Profile, Snapshot
    from config import load_config, config_path_from_argv

    plt.style.use("default")
    plt.rcParams.update(
        {
            "mathtext.fontset": "stix",
            "font.family": "STIXGeneral",
            "font.size": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    _cfg_setup = load_config(config_path_from_argv())
    # Run mode = the unfolding whose weights/output we compare (e.g. angularities_noptd).
    # feature_mode = "angularities"
    feature_mode = _cfg_setup["feature_mode"]

    # Observable source: the angularity scalars + softdrop columns live ONLY in
    # the `angularities` arrows; noptd/bin_counts reuse them row-for-row.
    obs_feature_mode = "angularities"
    dataset_root = str(_cfg_setup.dataset_root)
    replica_reduce = _cfg_setup.get("replica_reduce", "median")
    replica_trim_frac = float(_cfg_setup.get("replica_trim_frac", 0.1))

    # This is the nominal real-data unfolding (SysVar.NONE); the comparison has no
    # other variation. (The closure flavours are handled by plot_closure.py.)
    sys_var = SysVar.NONE

    # kappa = 1 angularities -- the family the colleague flagged (vs Rg = sd_dR).
    kappa1 = ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2")
    x_var = "sd_dR"  # groomed radius R_g

    var_prof_ylabel = {
        "ch_ang_k1_b0.5": r"$\langle\lambda^{\kappa=1}_{\beta=0.5}\rangle$",
        "ch_ang_k1_b1": r"$\langle\lambda^{\kappa=1}_{\beta=1}\rangle$",
        "ch_ang_k1_b2": r"$\langle\lambda^{\kappa=1}_{\beta=2}\rangle$",
    }
    x_label = r"$R_g$"

    jpt_bins = get_jet_pt_bins(sys_var)
    # 0-based jet-pT slice index: 0=[10,15) 1=[15,20) 2=[20,30) 3=[30,60).
    CENTER_JPT = 2  # 20 < pT < 30 GeV/c -- the panel cited in the concern.

    # Iteration to read for the unfolded (gen) weights. The LIKE_DATA closure
    # favours iter 1 for the noptd run (see project_noptd_iteration_denoising);
    # get_unfolding_iter(NONE, .) returns nom_iter, so set it explicitly here and
    # keep it editable. arr_{2*iter} = gen weights.
    UNF_ITER = 2

    OUT_DIR = os.path.join("outputs", "unf_vs_data", feature_mode)


@app.function
def pt_label(ijpt):
    return rf"${jpt_bins[ijpt]:.0f} < p_{{\rm T,jet}} < {jpt_bins[ijpt + 1]:.0f}$ GeV/$c$"


@app.function
def read_arrow(path, buffers):
    """memory-map an arrow file, keep the buffer alive in `buffers`, return table."""
    buffers.append(pa.memory_map(path, "rb"))
    return pa.ipc.open_file(buffers[-1]).read_all()


@app.function
def collapse_replicas(x, dim=0):
    """Robust central value over the per-replica ensemble axis (Patch G). The
    replica *spread* keeps its plain std -- it is the uncertainty band."""
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
def make_prof(table, ycol, xcol, edges, mask, weights):
    """1-D Profile of <ycol>(xcol) on one jet-pT slice. `weights` is None,
    a 1-D tensor (data/MC), or a (num_replicas, N) tensor (unfolded ensemble)."""
    _m = np.asarray(mask)
    _x = table[xcol].to_numpy()[_m]
    _y = table[ycol].to_numpy()[_m]
    if weights is None:
        _w = None
    else:
        _w = weights[..., torch.as_tensor(_m)]
    # Single-axis profile -> snapshot directly. (.project() would sum away the
    # only axis; project is only for picking one axis out of a multi-D profile.)
    _prof, _ = Profile.create([_x], bins=[edges], y=_y, weights=_w, axis_names=(xcol,))
    return _prof.snapshot()


@app.cell
def _(ERR_MODE):
    def prof_points(snap, batched=False):
        """Central <lambda>(Rg) + displayed error from a ProfileSnapshot.

        ERR_MODE == "mean": statistical error ON THE MEAN = sqrt(variances /
          effective_counts) -- the correct uncertainty for a profile of means (with
          millions of jets it is ~0.5-1.5%); this is what shows whether a data/MC
          deviation is statistically resolved. THIS IS THE DEFAULT and the honest
          error for judging "does the unfolded result actually sit on pythia6?".
        ERR_MODE == "rms": sqrt(variances) = the per-bin spread of lambda (~30%), the
          histograms.py/poster convention (NOT the error on <lambda>). Kept so the
          poster look can be reproduced.

        For the batched (ensemble) unfolded snapshot the per-replica error is
        collapsed and the per-replica spread is added in quadrature."""

        def _err(variances, eff):
            if ERR_MODE == "mean":
                return (variances / eff.clamp_min(1e-9)).clamp_min(0).sqrt()
            return variances.clamp_min(0).sqrt()

        centers = snap.bin_centers[1:-1]
        if not batched:
            central = snap.values[1:-1]
            err = _err(snap.variances[1:-1], snap.effective_counts[1:-1])
        else:
            central = collapse_replicas(snap.values[:, 1:-1])
            stat = collapse_replicas(_err(snap.variances[:, 1:-1], snap.effective_counts[:, 1:-1]))
            spread = snap.values[:, 1:-1].std(0)
            err = (stat.square() + spread.square()).sqrt()
        return (
            centers.numpy(),
            central.nan_to_num_(nan=0).numpy(),
            err.nan_to_num_(nan=0).numpy(),
        )

    return (prof_points,)


@app.function
def ratio_points(num, num_err, den, den_err):
    """num/den with relative errors added in quadrature (ratio_snapshot
    convention). Zero-protected."""
    den_safe = np.where(den == 0, np.nan, den)
    ratio = num / den_safe
    rel2 = (
        np.where(num == 0, 0.0, (num_err / np.where(num == 0, 1, num)) ** 2)
        + (den_err / den_safe) ** 2
    )
    return np.nan_to_num(ratio), np.nan_to_num(ratio * np.sqrt(rel2))


@app.cell
def _():
    # x-edges for the R_g profile: uniform linspace over the flat sd_dR range,
    # matching histograms.py prof_bins_perpt (n=10), so this is directly
    # comparable to the poster's vs-Rg panels.
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as _f:
        _flat = json.load(_f)
    _lo, _hi = float(_flat[x_var][0]), float(_flat[x_var][-1])
    x_edges = np.linspace(_lo, _hi, 11).tolist()
    print(f"R_g profile edges: {_lo:.3f} .. {_hi:.3f} (10 bins)")
    return (x_edges,)


@app.cell
def _():
    # ---- load the three input tables ----------------------------------------
    _buffers = []  # keep memory-maps alive for the session

    # RAW DATA (detector level): data.arrow, weight == 1.
    _data_path = os.path.join(dataset_root, "features", feature_mode, "data.arrow")
    data_table = read_arrow(_data_path, _buffers)

    # GEN observables (particle level): gen-matches (+) misses from the
    # angularities embedding. Row order matches w_unfolding.
    _emb = os.path.join(dataset_root, "features", obs_feature_mode, "embedding", str(sys_var))
    gen_table = pa.concat_tables(
        (
            read_arrow(os.path.join(_emb, "gen-matches.arrow"), _buffers),
            read_arrow(os.path.join(_emb, "misses.arrow"), _buffers),
        )
    )

    # PYTHIA6 reco (detector level): reco-matches (+) fakes = the full p6 reco
    # spectrum, cross-section weighted by the embedding `weight` column.
    reco_table = pa.concat_tables(
        (
            read_arrow(os.path.join(_emb, "reco-matches.arrow"), _buffers),
            read_arrow(os.path.join(_emb, "fakes.arrow"), _buffers),
        )
    )

    print(f"data(reco)={data_table.num_rows}  gen={gen_table.num_rows}  p6reco={reco_table.num_rows}")
    return data_table, gen_table, reco_table


@app.cell
def _(gen_table):
    # ---- weights -------------------------------------------------------------
    # Unfolded (ensemble): arr_{2*UNF_ITER} from the run-mode w_unfolding.npz.
    _unf_dir = os.path.join(dataset_root, "features", feature_mode, "embedding", str(sys_var))
    _wfile = os.path.join(_unf_dir, "w_unfolding.npz")
    print(f"Unfolded weights: {_wfile}  (iter {UNF_ITER})")
    _wd = np.load(_wfile)
    unf_weights = torch.as_tensor(_wd[f"arr_{2 * UNF_ITER}"], dtype=torch.float32)
    assert unf_weights.shape[1] == gen_table.num_rows, (
        f"w_unfolding cols {unf_weights.shape[1]} != gen rows {gen_table.num_rows}"
    )

    # PYTHIA6 gen weights = the prior (gen `weight` column, cross-section).
    p6_gen_weights = torch.as_tensor(gen_table["weight"].to_numpy(), dtype=torch.float32)
    return p6_gen_weights, unf_weights


@app.cell
def _(data_table, gen_table, reco_table):
    # ---- per jet-pT slice masks ---------------------------------------------
    def _masks(tbl):
        pt = np.asarray(tbl["pt"].to_numpy())
        return [(pt >= jpt_bins[i]) & (pt < jpt_bins[i + 1]) for i in range(len(jpt_bins) - 1)]


    data_masks = _masks(data_table)
    gen_masks = _masks(gen_table)
    reco_masks = _masks(reco_table)
    p6reco_weights = torch.as_tensor(reco_table["weight"].to_numpy(), dtype=torch.float32)
    return data_masks, gen_masks, p6reco_weights, reco_masks


@app.cell
def _(
    data_masks,
    data_table,
    gen_masks,
    gen_table,
    p6_gen_weights,
    p6reco_weights,
    prof_points,
    reco_masks,
    reco_table,
    unf_weights,
    x_edges,
):
    # ---- build all profiles --------------------------------------------------
    # For each (var, jet-pT slice) compute four <lambda>(Rg) profiles:
    #   det:  data (reco, w=1)          vs  p6 reco (cross-section weighted)
    #   part: unfolded (gen, ensemble)  vs  p6 gen  (prior)
    # `incl` uses the ungroomed angularity ch_ang_*, `sd` the groomed sd_ch_ang_*;
    # both binned in R_g (sd_dR). Mirrors the poster's incl/SD split.
    profs = {}  # (level, mode, var, ijpt) -> (centers, central, err, batched?)
    _njpt = len(jpt_bins) - 1
    for _var in kappa1:
        _sdvar = f"sd_{_var}"
        for _ijpt in range(_njpt):
            # detector level
            for _mode, _ycol in (("incl", _var), ("sd", _sdvar)):
                _d = make_prof(data_table, _ycol, x_var, x_edges, data_masks[_ijpt], None)
                profs[("det_data", _mode, _var, _ijpt)] = (*prof_points(_d), False)
                _r = make_prof(reco_table, _ycol, x_var, x_edges, reco_masks[_ijpt], p6reco_weights)
                profs[("det_p6", _mode, _var, _ijpt)] = (*prof_points(_r), False)
            # particle level
            for _mode, _ycol in (("incl", _var), ("sd", _sdvar)):
                _u = make_prof(gen_table, _ycol, x_var, x_edges, gen_masks[_ijpt], unf_weights)
                profs[("part_unf", _mode, _var, _ijpt)] = (*prof_points(_u, batched=True), True)
                _g = make_prof(gen_table, _ycol, x_var, x_edges, gen_masks[_ijpt], p6_gen_weights)
                profs[("part_p6", _mode, _var, _ijpt)] = (*prof_points(_g), False)
        print(f"profiles done: {_var}")
    return (profs,)


@app.cell
def _(profs):
    # ---- drawing helper ------------------------------------------------------
    # mode -> (color, marker) : inclusive = blue ^, softdrop = red o (poster scheme)
    _mode_style = {"incl": ("blue", "^"), "sd": ("red", "o")}


    def draw_level(ax_main, ax_ratio, var, ijpt, data_key, mc_key, mc_label):
        """Draw one level (detector or particle). `data_key`/`mc_key` are the
        ('<lvl>_<who>') prefixes into `profs`. incl (blue) + SD (red)."""
        for _mode in ("incl", "sd"):
            _clr, _mk = _mode_style[_mode]
            _xc, _yc, _ye, _ = profs[(data_key, _mode, var, ijpt)]
            _xm, _ym, _yme, _ = profs[(mc_key, _mode, var, ijpt)]
            # measurement points (data / unfolded)
            ax_main.errorbar(
                _xc,
                _yc,
                yerr=_ye,
                color=_clr,
                marker=_mk,
                linestyle="none",
                markersize=5,
                capsize=2,
                label=None,
            )
            # pythia6 curve (dotted, matches the poster's pythia6 linestyle)
            ax_main.plot(_xm, _ym, color=_clr, linestyle=":", linewidth=2)
            # ratio (measurement / p6)
            _r, _re = ratio_points(_yc, _ye, _ym, _yme)
            ax_ratio.errorbar(
                _xc,
                _r,
                yerr=_re,
                color=_clr,
                marker=_mk,
                linestyle="none",
                markersize=4,
                capsize=2,
            )
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.4)
        ax_ratio.set_ylim(0.9, 1.1)

    return (draw_level,)


@app.cell
def _(draw_level):
    # ---- HEADLINE figure: 3 vars (cols) x 4 rows --------------------------------
    # rows: detector main / detector ratio / particle main / particle ratio.
    # Top half = "does data already match p6 at RECO?"; bottom half = "is unfolded
    # ~ p6 at GEN just propagation of that?". Built at CENTER_JPT (20-30 GeV).
    _ijpt = CENTER_JPT
    _n = len(kappa1)
    fig_headline = plt.figure(figsize=(5 * _n, 11))
    _axs = fig_headline.subplots(
        4,
        _n,
        height_ratios=[3, 1, 3, 1],
        sharex="col",
        squeeze=False,
        gridspec_kw=dict(hspace=0.0, wspace=0.28),
    )
    for _k, _var in enumerate(kappa1):
        _ax_dm, _ax_dr, _ax_pm, _ax_pr = _axs[0, _k], _axs[1, _k], _axs[2, _k], _axs[3, _k]
        draw_level(_ax_dm, _ax_dr, _var, _ijpt, "det_data", "det_p6", "pythia6")
        draw_level(_ax_pm, _ax_pr, _var, _ijpt, "part_unf", "part_p6", "pythia6")
        _ax_pr.set_xlabel(x_label)
        _ax_dm.set_title(var_prof_ylabel[_var], fontsize=16)
        if _k == 0:
            _ax_dm.set_ylabel("detector level\n(raw data)", fontsize=14)
            _ax_dr.set_ylabel(r"$\frac{\rm data}{\rm p6\,reco}$", fontsize=13)
            _ax_pm.set_ylabel("particle level\n(unfolded)", fontsize=14)
            _ax_pr.set_ylabel(r"$\frac{\rm unf}{\rm p6\,gen}$", fontsize=13)
    _axs[0, 0].text(
        0.04,
        0.96,
        "STAR Preliminary",
        transform=_axs[0, 0].transAxes,
        color="red",
        fontweight="bold",
        va="top",
        fontsize=14,
    )
    _axs[0, 0].text(
        0.04,
        0.86,
        pt_label(_ijpt) + "\n" + r"$p+p\ \sqrt{s}=200$ GeV",
        transform=_axs[0, 0].transAxes,
        va="top",
        fontsize=12,
    )
    _axs[0, _n - 1].legend(handles=row_legend("raw data", "reco"), loc="upper left", fontsize=11)
    _axs[2, _n - 1].legend(handles=row_legend("unfolded", "gen"), loc="upper left", fontsize=11)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig_headline.savefig(
        os.path.join(OUT_DIR, f"fig_unf_vs_data_grid_jpt{_ijpt}.pdf"), bbox_inches="tight"
    )
    print("saved:", os.path.join(OUT_DIR, f"fig_unf_vs_data_grid_jpt{_ijpt}.pdf"))
    fig_headline
    return


@app.cell
def _(draw_level):
    # ---- per-variable, all-pT figures (one PDF per kappa=1 angularity) -------
    # Each: 4 rows (det main/ratio, part main/ratio) x 4 jet-pT columns. Lets you
    # check the inheritance holds across the whole pT range, not just 20-30 GeV.
    _njpt = len(jpt_bins) - 1
    saved_singles = []
    for _var in kappa1:
        _fig = plt.figure(figsize=(4.2 * _njpt, 11))
        _axs2 = _fig.subplots(
            4,
            _njpt,
            height_ratios=[3, 1, 3, 1],
            sharex="col",
            squeeze=False,
            gridspec_kw=dict(hspace=0.0, wspace=0.30),
        )
        for _j in range(_njpt):
            _adm, _adr, _apm, _apr = _axs2[0, _j], _axs2[1, _j], _axs2[2, _j], _axs2[3, _j]
            draw_level(_adm, _adr, _var, _j, "det_data", "det_p6", "pythia6")
            draw_level(_apm, _apr, _var, _j, "part_unf", "part_p6", "pythia6")
            _apr.set_xlabel(x_label)
            _adm.set_title(pt_label(_j), fontsize=12)
        _axs2[0, 0].set_ylabel("data (reco)\n" + var_prof_ylabel[_var], fontsize=13)
        _axs2[1, 0].set_ylabel(r"$\frac{\rm data}{\rm p6\,reco}$", fontsize=12)
        _axs2[2, 0].set_ylabel("unfolded\n" + var_prof_ylabel[_var], fontsize=13)
        _axs2[3, 0].set_ylabel(r"$\frac{\rm unf}{\rm p6\,gen}$", fontsize=12)
        _axs2[0, _njpt - 1].legend(
            handles=row_legend("raw data", "reco"), loc="upper left", fontsize=10
        )
        _axs2[2, _njpt - 1].legend(handles=row_legend("unfolded", "gen"), loc="upper left", fontsize=10)
        _path = os.path.join(OUT_DIR, f"fig_unf_vs_data_{_var}.pdf")
        _fig.savefig(_path, bbox_inches="tight")
        saved_singles.append(_path)
    print("saved per-var figures:", *saved_singles, sep="\n  ")
    return


@app.cell
def _():
    # Displayed error convention (see prof_points). "mean" = statistical error on
    # <lambda> (sqrt(var/eff_counts), ~0.5-1.5%) -- the honest uncertainty that shows
    # the data/pythia6 deviation is resolved. "rms" = per-bin spread of lambda (~30%,
    # the poster/histograms.py convention).
    ERR_MODE = "mean"
    return (ERR_MODE,)


@app.function
# Shared row-specific legend builder. The markers/curve mean different things
# per level (detector = raw data vs pythia6 RECO; particle = unfolded vs pythia6
# GEN), so each main row gets its own spelled-out legend.
def row_legend(measured, mc_level):
    return [
        Line2D([], [], color="blue", marker="^", linestyle="none", label=f"inclusive ({measured})"),
        Line2D([], [], color="red", marker="o", linestyle="none", label=f"SoftDrop ({measured})"),
        Line2D([], [], color="black", linestyle=":", label=f"pythia6 ({mc_level})"),
    ]


@app.cell
def _():
    # ---- 1D distribution helpers (the poster's 1/N dN/dlambda densities) -----
    from thoda import Histogram

    # Lambda-axis labels + shared y-axis label for the angularity distributions.
    dist_lambda_label = {
        "ch_ang_k1_b0.5": r"$\lambda^{\kappa=1}_{\beta=0.5}$",
        "ch_ang_k1_b1": r"$\lambda^{\kappa=1}_{\beta=1}$",
        "ch_ang_k1_b2": r"$\lambda^{\kappa=1}_{\beta=2}$",
    }
    dist_ylabel = r"$\frac{1}{N}\frac{dN}{d\lambda}$"


    def make_hist(table, col, edges, mask, weights):
        """Normalized 1-D distribution (density=True -> 1/N dN/dlambda) of `col` on
        one jet-pT slice. `weights`: None, 1-D (data/MC), or (R, N) (unfolded)."""
        _m = np.asarray(mask)
        _v = table[col].to_numpy()[_m]
        _w = None if weights is None else weights[..., torch.as_tensor(_m)]
        _h, _ = Histogram.create([_v], bins=[edges], weights=_w, axis_names=(col,))
        return _h.snapshot()


    def hist_points(snap, batched=False):
        """Central density + error from a HistogramSnapshot. For a histogram
        sqrt(variances) IS the per-bin statistical error (unlike a profile), so no
        effective_counts division. Batched (unfolded): collapse replicas, add the
        per-replica spread in quadrature. Returns (centers, half_width, central, err)."""
        centers = snap.bin_centers[1:-1]
        half = snap.bin_widths[1:-1] / 2.0
        if not batched:
            central = snap.values[1:-1]
            err = snap.variances[1:-1].clamp_min(0).sqrt()
        else:
            central = collapse_replicas(snap.values[:, 1:-1])
            stat = collapse_replicas(snap.variances[:, 1:-1].clamp_min(0).sqrt())
            spread = snap.values[:, 1:-1].std(0)
            err = (stat.square() + spread.square()).sqrt()
        return (
            centers.numpy(),
            half.numpy(),
            central.nan_to_num_(nan=0).numpy(),
            err.nan_to_num_(nan=0).numpy(),
        )

    return dist_lambda_label, dist_ylabel, hist_points, make_hist


@app.cell
def _():
    # Adaptive per-(var, jet-pT) distribution edges, the SAME Bayesian-block edges
    # histograms.py/the poster use (runtime-files/bins_perpt.json), so these
    # distributions match the poster binning. sd_<ang> shares its inclusive edges.
    with open("./runtime-files/bins_perpt.json", "rb") as _f:
        _bp = json.load(_f)
    dist_edges = {_c: {int(_k): _v for _k, _v in _d.items()} for _c, _d in _bp.items()}
    for _ang in kappa1:
        dist_edges.setdefault(f"sd_{_ang}", dist_edges[_ang])
    print("dist edges loaded for:", sorted(dist_edges))
    return (dist_edges,)


@app.cell
def _(
    data_masks,
    data_table,
    dist_edges,
    gen_masks,
    gen_table,
    hist_points,
    make_hist,
    p6_gen_weights,
    p6reco_weights,
    reco_masks,
    reco_table,
    unf_weights,
):
    # ---- build all 1D distributions -----------------------------------------
    # Same four comparisons as the profiles, but 1/N dN/dlambda instead of <lambda>:
    #   det:  data (reco) vs p6 reco ;  part: unfolded (gen) vs p6 gen.
    # incl uses ch_ang_*, sd uses sd_ch_ang_*; both on the inclusive lambda edges.
    dists = {}  # (level, mode, var, ijpt) -> (centers, half, central, err, batched?)
    _njpt = len(jpt_bins) - 1
    for _var in kappa1:
        _sdvar = f"sd_{_var}"
        for _ijpt in range(_njpt):
            for _mode, _col in (("incl", _var), ("sd", _sdvar)):
                _e = dist_edges[_col][_ijpt]
                _d = make_hist(data_table, _col, _e, data_masks[_ijpt], None)
                dists[("det_data", _mode, _var, _ijpt)] = (*hist_points(_d), False)
                _r = make_hist(reco_table, _col, _e, reco_masks[_ijpt], p6reco_weights)
                dists[("det_p6", _mode, _var, _ijpt)] = (*hist_points(_r), False)
                _u = make_hist(gen_table, _col, _e, gen_masks[_ijpt], unf_weights)
                dists[("part_unf", _mode, _var, _ijpt)] = (*hist_points(_u, batched=True), True)
                _g = make_hist(gen_table, _col, _e, gen_masks[_ijpt], p6_gen_weights)
                dists[("part_p6", _mode, _var, _ijpt)] = (*hist_points(_g), False)
        print(f"dist done: {_var}")
    return (dists,)


@app.cell
def _(dists):
    # ---- 1D distribution drawing helper -------------------------------------
    def draw_dist_level(ax_main, ax_ratio, var, ijpt, data_key, mc_key):
        """One level: 1/N dN/dlambda data/unfolded points (with the p6 step curve),
        and below it the (data or unfolded)/p6 ratio. incl=blue, SD=red."""
        for _mode, _clr, _mk in (("incl", "blue", "^"), ("sd", "red", "o")):
            _xc, _hw, _yc, _ye, _ = dists[(data_key, _mode, var, ijpt)]
            _xm, _hwm, _ym, _yme, _ = dists[(mc_key, _mode, var, ijpt)]
            ax_main.errorbar(
                _xc,
                _yc,
                xerr=_hw,
                yerr=_ye,
                color=_clr,
                marker=_mk,
                linestyle="none",
                markersize=4,
                capsize=2,
            )
            ax_main.plot(_xm, _ym, color=_clr, linestyle=":", linewidth=2, drawstyle="steps-mid")
            _r, _re = ratio_points(_yc, _ye, _ym, _yme)
            ax_ratio.errorbar(
                _xc,
                _r,
                yerr=_re,
                color=_clr,
                marker=_mk,
                linestyle="none",
                markersize=4,
                capsize=2,
            )
        ax_main.set_yscale("log")
        ax_ratio.axhline(1.0, color="black", linestyle="--", linewidth=1.5, alpha=0.4)
        ax_ratio.set_ylim(0.5, 1.5)

    return (draw_dist_level,)


@app.cell
def _(dist_lambda_label, dist_ylabel, draw_dist_level):
    # ---- HEADLINE distribution figure: 3 vars (cols) x 4 rows ---------------
    # Same det/particle split as the profile grid, but for the poster's 1/N dN/dlambda
    # distributions. Built at CENTER_JPT (20-30 GeV).
    _ijpt = CENTER_JPT
    _n = len(kappa1)
    fig_dist_headline = plt.figure(figsize=(5 * _n, 11))
    _axs = fig_dist_headline.subplots(
        4,
        _n,
        height_ratios=[3, 1, 3, 1],
        sharex="col",
        squeeze=False,
        gridspec_kw=dict(hspace=0.0, wspace=0.28),
    )
    for _k, _var in enumerate(kappa1):
        _ax_dm, _ax_dr, _ax_pm, _ax_pr = _axs[0, _k], _axs[1, _k], _axs[2, _k], _axs[3, _k]
        draw_dist_level(_ax_dm, _ax_dr, _var, _ijpt, "det_data", "det_p6")
        draw_dist_level(_ax_pm, _ax_pr, _var, _ijpt, "part_unf", "part_p6")
        _ax_pr.set_xlabel(dist_lambda_label[_var], fontsize="x-large")
        _ax_dm.set_title(dist_lambda_label[_var], fontsize=16)
        if _k == 0:
            _ax_dm.set_ylabel("detector level (raw data)\n" + dist_ylabel, fontsize=13)
            _ax_dr.set_ylabel(r"$\frac{\rm data}{\rm p6\,reco}$", fontsize=13)
            _ax_pm.set_ylabel("particle level (unfolded)\n" + dist_ylabel, fontsize=13)
            _ax_pr.set_ylabel(r"$\frac{\rm unf}{\rm p6\,gen}$", fontsize=13)
    _axs[0, 0].text(
        0.04,
        0.96,
        "STAR Preliminary",
        transform=_axs[0, 0].transAxes,
        color="red",
        fontweight="bold",
        va="top",
        fontsize=14,
    )
    _axs[0, 0].text(
        0.04,
        0.80,
        pt_label(_ijpt) + "\n" + r"$p+p\ \sqrt{s}=200$ GeV",
        transform=_axs[0, 0].transAxes,
        va="top",
        fontsize=12,
    )
    _axs[0, _n - 1].legend(handles=row_legend("raw data", "reco"), loc="upper right", fontsize=11)
    _axs[2, _n - 1].legend(handles=row_legend("unfolded", "gen"), loc="upper right", fontsize=11)

    os.makedirs(OUT_DIR, exist_ok=True)
    fig_dist_headline.savefig(
        os.path.join(OUT_DIR, f"fig_unf_vs_data_dist_grid_jpt{_ijpt}.pdf"), bbox_inches="tight"
    )
    print("saved:", os.path.join(OUT_DIR, f"fig_unf_vs_data_dist_grid_jpt{_ijpt}.pdf"))
    fig_dist_headline
    return


@app.cell
def _(dist_lambda_label, dist_ylabel, draw_dist_level):
    # ---- per-variable, all-pT distribution figures --------------------------
    _njpt = len(jpt_bins) - 1
    saved_dist_singles = []
    for _var in kappa1:
        _fig = plt.figure(figsize=(4.2 * _njpt, 11))
        _axs2 = _fig.subplots(
            4,
            _njpt,
            height_ratios=[3, 1, 3, 1],
            sharex="col",
            squeeze=False,
            gridspec_kw=dict(hspace=0.0, wspace=0.30),
        )
        for _j in range(_njpt):
            _adm, _adr, _apm, _apr = _axs2[0, _j], _axs2[1, _j], _axs2[2, _j], _axs2[3, _j]
            draw_dist_level(_adm, _adr, _var, _j, "det_data", "det_p6")
            draw_dist_level(_apm, _apr, _var, _j, "part_unf", "part_p6")
            _apr.set_xlabel(dist_lambda_label[_var], fontsize="large")
            _adm.set_title(pt_label(_j), fontsize=12)
        _axs2[0, 0].set_ylabel("data (reco)\n" + dist_ylabel, fontsize=13)
        _axs2[1, 0].set_ylabel(r"$\frac{\rm data}{\rm p6\,reco}$", fontsize=12)
        _axs2[2, 0].set_ylabel("unfolded\n" + dist_ylabel, fontsize=13)
        _axs2[3, 0].set_ylabel(r"$\frac{\rm unf}{\rm p6\,gen}$", fontsize=12)
        _axs2[0, _njpt - 1].legend(
            handles=row_legend("raw data", "reco"), loc="upper right", fontsize=10
        )
        _axs2[2, _njpt - 1].legend(
            handles=row_legend("unfolded", "gen"), loc="upper right", fontsize=10
        )
        _path = os.path.join(OUT_DIR, f"fig_unf_vs_data_dist_{_var}.pdf")
        _fig.savefig(_path, bbox_inches="tight")
        saved_dist_singles.append(_path)
    print("saved per-var dist figures:", *saved_dist_singles, sep="\n  ")
    return


@app.cell(hide_code=True)
def grooming_cartoon():
    # ---------------------------------------------------------------------------
    # Edification (combined R_g grooming + one-prong/n-prong cartoons): four
    # archetype jets, all at p_T^jet ~ 30 GeV, ordered by R_g LEFT -> RIGHT. Numbers
    # cross-checked against the EXACT softdrop path used in preprocessing.py
    # (exclusive_jets_softdrop_grooming, z_cut=0.2, beta=0, R0=0.4, scalar-z).
    # Each jet is a hard core dressed (or not) by a soft, wide skirt that SoftDrop
    # peels; what survives -- the groomed core -- is what sets the girth lambda:
    #   (a) soft-wide halo + hard collinear core -> R_g tiny, strongly groomed
    #   (b) narrow & hard, ONE-prong             -> R_g small
    #   (c) diffuse & wide, n-PRONG              -> R_g mid
    #   (d) two hard wide prongs                 -> R_g large, grooming removes nothing
    # The pair (d) <-> (a) is the punchline: the HIGHEST-R_g jets (wide *and* hard
    # radiation) are the LEAST affected by grooming, since beta=0 makes the SoftDrop
    # cut a pure momentum balance and R_g is just the angle of the first split it keeps.
    # ---------------------------------------------------------------------------
    import marimo as mo
    from adjustText import adjust_text
    from matplotlib.backends.backend_agg import FigureCanvasAgg


    def _run_softdrop(consts, _R=0.4):
        """Groom a toy jet through the real code path; also return the surviving
        (groomed) constituents so the kept/dropped mask is DERIVED, not guessed.
        consts = [(pt,eta,phi),...]."""
        import awkward as _ak, vector as _vec, fastjet as _fj

        _vec.register_awkward()
        _arr = _ak.with_name(
            _ak.Array(
                [
                    [
                        dict(
                            px=pt * np.cos(phi),
                            py=pt * np.sin(phi),
                            pz=pt * np.sinh(eta),
                            E=pt * np.cosh(eta),
                        )
                        for (pt, eta, phi) in consts
                    ]
                ]
            ),
            "Momentum4D",
        )
        _cs = _fj.ClusterSequence(_arr, _fj.JetDefinition(_fj.antikt_algorithm, _R, _fj.E_scheme))
        _sd = _cs.exclusive_jets_softdrop_grooming(symmetry_cut=0.2, R0=_R)
        _c = _sd.constituents
        _px = np.asarray(_ak.flatten(_c.px, axis=1))
        _py = np.asarray(_ak.flatten(_c.py, axis=1))
        _pz = np.asarray(_ak.flatten(_c.pz, axis=1))
        _gpt = np.hypot(_px, _py)
        return dict(
            Rg=float(_sd.deltaRsoftdrop[0]),
            zg=float(_sd.symmetrysoftdrop[0]),
            n_out=int(_ak.count(_c.E, axis=1)[0]),
            groomed=list(zip(_gpt, np.arcsinh(_pz / _gpt), np.arctan2(_py, _px))),
        )


    def _jet_axis(consts):
        """Jet axis (eta, phi) from the constituents' summed 4-momentum (massless)."""
        _pt = np.array([c[0] for c in consts])
        _e = np.array([c[1] for c in consts])
        _p = np.array([c[2] for c in consts])
        _px = (_pt * np.cos(_p)).sum()
        _py = (_pt * np.sin(_p)).sum()
        _pz = (_pt * np.sinh(_e)).sum()
        _ptj = np.hypot(_px, _py)
        return float(np.arcsinh(_pz / _ptj)), float(np.arctan2(_py, _px))


    def _jet_pt(consts):
        _pt = np.array([c[0] for c in consts])
        _p = np.array([c[2] for c in consts])
        return float(np.hypot((_pt * np.cos(_p)).sum(), (_pt * np.sin(_p)).sum()))


    def _lambda11(consts, _R=0.4):
        """Girth lambda_1^1 about the jet axis (summed-4-momentum direction)."""
        _pt = np.array([c[0] for c in consts])
        _e = np.array([c[1] for c in consts])
        _p = np.array([c[2] for c in consts])
        _ae, _ap = _jet_axis(consts)
        _dR = np.sqrt((_e - _ae) ** 2 + (_p - _ap) ** 2)
        return float(((_pt / _pt.sum()) * (_dR / _R)).sum())


    def _kept_mask(jet, groomed):
        """Match each input constituent to a surviving groomed constituent."""
        return [
            any(abs(pt - a) < 1e-2 and abs(e - b) < 1e-2 and abs(p - d) < 1e-2 for (a, b, d) in groomed)
            for (pt, e, p) in jet
        ]


    # Four toy jets (pt[GeV], eta, phi), each ~30 GeV.
    #   A = two hard wide clusters (each with collinear substructure): the widest
    #       C/A split balances them (z_g>0.2) so grooming stops on the FIRST split
    #       and keeps everything -> large R_g, lambda untouched.
    #   B = hard collinear core + 3 soft wide-angle constituents: every wide split
    #       is soft so all 3 peel -> tiny R_g, lambda collapses.
    #   C = narrow & hard collinear core (spread over dR~0.1 so individually visible)
    #       + a soft wide skirt of 3 -> small R_g, one-prong.
    #   D = diffuse cross-like core of 4 comparable-pT hard prongs + a soft wide
    #       skirt of 3 in the diagonal gaps -> mid R_g, n-prong.
    _jetA = [
        (8.0, 0.00, 0.00),
        (6.0, 0.09, 0.06),
        (4.0, 0.03, -0.09),  # cluster 1 (phi~0), pT~18, constituents spread for visibility
        (7.0, 0.02, 0.34),
        (5.0, 0.12, 0.30),  # cluster 2 (phi~0.32), pT~12  ->  jet pT ~ 30 GeV
    ]
    _jetB = [
        (17.0, 0.00, 0.00),
        (10.0, 0.04, 0.00),  # core, dR=0.04  pT~27
        (1.5, 0.00, 0.36),
        (1.0, 0.30, 0.20),
        (0.5, -0.05, 0.34),  # 3 soft wide  ->  jet pT ~ 30 GeV
    ]
    _jetC = [
        (10.0, 0.00, 0.00),
        (8.0, 0.06, 0.04),
        (5.0, 0.04, 0.08),
        (3.0, 0.08, 0.06),  # hard collinear core (kept)  ~26 GeV
        (2.0, 0.20, 0.16),
        (1.2, -0.14, 0.20),
        (0.8, 0.20, -0.10),  # soft wide skirt (groomed away)  ->  jet pT ~ 30 GeV
    ]
    _jetD = [
        (7.0, 0.00, 0.15),
        (7.0, 0.00, -0.15),
        (7.0, 0.15, 0.00),
        (6.0, -0.15, 0.00),  # diffuse cross core: 4 comparable prongs (kept)
        (1.4, 0.26, 0.26),
        (1.0, -0.34, -0.12),
        (0.7, 0.26, -0.26),  # soft wide skirt (1.0 raised to clear the lambda box)  ->  jet pT ~ 30 GeV
    ]

    _jets = {"A": _jetA, "B": _jetB, "C": _jetC, "D": _jetD}
    _res = {_k: _run_softdrop(_v) for _k, _v in _jets.items()}
    _kept = {_k: _kept_mask(_jets[_k], _res[_k]["groomed"]) for _k in _jets}
    _lamun = {_k: _lambda11(_jets[_k]) for _k in _jets}
    _lamgr = {_k: _lambda11([c for c, _kk in zip(_jets[_k], _kept[_k]) if _kk]) for _k in _jets}

    # order panels by INCREASING R_g (left -> right): B(0.04) C(0.08) D(0.21) A(0.34)
    _order = sorted(_jets, key=lambda _k: _res[_k]["Rg"])
    _names = {
        "A": "two hard wide prongs",
        "B": "soft halo + hard core",
        "C": "narrow & hard (one-prong)",
        "D": "diffuse & wide (n-prong)",
    }
    _tags = ["(a)", "(b)", "(c)", "(d)"]

    # match the HP2026 poster font: sans-serif (Helvetica/Arial/DejaVu Sans) with the
    # stixsans math fontset (the override applied in plot_hp2026_prelims.py).
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    )

    # --- cartoon: constituents in the (eta, phi) plane, sized by pT ------------
    # poster layout: shared y across the row, panels flush (wspace=0); square
    # figure aspect keeps the R=0.4 cones circular.
    _figc, _axc = plt.subplots(
        1, 4, figsize=(15.24, 4.7), sharex=True, sharey=True, gridspec_kw={"wspace": 0.0}
    )
    # attach a renderer-backed (Agg) canvas so adjustText can measure label extents
    FigureCanvasAgg(_figc)
    for _i, _k in enumerate(_order):
        _ax = _axc[_i]
        _jet, _kp, _r = _jets[_k], _kept[_k], _res[_k]
        # origin = jet axis (summed 4-momentum); all coords shown relative to it
        _axe, _axp = _jet_axis(_jet)
        _jpt = _jet_pt(_jet)
        _e = np.array([c[1] for c in _jet]) - _axe
        _p = np.array([c[2] for c in _jet]) - _axp
        _pt = np.array([c[0] for c in _jet])
        _ax.plot(0, 0, "+", color="k", ms=12, mew=2.0, zorder=5)
        _texts = []
        _n = len(_pt)
        for _idx, (_xi, _yi, _pti, _kk) in enumerate(zip(_e, _p, _pt, _kp)):
            _ax.scatter(
                _xi,
                _yi,
                s=42 * _pti,
                c=("#2ca02c" if _kk else "#d62728"),
                alpha=0.55,
                edgecolors="k",
                linewidths=1.0,
                zorder=3,
            )
            # seed each label just outside its marker, radially out from the jet axis
            # (markers at the axis get a fanned-out fallback angle) so it starts off
            # the circle; adjustText then only resolves residual label-label overlaps.
            _rdata = np.sqrt(42 * _pti / np.pi) / 72.0 * (0.92 / 3.62)  # marker radius -> data units
            _d = np.hypot(_xi, _yi)
            if _d < 1e-3:
                _ang = 2 * np.pi * _idx / max(_n, 1)
                _ux, _uy = np.cos(_ang), np.sin(_ang)
            else:
                _ux, _uy = _xi / _d, _yi / _d
            _off = _rdata + 0.015
            _texts.append(
                _ax.text(
                    _xi + _ux * _off,
                    _yi + _uy * _off,
                    rf"${_pti:g}$",
                    fontsize=10,
                    ha="center",
                    va="center",
                    zorder=6,
                )
            )
        _lu, _lg = _lamun[_k], _lamgr[_k]
        _pct = 100.0 * (1.0 - _lg / _lu)
        _ax.text(
            0.04,
            0.115,
            (r"$\lambda^{1}_{1} = %.3f$" % _lu) + "\n" + (r"$\lambda^{1}_{1,\mathrm{g}} = %.3f$" % _lg),
            transform=_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=12,
        )
        # groomed fraction, bold + larger
        _ax.text(
            0.04,
            0.04,
            r"$\boldsymbol{\Delta\lambda^{1}_{1}/\lambda^{1}_{1} = %.0f\%%}$" % _pct,
            transform=_ax.transAxes,
            va="bottom",
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        # Rg (bold, larger) + zg in the upper-left corner (replaces the title)
        _ax.text(
            0.03,
            0.975,
            r"$\boldsymbol{R_g = %.2f}$" % _r["Rg"],
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=15,
            fontweight="bold",
        )
        _ax.text(
            0.03,
            0.86,
            r"$z_g = %.2f$" % _r["zg"],
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
        )
        # jet pT on its own, top-right corner
        _ax.text(
            0.97,
            0.97,
            r"$p_T^{\rm jet} = %.0f$ GeV/$c$" % _jpt,
            transform=_ax.transAxes,
            va="top",
            ha="right",
            fontsize=12,
        )
        _ax.set_xlabel(r"$\eta-\eta_{\rm jet}$")
        if _i == 0:
            _ax.set_ylabel(r"$\varphi-\varphi_{\rm jet}$", labelpad=10)
        # pad past R=0.4 so the whole jet cone clears the frame (square range)
        _ax.set_xlim(-0.46, 0.46)
        _ax.set_ylim(-0.46, 0.46)
        # square columns (fixed via subplots_adjust below) keep the cones circular
        _ax.set_aspect("equal", adjustable="box")
        # drop the +-0.4 edge ticks so adjacent (flush) panels don't collide
        _ax.set_xticks([-0.2, 0.0, 0.2])
        _ax.add_patch(plt.Circle((0, 0), 0.4, fill=False, ls=":", ec="0.6"))
        # repel the pT labels off the constituent positions and each other (limits
        # set above so adjustText knows the frame); thin leader lines when they move
        _ax.figure.canvas.draw()
        adjust_text(
            _texts,
            x=list(_e),
            y=list(_p),
            ax=_ax,
            force_text=(0.22, 0.32),
            force_static=(0.25, 0.32),
            force_explode=(0.08, 0.14),
            expand=(1.08, 1.2),
            max_move=4,
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6),
            ensure_inside_axes=True,
        )
    # legend on the last (highest-R_g) panel: green = kept, red = peeled by SoftDrop
    _legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#2ca02c",
            markeredgecolor="k",
            markersize=11,
            alpha=0.7,
            label="survives grooming",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#d62728",
            markeredgecolor="k",
            markersize=11,
            alpha=0.7,
            label="groomed away",
        ),
        Line2D(
            [0],
            [0],
            marker="+",
            linestyle="none",
            color="k",
            markersize=11,
            markeredgewidth=2.0,
            label="jet axis",
        ),
    ]
    _figc.legend(
        handles=_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        fontsize=12,
    )
    # make the rightmost panel's right edge exactly mirror the left edge:
    # same label (rotated to read down-the-axis) + the tick numbers on the right
    _axc[-1].yaxis.set_label_position("right")
    _axc[-1].set_ylabel(r"$\varphi-\varphi_{\rm jet}$", rotation=270, labelpad=29)
    _axc[-1].tick_params(axis="y", labelright=True)
    # margins chosen so each of the 4 columns is square -> box-equal stays flush
    _figc.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.13, wspace=0.0)

    os.makedirs(OUT_DIR, exist_ok=True)
    _figc.savefig(os.path.join(OUT_DIR, "fig_grooming_cartoon_4panel.pdf"), bbox_inches="tight")

    _rows = "\n".join(
        rf"| {_tags[_i]} {_names[_k]} | {len(_jets[_k])} | {_res[_k]['zg']:.2f} | {_res[_k]['Rg']:.2f} | "
        rf"{_res[_k]['n_out']}/{len(_jets[_k])} | **{_lamun[_k]:.3f} $\to$ {_lamgr[_k]:.3f}** "
        rf"($-${100 * (1 - _lamgr[_k] / _lamun[_k]):.0f}%) |"
        for _i, _k in enumerate(_order)
    )
    _md = mo.md(rf"""
    ### Four archetype jets, ordered by $R_g$: how grooming and the girth $\lambda$ respond to jet shape

    All four toy jets sit at $p_T^{{\rm jet}}\approx 30$ GeV and are run through the EXACT
    `exclusive_jets_softdrop_grooming` path ($z_{{\rm cut}}=0.2$, $\beta=0$, $R_0=0.4$, scalar-$z$).
    The girth $\lambda^{{1}}_{{1}}=\sum_i z_i\,(\Delta R_i/R)$ is a $p_T$-weighted angular spread about
    the jet axis. With $\beta=0$ the SoftDrop cut is a **pure momentum balance** ($z>0.2$, independent of
    angle), so $R_g$ is simply the angle of the *first* splitting that survives — and **a large $R_g$
    means the widest split was already hard, so grooming stops immediately and removes nothing**.

    | panel | constituents | $z_g$ | $R_g$ | kept | $\lambda^{{1}}_{{1}}$ ungroomed $\to$ groomed |
    |---|---|---|---|---|---|
    {_rows}

    Reading left $\to$ right (increasing $R_g$): **(a)** a soft wide halo around a hard collinear core
    — every wide split is soft, all of it peels, $R_g$ collapses and $\lambda$ drops sharply; **(b)** an
    intrinsically narrow, hard one-prong — tiny $\lambda$; **(c)** a diffuse wide *n*-prong — the
    $\Delta R$-weighted sum is large, $\lambda$ is several times bigger; **(d)** two hard, wide,
    *balanced* prongs — the first (widest) split passes the cut, so SoftDrop keeps **all** constituents
    and $\lambda$ is **untouched (0% groomed)**. Hence the punchline that motivated this: the
    **highest-$R_g$ jets are the least affected by grooming** — they must carry $\ge 20\%$ of their
    momentum in a wide-angle prong (wide *and* hard radiation), the rare "pathological" configuration.
    """)

    mo.vstack([_md, _figc])
    return FigureCanvasAgg, adjust_text, mo


@app.cell
def angular_ordering_cartoon(FigureCanvasAgg, adjust_text):
    # ---------------------------------------------------------------------------
    # Edification (companion to the R_g grooming cartoon above): how EARLY soft
    # wide-angle radiation CONSTRAINS the angular phase space of LATER splittings.
    #
    # Physics: a QCD parton shower is *angular ordered* (color coherence). A soft
    # gluon radiated at angle theta is emitted coherently by the NET color charge
    # of everything already collinear to it, so any subsequent radiation off either
    # resulting prong is confined to a cone of half-angle < theta. Emission angles
    # down a branch therefore form a strictly NESTED, DECREASING sequence
    # theta_1 > theta_2 > theta_3 ..., the earliest/widest split capping the rest.
    # Cambridge/Aachen reclustering (merge nearest-in-angle first) declusters the
    # jet back out widest-first -- the primary Lund sequence -- and its first split
    # IS R_g, tying this to the grooming cartoon above.
    # ---------------------------------------------------------------------------
    import marimo as _mo

    # match the HP2026 poster font: sans-serif + stixsans math (as in
    # plot_hp2026_prelims.py). `adjust_text` and `FigureCanvasAgg` are already
    # imported by the grooming_cartoon cell above -- reuse them, do not re-import.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "mathtext.fontset": "stixsans",
        }
    )


    def _dR(a, b):
        return float(np.hypot(a[1] - b[1], a[2] - b[2]))


    def _merge(a, b):
        # Cambridge/Aachen recombination: pt-weighted (eta, phi) centroid, pt summed.
        _ptsum = a[0] + b[0]
        return (
            _ptsum,
            (a[0] * a[1] + b[0] * b[1]) / _ptsum,
            (a[0] * a[2] + b[0] * b[2]) / _ptsum,
        )


    def _ca_primary_decluster(consts):
        """Cambridge/Aachen primary Lund declustering, pure-angle (dR) metric.

        Build the C/A tree by repeatedly merging the closest pair in dR, then walk
        DOWN from the root following the harder-pT child, recording the opening
        angle dR and Lund z of each split (widest/earliest first). The softer child
        at each step is the 'emission'."""
        _nodes = {}
        for _k, _c in enumerate(consts):
            _nodes[_k] = (_c[0], _c[1], _c[2], None, None)  # (pt, eta, phi, c1, c2)
        _active = list(range(len(consts)))
        _nxt = len(consts)
        while len(_active) > 1:
            _best = None
            for _ii in range(len(_active)):
                for _jj in range(_ii + 1, len(_active)):
                    _d = _dR(_nodes[_active[_ii]], _nodes[_active[_jj]])
                    if _best is None or _d < _best[0]:
                        _best = (_d, _ii, _jj)
            _, _ii, _jj = _best
            _ai, _bi = _active[_ii], _active[_jj]
            _m = _merge(_nodes[_ai], _nodes[_bi])
            _nodes[_nxt] = (_m[0], _m[1], _m[2], _ai, _bi)
            _active = [_x for _x in _active if _x not in (_ai, _bi)] + [_nxt]
            _nxt += 1
        _root = _active[0]
        _seq, _cur = [], _root
        while _nodes[_cur][3] is not None:
            _c1, _c2 = _nodes[_cur][3], _nodes[_cur][4]
            _n1, _n2 = _nodes[_c1], _nodes[_c2]
            _zi = min(_n1[0], _n2[0]) / (_n1[0] + _n2[0])
            _seq.append((_dR(_n1, _n2), _zi, _n1, _n2))  # widest first
            _cur = _c1 if _n1[0] >= _n2[0] else _c2  # follow the harder branch
        return _seq, _nodes[_root]


    # Toy: leading parton (hard core) + 3 angular-ordered emissions, widths chosen
    # so the primary C/A branch peels them widest-first (g1 soft+wide -> g3 collinear).
    _jet = [
        (16.0, 0.00, 0.00),  # hard core (leading parton)
        (2.0, 0.02, 0.34),  # g1: soft, WIDE-angle  (early emission)
        (5.0, 0.16, 0.05),  # g2: intermediate angle
        (3.0, -0.06, 0.03),  # g3: narrow / collinear (late emission)
    ]
    _seq, _root = _ca_primary_decluster(_jet)
    _thetas = [s[0] for s in _seq]  # theta_1 > theta_2 > theta_3
    _zs = [s[1] for s in _seq]
    _R = 0.4
    _core = _jet[0]

    _figo, (_axA, _axB) = plt.subplots(1, 2, figsize=(11.5, 5.3))
    # renderer-backed canvas so adjustText can measure label extents
    FigureCanvasAgg(_figo)

    # --- Panel A: nested coherence cones in the (eta, phi) plane ----------------
    _cone_fc = plt.cm.Blues(np.linspace(0.55, 0.22, len(_thetas)))
    for _th, _fc in zip(_thetas, _cone_fc):  # widest (lightest fill) first, drawn under
        _axA.add_patch(plt.Circle((0, 0), _th, fc=_fc, ec="none", alpha=0.5, zorder=1))
    for _th in _thetas:  # cone outlines; the theta values now label the emissions
        _axA.add_patch(plt.Circle((0, 0), _th, fill=False, ec="0.35", lw=1.0, zorder=2))
    _gen_color = ["#222222"] + list(plt.cm.autumn(np.linspace(0.05, 0.7, len(_jet) - 1)))
    _textsA, _exA, _eyA = [], [], []
    for _ci, _c in enumerate(_jet):
        _x, _y = _c[1] - _core[1], _c[2] - _core[2]
        _lab = "core" if _ci == 0 else rf"$g_{_ci}$"
        _axA.scatter(
            _x,
            _y,
            s=42 * _c[0],
            c=[_gen_color[_ci]],
            alpha=0.7,
            edgecolors="k",
            linewidths=1.0,
            zorder=4,
        )
        _exA.append(_x)
        _eyA.append(_y)
        if _ci == 0:
            # core: place its label by hand, just up-and-right of the big marker so it
            # hugs the point it labels (kept out of adjustText so it stays put).
            _axA.text(
                _x + 0.06,
                _y - 0.02,
                rf"{_lab} $p_T={_c[0]:g}$",
                fontsize=9,
                ha="left",
                va="center",
                zorder=6,
            )
            continue
        # emissions: seed each label just outside its marker (radially out from the
        # core) so it starts off the circle; adjustText then resolves overlaps.
        _rd = np.sqrt(42 * _c[0] / np.pi) / 72.0 * (0.92 / 3.4)
        _dd = np.hypot(_x, _y)
        _ux, _uy = (0.0, -1.0) if _dd < 1e-3 else (_x / _dd, _y / _dd)
        _o = _rd + 0.02
        _textsA.append(
            _axA.text(
                _x + _ux * _o,
                _y + _uy * _o,
                rf"$\theta_{_ci}={_thetas[_ci - 1]:.2f},\ p_T={_c[0]:g}$",
                fontsize=9,
                ha="center",
                va="center",
                zorder=6,
            )
        )
    _axA.plot(0, 0, "+", color="k", ms=12, mew=2.0, zorder=5)
    _axA.add_patch(plt.Circle((0, 0), _R, fill=False, ls=":", ec="0.6"))
    _axA.set_xlim(-0.46, 0.46)
    _axA.set_ylim(-0.46, 0.46)
    _axA.set_aspect("equal")
    _axA.set_xlabel(r"$\eta-\eta_{\rm core}$")
    _axA.set_ylabel(r"$\varphi-\varphi_{\rm core}$")
    # dodge the constituent labels off the markers/cones and each other
    _axA.figure.canvas.draw()
    adjust_text(
        _textsA,
        x=_exA,
        y=_eyA,
        ax=_axA,
        force_text=(0.4, 0.6),
        force_static=(0.3, 0.45),
        force_explode=(0.15, 0.25),
        expand=(1.2, 1.4),
        max_move=6,
        arrowprops=dict(arrowstyle="-", color="0.45", lw=0.6),
        ensure_inside_axes=True,
    )

    # --- Panel B: angular-ordering ladder (allowed band shrinks each step) ------
    _steps = np.arange(1, len(_thetas) + 1)
    _caps = [_R] + _thetas[:-1]  # theta_0 = R caps step 1; theta_{i-1} caps step i
    for _st, _cap in zip(_steps, _caps):
        _axB.add_patch(plt.Rectangle((_st - 0.36, 0), 0.72, _cap, fc="#2ca02c", alpha=0.13, ec="none"))
        _axB.add_patch(
            plt.Rectangle(
                (_st - 0.36, _cap), 0.72, _R - _cap, fc="#d62728", alpha=0.10, ec="none", hatch="//"
            )
        )
        _axB.hlines(_cap, _st - 0.36, _st + 0.36, color="0.4", lw=1.0, ls="--")
    _axB.plot(_steps, _thetas, "o-", color="#1f3b73", ms=9, lw=1.6, zorder=5)
    for _st, _th in zip(_steps, _thetas):
        _axB.annotate(
            rf"$\theta_{_st}={_th:.2f}$",
            (_st, _th),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=10,
        )
    _axB.axhline(_R, color="0.5", ls=":", lw=1.0)
    _axB.text(len(_thetas) + 0.42, _R, r"$R=0.4$", va="center", fontsize=9, color="0.4")
    _axB.set_xlim(0.5, len(_thetas) + 0.75)
    _axB.set_ylim(0, _R * 1.08)
    _axB.set_xticks(_steps)
    _axB.set_xlabel(r"splitting along leading branch (early $\to$ late)", fontsize=14)
    _axB.set_ylabel(r"emission angle $\theta_i=\Delta R_i$", fontsize=14)
    _figo.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    _figo.savefig(os.path.join(OUT_DIR, "fig_angular_ordering_cartoon.pdf"), bbox_inches="tight")

    _rows = "\n".join(
        rf"| {_i} | $g_{_i}$ | {_t:.2f} | {_z:.2f} | "
        + (rf"$\theta_{{{_i + 1}}}\leq\theta_{_i}$ |" if _i < len(_thetas) else "&mdash; |")
        for _i, (_t, _z) in enumerate(zip(_thetas, _zs), 1)
    )
    _md = _mo.md(rf"""
    ### How early soft wide-angle radiation constrains *later* splittings

    QCD radiation is **angular ordered** &mdash; a consequence of *color coherence*. A
    soft gluon emitted at angle $\theta$ off a color charge is radiated by the **net**
    charge of everything already collinear to it, so any *subsequent* emission off
    either of the two resulting prongs is confined to a cone of half-angle $<\theta$.
    The emission angles down a branch therefore form a **strictly nested, decreasing**
    sequence

    $$\theta_1 \;>\; \theta_2 \;>\; \theta_3 \;>\; \dots,$$

    the earliest, widest splitting setting the ceiling for everything that follows
    (**left:** each coherence cone sits inside the previous; **right:** every later
    angle is pinned below the green line the earlier one drew).

    This is exactly what **Cambridge/Aachen** reclustering exposes: it merges the
    nearest-in-angle pair first, so *declustering* from the root walks the shower back
    out **widest-angle-first** &mdash; the **primary Lund sequence**. The very first
    decluster is the widest splitting, i.e. **$R_g$** under SoftDrop ($\beta=0$):
    $R_g=\theta_1$ whenever that first split also passes the momentum cut. Here the
    early wide emission $g_1$ is genuinely **soft** ($z_1={_zs[0]:.2f}<z_{{\rm cut}}=0.2$),
    so SoftDrop peels it and $R_g$ falls back to $\theta_2={_thetas[1]:.2f}$ &mdash; this
    *is* the small-$R_g$ branch of the grooming cartoon above, now read off the same
    angular-ordered ladder.

    Toy leading parton $+$ 3 angular-ordered emissions, reclustered with C/A
    (pure-angle metric), primary (leading-$p_T$) branch:

    | split $i$ | emission | $\theta_i=\Delta R_i$ | Lund $z_i$ | constrains next |
    |---|---|---|---|---|
    {_rows}

    The angles fall monotonically by construction of the **shower**, *not* of the
    algorithm &mdash; C/A only *reveals* the ordering that coherence already imposed.
    Physically this is why the **groomed** observables track the ungroomed ones (the
    wide primary split that survives grooming dominates the angular structure), and why
    an angularity's $\beta$ exponent &mdash; the $\Delta R^\beta$ weight &mdash; selects
    *where along this nested ladder* it is most sensitive: large $\beta$ leans on the
    early wide splittings, small $\beta$ on the late collinear ones.
    """)

    _mo.vstack([_md, _figo])
    return


@app.cell(hide_code=True)
def poster_motivation_bullets(mo):
    mo.md(r"""
    ### Physical picture: angular ordering & grooming

    - **Color coherence $\Rightarrow$ angular ordering.** A soft gluon sees the *net* color charge of everything collinear to it, so each emission is confined *inside* the previous one's cone: $\theta_1\!>\!\theta_2\!>\!\theta_3$ *(right)*.
    - **The widest split — $R_g$ — caps the rest.** Every later (and every groomed) emission lives inside that first cone, so a single angle sets the angular phase space for the whole jet *(both)*.
    - **Grooming removes the same soft, wide skirt from every jet** *(left, four jets ordered by $R_g$)* — and the groomed fraction shrinks along the spectrum, vanishing once the widest split is itself hard.
    - **So $\langle\lambda\rangle$ vs $R_g$ is a test of coherence, not a redundant correlation:** does the data carry the nested angular structure the parton shower predicts?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why measure (charged) jet angularities? — physics motivations

    The two cartoons above are not just pedagogy; they encode *why* this measurement is
    worth doing. The **grooming cartoon** shows that the widest surviving splitting sets the
    jet's angular scale ($R_g$), and the **angular-ordering cartoon** shows that this early,
    wide emission **caps the angular phase space of every later splitting**
    ($\theta_1>\theta_2>\theta_3$). Measuring the charged angularities
    $\lambda^{\kappa}_{\beta}=\sum_{i\in\text{jet}} z_i^{\kappa}\,(\Delta R_i/R)^{\beta}$ —
    groomed *and* ungroomed, fully unfolded and differential in jet $p_T$ — turns those
    structural facts into quantitative tests of QCD.

    1. **Scan the QCD radiation pattern across the $(\kappa,\beta)$ plane.**
       A single jet image is projected onto a *family* of observables: the energy weight
       $\kappa$ and the angular weight $\beta$ dial *which* emissions dominate. As the
       angular-ordering ladder makes explicit, **large $\beta$ leans on the early, wide
       splittings; small $\beta$ on the late, collinear ones.** Measuring the full
       $(\kappa,\beta)$ grid (here $\kappa=1,2$; $\beta=0,\tfrac12,1,2$, plus $p_T^D=\lambda^2_0$
       and the neutral-energy fraction) maps the *differential* structure of the parton
       shower rather than a single moment of it, directly probing the DGLAP/coherent-branching
       splitting kernels.

    2. **Test color coherence / angular ordering experimentally.**
       Angular ordering ($\theta_{i}\le\theta_{i-1}$) is a *prediction* of coherent QCD, not an
       axiom. Because $R_g$ is the first/widest angle and every groomed angularity lives
       *inside* that cone, the **measured correlation between $R_g$ and the (groomed) angularities**
       is a direct handle on whether the data carry the nested angular structure coherence
       predicts — and a vacuum reference for the *de*coherence expected in a medium.

    3. **Separate perturbative from non-perturbative physics with grooming.**
       SoftDrop ($z_{\rm cut}=0.2,\ \beta=0$) removes exactly the early *soft* wide-angle
       radiation of the cartoons — the region most contaminated by hadronization, the
       underlying event, and non-global logarithms. Comparing **groomed vs. ungroomed**
       angularities quantifies the non-perturbative shift and yields observables that are far
       more directly calculable in resummed pQCD. (The "highest-$R_g$ jets are *least* groomed"
       insight even tells you *which* jets are perturbatively cleanest.)

    4. **Provide a precision pp vacuum baseline at RHIC energy.**
       Jet-substructure modification in heavy-ion collisions is always quoted *relative to* a
       pp reference. A fully-corrected, multi-differential measurement of charged angularities
       in $\sqrt{s}=200$ GeV $pp$ is that baseline — and angular ordering is precisely the
       structure medium-induced color decoherence is expected to disrupt, so the vacuum
       measurement defines the null hypothesis for quenching searches.

    5. **Discriminate and tune parton-shower Monte Carlos.**
       The unfolded result is confronted with **PYTHIA 6, PYTHIA 8, and HERWIG 7** — angular-
       ordered vs. dipole/$p_T$-ordered showers with different hadronization models. A
       multi-differential angularity measurement across jet $p_T$ and the $(\kappa,\beta)$
       plane is sensitive enough to *separate* these models and constrain their tunes, especially
       at RHIC kinematics (larger $\alpha_s$, smaller boost, different phase space than the LHC).

    6. **Quantify the *information content* of jet substructure.**
       The `angularities_noptd` and `angularities_minimal` cross-checks ask **how little of the
       substructure is actually needed** to reproduce the unfolded result. This is a genuine
       physics question about the dimensionality of the relevant emission manifold — whether
       mass $M$, $M_g$, $R_g$ and $p_T^D$ are independently informative or largely redundant
       given the angular ladder — and it connects this classical measurement to the modern
       optimal-observable / ML-tagging program.

    **Bonus — charged-particle observables & continuity with published STAR.** Using charged
    constituents buys tracking-grade angular resolution (sharpening the $\Delta R$ weighting that
    $\beta$ controls) and tests track-function universality and charge-dependent fragmentation;
    the groomed mass, $z_g$ and $R_g$ here also extend the previously published STAR jet-substructure
    results into the full angularity family.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Poster review-response — specific, located edits (`slides/hp2026-poster/poster.tex`)

    > *"…the motivation for generalized angularities is just that they are IRC safe, while
    > there is no motivation listed for showing the different values of $\beta$, or the
    > correlation of mean angularity with $R_g$. The Summary doesn't have any physics
    > statements other than that the Monte Carlo is a bit off."*

    Both named gaps are exactly the physics in the two cartoons above. Four located edits
    close the comment word-for-word. (Your **Abstract** already states the real physics —
    angular scale ↔ time/energy scale, p/npQCD separation via grooming — but that thread
    never reaches the body boxes; these edits pull it through.)

    ---

    ### ① Box **"Generalized angularities"** (≈ L137–148) — fixes *"motivation is just IRC-safe"* **and** *"why different $\beta$"*
    Right now `κ=1 → IRC-safe` is the *only* motivation. Add the $(\kappa,\beta)$ reading —
    $\kappa$ = soft/hard weight, $\beta$ = **angular reach** — and say outright that $\beta$
    slices the angular-ordered shower. Paste under the `\lambda` definition / into the itemize:

    ```latex
    \kappa:\ \text{soft vs hard fragmentation}\qquad
    \beta:\ \text{collinear vs wide-angle emission}\notag
    ```
    ```latex
    \item \textbf{Why scan }\boldsymbol{\beta}\textbf{:} larger $\beta$ up-weights wider
          angles, so $\beta$ tunes the angular reach of $\lambda$ — small $\beta$ probes
          late/collinear splittings, large $\beta$ the early, soft, wide-angle radiation
          (angular ordering).
    \item \textbf{Momentum dispersion:} $\lambda_2^0\ (p_T^D)$ — soft vs hard fragmentation
    ```
    (The $\lambda^2_0/p_T^D$ bullet is currently **commented out** at L147, yet $\lambda^2_0$
    appears in your Summary list — restore it *with* this motivation.)

    ### ② Box **"$\lambda_\beta^\kappa$ vs $R_g$"** (≈ L305–308) — fixes *"no motivation for the correlation with $R_g$"*
    Add a **leading motivation** bullet, then upgrade the two observations into physics:

    ```latex
    \item \textbf{Why }\boldsymbol{R_g}\textbf{:} $R_g$ = angle of the widest (first) hard
          splitting; angular ordering confines all later emission inside that cone, so
          $\langle\lambda\rangle(R_g)$ probes how the leading splitting sets the angular
          phase space — a test of \textbf{color coherence}, not a redundant correlation.
    \item High-$R_g$ jets are unaffected by grooming: their widest split already passes
          $z_{\rm cut}$ (groomed $=$ ungroomed) $\Rightarrow$ genuinely two-prong.
    \item $\langle\lambda^{\kappa=1}_\beta\rangle$ rises monotonically with $R_g$, steepening
          with $\beta$ — the angular-ordering fingerprint (wider leading split $\to$ more
          angular phase space for radiation).
    ```
    *Backup if pushed:* your own checks show this trend is **data-driven, not inherited from
    the PYTHIA6 prior** (prior-independence / prior-domination defense).

    ### ③ Box **"$\lambda_\beta^\kappa$ distributions"** (≈ L289) — give the grooming shift a mechanism
    `Shift increases with β` is an observation; make the cause explicit (ties to the grooming cartoon):

    ```latex
    \item Shift grows with $\beta$: SoftDrop removes the early, soft, wide-angle radiation
          that dominates large-$\beta$ angularities (npQCD $\to$ pQCD).
    ```

    ### ④ Box **"Summary"** (≈ L316–321) — replace *"MC is a bit off"* with physics conclusions
    Swap the descriptive bullets for statements about the *shower*:

    ```latex
    \item $\beta$-dependence of $\langle\lambda\rangle$ maps the wide-angle $\to$ collinear
          structure of the shower; grooming shifts confirm removal of early soft wide radiation.
    \item $\langle\lambda\rangle$ increases with $R_g$ as required by \textbf{angular ordering}:
          the leading splitting sets the angular scale for all subsequent radiation (coherence).
    \item Generators reproduce the collinear regime but \textbf{under-estimate wide-angle
          radiation} (high $\beta$, large $R_g$) — they mis-model early, soft, wide emission,
          not a generic normalization offset.
    ```

    ### ⑤ Box **"Abstract"** (≈ L121–123) — lead with the physics, demote quenching to one line
    Generic phrasing ("various jet angularities", "model calculations") buries the hook.
    Paste-ready rewrite (keeps both `\cite`s; opens on angular ordering):

    ```latex
    Jet substructure encodes the QCD shower's history --- wide-angle, early emissions
    constrain the collinear, later ones through angular ordering. Generalized angularities
    \(\lambda_\beta^\kappa\) read out this structure on a tunable plane --- \(\kappa\)
    weighting soft vs.\ hard fragmentation, \(\beta\) the collinear-to-wide-angle reach ---
    while SoftDrop grooming \cite{SoftDrop} strips the soft, wide-angle radiation where
    non-perturbative QCD dominates, exposing the perturbative core. We present
    fully-corrected (MultiFold) inclusive and groomed
    \(\lambda^{\kappa=1}_{\beta\in\{0.5,1,2\}}\) and \(\lambda^{\kappa=2}_{0}\) from
    \(p+p\) collisions at \(\sqrt{s}=200\) GeV in STAR, together with their dependence on
    the groomed radius \(R_g\) --- a direct test of how the first, widest splitting sets
    the angular phase space (color coherence). Confronting PYTHIA and HERWIG pinpoints
    where shower models mis-handle wide-angle emission. At RHIC energies these measurements
    reach a wider-angle, more non-perturbative regime than the LHC \cite{ALICE:2021njq},
    fixing the vacuum baseline for jet-quenching signatures such as broadening.
    ```
    Key moves: open on **angular ordering** (not the vague time/energy line); fold in the
    **$(\kappa,\beta)$ reading + $\beta$ reach** and the **$\langle\lambda\rangle$–$R_g$
    coherence test** so the reviewer's two gaps are answered before the body; turn "model
    calculations" into a physics statement; compress quenching to the final clause.

    ---

    **If space is tight:** ① and ② are non-negotiable (the reviewer's two named gaps); ④
    makes the Summary pass; ③ is a one-line upgrade. Together they answer every clause of the
    comment, and they re-use motivations #1 (β scan), #2 (coherence/$R_g$) and #3 (grooming =
    pQCD/npQCD) from the cell above.
    """)
    return


if __name__ == "__main__":
    app.run()
