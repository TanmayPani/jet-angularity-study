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


@app.cell
def _():
    # ---------------------------------------------------------------------------
    # Edification: why the HIGHEST-R_g jets are the LEAST affected by grooming.
    #
    # Colleague's question -- "jets with the highest R_g would be the least
    # affected by grooming? These would have to be pathological jets, with
    # wide-angle but high-z radiation, yes?"  Answer: yes, exactly. Worked through
    # by hand on two 5-constituent jets, numbers cross-checked against the EXACT
    # softdrop path used in preprocessing.py (fastjet
    # exclusive_jets_softdrop_grooming, z_cut=0.2, beta=0, R0=0.4, scalar-z).
    # ---------------------------------------------------------------------------
    import marimo as mo


    def _run_softdrop(consts, _R=0.4):
        """Groom a toy jet through the real code path. consts = [(pt,eta,phi),...]."""
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
        return dict(
            Rg=float(_sd.deltaRsoftdrop[0]),
            zg=float(_sd.symmetrysoftdrop[0]),
            n_out=int(_ak.count(_sd.constituents.E, axis=1)[0]),
        )


    def _lambda11(consts, _R=0.4):
        """Girth lambda_1^1 about the jet axis (summed-4-momentum direction)."""
        _pt = np.array([c[0] for c in consts])
        _e = np.array([c[1] for c in consts])
        _p = np.array([c[2] for c in consts])
        _ae, _ap = _jet_axis(consts)
        _dR = np.sqrt((_e - _ae) ** 2 + (_p - _ap) ** 2)
        return float(((_pt / _pt.sum()) * (_dR / _R)).sum())


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
        """Jet transverse momentum from the summed 4-momentum."""
        _pt = np.array([c[0] for c in consts])
        _p = np.array([c[2] for c in consts])
        return float(np.hypot((_pt * np.cos(_p)).sum(), (_pt * np.sin(_p)).sum()))


    # Two minimal 5-constituent jets (pt [GeV], eta, phi).
    #   A = two wide-separated HARD clusters (each with collinear substructure):
    #       the widest C/A split balances the two clusters -> passes z>z_cut.
    #   B = hard collinear core + a halo of 3 soft wide-angle constituents:
    #       every wide split is soft -> all peeled, stop at the core.
    _jetA = [
        (7.0, 0.0, 0.00),
        (5.0, 0.03, 0.00),
        (3.0, 0.0, 0.03),  # cluster 1, pT~15
        (6.0, 0.0, 0.35),
        (4.0, 0.03, 0.35),
    ]  # cluster 2, pT~10  ->  jet pT ~ 25 GeV
    _jetB = [
        (14.0, 0.0, 0.00),
        (9.0, 0.04, 0.00),  # core, dR=0.04
        (1.2, 0.0, 0.36),
        (0.8, 0.30, 0.20),
        (0.5, -0.05, 0.34),
    ]  # 3 soft wide  ->  jet pT ~ 25 GeV

    _rA, _rB = _run_softdrop(_jetA), _run_softdrop(_jetB)
    # A keeps everything; B keeps only the 2-constituent core.
    _keptA = [True] * 5
    _keptB = [True, True, False, False, False]
    _lamA_un, _lamA_gr = _lambda11(_jetA), _lambda11(_jetA)  # identical (nothing dropped)
    _lamB_un, _lamB_gr = _lambda11(_jetB), _lambda11(_jetB[:2])  # groomed = core only

    # --- cartoon: constituents in the (eta, phi) plane, sized by pT ------------
    _figc, _axc = plt.subplots(1, 2, figsize=(11, 5.2))
    _specs = [
        dict(
            ax=_axc[0],
            jet=_jetA,
            res=_rA,
            kept=_keptA,
            name="A: Two pronged",
            groups=[_jetA[:3], _jetA[3:]],
            lam=(_lamA_un, _lamA_gr),
            note=(r"$z=%.2f>z_{cut}-0.2$" % _rA["zg"]) + ("\n $R_g=%.2f$" % _rA["Rg"]),
        ),
        dict(
            ax=_axc[1],
            jet=_jetB,
            res=_rB,
            kept=_keptB,
            name="B: Hard core + soft-wide radiation",
            groups=[_jetB[:2]],
            lam=(_lamB_un, _lamB_gr),
            note=("$z=%.2f>z_{cut}=0.2$" % _rB["zg"]) + ("\n $R_g=%.2f$" % _rB["Rg"]),
        ),
    ]
    for _s in _specs:
        _ax, _jet, _kept = _s["ax"], _s["jet"], _s["kept"]
        # origin = jet axis (summed 4-momentum); all coords shown relative to it
        _axe, _axp = _jet_axis(_jet)
        _jpt = _jet_pt(_jet)
        _e = np.array([c[1] for c in _jet]) - _axe
        _p = np.array([c[2] for c in _jet]) - _axp
        _pt = np.array([c[0] for c in _jet])
        # dashed guide lines join SUBJET axes (also relative to the jet axis):
        #   A = the two hard-cluster axes; B = core axis -> each dropped soft prong.
        if len(_s["groups"]) == 2:
            _g1, _g2 = _jet_axis(_s["groups"][0]), _jet_axis(_s["groups"][1])
            _ax.plot(
                [_g1[0] - _axe, _g2[0] - _axe], [_g1[1] - _axp, _g2[1] - _axp], "k--", lw=1.3, zorder=2
            )
        else:
            _gc = _jet_axis(_s["groups"][0])
            _gce, _gcp = _gc[0] - _axe, _gc[1] - _axp
            for _xi, _yi, _k in zip(_e, _p, _kept):
                if not _k:
                    _ax.plot([_gce, _xi], [_gcp, _yi], "--", color="#d62728", lw=1.0, zorder=2)
        _ax.plot(0, 0, "+", color="k", ms=12, mew=2.0, zorder=5)
        _ax.annotate(
            "jet axis",
            (0, 0),
            textcoords="offset points",
            xytext=(-8, 9),
            ha="right",
            fontsize=9,
            color="0.3",
        )
        for _xi, _yi, _pti, _k in zip(_e, _p, _pt, _kept):
            _ax.scatter(
                _xi,
                _yi,
                s=42 * _pti,
                c=("#2ca02c" if _k else "#d62728"),
                alpha=0.55,
                edgecolors="k",
                linewidths=1.0,
                zorder=3,
            )
            _ax.annotate(
                rf"${_pti:g}$", (_xi, _yi), textcoords="offset points", xytext=(7, 6), fontsize=10
            )
        _ax.text(
            0.6,
            0.96,
            _s["note"],
            transform=_ax.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            bbox=dict(boxstyle="round", fc="#fff7e6", ec="0.5"),
        )
        _lu, _lg = _s["lam"]
        _ax.text(
            0.96,
            0.04,
            r"girth $\lambda^{1}_{1}$:"
            + "\n"
            + (r"  ungroomed $= %.3f$" % _lu)
            + "\n"
            + (r"  groomed $\;\;= %.3f$" % _lg),
            transform=_ax.transAxes,
            va="bottom",
            ha="right",
            fontsize=12,
            bbox=dict(boxstyle="round", fc="#eef5ff", ec="0.5"),
        )
        _ax.set_title(
            _s["name"] + rf"  ($p_T^{{\rm jet}}\approx{_jpt:.0f}$ GeV, kept {_s['res']['n_out']}/5)",
            fontsize=12.5,
        )
        _ax.set_xlabel(r"$\eta-\eta_{\rm jet}$")
        _ax.set_ylabel(r"$\varphi-\varphi_{\rm jet}$")
        _ax.set_xlim(-0.4, 0.4)
        _ax.set_ylim(-0.4, 0.4)
        _ax.add_patch(plt.Circle((0, 0), 0.4, fill=False, ls=":", ec="0.6"))
    _figc.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    _figc.savefig(os.path.join(OUT_DIR, "fig_rg_grooming_cartoon.pdf"), bbox_inches="tight")

    _md = mo.md(rf"""
    ### Why the highest-$R_g$ jets are the *least* affected by grooming

    SoftDrop here ($z_{{\rm cut}}=0.2$, $\beta=0$, $R_0=0.4$, scalar-$z$) walks the
    angular-ordered (C/A) tree from the **widest** splitting inward and at each two-prong
    splitting tests

    $$z=\frac{{\min(p_{{T,1}},p_{{T,2}})}}{{p_{{T,1}}+p_{{T,2}}}}\;>\;z_{{\rm cut}}\Big(\tfrac{{\Delta R_{{12}}}}{{R_0}}\Big)^{{\beta}}\;\overset{{\beta=0}}{{=}}\;z_{{\rm cut}}=0.2 .$$

    With **$\beta=0$ the cut is purely a momentum balance, independent of angle**, and
    $R_g$ is simply the angle of the *first* splitting that passes. Hence:

    * **Large $R_g$:** the very first, widest splitting already had $z>0.2$ &rarr; grooming
      **stops immediately and removes nothing** &rarr; the groomed jet *is* the full jet, so
      **every groomed angularity equals its ungroomed value** (least affected, by construction).
    * **Small $R_g$:** the wide splittings were soft ($z<0.2$) and got peeled away; SoftDrop
      only stopped deep in the collinear core &rarr; strongly groomed.

    So a high-$R_g$ jet **must** carry $\ge 20\%$ of its momentum in a **wide-angle prong** —
    wide *and hard* radiation. Ordinary QCD wide emission is soft, so these are rare,
    "pathological" configurations. Two **5-constituent** jets (run through the exact
    `exclusive_jets_softdrop_grooming` path, $R=0.4$):

    | | constituents | stop-split $z_g$ | $R_g$ | kept | $\lambda^{{1}}_{{1}}$ ungroomed &rarr; groomed |
    |---|---|---|---|---|---|
    | **A** two hard wide clusters | 5 | {_rA["zg"]:.2f} | {_rA["Rg"]:.2f} | {_rA["n_out"]}/5 | **{_lamA_un:.3f} &rarr; {_lamA_gr:.3f}**  (0% change) |
    | **B** core + 3 soft wide | 5 | {_rB["zg"]:.2f} | {_rB["Rg"]:.2f} | {_rB["n_out"]}/5 | {_lamB_un:.3f} &rarr; {_lamB_gr:.3f}  (&minus;{100 * (1 - _lamB_gr / _lamB_un):.0f}%) |

    Jet **A**: the widest C/A split balances the two hard clusters ($z_g={_rA["zg"]:.2f}>0.2$),
    so SoftDrop stops on the *first* split, keeps **all 5** constituents, and $R_g={_rA["Rg"]:.2f}$
    is large — grooming is a no-op and $\lambda$ is untouched.
    Jet **B**: every wide split is a soft halo prong ($z<0.2$), so all **3** are peeled until
    SoftDrop reaches the $z_g={_rB["zg"]:.2f}$ collinear core; $R_g$ collapses to ${_rB["Rg"]:.2f}$
    and $\lambda$ drops by &minus;{100 * (1 - _lamB_gr / _lamB_un):.0f}%. **Maximal $R_g$ &hArr; minimal grooming.**
    """)

    mo.vstack([_md, _figc])
    return


if __name__ == "__main__":
    app.run()
