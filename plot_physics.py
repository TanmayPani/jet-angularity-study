import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FormatStrFormatter, MaxNLocator
    # from adjustText import adjust_text

    plt.style.use("default")
    # --- old: minimal style stub ---
    # plt.rcParams["savefig.facecolor"] = "white"
    # plt.rcParams["savefig.edgecolor"] = "white"
    # --- new: publication-ready (STAR/journal) styling. Plain matplotlib, no
    # mplhep. Only affects frame/ticks/font/sizes -- the mode-differentiation
    # scheme (incl/groomed colors, ^/o markers, pythia/herwig linestyles) is set
    # per-call and is untouched here. ---
    plt.rcParams.update({
        # STIX serif (Times-like), shared by the $...$ math labels and plain text;
        # bundled with matplotlib (no usetex, no font install).
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        # base sizes; the code's named relative sizes (x-large, small, ...) scale
        # off font.size, tick numbers take [xy]tick.labelsize.
        "font.size": 16,
        "axes.titlesize": 16,
        "axes.labelsize": 20,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        # HEP ticks: inward, on all four sides, with minor ticks.
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 7,
        "ytick.major.size": 7,
        "xtick.minor.size": 4,
        "ytick.minor.size": 4,
        "xtick.major.width": 1.1,
        "ytick.major.width": 1.1,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        # heavier frame, ROOT/journal look.
        "axes.linewidth": 1.2,
        # keep white background for the saved PDFs.
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    })

    import torch

    from systematics import SysVar, get_jet_pt_bins
    from config import load_config

    _cfg_setup = load_config()
    feature_mode = _cfg_setup["feature_mode"]

    common_vars = (
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

    var_xlabel = {
        "m": r"$M_{jet}$ (GeV)",
        "sd_m": r"$M_{jet, g}$ (GeV)",
        "sd_dR": r"$\Delta R_{g}$",
        "sd_symmetry": r"$z_{g}$",
        "ch_ang_k1_b0.5": r"$\lambda^{\kappa = 1}_{\beta = 0.5}$ (LHA)",
        "ch_ang_k1_b1": r"$\lambda^{\kappa = 1}_{\beta = 1}$ (girth)",
        "ch_ang_k1_b2": r"$\lambda^{\kappa = 1}_{\beta = 2}$ (thrust)",
        "ch_ang_k2_b0": r"$\lambda^{\kappa = 2}_{\beta = 0}$ ($(p_T^D)^2$)",
        "sd_ch_ang_k1_b0.5": r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 0.5}$ (LHA, SD)",
        "sd_ch_ang_k1_b1": r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 1}$ (girth, SD)",
        "sd_ch_ang_k1_b2": r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 2}$ (thrust, SD)",
        "sd_ch_ang_k2_b0": r"$\lambda^{\kappa = 2, \rm SD}_{\beta = 0}$ ($(p_T^D)^2$, SD)",
    }

    var_hist_ylabel = {
        "pt": r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{dp_{\rm T, jet}}\,(\mathrm{GeV}/c)^{-1}$",
        "ch_ang_k1_b0.5": r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 0.5}}$ ",
        "ch_ang_k1_b1": r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 1}}$",
        "ch_ang_k1_b2": r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 2}}$",
        "ch_ang_k2_b0": r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 2}_{\beta = 0}}$",
    }

    var_prof_ylabel = {
        "ch_ang_k1_b0.5": r"$\langle\lambda^{\kappa = 1}_{\beta = 0.5}\rangle$ ",
        "ch_ang_k1_b1": r"$\langle\lambda^{\kappa = 1}_{\beta = 1}\rangle$",
        "ch_ang_k1_b2": r"$\langle\lambda^{\kappa = 1}_{\beta = 2}\rangle$",
        "ch_ang_k2_b0": r"$\langle\lambda^{\kappa = 2}_{\beta = 0}\rangle$",
    }

    var_xlim = {
        "pt": (10.0, 70.0),
        "m": (1.0, 10.0),
        "sd_m": (1.0, 10.0),
        "sd_dR": (0.05, 0.4),
        "sd_symmetry": (0.18, 0.52),
        "ch_ang_k1_b0.5": (0.0, 0.72),
        "ch_ang_k1_b1": (0.0, 0.62),
        "ch_ang_k1_b2": (0.0, 0.42),
        "ch_ang_k2_b0": (0.0, 0.65),
        "sd_ch_ang_k1_b0.5": (0.0, 0.72),
        "sd_ch_ang_k1_b1": (0.0, 0.62),
        "sd_ch_ang_k1_b2": (0.0, 0.42),
        "sd_ch_ang_k2_b0": (0.0, 0.65),
    }

    # ----------------------------------------------------------------------- #
    # Tunable y-ranges (set by visual feedback for the STAR preliminary).
    #
    # Each entry is keyed by variable name and may be EITHER:
    #   * a single (lo, hi)              -> shared across every jet-pT panel
    #   * {ijpt: (lo, hi), "default": (lo, hi)}  -> per jet-pT-bin override
    # `ijpt` is the *plotted* panel index (0 = the leftmost panel after
    # `jpt_bins_to_omit`). A variable absent from a dict keeps the old behaviour
    # (main panels autoscale; ratio panels use (0.5, 1.5)). When a variable has a
    # per-pT dict, that figure drops `sharey="row"` so each panel scales on its
    # own (see `sharey_for`).
    # ----------------------------------------------------------------------- #
    var_ylim: dict = {}  # main distribution panel; {} -> autoscale
    var_prof_ylim: dict = {
        "ch_ang_k1_b0.5": (0.15, 0.5),
        "ch_ang_k1_b2": (0, 0.35),
    }  # profile (<lambda> vs x) main panel
    var_ratio_ylim: dict = {}  # MC/Data ratio panels; fallback (0.5, 1.5)
    var_prof_ratio_ylim: dict = {
        "ch_ang_k1_b0.5": (0.8, 1.2),
        "ch_ang_k1_b1": (0.8, 1.2),
        "ch_ang_k1_b2": (0.8, 1.2),
        # "ch_ang_k2_b0": (0.8, 1.2),
    }  # profile MC/Data ratio panels; fallback (0.5, 1.5)
    var_logy: set = set(("ch_ang_k1_b2",))  # variables drawn with a log-y main panel

    prefix_dir = Path("./outputs/histograms")

    mc_labels = ("pythia6", "pythia8", "herwig7")
    mc_hist_styles = {
        "pythia6": {"linestyle": "dotted"},
        # pythia8 was "dashdot", which reads as near-solid at ratio-panel scale
        # and blended with herwig7's solid line. Use an open long-dash pattern so
        # the three MC curves (dotted / long-dash / solid) stay distinguishable.
        "pythia8": {"linestyle": (0, (3, 1.5))},
        "herwig7": {"linestyle": "solid"},
    }
    mc_proxy_handles = {
        mc: Line2D([], [], color="black", linewidth=2, **mc_hist_styles[mc])
        for mc in mc_labels
    }
    plot_ratio_sys_err = False
    # Master switch for the data systematic-uncertainty overlays (main-panel
    # error boxes + gray ratio band). Set False to suppress them entirely — e.g.
    # right after a binning change, when outputs/histograms/sys_errors/ is stale
    # and must be regenerated with systematics.py before the bands are valid.
    plot_sys_err = True
    jpt_bins_to_omit = (0,)

    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols = len(jpt_bins) - 1 - len(jpt_bins_to_omit)
    fig_scale = 6
    figsize = (num_cols * fig_scale, fig_scale * 2)


@app.function
def resolve_ylim(spec, ijpt, default=None):
    """Pick a (lo, hi) from a y-range spec for plotted panel ``ijpt``.

    ``spec`` is ``None`` | ``(lo, hi)`` | ``{ijpt: (lo, hi), "default": (lo, hi)}``.
    Returns ``default`` when nothing matches.
    """
    if spec is None:
        return default
    if isinstance(spec, dict):
        val = spec.get(ijpt)
        if val is None:
            val = spec.get("default")
        return val if val is not None else default
    return spec


@app.function
def has_perpt(spec):
    """True if ``spec`` carries per-jet-pT (integer-keyed) overrides."""
    return isinstance(spec, dict) and any(isinstance(k, int) for k in spec)


@app.function
def sharey_for(*specs):
    """``"row"`` (shared y per row) unless any spec has per-pT overrides, in
    which case each panel needs its own y-axis -> ``False``."""
    return False if any(has_perpt(s) for s in specs) else "row"


@app.function
def add_star_preliminary(top_axes, x=0.03, y=0.97):
    """STAR Preliminary watermark + collision-system annotation, spread one item
    per top-row panel (axes-fraction coords, top-left). Splitting the strings
    across panels and keeping each short prevents them from reaching the centered
    jet-pT column labels."""
    # --- old: both lines crammed into panel 0 via figure coords (overlapped the
    # centered jet-pT label) ---
    # fig.text(x, y, "STAR Preliminary", fontsize="xx-large", fontweight="bold",
    #          ha="left", va="top")
    # fig.text(x, y - 0.03, r"$p$+$p$  $\sqrt{s}=200$ GeV   anti-$k_{T}$  $R=0.4$",
    #          fontsize="large", ha="left", va="top")
    items = [
        ("STAR Preliminary", {"fontsize": "large", "fontweight": "bold"}),
        (r"$p$+$p$  $\sqrt{s}=200$ GeV", {"fontsize": "small"}),
        (r"anti-$k_{T}$  $R=0.4$", {"fontsize": "small"}),
    ]
    top_axes = list(top_axes)
    n = len(top_axes)
    for i, (text, kw) in enumerate(items):
        if n >= len(items):
            # one item per panel, top-left
            ax = top_axes[i]
            ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", **kw)
        else:
            # fewer panels than items: stack leftovers on panel 0 at descending y
            ax = top_axes[0]
            ax.text(
                x, y - 0.07 * i, text, transform=ax.transAxes,
                ha="left", va="top", **kw,
            )


@app.function
def prune_ratio_panel_yticks(axs):
    """Drop the topmost y-tick label of each stacked sub-panel (rows 1.., the
    always-linear MC/Data ratio panels) so it doesn't crowd the bottom y-tick label
    of the panel above at the shared (hspace=0) boundary. Row 0 — the distribution
    panel, which may be log-y — is left untouched (only the ratio panels are
    pruned, which are always linear). steps=... mirrors matplotlib's AutoLocator so
    the tick selection is unchanged apart from the removed top tick."""
    for _r in range(1, axs.shape[0]):
        for _ax in np.atleast_1d(axs[_r]):
            _ax.yaxis.set_major_locator(
                MaxNLocator(nbins="auto", steps=[1, 2, 2.5, 5, 10], prune="upper")
            )


@app.function
def plot_data_points(ax, plot_type, hdict, **kwargs):
    hdict["bin_count"] = hdict["bin_count"].nan_to_num_(nan=0, posinf=0, neginf=0)
    hdict["bin_count_err"] = hdict["bin_count_err"].nan_to_num_(
        nan=0, posinf=0, neginf=0
    )
    if "bin_count_std" in hdict:
        hdict["bin_count_std"] = hdict["bin_count_std"].nan_to_num_(
            nan=0, posinf=0, neginf=0
        )

    _step_mode = False
    _ls = None
    if plot_type == "errorbar":
        # MC/Data ratio panels pass a real linestyle (dotted/dashdot/solid via
        # mc_hist_styles); the top-panel data errorbars instead use
        # linestyle="none" (marker mode). When a real linestyle is present, draw
        # the connecting line as a step (steps-mid -> vertical transitions at the
        # bin edges) and carry the same dash pattern onto the error bars below.
        _ls = kwargs.get("linestyle", kwargs.get("ls"))
        _step_mode = _ls is not None and _ls != "none"
        # x-error bars (bin-width whiskers) only on the non-ratio panels; the
        # stepped ratio panels already span each bin via steps-mid, so the xerr
        # there is redundant/cluttering -> omit it.
        if not _step_mode:
            kwargs["xerr"] = hdict["half_bin_width"]
        kwargs["yerr"] = (hdict.get("bin_count_std", hdict["bin_count_err"]),)
        if _step_mode:
            kwargs.setdefault("drawstyle", "steps-mid")

    ax_arts = getattr(ax, plot_type)(
        hdict["bin_center"],
        hdict["bin_count"],
        **kwargs,
    )

    if _step_mode:
        # ax.errorbar -> (line, caplines, barlinecols); barlinecols are the
        # vertical/horizontal error-bar LineCollections. Apply the same linestyle
        # so the error bars match the step line's dash pattern.
        for _barlinecol in ax_arts[2]:
            _barlinecol.set_linestyle(_ls)

    if plot_type == "plot":
        return ax_arts[0]
    return ax_arts


@app.function
def plot_error_bars(ax, hdict, sys_err, **kwargs):
    bin_edges = hdict["bin_center"] - hdict["half_bin_width"]
    last_bin_edge = (hdict["bin_center"][-1] + hdict["half_bin_width"][-1]).unsqueeze_(
        0
    )
    bin_edges = torch.concatenate((bin_edges, last_bin_edge))

    sys_err.nan_to_num_(nan=0, posinf=0, neginf=0)
    return ax.stairs(
        hdict["bin_count"] + sys_err,
        bin_edges,
        baseline=(hdict["bin_count"] - sys_err).numpy(),
        **kwargs,
    )


@app.function
def plot_hist(
    ax,
    plot_type,
    hdict,
    sys_err_hdict=None,
    points_kwargs=None,
    errbar_kwargs=None,
    **kwargs,
):
    points_kwargs = points_kwargs or {}
    points_artist = plot_data_points(
        ax,
        plot_type,
        hdict,
        **points_kwargs,
        **kwargs,
    )
    if sys_err_hdict is None or not plot_sys_err:
        return points_artist

    # Guard against a stale systematics snapshot whose binning predates the
    # current (per-pT) histogram binning. The sys-error files under
    # outputs/histograms/sys_errors/ are produced by systematics.py and must be
    # regenerated after a binning change; until then skip the band rather than
    # crash on the length mismatch.
    if sys_err_hdict["total_sys"].shape[0] != hdict["bin_count"].shape[0]:
        print(
            "  [warn] sys-error bin count "
            f"{tuple(sys_err_hdict['total_sys'].shape)} != hist "
            f"{tuple(hdict['bin_count'].shape)}; skipping sys band "
            "(re-run systematics.py for this binning)"
        )
        return points_artist

    errbar_kwargs = errbar_kwargs or {}
    errband_artist = plot_error_bars(
        ax,
        hdict,
        sys_err_hdict["total_sys"],
        **errbar_kwargs,
        **kwargs,
    )

    return (errband_artist, points_artist)


@app.function
def plot_profile_single(
    ax,
    plot_type,
    file_path,
    sys_err_path=None,
    label="MC",
    **kwargs,
):
    hdict = torch.load(file_path, mmap=True)
    if sys_err_path is not None:
        points_label: str = rf"$\langle\lambda^{label}\rangle\pm\delta_{{sys.}}(\langle \lambda^{label} \rangle)$"
        # --- old: single hardcoded star marker for both incl and groomed ---
        # points_kwargs = dict(
        #     linestyle="none",
        #     marker="*",
        #     markersize=10,
        #     markeredgecolor="white",
        # )
        # --- new: match the histogram marker scheme (caller picks the marker:
        # "^" for incl, "o" for groomed), same defaults as plot_hist_single ---
        marker = kwargs.pop("marker", "o")
        markersize = kwargs.pop("markersize", 5)
        markeredgecolor = kwargs.pop("markeredgecolor", "white")
        points_kwargs = dict(
            linestyle="none",
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
        )
        sys_err_dict = torch.load(sys_err_path, mmap=True)
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_label: str = rf"$\langle\lambda^{label}\rangle$"
        # Respect a caller-supplied linewidth (see plot_hist_single note); pop
        # it so it isn't forwarded twice into plot_data_points.
        points_kwargs = dict(linewidth=kwargs.pop("linewidth", 2))
        sys_err_dict = None
        errbar_kwargs = None

    artists = {}
    artists[points_label] = plot_hist(
        ax,
        plot_type,
        hdict,
        sys_err_dict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        **kwargs,
    )
    return artists


@app.function
def plot_profile(
    var_name,
    x_var_name,
    save_figs=True,
    plot_mc=True,
):
    has_incl = var_name != x_var_name
    n_rows = 3 if has_incl else 2
    height_ratios = [3, 1, 1] if has_incl else [3, 1]
    incl_row = 1 if has_incl else None
    sd_row = 2 if has_incl else 1

    _prof_spec = var_prof_ylim.get(var_name)
    _ratio_spec = var_prof_ratio_ylim.get(var_name)

    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(
        n_rows,
        num_cols,
        height_ratios=height_ratios,
        sharey=sharey_for(_prof_spec, _ratio_spec),
        sharex="col",
        gridspec_kw=dict(hspace=0, wspace=0, right=0.9, left=0.2),
    )

    ax_art_map = {}

    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue
        ijpt += 1
        top_ax = axs[0, ijpt]
        ax_arts = plot_profile_single(
            top_ax,
            "errorbar",
            file_path=prefix_dir
            / str(SysVar.NONE)
            / feature_mode
            / var_name
            / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
            sys_err_path=prefix_dir
            / "sys_errors"
            / feature_mode
            / var_name
            / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
            color="red",
            marker="o",  # groomed: circle (matches histogram scheme)
            label="{SD}",
        )
        if has_incl:
            ax_arts.update(
                plot_profile_single(
                    top_ax,
                    "errorbar",
                    file_path=prefix_dir
                    / str(SysVar.NONE)
                    / feature_mode
                    / var_name
                    / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    sys_err_path=prefix_dir
                    / "sys_errors"
                    / feature_mode
                    / var_name
                    / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    color="blue",
                    marker="^",  # incl: triangle (matches histogram scheme)
                    label="{incl.}",
                )
            )

        if plot_mc:
            for mc in mc_labels:
                plot_profile_single(
                    top_ax,
                    "plot",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    label="".join(("{", mc, "}")),
                    color="red",
                    **(mc_hist_styles[mc]),
                )

                if has_incl:
                    plot_profile_single(
                        top_ax,
                        "plot",
                        file_path=prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                        color="blue",
                        **(mc_hist_styles[mc]),
                    )

                plot_hist_single(
                    axs[sd_row, ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"ratio_prof_sd_vs_{x_var_name}_data_vs_{mc}_jpt{ijpt_true}.pt",
                    sys_err_path=make_ratio_sys_hdict(
                        prefix_dir
                        / "sys_errors"
                        / feature_mode
                        / var_name
                        / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
                        prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    )
                    if plot_ratio_sys_err
                    else None,
                    color="red",
                    label=mc,
                    **(mc_hist_styles[mc]),
                )

                if has_incl:
                    plot_hist_single(
                        axs[incl_row, ijpt],
                        "errorbar",
                        file_path=prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"ratio_prof_incl_vs_{x_var_name}_data_vs_{mc}_jpt{ijpt_true}.pt",
                        sys_err_path=make_ratio_sys_hdict(
                            prefix_dir
                            / "sys_errors"
                            / feature_mode
                            / var_name
                            / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                            prefix_dir
                            / mc
                            / feature_mode
                            / var_name
                            / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                        )
                        if plot_ratio_sys_err
                        else None,
                        color="blue",
                        label=mc,
                        **(mc_hist_styles[mc]),
                    )

        if ijpt == 0:
            ax_art_map.update(ax_arts)

        # --- old: jet-pT range as panel title (above the axes) ---
        # top_ax.set_title(
        #     rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        # )
        # --- new: jet-pT range inside the plot body (top-center) ---
        top_ax.text(
            0.5,
            0.90,
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
            transform=top_ax.transAxes,
            ha="center",
            va="top",
            fontsize="small",
        )
        _prof_ylim = resolve_ylim(_prof_spec, ijpt)
        if x_var_name in {var_name, "sd_dR"}:
            if x_var_name in var_xlim:
                lims = list(var_xlim[x_var_name])
            else:
                lims = [
                    np.min([top_ax.get_xlim(), top_ax.get_ylim()]),
                    np.max([top_ax.get_xlim(), top_ax.get_ylim()]),
                ]
            if x_var_name == "sd_dR":
                # vs ΔR_g: the natural reference is the single-emission angular
                # scaling y = (ΔR_g)^β, not the x=y diagonal. β is read off the
                # y-axis angularity name (..._b<beta>).
                _beta = float(var_name.split("_b")[-1])
                _xs = np.linspace(lims[0], lims[1], 200)
                top_ax.plot(_xs, _xs ** _beta, "--", color="black", alpha=0.3)
            else:
                # vs itself: x=y diagonal.
                top_ax.plot(lims, lims, "--", color="black", alpha=0.3)
            top_ax.set_xlim(lims)
            top_ax.set_ylim(lims)
        elif x_var_name in var_xlim:
            top_ax.set_xlim(*var_xlim[x_var_name])
        if _prof_ylim is not None:
            top_ax.set_ylim(*_prof_ylim)

        # gray band = sys_err(Data)/Data around unity, per ratio panel
        ratio_band_paths = {
            sd_row: (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
            ),
        }
        if has_incl:
            ratio_band_paths[incl_row] = (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
            )

        _ratio_ylim = resolve_ylim(_ratio_spec, ijpt, default=(0.5, 1.5))
        for _r in range(1, n_rows):
            plot_data_sys_band(axs[_r, ijpt], *ratio_band_paths[_r])
            axs[_r, ijpt].axhline(
                y=1, linewidth=2, color="black", linestyle="--", alpha=0.3
            )
            axs[_r, ijpt].set_ylim(*_ratio_ylim)
        axs[n_rows - 1, ijpt].set_xlabel(var_xlabel[x_var_name], fontsize="x-large")
        axs[n_rows - 1, ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[0, 0].set_ylabel(var_prof_ylabel[var_name], fontsize="x-large")
    if has_incl:
        axs[incl_row, 0].set_ylabel(
            r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large"
        )
    axs[sd_row, 0].set_ylabel(
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large"
    )

    if plot_mc:
        ax_art_map.update(mc_proxy_handles)
    # --- old: figure-level legend below the panels ---
    # fig.legend(
    #     list(ax_art_map.values()),
    #     list(ax_art_map.keys()),
    #     frameon=False,
    #     fontsize="large",
    #     loc="upper center",
    #     ncol=len(ax_art_map),
    #     bbox_to_anchor=(0.5, 0.0),
    # )
    # In-panel legend in the rightmost top panel. The empty corner depends on the
    # profile shape: rising profiles (sd_dR) leave the lower-right clear; falling
    # ones (z_g / sd_symmetry) leave the upper-right clear but it must be dropped
    # below the centered jet-pT label (y~0.83). Top-left is reserved for the
    # add_star_preliminary system label, so never use "upper left".
    _prof_leg = {
        "sd_dR": ("lower right", None),
        "sd_m": ("lower right", None),
        "sd_symmetry": ("upper right", (0.99, 0.83)),
    }.get(x_var_name, ("upper right", (0.99, 0.83)))
    axs[0, -1].legend(
        list(ax_art_map.values()),
        list(ax_art_map.keys()),
        frameon=False,
        fontsize="x-small",
        loc=_prof_leg[0],
        bbox_to_anchor=_prof_leg[1],
    )

    prune_ratio_panel_yticks(axs)
    add_star_preliminary(axs[0])

    if save_figs:
        fig_save_dir = prefix_dir / "plots" / feature_mode / var_name
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_save_dir / f"prof_ang_vs_{x_var_name}.pdf"
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


@app.function
def make_ratio_sys_hdict(data_sys_path, mc_hist_path):
    data_sys = torch.load(data_sys_path, mmap=True)
    mc_hist = torch.load(mc_hist_path, mmap=True)
    mc_count = mc_hist["bin_count"]
    tiny = torch.finfo(mc_count.dtype).tiny
    total_sys = data_sys["total_sys"].clone()
    total_sys.div_(mc_count.clone().clamp_(min=tiny))
    total_sys.masked_fill_(mc_count == 0, 0)
    total_sys.nan_to_num_(nan=0, posinf=0, neginf=0)
    return {"total_sys": total_sys}


@app.function
def plot_data_sys_band(ax, data_hist_path, data_sys_path, **kwargs):
    """Gray fractional-systematic band around y=1 for a ratio panel.

    Draws 1 +/- sys_err(Data)/Data as a filled stairs band centered on unity,
    i.e. the data's own systematic uncertainty propagated into the MC/Data ratio.
    The fractional band is identical for MC/Data and Data/MC (MC is exact here),
    so unity-centered |sys_err(Data)/Data| is unchanged by the numerator swap.
    """
    if not plot_sys_err:
        return None

    hdict = torch.load(data_hist_path, mmap=True)
    sdict = torch.load(data_sys_path, mmap=True)

    count = hdict["bin_count"].clone().nan_to_num_(nan=0, posinf=0, neginf=0)
    # Skip a stale systematics snapshot whose binning predates the current
    # per-pT binning (re-run systematics.py to refresh the band).
    if sdict["total_sys"].shape[0] != count.shape[0]:
        print(
            "  [warn] sys band bin count "
            f"{tuple(sdict['total_sys'].shape)} != hist {tuple(count.shape)}; "
            "skipping (re-run systematics.py for this binning)"
        )
        return None
    tiny = torch.finfo(count.dtype).tiny
    frac = sdict["total_sys"].clone()
    frac.div_(count.clamp(min=tiny))
    frac.masked_fill_(count == 0, 0)
    frac.nan_to_num_(nan=0, posinf=0, neginf=0)

    bin_edges = hdict["bin_center"] - hdict["half_bin_width"]
    last_bin_edge = (hdict["bin_center"][-1] + hdict["half_bin_width"][-1]).unsqueeze(0)
    bin_edges = torch.concatenate((bin_edges, last_bin_edge))

    band_kwargs = dict(fill=True, color="gray", alpha=0.3, linewidth=0, zorder=0)
    band_kwargs.update(kwargs)
    return ax.stairs(
        (1 + frac).numpy(),
        bin_edges.numpy(),
        baseline=(1 - frac).numpy(),
        **band_kwargs,
    )


@app.function
def plot_hist_single(
    ax,
    plot_type,
    file_path,
    sys_err_path=None,
    label="hist",
    **kwargs,
):

    hdict = torch.load(file_path, mmap=True)
    line_style_mode = "linestyle" in kwargs or "ls" in kwargs

    if line_style_mode:
        # Respect a caller-supplied linewidth (e.g. per-MC width bumps to
        # compensate for dotted/dashed reading thinner than solid); pop it out
        # of kwargs so it isn't also forwarded into plot_data_points (which
        # would raise "multiple values for keyword argument 'linewidth'").
        points_kwargs = dict(linewidth=kwargs.pop("linewidth", 2))
    else:
        marker = kwargs.pop("marker", "o")
        markersize = kwargs.pop("markersize", 5)
        markeredgecolor = kwargs.pop("markeredgecolor", "white")
        points_kwargs = dict(
            linestyle="none",
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
        )

    if sys_err_path is not None:
        if isinstance(sys_err_path, dict):
            sys_err_hdict = sys_err_path
        else:
            sys_err_hdict = torch.load(sys_err_path, mmap=True)
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        sys_err_hdict = None
        errbar_kwargs = None

    ax_arts = plot_hist(
        ax,
        plot_type,
        hdict,
        sys_err_hdict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        **kwargs,
    )

    return {
        rf"${label} \pm \delta_{{sys}}$"
        if sys_err_hdict is not None
        else label: ax_arts,
    }


@app.function
def plot_hists(
    var_name,
    save_figs=True,
    plot_mc=True,
):
    has_sd_pair = f"sd_{var_name}" in common_vars
    n_rows = 3 if has_sd_pair else 2
    height_ratios = [3, 1, 1] if has_sd_pair else [3, 1]
    incl_ratio_row = 1
    sd_ratio_row = 2 if has_sd_pair else None

    _ylim_spec = var_ylim.get(var_name)
    _ratio_spec = var_ratio_ylim.get(var_name)

    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(
        n_rows,
        num_cols,
        height_ratios=height_ratios,
        sharey=sharey_for(_ylim_spec, _ratio_spec),
        sharex="col",
        gridspec_kw=dict(
            hspace=0,
            wspace=0,
            right=0.9,
            left=0.2,
            top=0.9,
            bottom=0.2,
        ),
    )

    hist_ax_arts = {}
    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue

        ijpt += 1
        top_ax = axs[0, ijpt]

        ax_arts = plot_hist_single(
            top_ax,
            "errorbar",
            file_path=prefix_dir
            / str(SysVar.NONE)
            / feature_mode
            / var_name
            / f"hist_jpt{ijpt_true}.pt",
            sys_err_path=prefix_dir
            / "sys_errors"
            / feature_mode
            / var_name
            / f"hist_jpt{ijpt_true}.pt",
            label="inclusive" if has_sd_pair else "data",
            color="blue",
        )

        if plot_mc:
            for mc in mc_labels:
                plot_hist_single(
                    top_ax,
                    "plot",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"hist_jpt{ijpt_true}.pt",
                    sys_err_path=None,
                    label=mc if not has_sd_pair else f"{mc} (incl.)",
                    color="blue",
                    **(mc_hist_styles[mc]),
                )

                plot_hist_single(
                    axs[incl_ratio_row, ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"ratio_hist_data_vs_{mc}_jpt{ijpt_true}.pt",
                    sys_err_path=make_ratio_sys_hdict(
                        prefix_dir
                        / "sys_errors"
                        / feature_mode
                        / var_name
                        / f"hist_jpt{ijpt_true}.pt",
                        prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"hist_jpt{ijpt_true}.pt",
                    )
                    if plot_ratio_sys_err
                    else None,
                    color="blue",
                    label=mc,
                    **(mc_hist_styles[mc]),
                )

        if has_sd_pair:
            ax_arts.update(
                plot_hist_single(
                    top_ax,
                    "errorbar",
                    file_path=prefix_dir
                    / str(SysVar.NONE)
                    / feature_mode
                    / f"sd_{var_name}"
                    / f"hist_jpt{ijpt_true}.pt",
                    sys_err_path=prefix_dir
                    / "sys_errors"
                    / feature_mode
                    / f"sd_{var_name}"
                    / f"hist_jpt{ijpt_true}.pt",
                    label="groomed",
                    color="red",
                )
            )

            if plot_mc:
                for mc in mc_labels:
                    plot_hist_single(
                        top_ax,
                        "plot",
                        file_path=prefix_dir
                        / mc
                        / feature_mode
                        / f"sd_{var_name}"
                        / f"hist_jpt{ijpt_true}.pt",
                        sys_err_path=None,
                        label=f"{mc} (groomed)",
                        color="red",
                        **(mc_hist_styles[mc]),
                    )

                    plot_hist_single(
                        axs[sd_ratio_row, ijpt],
                        "errorbar",
                        file_path=prefix_dir
                        / mc
                        / feature_mode
                        / f"sd_{var_name}"
                        / f"ratio_hist_data_vs_{mc}_jpt{ijpt_true}.pt",
                        sys_err_path=make_ratio_sys_hdict(
                            prefix_dir
                            / "sys_errors"
                            / feature_mode
                            / f"sd_{var_name}"
                            / f"hist_jpt{ijpt_true}.pt",
                            prefix_dir
                            / mc
                            / feature_mode
                            / f"sd_{var_name}"
                            / f"hist_jpt{ijpt_true}.pt",
                        )
                        if plot_ratio_sys_err
                        else None,
                        color="red",
                        label=mc,
                        **(mc_hist_styles[mc]),
                    )

        if ijpt == 2:
            hist_ax_arts.update(ax_arts)

        # --- old: jet-pT range as panel title (above the axes) ---
        # top_ax.set_title(
        #     rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        # )
        # --- new: jet-pT range inside the plot body (top-center) ---
        top_ax.text(
            0.5,
            0.90,
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
            transform=top_ax.transAxes,
            ha="center",
            va="top",
            fontsize="small",
        )

        # gray band = sys_err(Data)/Data around unity, per ratio panel
        ratio_band_paths = {
            incl_ratio_row: (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"hist_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"hist_jpt{ijpt_true}.pt",
            ),
        }
        if has_sd_pair:
            ratio_band_paths[sd_ratio_row] = (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / f"sd_{var_name}"
                / f"hist_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / f"sd_{var_name}"
                / f"hist_jpt{ijpt_true}.pt",
            )

        _ratio_ylim = resolve_ylim(_ratio_spec, ijpt, default=(0.5, 1.5))
        for _r in range(1, n_rows):
            plot_data_sys_band(axs[_r, ijpt], *ratio_band_paths[_r])
            axs[_r, ijpt].axhline(
                y=1, linewidth=2, color="black", linestyle="--", alpha=0.3
            )
            axs[_r, ijpt].set_ylim(*_ratio_ylim)
        axs[n_rows - 1, ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[n_rows - 1, ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        if var_name in var_xlim:
            top_ax.set_xlim(*var_xlim[var_name])
        if var_name in var_logy:
            top_ax.set_yscale("log")
        _ylim = resolve_ylim(_ylim_spec, ijpt)
        if _ylim is not None:
            top_ax.set_ylim(*_ylim)

    if plot_mc:
        hist_ax_arts.update(mc_proxy_handles)
    # --- old: figure-level legend below the panels ---
    # fig.legend(
    #     list(hist_ax_arts.values()),
    #     list(hist_ax_arts.keys()),
    #     frameon=False,
    #     fontsize="large",
    #     loc="upper center",
    #     ncol=len(hist_ax_arts),
    #     bbox_to_anchor=(0.5, 0.0),
    # )
    # In-panel legend in the rightmost top panel; distributions fall off to the
    # right so the right side is clear (top-left holds the system label). Drop it
    # just below the centered jet-pT label (y~0.83) so it clears that annotation.
    axs[0, -1].legend(
        list(hist_ax_arts.values()),
        list(hist_ax_arts.keys()),
        frameon=False,
        fontsize="small",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.83),
    )

    prune_ratio_panel_yticks(axs)
    add_star_preliminary(axs[0])

    if has_sd_pair:
        axs[incl_ratio_row, 0].set_ylabel(
            r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large"
        )
        axs[sd_ratio_row, 0].set_ylabel(
            r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large"
        )
    else:
        axs[incl_ratio_row, 0].set_ylabel(
            r"$\frac{\mathrm{MC}}{\mathrm{Data}}$", fontsize="x-large"
        )

    if save_figs:
        fig_save_dir = prefix_dir / "plots" / feature_mode / var_name
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_save_dir / f"hist_{var_name}.pdf"
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


@app.function
def plot_ratio(
    var_name,
    save_figs=True,
    plot_mc=True,
):
    _ylim_spec = var_ylim.get(var_name)
    _ratio_spec = var_ratio_ylim.get(var_name)

    fig = plt.figure(figsize=figsize)
    axs = fig.subplots(
        3,
        num_cols,
        height_ratios=[3, 1, 1],
        sharey=sharey_for(_ylim_spec, _ratio_spec),
        sharex="col",
        gridspec_kw=dict(
            hspace=0,
            wspace=0,
            right=0.9,
            left=0.2,
            top=0.9,
            bottom=0.2,
        ),
    )

    hist_ax_arts = {}

    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue
        ijpt += 1

        ax_arts = {}
        ax_arts.update(
            plot_hist_single(
                axs[0, ijpt],
                "errorbar",
                file_path=prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"hist_sd_ang_jpt{ijpt_true}.pt",
                sys_err_path=prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"hist_sd_ang_jpt{ijpt_true}.pt",
                color="red",
                label="groomed",
            )
        )
        ax_arts.update(
            plot_hist_single(
                axs[0, ijpt],
                "errorbar",
                file_path=prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"hist_ang_jpt{ijpt_true}.pt",
                sys_err_path=prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"hist_ang_jpt{ijpt_true}.pt",
                color="blue",
                marker="^",
                label="incl.",
            )
        )

        if plot_mc:
            for mc in mc_labels:
                plot_hist_single(
                    axs[0, ijpt],
                    "plot",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"hist_sd_ang_jpt{ijpt_true}.pt",
                    sys_err_path=None,
                    color="red",
                    label=f"{mc} (groomed)",
                    **(mc_hist_styles[mc]),
                )
                plot_hist_single(
                    axs[0, ijpt],
                    "plot",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"hist_ang_jpt{ijpt_true}.pt",
                    sys_err_path=None,
                    color="blue",
                    label=f"{mc} (incl.)",
                    **(mc_hist_styles[mc]),
                )

                plot_hist_single(
                    axs[1, ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"ratio_ang_data_vs_{mc}_jpt{ijpt_true}.pt",
                    sys_err_path=make_ratio_sys_hdict(
                        prefix_dir
                        / "sys_errors"
                        / feature_mode
                        / var_name
                        / f"hist_ang_jpt{ijpt_true}.pt",
                        prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"hist_ang_jpt{ijpt_true}.pt",
                    )
                    if plot_ratio_sys_err
                    else None,
                    color="blue",
                    label=mc,
                    **(mc_hist_styles[mc]),
                )
                plot_hist_single(
                    axs[2, ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / mc
                    / feature_mode
                    / var_name
                    / f"ratio_sd_ang_data_vs_{mc}_jpt{ijpt_true}.pt",
                    sys_err_path=make_ratio_sys_hdict(
                        prefix_dir
                        / "sys_errors"
                        / feature_mode
                        / var_name
                        / f"hist_sd_ang_jpt{ijpt_true}.pt",
                        prefix_dir
                        / mc
                        / feature_mode
                        / var_name
                        / f"hist_sd_ang_jpt{ijpt_true}.pt",
                    )
                    if plot_ratio_sys_err
                    else None,
                    color="red",
                    label=mc,
                    **(mc_hist_styles[mc]),
                )

        if ijpt == 2:
            hist_ax_arts.update(ax_arts)

        # --- old: jet-pT range as panel title (above the axes) ---
        # axs[0, ijpt].set_title(
        #     rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        # )
        # --- new: jet-pT range inside the plot body (top-center) ---
        axs[0, ijpt].text(
            0.5,
            0.90,
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
            transform=axs[0, ijpt].transAxes,
            ha="center",
            va="top",
            fontsize="small",
        )

        # gray band = sys_err(Data)/Data around unity, per ratio panel
        ratio_band_paths = {
            1: (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"hist_ang_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"hist_ang_jpt{ijpt_true}.pt",
            ),
            2: (
                prefix_dir
                / str(SysVar.NONE)
                / feature_mode
                / var_name
                / f"hist_sd_ang_jpt{ijpt_true}.pt",
                prefix_dir
                / "sys_errors"
                / feature_mode
                / var_name
                / f"hist_sd_ang_jpt{ijpt_true}.pt",
            ),
        }

        _ratio_ylim = resolve_ylim(_ratio_spec, ijpt, default=(0.5, 1.5))
        for _r in (1, 2):
            plot_data_sys_band(axs[_r, ijpt], *ratio_band_paths[_r])
            axs[_r, ijpt].axhline(
                y=1, linewidth=2, color="black", linestyle="--", alpha=0.3
            )
            axs[_r, ijpt].set_ylim(*_ratio_ylim)
        axs[2, ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[2, ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        if var_name in var_xlim:
            axs[0, ijpt].set_xlim(*var_xlim[var_name])
        if var_name in var_logy:
            axs[0, ijpt].set_yscale("log")
        _ylim = resolve_ylim(_ylim_spec, ijpt)
        if _ylim is not None:
            axs[0, ijpt].set_ylim(*_ylim)

    if plot_mc:
        hist_ax_arts.update(mc_proxy_handles)
    # --- old: figure-level legend below the panels ---
    # fig.legend(
    #     list(hist_ax_arts.values()),
    #     list(hist_ax_arts.keys()),
    #     frameon=False,
    #     fontsize="large",
    #     loc="upper center",
    #     ncol=len(hist_ax_arts),
    #     bbox_to_anchor=(0.5, 0.0),
    # )
    # In-panel legend in the rightmost top panel; distributions fall off to the
    # right so the right side is clear (top-left holds the system label). Drop it
    # just below the centered jet-pT label (y~0.83) so it clears that annotation.
    axs[0, -1].legend(
        list(hist_ax_arts.values()),
        list(hist_ax_arts.keys()),
        frameon=False,
        fontsize="small",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.83),
    )
    prune_ratio_panel_yticks(axs)
    add_star_preliminary(axs[0])
    axs[0, 0].set_ylabel(var_hist_ylabel[var_name], fontsize="xx-large")
    axs[1, 0].set_ylabel(
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large"
    )
    axs[2, 0].set_ylabel(
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large"
    )

    if save_figs:
        fig_save_dir = prefix_dir / "plots" / feature_mode / var_name
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_save_dir / "ratio_incl_vs_hc.pdf"
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


@app.cell
def _():
    save_figs = True
    plot_mc = True
    # if plot_mc:
    #    fig = plt.figure(
    #        figsize=(fig_scale, fig_scale),
    #    )

    #    ax = fig.add_subplot()
    #    plot_hist_single(
    #        ax,
    #        "plot",
    #        "herwig",
    #        "pt",
    #        is_mc=True,
    #        label="HERWIG7",
    #        color="red",
    #        linestyle="solid",
    #    )
    #    ax.set_yscale("log")
    #    fig_save_dir = os.path.join(prefix, "plots", "pt")
    #    os.makedirs(fig_save_dir, exist_ok=True)
    #    fig_save_path = os.path.join(fig_save_dir, "hist.pdf")
    #    fig.savefig(fig_save_path, bbox_inches="tight")

    for var_name in ("m", "sd_dR", "sd_symmetry"):
        plot_hists(var_name, save_figs=save_figs, plot_mc=plot_mc)
    for var_name in angularities:
        plot_ratio(var_name, save_figs=save_figs, plot_mc=plot_mc)
        for x_var_name in common_vars + (var_name,):
            plot_profile(var_name, x_var_name, save_figs=save_figs, plot_mc=plot_mc)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
