import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="full",
    layout_file="layouts/plot_hp2026_prelims.slides.json",
)


@app.cell
def _():
    import json
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
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

    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    # Import the data-differentiation scheme and low-level draw helpers from
    # the deck plotter. Importing plot_physics runs its setup block, which loads
    # config.json, sets STIX-serif rcParams, and defines all module-level dicts.

    from plot_physics import (
        resolve_ylim,
        prune_ratio_panel_yticks,
        plot_data_points,
        plot_error_bars,
        plot_hist,
        plot_profile_single,
        plot_data_sys_band,
        plot_hist_single,
    )
    from systematics import SysVar

    # Override font to sans-serif to match the poster (tikzposter Helvetica /
    # sansmath). This runs AFTER the STIX-serif block set by plot_physics, so
    # sans wins. Helvetica may not be installed; matplotlib falls back to
    # Arial → DejaVu Sans.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "mathtext.fontset": "stixsans",
    })
    # HEP tick / frame settings are already set by plot_physics; they carry over.

    # Center panel = 20–30 GeV/c (bin index 2 in jpt_bins = (10,15,20,30,60))
    CENTER_JPT = 2

    OUT_DIR = Path("outputs/hp2026_poster") / feature_mode

    # Poster-local y-range overrides for identical axes across pT bins
    POSTER_PROF_YLIM: dict = {
        # --- old: data band (~0.13-0.27) squished into the middle third ---
        # "ch_ang_k2_b0": (0.0, 0.55),
        "ch_ang_k2_b0": (0.11, 0.32),
    }
    POSTER_PROF_RATIO_YLIM: dict = {
        "ch_ang_k2_b0": (0.6, 1.2),
    }

    # ----------------------------------------------------------------------- #
    # Shared poster annotation strings + legend styling. Lifted out of the
    # per-figure cells so the content lives in one place; each cell still
    # positions the text itself (the (x, y) and per-call fontsize stay inline).
    # ----------------------------------------------------------------------- #
    PRELIM_TEXT = "STAR Preliminary"
    PRELIM_KW = dict(
        ha="left", va="top", fontsize="medium",
        fontweight="bold", color="red", fontstyle="italic",
    )
    SYSTEM_INFO_TEXT = "\n".join((
        r"$p$+$p$ @ $\sqrt{s}=200$ GeV",
        r"anti-$k_{\rm T}$ full jets, $R=0.4$",
        r"SoftDrop $z_{\rm cut} = 0.2$, $\beta = 0$",
    ))
    KINEMATIC_CUTS_TEXT = "\n".join((
        r"$| \eta_{\rm jet}| + R < 1.0$",
        r"$N^{\rm constit.}_{\rm charged, jet} > 1$",
    ))
    LEGEND_KW = dict(frameon=False, fontsize="small")


    # ----------------------------------------------------------------------- #
    # Reusable legend, defined once. The handle set (sys-box + open marker for the
    # two data series, plus the MC line proxies) is identical in every figure, so
    # build it here and just draw it per-axes:
    #   ax.legend(POSTER_LEGEND_HANDLES, <DIST|PROF>_LEGEND_LABELS, loc=.., **LEGEND_KW)
    # Only the data labels differ between the distribution and profile figures.
    # ----------------------------------------------------------------------- #
    def _data_legend_proxy(color, marker):
        """Axes-independent proxy mirroring a data swatch: a semi-transparent sys
        box behind an open marker with an errorbar whisker."""
        return (
            Patch(facecolor=color, alpha=0.5, edgecolor="none"),
            Line2D(
                [], [], color=color, marker=marker, linestyle="-",
                linewidth=1.2, markersize=5, markerfacecolor=color,
                markeredgecolor="white",
            ),
        )

    POSTER_LEGEND_HANDLES = [
        _data_legend_proxy("red", "o"),
        _data_legend_proxy("blue", "^"),
        *mc_proxy_handles.values(),
    ]
    DIST_LEGEND_LABELS = [
        r"$groomed \pm \delta_{sys}$",
        r"$incl. \pm \delta_{sys}$",
        *mc_proxy_handles.keys(),
    ]
    PROF_LEGEND_LABELS = [
        r"$\langle\lambda^{SD}\rangle\pm\delta_{sys.}(\langle \lambda^{SD} \rangle)$",
        r"$\langle\lambda^{incl.}\rangle\pm\delta_{sys.}(\langle \lambda^{incl.} \rangle)$",
        *mc_proxy_handles.keys(),
    ]
    return (
        CENTER_JPT,
        DIST_LEGEND_LABELS,
        FormatStrFormatter,
        KINEMATIC_CUTS_TEXT,
        LEGEND_KW,
        Line2D,
        OUT_DIR,
        POSTER_LEGEND_HANDLES,
        POSTER_PROF_RATIO_YLIM,
        POSTER_PROF_YLIM,
        PRELIM_KW,
        PRELIM_TEXT,
        PROF_LEGEND_LABELS,
        SYSTEM_INFO_TEXT,
        SysVar,
        angularities,
        feature_mode,
        get_jet_pt_bins,
        jpt_bins,
        json,
        load_config,
        mc_hist_styles,
        mc_labels,
        np,
        plot_data_points,
        plot_data_sys_band,
        plot_hist_single,
        plot_profile_single,
        plt,
        prefix_dir,
        prune_ratio_panel_yticks,
        resolve_ylim,
        torch,
        var_hist_ylabel,
        var_logy,
        var_prof_ratio_ylim,
        var_prof_ylabel,
        var_prof_ylim,
        var_xlabel,
        var_xlim,
    )


@app.cell
def _(
    SysVar,
    feature_mode,
    jpt_bins,
    mc_hist_styles,
    mc_labels,
    np,
    plot_data_sys_band,
    plot_hist_single,
    plot_profile_single,
    prefix_dir,
    prune_ratio_panel_yticks,
):
    # Local-only helper (no plot_physics equivalent). The byte-identical
    # draw/scale helpers (resolve_ylim, prune_ratio_panel_yticks,
    # plot_data_points, plot_error_bars, plot_hist, plot_profile_single,
    # plot_data_sys_band, plot_hist_single) are imported from plot_physics in
    # the Hbol setup cell instead of being inlined here.
    def pt_label(jpt_true):
        """Axis annotation string for a pT bin, e.g. '$20 < p_{T,jet} < 30$ GeV/$c$'."""
        return (
            rf"${jpt_bins[jpt_true]:.0f} < p_{{\rm T,jet}}"
            rf" < {jpt_bins[jpt_true + 1]:.0f}$ GeV/$c$"
        )



    def draw_profile_panels(ax_main, ax_incl, ax_sd, var_name, x_var_name, jpt_true):
        """Draw the SD + inclusive <lambda> data points (with sys), the MC profile curves,
        and the two MC/Data ratio panels with data sys-bands. Returns the {label: handle}
        dict for the legend. Pure data-drawing only -- limits/annotations/legend stay in
        the calling cell."""
        ax_arts = {}
        ax_arts.update(
            plot_profile_single(
                ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"prof_sd_vs_{x_var_name}_jpt{jpt_true}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / var_name / f"prof_sd_vs_{x_var_name}_jpt{jpt_true}.pt",
                color="red", marker="o", label="{SD}",
            )
        )
        ax_arts.update(
            plot_profile_single(
                ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"prof_incl_vs_{x_var_name}_jpt{jpt_true}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / var_name / f"prof_incl_vs_{x_var_name}_jpt{jpt_true}.pt",
                color="blue", marker="^", label="{incl.}",
            )
        )
        for mc in mc_labels:
            plot_profile_single(
                ax_main, "plot",
                file_path=prefix_dir / mc / feature_mode / var_name / f"prof_sd_vs_{x_var_name}_jpt{jpt_true}.pt",
                label="".join(("{", mc, "}")), color="red", **(mc_hist_styles[mc]),
            )
            plot_profile_single(
                ax_main, "plot",
                file_path=prefix_dir / mc / feature_mode / var_name / f"prof_incl_vs_{x_var_name}_jpt{jpt_true}.pt",
                color="blue", **(mc_hist_styles[mc]),
            )
            plot_hist_single(
                ax_sd, "errorbar",
                file_path=prefix_dir / mc / feature_mode / var_name / f"ratio_prof_sd_vs_{x_var_name}_data_vs_{mc}_jpt{jpt_true}.pt",
                sys_err_path=None, color="red", label=mc, **(mc_hist_styles[mc]),
            )
            plot_hist_single(
                ax_incl, "errorbar",
                file_path=prefix_dir / mc / feature_mode / var_name / f"ratio_prof_incl_vs_{x_var_name}_data_vs_{mc}_jpt{jpt_true}.pt",
                sys_err_path=None, color="blue", label=mc, **(mc_hist_styles[mc]),
            )
        for ax_r, pfx, clr in ((ax_incl, "prof_incl_vs", "blue"), (ax_sd, "prof_sd_vs", "red")):
            plot_data_sys_band(
                ax_r,
                prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"{pfx}_{x_var_name}_jpt{jpt_true}.pt",
                prefix_dir / "sys_errors" / feature_mode / var_name / f"{pfx}_{x_var_name}_jpt{jpt_true}.pt",
                color=clr,
            )
            ax_r.axhline(y=1, linewidth=2, color="black", linestyle="--", alpha=0.3)
        return ax_arts


    def draw_dist_panels(ax_main, ax_incl, ax_sd, var_name, jpt_true):
        """Draw the groomed + inclusive data distributions (with sys), the MC curves, and
        the two MC/Data ratio panels (fixed 0.5-1.5 ylim) with data sys-bands. Returns the
        {label: handle} dict for the legend. Pure data-drawing only."""
        ax_arts = {}
        ax_arts.update(
            plot_hist_single(
                ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"hist_sd_ang_jpt{jpt_true}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / var_name / f"hist_sd_ang_jpt{jpt_true}.pt",
                color="red", label="groomed",
            )
        )
        ax_arts.update(
            plot_hist_single(
                ax_main, "errorbar",
                file_path=prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"hist_ang_jpt{jpt_true}.pt",
                sys_err_path=prefix_dir / "sys_errors" / feature_mode / var_name / f"hist_ang_jpt{jpt_true}.pt",
                color="blue", marker="^", label="incl.",
            )
        )
        for mc in mc_labels:
            plot_hist_single(
                ax_main, "plot",
                file_path=prefix_dir / mc / feature_mode / var_name / f"hist_sd_ang_jpt{jpt_true}.pt",
                color="red", label=f"{mc} (groomed)", **(mc_hist_styles[mc]),
            )
            plot_hist_single(
                ax_main, "plot",
                file_path=prefix_dir / mc / feature_mode / var_name / f"hist_ang_jpt{jpt_true}.pt",
                color="blue", label=f"{mc} (incl.)", **(mc_hist_styles[mc]),
            )
            plot_hist_single(
                ax_incl, "errorbar",
                file_path=prefix_dir / mc / feature_mode / var_name / f"ratio_ang_data_vs_{mc}_jpt{jpt_true}.pt",
                sys_err_path=None, color="blue", label=mc, **(mc_hist_styles[mc]),
            )
            plot_hist_single(
                ax_sd, "errorbar",
                file_path=prefix_dir / mc / feature_mode / var_name / f"ratio_sd_ang_data_vs_{mc}_jpt{jpt_true}.pt",
                sys_err_path=None, color="red", label=mc, **(mc_hist_styles[mc]),
            )
        for ax_r, stem, clr in ((ax_incl, "hist_ang", "blue"), (ax_sd, "hist_sd_ang", "red")):
            plot_data_sys_band(
                ax_r,
                prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"{stem}_jpt{jpt_true}.pt",
                prefix_dir / "sys_errors" / feature_mode / var_name / f"{stem}_jpt{jpt_true}.pt",
                color=clr,
            )
            ax_r.axhline(y=1, linewidth=2, color="black", linestyle="--", alpha=0.3)
            ax_r.set_ylim(0.5, 1.5)
        return ax_arts


    def finalize_ratio_panels(axs_col, set_ratio_ylabels=True):
        """Prune the ratio-panel y-ticks and (optionally) set the two MC/Data ratio
        ylabels. The main-panel ylabel (hist vs prof) stays in the calling cell. For grids,
        pass set_ratio_ylabels=(column is first) so only column 0 carries the ratio ylabels."""
        prune_ratio_panel_yticks(np.array([[axs_col[0]], [axs_col[1]], [axs_col[2]]]))
        if set_ratio_ylabels:
            axs_col[1].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large")
            axs_col[2].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large")


    return (
        draw_dist_panels,
        draw_profile_panels,
        finalize_ratio_panels,
        pt_label,
    )


@app.cell
def _(
    CENTER_JPT,
    DIST_LEGEND_LABELS,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    PRELIM_KW,
    PRELIM_TEXT,
    SYSTEM_INFO_TEXT,
    draw_dist_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    var_hist_ylabel,
    var_logy: set,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k1_b0.5"
    fig_dist_ch_ang_k1_b0_5 = plt.figure(figsize=(6.5, 10))
    _axs = fig_dist_ch_ang_k1_b0_5.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT

    _ax_arts = draw_dist_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _jpt_true)

    _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if _var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_var_name])
    if _var_name in var_logy:
        _ax_main.set_yscale("log")

    # --- headroom (inlined from add_top_headroom) ---
    _hr_lo, _hr_hi = _ax_main.get_ylim()
    if _ax_main.get_yscale() == "log":
        _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
    else:
        _ax_main.set_ylim(_hr_lo, _hr_hi + 0.3 * (_hr_hi - _hr_lo))
    # --- annotation block (inlined from annotate_corner) ---
    _ax_main.text(0.55, 0.97, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.98, 
        SYSTEM_INFO_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", 
        va="top",
    )
    _ax_main.text(
        0.3, 0.25, 
        KINEMATIC_CUTS_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", va="top",
    )
    _ax_main.text(0.2, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _ax_main.legend(
        POSTER_LEGEND_HANDLES, DIST_LEGEND_LABELS,
        **LEGEND_KW,
        loc='best', bbox_to_anchor=(0.6, 0.6, 0.3, 0.3)
        #loc="lower right", bbox_to_anchor=(0.99, 0.99),
    )

    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dist_ch_ang_k1_b0_5.savefig(OUT_DIR / "fig_dist_ch_ang_k1_b0_5.pdf", bbox_inches="tight")

    fig_dist_ch_ang_k1_b0_5
    return


@app.cell
def _(
    CENTER_JPT,
    DIST_LEGEND_LABELS,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    PRELIM_KW,
    PRELIM_TEXT,
    SYSTEM_INFO_TEXT,
    draw_dist_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    var_hist_ylabel,
    var_logy: set,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k1_b1"
    fig_dist_ch_ang_k1_b1 = plt.figure(figsize=(6.5, 10))
    _axs = fig_dist_ch_ang_k1_b1.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT

    _ax_arts = draw_dist_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _jpt_true)

    _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if _var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_var_name])
    if _var_name in var_logy:
        _ax_main.set_yscale("log")

    # --- headroom (inlined from add_top_headroom) ---
    _hr_lo, _hr_hi = _ax_main.get_ylim()
    if _ax_main.get_yscale() == "log":
        _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
    else:
        _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))
    # --- annotation block (inlined from annotate_corner) ---
    _ax_main.text(0.55, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.98, 
        SYSTEM_INFO_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", 
        va="top",
    )
    _ax_main.text(
        0.65, 0.9, 
        KINEMATIC_CUTS_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", va="top",
    )
    _ax_main.text(0.45, 0.73, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _ax_main.legend(
        POSTER_LEGEND_HANDLES, DIST_LEGEND_LABELS,
        **LEGEND_KW,
        loc='best', bbox_to_anchor=(0.5, 0.0, 0.5, 0.5)
        #loc="lower right", bbox_to_anchor=(0.99, 0.99),
    )

    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dist_ch_ang_k1_b1.savefig(OUT_DIR / "fig_dist_ch_ang_k1_b1.pdf", bbox_inches="tight")

    fig_dist_ch_ang_k1_b1
    return


@app.cell
def _(
    CENTER_JPT,
    DIST_LEGEND_LABELS,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    PRELIM_KW,
    PRELIM_TEXT,
    SYSTEM_INFO_TEXT,
    draw_dist_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    var_hist_ylabel,
    var_logy: set,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k1_b2"
    fig_dist_ch_ang_k1_b2 = plt.figure(figsize=(6.5, 10))
    _axs = fig_dist_ch_ang_k1_b2.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT

    _ax_arts = draw_dist_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _jpt_true)

    _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if _var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_var_name])
    if _var_name in var_logy:
        _ax_main.set_yscale("log")

    # --- headroom (inlined from add_top_headroom) ---
    _hr_lo, _hr_hi = _ax_main.get_ylim()
    if _ax_main.get_yscale() == "log":
        _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
    else:
        _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))
    # --- annotation block (inlined from annotate_corner) ---
    _ax_main.text(0.55, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.98, 
        SYSTEM_INFO_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", 
        va="top",
    )
    _ax_main.text(
        0.03, 0.4, 
        KINEMATIC_CUTS_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", va="top",
    )
    _ax_main.text(0.03, 0.25, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _ax_main.legend(
        POSTER_LEGEND_HANDLES, DIST_LEGEND_LABELS,
        **LEGEND_KW,
        loc='best', bbox_to_anchor=(0.5, 0.5, 0.5, 0.5)
        #loc="lower right", bbox_to_anchor=(0.99, 0.99),
    )


    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dist_ch_ang_k1_b2.savefig(OUT_DIR / "fig_dist_ch_ang_k1_b2.pdf", bbox_inches="tight")

    fig_dist_ch_ang_k1_b2
    return


@app.cell
def _(
    CENTER_JPT,
    DIST_LEGEND_LABELS,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    PRELIM_KW,
    PRELIM_TEXT,
    SYSTEM_INFO_TEXT,
    draw_dist_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    var_hist_ylabel,
    var_logy: set,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k2_b0"
    fig_dist_ch_ang_k2_b0 = plt.figure(figsize=(6.5, 10))
    _axs = fig_dist_ch_ang_k2_b0.subplots(
        3, 1,
        height_ratios=[3, 1, 1],
        sharex=True,
        gridspec_kw=dict(hspace=0),
    )
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT

    _ax_arts = draw_dist_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _jpt_true)

    _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    if _var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_var_name])
    if _var_name in var_logy:
        _ax_main.set_yscale("log")

    # --- headroom (inlined from add_top_headroom) ---
    _hr_lo, _hr_hi = _ax_main.get_ylim()
    if _ax_main.get_yscale() == "log":
        _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
    else:
        _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))
    # --- annotation block (inlined from annotate_corner) ---
    _ax_main.text(0.55, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.98, 
        SYSTEM_INFO_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", 
        va="top",
    )
    _ax_main.text(
        0.7, 0.9, 
        KINEMATIC_CUTS_TEXT, 
        transform=_ax_main.transAxes, 
        ha="left", va="top",
    )
    _ax_main.text(0.45, 0.73, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _ax_main.legend(
        POSTER_LEGEND_HANDLES, DIST_LEGEND_LABELS,
        **LEGEND_KW,
        loc='best', bbox_to_anchor=(0.5, 0.0, 0.5, 0.5)
        #loc="lower right", bbox_to_anchor=(0.99, 0.99),
    )


    _axs[0].set_ylabel(var_hist_ylabel[_var_name], fontsize="xx-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dist_ch_ang_k2_b0.savefig(OUT_DIR / "fig_dist_ch_ang_k2_b0.pdf", bbox_inches="tight")

    fig_dist_ch_ang_k2_b0
    return


@app.cell
def _(
    CENTER_JPT,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):
    _var_name = "ch_ang_k1_b0.5"
    _x_var_name = "sd_dR"
    fig_prof_ch_ang_k1_b0_5 = plt.figure(figsize=(6, 10))
    _axs = fig_prof_ch_ang_k1_b0_5.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)

    _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.5 * (_hr_hi - _hr_lo))

    # --- annotation block (matches single-jetpt histogram cells) ---
    _ax_main.text(0.5, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.95,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )

    _ax_main.text(
        0.2, 0.25,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(0.1, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _leg_loc, _leg_bbox = {
        "sd_dR": ("best", (0.5, 0.0, 0.5, 0.5)),
        "sd_m": ("lower right", None),
        "sd_symmetry": ("upper right", (0.99, 0.99)),
    }.get(_x_var_name, ("upper right", (0.99, 0.99)))


    _ax_main.legend(
        POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
        loc=_leg_loc, bbox_to_anchor=_leg_bbox, **LEGEND_KW
    )

    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_prof_ch_ang_k1_b0_5.savefig(OUT_DIR / "fig_prof_ch_ang_k1_b0_5.pdf", bbox_inches="tight")

    fig_prof_ch_ang_k1_b0_5
    return


@app.cell
def _(
    CENTER_JPT,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):
    _var_name = "ch_ang_k1_b1"
    _x_var_name = "sd_dR"
    fig_prof_ch_ang_k1_b1 = plt.figure(figsize=(6, 10))
    _axs = fig_prof_ch_ang_k1_b1.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)

    _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

    # --- annotation block (matches single-jetpt histogram cells) ---
    _ax_main.text(0.5, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.95,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )

    _ax_main.text(
        0.03, 0.73,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(0.03, 0.58, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")



    _leg_loc, _leg_bbox = {
        "sd_dR": ("best", (0.5, 0.0, 0.5, 0.5)),
        "sd_m": ("lower right", None),
        "sd_symmetry": ("upper right", (0.99, 0.99)),
    }.get(_x_var_name, ("upper right", (0.99, 0.99)))

    _ax_main.legend(
        POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
        loc=_leg_loc, bbox_to_anchor=_leg_bbox, **LEGEND_KW,
    )

    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_prof_ch_ang_k1_b1.savefig(OUT_DIR / "fig_prof_ch_ang_k1_b1.pdf", bbox_inches="tight")

    fig_prof_ch_ang_k1_b1
    return


@app.cell
def _(
    CENTER_JPT,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):
    _var_name = "ch_ang_k1_b2"
    _x_var_name = "sd_dR"
    fig_prof_ch_ang_k1_b2 = plt.figure(figsize=(6, 10))
    _axs = fig_prof_ch_ang_k1_b2.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = CENTER_JPT
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)

    _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

    # --- annotation block (matches single-jetpt histogram cells) ---
    _ax_main.text(0.5, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.03, 0.94,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )

    _ax_main.text(
        0.65, 0.89,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(0.2, 0.71, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")


    _leg_loc, _leg_bbox = {
        "sd_dR": ("best", (0.05, 0.1, 0.5, 0.5)),
        "sd_m": ("lower right", None),
        "sd_symmetry": ("upper right", (0.99, 0.99)),
    }.get(_x_var_name, ("upper right", (0.99, 0.99)))

    _ax_main.legend(
        POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
        loc=_leg_loc, bbox_to_anchor=_leg_bbox, **LEGEND_KW,
    )

    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_prof_ch_ang_k1_b2.savefig(OUT_DIR / "fig_prof_ch_ang_k1_b2.pdf", bbox_inches="tight")

    fig_prof_ch_ang_k1_b2
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):
    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    fig_zg_1 = plt.figure(figsize=(6, 10))
    _axs = fig_zg_1.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = 1
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)
    _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

    # --- annotation block (separate text blocks, matches sd_dR profile cells) ---
    _ax_main.text(0.52, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.02, 0.98,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(
        0.65, 0.93,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(0.32, 0.77, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")

    _leg1 = _ax_main.legend(
        POSTER_LEGEND_HANDLES[:2], PROF_LEGEND_LABELS[:2], loc="best", bbox_to_anchor=(0.0, 0.06, 0.5, 0.5), ncols=2, columnspacing=1, **LEGEND_KW
    )
    _ax_main.legend(
        POSTER_LEGEND_HANDLES[2:], PROF_LEGEND_LABELS[2:], loc="best", bbox_to_anchor=(0.0, 0.0, 0.5, 0.5), ncols=3, **LEGEND_KW
    )
    _ax_main.add_artist(_leg1)
    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_1.savefig(OUT_DIR / "fig_zg_1.pdf", bbox_inches="tight")

    fig_zg_1
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlim,
):
    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    fig_zg_2 = plt.figure(figsize=(6, 10))
    _axs = fig_zg_2.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = 2
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)
    _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

    # --- annotation block (separate text blocks, matches sd_dR profile cells) ---
    _ax_main.text(0.52, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.02, 0.98,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(
        0.65, 0.93,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )

    _ax_main.text(0.03, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")

    _leg1 = _ax_main.legend(
        POSTER_LEGEND_HANDLES[:2], PROF_LEGEND_LABELS[:2], loc="best", bbox_to_anchor=(0.49, 0.25, 0.5, 0.5), **LEGEND_KW
    )
    _ax_main.legend(
        POSTER_LEGEND_HANDLES[2:], PROF_LEGEND_LABELS[2:], loc="best", bbox_to_anchor=(0.5, 0.31, 0.5, 0.5), ncols=3, columnspacing=1, **LEGEND_KW
    )
    _ax_main.add_artist(_leg1)

    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_2.savefig(OUT_DIR / "fig_zg_2.pdf", bbox_inches="tight")

    fig_zg_2
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):
    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    fig_zg_3 = plt.figure(figsize=(6, 10))
    _axs = fig_zg_3.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
    _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]
    _jpt_true = 3
    _ijpt = _jpt_true - 1

    _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
    _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

    _ax_arts = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

    _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
    _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

    if _x_var_name in var_xlim:
        _ax_main.set_xlim(*var_xlim[_x_var_name])

    if _prof_ylim_val is not None:
        _ax_main.set_ylim(*_prof_ylim_val)
    _ax_incl.set_ylim(*_ratio_ylim_val)
    _ax_sd.set_ylim(*_ratio_ylim_val)
    _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
    _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

    if _prof_ylim_val is None:
        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

    # --- annotation block (separate text blocks, matches sd_dR profile cells) ---
    _ax_main.text(0.52, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
    _ax_main.text(
        0.02, 0.98,
        SYSTEM_INFO_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(
        0.65, 0.93,
        KINEMATIC_CUTS_TEXT,
        transform=_ax_main.transAxes,
        ha="left",
        va="top",
    )
    _ax_main.text(0.03, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")


    _leg1 = _ax_main.legend(
        POSTER_LEGEND_HANDLES[:2], PROF_LEGEND_LABELS[:2], loc="best", bbox_to_anchor=(0.49, 0.23, 0.5, 0.5), **LEGEND_KW
    )
    _ax_main.legend(
        POSTER_LEGEND_HANDLES[2:], PROF_LEGEND_LABELS[2:], loc="best", bbox_to_anchor=(0.5, 0.3, 0.5, 0.5), ncols=3, columnspacing=1, **LEGEND_KW
    )
    _ax_main.add_artist(_leg1)

    _axs[0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")
    finalize_ratio_panels(_axs)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_3.savefig(OUT_DIR / "fig_zg_3.pdf", bbox_inches="tight")

    fig_zg_3
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    np,
    plt,
    prune_ratio_panel_yticks,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    _jpt_trues = [1, 2]
    _n = len(_jpt_trues)

    fig_zg_12 = plt.figure(figsize=(6 * _n, 10))
    if _n == 1:
        _axs_raw = fig_zg_12.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
        _col_axs = [_axs_raw]
    else:
        _axs_raw = fig_zg_12.subplots(3, _n, height_ratios=[3, 1, 1], sharey="row", sharex="col", squeeze=False, gridspec_kw=dict(hspace=0, wspace=0))
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]

    for _k, _jpt_true in enumerate(_jpt_trues):

        _ax_main, _ax_incl, _ax_sd = _col_axs[_k][0], _col_axs[_k][1], _col_axs[_k][2]
        _ijpt = _jpt_true - 1

        _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
        _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

        _ax_art_map = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

        _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
        _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

        if _x_var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_x_var_name])

        if _prof_ylim_val is not None:
            _ax_main.set_ylim(*_prof_ylim_val)
        _ax_incl.set_ylim(*_ratio_ylim_val)
        _ax_sd.set_ylim(*_ratio_ylim_val)
        _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        if _prof_ylim_val is None:
            # --- headroom (inlined from add_top_headroom) ---
            _hr_lo, _hr_hi = _ax_main.get_ylim()
            if _ax_main.get_yscale() == "log":
                _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
            else:
                _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))

        # --- annotation distributed across columns: system-info -> first column,
        #     kinematic cuts -> second column, legend -> last column, pt per column ---
        _is_first = _k == 0
        _is_second = _k == 1
        _is_last = _k == _n - 1

        if _is_first:
            _ax_main.text(0.03, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
            _ax_main.text(
                0.4, 0.90,
                SYSTEM_INFO_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )
            _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

        if _is_second:
            _ax_main.text(
                0.1, 0.9,
                KINEMATIC_CUTS_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )

        # pt label in every column (each its own jet-pt bin)
        _ax_main.text(0.03, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")


        _leg_loc, _leg_bbox = {
            "sd_dR": ("lower right", None),
            "sd_m": ("lower right", None),
            "sd_symmetry": ("best", (0.5, 0.5, 0.5, 0.5)),
        }.get(_x_var_name, ("upper right", (0.99, 0.99)))

        if _is_last:
            _ax_main.legend(
                POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
                loc=_leg_loc, bbox_to_anchor=_leg_bbox, **{**LEGEND_KW, "fontsize": "small"},
            )

        prune_ratio_panel_yticks(np.array([[_col_axs[_k][0]], [_col_axs[_k][1]], [_col_axs[_k][2]]]))


    _col_axs[0][1].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large")

    # --- mirror the full left-side y-axis (ticks, tick numbers, axis label) onto
    #     the right outer side of the last column ---
    _last = _n - 1
    for _r, _rlabel in enumerate((
        var_prof_ylabel[_var_name],
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$",
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$",
    )):
        _rax = _col_axs[_last][_r]
        _rax.tick_params(axis="y", which="both", right=True, labelright=True)
        _rax.yaxis.set_label_position("right")
        _rax.set_ylabel(_rlabel, fontsize="x-large")


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_12.savefig(OUT_DIR / "fig_zg_12.pdf", bbox_inches="tight")

    fig_zg_12
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    np,
    plt,
    prune_ratio_panel_yticks,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    _jpt_trues = [2, 3]
    _n = len(_jpt_trues)

    fig_zg_23 = plt.figure(figsize=(6 * _n, 10))
    if _n == 1:
        _axs_raw = fig_zg_23.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
        _col_axs = [_axs_raw]
    else:
        _axs_raw = fig_zg_23.subplots(3, _n, height_ratios=[3, 1, 1], sharey="row", sharex="col", squeeze=False, gridspec_kw=dict(hspace=0, wspace=0))
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]

    for _k, _jpt_true in enumerate(_jpt_trues):

        _ax_main, _ax_incl, _ax_sd = _col_axs[_k][0], _col_axs[_k][1], _col_axs[_k][2]
        _ijpt = _jpt_true - 1

        _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
        _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

        _ax_art_map = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

        _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
        _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

        if _x_var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_x_var_name])

        if _prof_ylim_val is not None:
            _ax_main.set_ylim(*_prof_ylim_val)
        _ax_incl.set_ylim(*_ratio_ylim_val)
        _ax_sd.set_ylim(*_ratio_ylim_val)
        _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        if _prof_ylim_val is None:
            # --- headroom (inlined from add_top_headroom) ---
            _hr_lo, _hr_hi = _ax_main.get_ylim()
            if _ax_main.get_yscale() == "log":
                _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
            else:
                _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0*(_hr_hi - _hr_lo))

        # --- annotation distributed across columns: system-info -> first column,
        #     kinematic cuts -> second column, legend -> last column, pt per column ---
        _is_first = _k == 0
        _is_second = _k == 1
        _is_last = _k == _n - 1

        if _is_first:
            _ax_main.text(0.03, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
            _ax_main.text(
                0.03, 0.93,
                SYSTEM_INFO_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )
            _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

        if _is_second:
            _ax_main.text(
                0.1, 0.9,
                KINEMATIC_CUTS_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )

        # pt label in every column (each its own jet-pt bin)
        _ax_main.text(0.33, 0.7, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")


        _leg_loc, _leg_bbox = {
            "sd_dR": ("lower right", None),
            "sd_m": ("lower right", None),
            "sd_symmetry": ("best", (0.5, 0.5, 0.5, 0.5)),
        }.get(_x_var_name, ("upper right", (0.99, 0.99)))

        if _is_last:
            _ax_main.legend(
                POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
                loc=_leg_loc, bbox_to_anchor=_leg_bbox, **{**LEGEND_KW, "fontsize": "x-small"},
            )

        prune_ratio_panel_yticks(np.array([[_col_axs[_k][0]], [_col_axs[_k][1]], [_col_axs[_k][2]]]))


    _col_axs[0][1].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large")

    # --- mirror the full left-side y-axis (ticks, tick numbers, axis label) onto
    #     the right outer side of the last column ---
    _last = _n - 1
    for _r, _rlabel in enumerate((
        var_prof_ylabel[_var_name],
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$",
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$",
    )):
        _rax = _col_axs[_last][_r]
        _rax.tick_params(axis="y", which="both", right=True, labelright=True)
        _rax.yaxis.set_label_position("right")
        _rax.set_ylabel(_rlabel, fontsize="x-large")


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_23.savefig(OUT_DIR / "fig_zg_23.pdf", bbox_inches="tight")

    fig_zg_23
    return


@app.cell
def _(
    CENTER_JPT,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    np,
    plt,
    prune_ratio_panel_yticks,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):

    _kappa1 = ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2")
    _n = len(_kappa1)
    fig_grid = plt.figure(figsize=(6 * _n, 10))
    _axs = fig_grid.subplots(3, _n, height_ratios=[3, 1, 1], sharex="col", sharey=False, squeeze=False, gridspec_kw=dict(hspace=0, wspace=0.30))
    _col_axs = [_axs[:, _k] for _k in range(_n)]
    _x_var_name = "sd_dR"
    _jpt_true = CENTER_JPT
    _ijpt = _jpt_true - 1

    for _k, _var_name in enumerate(_kappa1):
        _ax_main, _ax_incl, _ax_sd = _col_axs[_k][0], _col_axs[_k][1], _col_axs[_k][2]

        _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
        _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

        _ax_art_map = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

        _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
        _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

        if _x_var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_x_var_name])

        if _prof_ylim_val is not None:
            _ax_main.set_ylim(*_prof_ylim_val)
        _ax_incl.set_ylim(0.9, 1.1)
        _ax_sd.set_ylim(0.9, 1.1)

        _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        if _prof_ylim_val is None:
            # --- headroom (inlined from add_top_headroom) ---
            _hr_lo, _hr_hi = _ax_main.get_ylim()
            if _ax_main.get_yscale() == "log":
                _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
            else:
                # --- old: 0.25 headroom; tightened to match finalized singles ---
                # _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))
                _ax_main.set_ylim(_hr_lo, _hr_hi + 0.0 * (_hr_hi - _hr_lo))

        # --- annotation distributed across columns: system-info -> first column,
        #     kinematic cuts (incl. jet-pt range, same for all columns) -> second column,
        #     legend -> last column ---
        _is_first = _k == 0
        _is_second = _k == 1
        _is_last = _k == _n - 1

        if _is_first:
            _ax_main.text(0.03, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
            _ax_main.text(
                0.03, 0.90,
                SYSTEM_INFO_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )

        if _is_second:
            _ax_main.text(
                0.1, 0.95,
                KINEMATIC_CUTS_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )
            _ax_main.text(0.08, 0.8, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")


        _leg_loc, _leg_bbox = {
            "sd_dR": ("upper left", None),
            "sd_m": ("lower right", None),
            "sd_symmetry": ("upper right", (0.99, 0.99)),
        }.get(_x_var_name, ("upper right", (0.99, 0.99)))

        if _is_last:
            _ax_main.legend(
                POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS,
                loc=_leg_loc, bbox_to_anchor=_leg_bbox, **LEGEND_KW,
            )

        prune_ratio_panel_yticks(np.array([[_col_axs[_k][0]], [_col_axs[_k][1]], [_col_axs[_k][2]]]))

        _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

    _col_axs[0][1].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large")


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_grid.savefig(OUT_DIR / "fig_grid.pdf", bbox_inches="tight")

    fig_grid
    return


@app.cell
def _(
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    LEGEND_KW,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    POSTER_PROF_RATIO_YLIM: dict,
    POSTER_PROF_YLIM: dict,
    PRELIM_KW,
    PRELIM_TEXT,
    PROF_LEGEND_LABELS,
    SYSTEM_INFO_TEXT,
    draw_profile_panels,
    np,
    plt,
    prune_ratio_panel_yticks,
    pt_label,
    resolve_ylim,
    var_prof_ratio_ylim: dict,
    var_prof_ylabel,
    var_prof_ylim: dict,
    var_xlabel,
    var_xlim,
):

    _var_name = "ch_ang_k2_b0"
    _x_var_name = "sd_symmetry"
    _jpt_trues = [1, 2, 3]
    _n = len(_jpt_trues)

    fig_zg_123 = plt.figure(figsize=(6 * _n, 10))
    if _n == 1:
        _axs_raw = fig_zg_123.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=dict(hspace=0))
        _col_axs = [_axs_raw]
    else:
        _axs_raw = fig_zg_123.subplots(3, _n, height_ratios=[3, 1, 1], sharey="row", sharex="col", squeeze=False, gridspec_kw=dict(hspace=0, wspace=0))
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]

    for _k, _jpt_true in enumerate(_jpt_trues):

        _ax_main, _ax_incl, _ax_sd = _col_axs[_k][0], _col_axs[_k][1], _col_axs[_k][2]
        _ijpt = _jpt_true - 1

        _prof_spec = POSTER_PROF_YLIM.get(_var_name, var_prof_ylim.get(_var_name))
        _ratio_spec = POSTER_PROF_RATIO_YLIM.get(_var_name, var_prof_ratio_ylim.get(_var_name))

        _ax_art_map = draw_profile_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _x_var_name, _jpt_true)

        _prof_ylim_val = resolve_ylim(_prof_spec, _ijpt)
        _ratio_ylim_val = resolve_ylim(_ratio_spec, _ijpt, default=(0.5, 1.5))

        if _x_var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_x_var_name])

        if _prof_ylim_val is not None:
            _ax_main.set_ylim(*_prof_ylim_val)
        _ax_incl.set_ylim(*_ratio_ylim_val)
        _ax_sd.set_ylim(*_ratio_ylim_val)
        _ax_main.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        _ax_sd.set_xlabel(var_xlabel[_x_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))

        if _prof_ylim_val is None:
            # --- headroom (inlined from add_top_headroom) ---
            _hr_lo, _hr_hi = _ax_main.get_ylim()
            if _ax_main.get_yscale() == "log":
                _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
            else:
                _ax_main.set_ylim(_hr_lo, _hr_hi + 0.25 * (_hr_hi - _hr_lo))

        # --- annotation distributed across columns: system-info -> first column,
        #     kinematic cuts -> second column, legend -> last column, pt per column ---
        _is_first = _k == 0
        _is_second = _k == 1
        _is_last = _k == _n - 1

        if _is_first:
            _ax_main.text(0.03, 0.98, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
            _ax_main.text(
                0.4, 0.90,
                SYSTEM_INFO_TEXT,
                transform=_ax_main.transAxes, ha="left", va="top",
            )
            _col_axs[_k][0].set_ylabel(var_prof_ylabel[_var_name], fontsize="x-large")

        if _is_second:
            _ax_main.text(
                0.4, 0.9,
                KINEMATIC_CUTS_TEXT, fontsize="large",
                transform=_ax_main.transAxes, ha="left", va="top",
            )

        # pt label in every column (each its own jet-pt bin)
        _ax_main.text(0.03, 0.1, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")


        _leg_loc, _leg_bbox = {
            "sd_dR": ("lower right", None),
            "sd_m": ("lower right", None),
            "sd_symmetry": ("best", (0.5, 0.5, 0.5, 0.5)),
        }.get(_x_var_name, ("upper right", (0.99, 0.99)))

        if _is_last:
            _ax_main.legend(
                POSTER_LEGEND_HANDLES, PROF_LEGEND_LABELS, ncols=2, reverse=True,
                loc=_leg_loc, bbox_to_anchor=_leg_bbox, **{**LEGEND_KW, "fontsize": "small"},
            )

        prune_ratio_panel_yticks(np.array([[_col_axs[_k][0]], [_col_axs[_k][1]], [_col_axs[_k][2]]]))


    _col_axs[0][1].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$", fontsize="x-large")
    _col_axs[0][2].set_ylabel(r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$", fontsize="x-large")

    # --- mirror the full left-side y-axis (ticks, tick numbers, axis label) onto
    #     the right outer side of the last column ---
    _last = _n - 1
    for _r, _rlabel in enumerate((
        var_prof_ylabel[_var_name],
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(incl.)}$",
        r"$\frac{\mathrm{MC}}{\mathrm{Data}}\,\mathrm{(SD)}$",
    )):
        _rax = _col_axs[_last][_r]
        _rax.tick_params(axis="y", which="both", right=True, labelright=True)
        _rax.yaxis.set_label_position("right")
        _rax.set_ylabel(_rlabel, fontsize="x-large")


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_zg_123.savefig(OUT_DIR / "fig_zg_123.pdf", bbox_inches="tight")

    fig_zg_123
    return


@app.cell
def _(
    CENTER_JPT,
    DIST_LEGEND_LABELS,
    FormatStrFormatter,
    KINEMATIC_CUTS_TEXT,
    OUT_DIR,
    POSTER_LEGEND_HANDLES,
    PRELIM_KW,
    PRELIM_TEXT,
    SYSTEM_INFO_TEXT,
    draw_dist_panels,
    finalize_ratio_panels,
    plt,
    pt_label,
    var_hist_ylabel,
    var_logy: set,
    var_xlabel,
    var_xlim,
):

    # --- 2x2 grid of all four distribution histograms (each tile = main + 2 ratio
    #     panels via draw_dist_panels). Columns butt flush (wspace=0): the right
    #     column's y axis labels/ticks are moved to the right outer side. Rows are
    #     packed with minimal hspace (just enough for the per-tile x-axis label).
    #     Annotations distributed: tile0 -> STAR + system info, tile1 -> kinematic
    #     cuts + pt, tile2 -> legend. All four share CENTER_JPT. ---
    _dist_vars = ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0"]
    _jpt_true = CENTER_JPT

    fig_dist_grid = plt.figure(figsize=(13, 16))
    _subfigs = fig_dist_grid.subfigures(2, 2, wspace=0.0, hspace=0.0)

    for _idx, _var_name in enumerate(_dist_vars):
        _col = _idx % 2
        _is_right = _col == 1
        _sf = _subfigs.flat[_idx]

        # per-tile subplot margins: butt the two columns together at the shared
        # center (left col extends to right=1.0, right col starts at left=0.0);
        # outer side carries the labels. bottom leaves room for the x-axis label.
        _gkw = dict(hspace=0.0, top=0.985, bottom=0.085)
        if _is_right:
            _gkw.update(left=0.01, right=0.86)
        else:
            _gkw.update(left=0.14, right=0.99)

        _axs = _sf.subplots(3, 1, height_ratios=[3, 1, 1], sharex=True, gridspec_kw=_gkw)
        _ax_main, _ax_incl, _ax_sd = _axs[0], _axs[1], _axs[2]

        _ax_arts = draw_dist_panels(_ax_main, _ax_incl, _ax_sd, _var_name, _jpt_true)

        _ax_sd.set_xlabel(var_xlabel[_var_name], fontsize="x-large")
        _ax_sd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        if _var_name in var_xlim:
            _ax_main.set_xlim(*var_xlim[_var_name])
        if _var_name in var_logy:
            _ax_main.set_yscale("log")

        # --- headroom (inlined from add_top_headroom) ---
        _hr_lo, _hr_hi = _ax_main.get_ylim()
        if _ax_main.get_yscale() == "log":
            _ax_main.set_ylim(_hr_lo, _hr_hi * 5.0)
        else:
            _ax_main.set_ylim(_hr_lo, _hr_hi + 0.3 * (_hr_hi - _hr_lo))

        # --- annotations distributed across tiles ---
        if _idx == 0:
            _ax_main.text(0.55, 0.97, PRELIM_TEXT, transform=_ax_main.transAxes, **PRELIM_KW)
            _ax_main.text(0.03, 0.98, SYSTEM_INFO_TEXT, transform=_ax_main.transAxes, ha="left", va="top")
        if _idx == 1:
            _ax_main.text(0.7, 0.95, KINEMATIC_CUTS_TEXT, transform=_ax_main.transAxes, ha="left", va="top", fontsize="large")
            _ax_main.text(0.42, 0.7, pt_label(_jpt_true), transform=_ax_main.transAxes, ha="left", va="top", fontsize="x-large")
        if _idx == 3:
            _ax_main.legend(
                POSTER_LEGEND_HANDLES, DIST_LEGEND_LABELS,
                loc="best", bbox_to_anchor=(0.5, 0.55, 0.45, 0.4), frameon=False, reverse=True, ncols=2, #fontsize="large"
            )

        _ax_main.set_ylabel(var_hist_ylabel[_var_name], fontsize="x-large")
        finalize_ratio_panels(_axs)

        # right column: move y axis labels + tick labels to the right outer side so
        # the columns can sit flush with no inner-edge labels.
        if _is_right:
            for _a in _axs:
                _a.yaxis.set_label_position("right")
                _a.tick_params(axis="y", left=True, right=True, labelleft=False, labelright=True)


    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_dist_grid.savefig(OUT_DIR / "fig_dist_grid.pdf", bbox_inches="tight")

    fig_dist_grid
    return


@app.cell
def _(SysVar, angularities, get_jet_pt_bins, json, load_config, np, torch):
    # ===================================================================== #
    # Closure-plot data pipeline for the poster figures (LOADER cell).
    #
    # The poster figures above draw DATA vs MC from precomputed .pt snapshots
    # (outputs/histograms). The closure figures below instead draw the multifold
    # UNFOLDED result vs the TRUTH (particle-level prior), reusing the same poster
    # layout / styling. Two closure flavours:
    #   * "self"      -> SysVar.UNFOLDING_PRIOR_SAME  (AB-split on nominal)
    #   * "data-like" -> SysVar.UNFOLDING_PRIOR_LIKE_DATA (omniseq-reweighted prior)
    #
    # This cell ports the GEN/particle-level data pipeline from plot_closure.py /
    # plot_ab_closure.py. The figure BUILDERS live in the next cell so that
    # cosmetic edits there don't force a re-run of the heavy build_closure(...)
    # calls below them.
    # ===================================================================== #
    import pyarrow as pa
    from tensordict import TensorDict

    from histograms import (
        histogram,
        profile,
        ratio_snapshot,
        closure_state_dict,
        snapshot_state_dict,
    )

    # Iteration shown for closure (matches plot_closure.py's best_iter). The npz
    # packs gen weights as even arrays arr_{2*i}; we load only this iteration so the
    # 5.6 GB LIKE_DATA file is never fully materialised.
    CLOSURE_ITER = 1


    def take_table(table, indices):
        return table.take(pa.array(indices, type=pa.int64()))


    def gen_table_weight(table):
        return torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)


    def build_closure(sys_var, iteration=CLOSURE_ITER):
        """Load the GEN-side unfolded + truth snapshots for one closure flavour.

        Returns a dict with per-pt-bin snapshot lists for the dist observables
        (4 angularities + their sd_ variants) and the zg profile observables
        (ch_ang_k2_b0 / sd_ch_ang_k2_b0 vs sd_symmetry). Both flavours reuse the
        nominal embedding arrows for the OBSERVABLES; only the weights differ."""
        # --- bins (identical construction to plot_closure.py) ---
        with open("./runtime-files/bins_p00.02_N100000.json", "rb") as _f:
            bins = json.load(_f)
        bins["pt"] = get_jet_pt_bins(SysVar.NONE)
        for _ang in angularities:
            bins[f"sd_{_ang}"] = bins[_ang]
        clo_jpt_bins = bins["pt"]
        n_pt = len(clo_jpt_bins) - 1

        _cfg = load_config()
        _src = _cfg.dataset_root / "features" / "angularities"
        _unf_dir = _src / "embedding" / str(sys_var)
        _nom = _src / "embedding" / str(SysVar.NONE)

        # GEN observables = nominal [matches | misses] arrows, memory-mapped.
        _bufs = []

        def _read(p):
            _bufs.append(pa.memory_map(str(p)))
            return pa.ipc.open_file(_bufs[-1]).read_all()

        _gm = _read(_nom / "gen-matches.arrow")
        _mi = _read(_nom / "misses.arrow")
        gen_table = pa.concat_tables((_gm, _mi))
        n_gen = len(gen_table)

        # load ONLY the chosen iteration's gen weights (even-indexed array).
        _wz = np.load(_unf_dir / "w_unfolding.npz")
        _max_iter = (len(_wz.files) // 2) - 1
        _it = min(iteration, _max_iter)
        gen_w_full = torch.as_tensor(_wz[f"arr_{2 * _it}"], dtype=torch.float32)
        _wz.close()

        if sys_var == SysVar.UNFOLDING_PRIOR_SAME:
            # AB-split: B-side indices are in tensordict space; map back to arrow
            # rows via (len(part_lvl td) - n_gen). Truth = A-side, own weight column.
            _td_root = _src / "tensordicts" / str(SysVar.NONE)
            _gen_off = len(TensorDict.load_memmap(_td_root / "part_lvl")) - n_gen
            _idx = np.load(_unf_dir / "index_split.npz")
            _b_match = _idx["partlvl_matched_indices"].astype(np.int64) - _gen_off
            _b_miss = _idx["partlvl_missed_indices"].astype(np.int64) - _gen_off
            gen_unf_order = np.concatenate([_b_match, _b_miss])
            gen_truth_order = np.setdiff1d(
                np.arange(n_gen, dtype=np.int64), gen_unf_order, assume_unique=False
            )
            truth_gen_weights = None
        else:
            # LIKE_DATA: no split; unfolded runs over the full nominal table in its
            # natural [matches | misses] order. Truth weights are the reweighted
            # prior weights baked into the LIKE_DATA arrows.
            gen_unf_order = np.arange(n_gen, dtype=np.int64)
            gen_truth_order = gen_unf_order
            _pb = []

            def _readp(p):
                _pb.append(pa.memory_map(str(p)))
                return pa.ipc.open_file(_pb[-1]).read_all()

            _gm_p = _readp(_unf_dir / "gen-matches.arrow")
            _mi_p = _readp(_unf_dir / "misses.arrow")
            truth_gen_weights = torch.as_tensor(
                np.concatenate([_gm_p["weight"].to_numpy(), _mi_p["weight"].to_numpy()]),
                dtype=torch.float32,
            )

        unf_w = gen_w_full[:, : len(gen_unf_order)]

        # Pre-take the unfolded / truth subsets once (skip the copy when the order
        # is the full identity arange, as for LIKE_DATA).
        _full = np.arange(n_gen, dtype=np.int64)
        unf_tbl = (
            gen_table if np.array_equal(gen_unf_order, _full)
            else take_table(gen_table, gen_unf_order)
        )
        truth_tbl = (
            gen_table if np.array_equal(gen_truth_order, _full)
            else take_table(gen_table, gen_truth_order)
        )
        if truth_gen_weights is None:
            truth_w = gen_table_weight(truth_tbl)
        else:
            truth_w = truth_gen_weights[torch.as_tensor(gen_truth_order)]

        print(
            f"[closure:{sys_var}] iter={_it} n_gen={n_gen} "
            f"n_unf={len(gen_unf_order)} n_truth={len(gen_truth_order)} "
            f"replicas={unf_w.shape[0]}"
        )

        dist_vars = list(angularities)
        hist_obs = list(dist_vars) + [f"sd_{_v}" for _v in dist_vars]
        # --- old: zg profiles only ---
        # prof_specs = [("ch_ang_k2_b0", "sd_symmetry"), ("sd_ch_ang_k2_b0", "sd_symmetry")]
        prof_specs = [("ch_ang_k2_b0", "sd_symmetry"), ("sd_ch_ang_k2_b0", "sd_symmetry")]
        # sd_dR profiles for the 3-column kappa=1 closure grid (incl.=ungroomed lambda,
        # SD=groomed sd_ lambda, both binned in the groomed splitting angle dR_g).
        for _dr_v in ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2"):
            prof_specs += [(_dr_v, "sd_dR"), (f"sd_{_dr_v}", "sd_dR")]

        hist_unf, hist_truth, hist_ratio = {}, {}, {}
        for _obs in hist_obs:
            _hu = histogram(unf_tbl, bins, ("pt", _obs), unf_w).unbind("pt")[1:-1]
            _ht = histogram(truth_tbl, bins, ("pt", _obs), truth_w).unbind("pt")[1:-1]
            hist_unf[_obs] = [_h.snapshot() for _h in _hu]
            hist_truth[_obs] = [_h.snapshot() for _h in _ht]
            hist_ratio[_obs] = [
                ratio_snapshot(_u, _t) for _u, _t in zip(hist_unf[_obs], hist_truth[_obs])
            ]

        prof_unf, prof_truth, prof_ratio = {}, {}, {}
        for _yobs, _x in prof_specs:
            _pu = profile(unf_tbl, bins, ("pt", _x), _yobs, unf_w).unbind("pt")[1:-1]
            _pt = profile(truth_tbl, bins, ("pt", _x), _yobs, truth_w).unbind("pt")[1:-1]
            prof_unf[(_yobs, _x)] = [_h.snapshot() for _h in _pu]
            prof_truth[(_yobs, _x)] = [_h.snapshot() for _h in _pt]
            prof_ratio[(_yobs, _x)] = [
                ratio_snapshot(_u, _t)
                for _u, _t in zip(prof_unf[(_yobs, _x)], prof_truth[(_yobs, _x)])
            ]

        return dict(
            sys_var=str(sys_var), iteration=_it, jpt_bins=clo_jpt_bins, n_pt=n_pt,
            hist_unf=hist_unf, hist_truth=hist_truth, hist_ratio=hist_ratio,
            prof_unf=prof_unf, prof_truth=prof_truth, prof_ratio=prof_ratio,
            _bufs=_bufs,
        )


    return (
        CLOSURE_ITER,
        build_closure,
        closure_state_dict,
        histogram,
        pa,
        profile,
        ratio_snapshot,
        snapshot_state_dict,
    )


@app.cell
def _(SysVar, build_closure):
    # Heavy: load GEN-side unfolded + truth snapshots for both closure flavours.
    # Reuses the nominal embedding arrows for observables; loads only iteration
    # CLOSURE_ITER of each w_unfolding.npz. Runs once; the four closure figure
    # cells below are cheap given these.
    closure_same = build_closure(SysVar.UNFOLDING_PRIOR_SAME)
    closure_like = build_closure(SysVar.UNFOLDING_PRIOR_LIKE_DATA)
    print("closure_same:", closure_same["sys_var"], "iter", closure_same["iteration"])
    print("closure_like:", closure_like["sys_var"], "iter", closure_like["iteration"])
    return closure_like, closure_same


@app.cell
def _(closure_same, make_closure_zg):
    # zg profile closure -- SELF (UNFOLDING_PRIOR_SAME, AB-split on nominal)
    fig_zg_123_closure_same = make_closure_zg(
        closure_same, "Self Closure", "fig_zg_123_closure_same.pdf"
    )
    fig_zg_123_closure_same
    return


@app.cell
def _(closure_like, make_closure_zg):
    # zg profile closure -- DATA-LIKE (UNFOLDING_PRIOR_LIKE_DATA, omniseq prior)
    fig_zg_123_closure_like = make_closure_zg(
        closure_like, "Pseudodata Closure", "fig_zg_123_closure_like.pdf"
    )
    fig_zg_123_closure_like
    return


@app.cell
def _(closure_same, make_closure_dist_grid):
    # 2x2 distribution closure -- SELF (UNFOLDING_PRIOR_SAME, AB-split on nominal)
    fig_dist_grid_closure_same = make_closure_dist_grid(
        closure_same, "Self Closure", "fig_dist_grid_closure_same.pdf"
    )
    fig_dist_grid_closure_same
    return


@app.cell
def _(closure_like, make_closure_dist_grid):
    # 2x2 distribution closure -- DATA-LIKE (UNFOLDING_PRIOR_LIKE_DATA, omniseq prior)
    fig_dist_grid_closure_like = make_closure_dist_grid(
        closure_like, "Pseudodata Closure", "fig_dist_grid_closure_like.pdf"
    )
    fig_dist_grid_closure_like
    return


@app.cell
def _(
    CENTER_JPT,
    FormatStrFormatter,
    Line2D,
    OUT_DIR,
    PRELIM_KW,
    closure_state_dict,
    np,
    plot_data_points,
    plt,
    prune_ratio_panel_yticks,
    pt_label,
    snapshot_state_dict,
    var_prof_ylabel,
    var_xlabel,
    var_xlim,
):
    # ===================================================================== #
    # Closure figure BUILDERS (kept separate from build_closure so cosmetic edits
    # here don't re-trigger the heavy build). RATIO-ONLY: the top distribution/
    # profile comparison panel is dropped; each figure keeps only the two
    # unfolded/truth ratio panels (incl. = blue, SD/groomed = red). The closure
    # type stands in the headline (PRELIM) slot; the filled/open unf-vs-truth
    # legend no longer applies (ratio panels carry a single starred series) so it
    # is omitted -- the ylabels name each panel.
    # ===================================================================== #
    def _closure_proxy(color, marker, filled):
        return Line2D(
            [], [], color=color, marker=marker, linestyle="none", markersize=8,
            markerfacecolor=(color if filled else "none"), markeredgecolor=color,
        )


    # Retained (unused by the ratio-only figures) in case the comparison panel /
    # legend is reinstated later.
    CLOSURE_PROF_HANDLES = [
        _closure_proxy("blue", "^", True), _closure_proxy("blue", "^", False),
        _closure_proxy("red", "o", True), _closure_proxy("red", "o", False),
    ]
    CLOSURE_PROF_LABELS = ["unf. (incl.)", "truth (incl.)", "unf. (SD)", "truth (SD)"]
    CLOSURE_DIST_LABELS = ["unf. (incl.)", "truth (incl.)", "unf. (groomed)", "truth (groomed)"]


    def make_closure_zg(C, tag, fname):
        """3-column zg profile closure (ch_ang_k2_b0 vs sd_symmetry, pt bins 1-3),
        mirroring the `fig_zg_123` layout but RATIO-ONLY (incl. + SD unf/truth ratio
        panels; the top <lambda> comparison panel is removed)."""
        _var, _x = "ch_ang_k2_b0", "sd_symmetry"
        _jpts = [1, 2, 3]
        _n = len(_jpts)
        fig = plt.figure(figsize=(6 * _n, 5.5))
        _axs_raw = fig.subplots(
            2, _n, height_ratios=[1, 1], sharey="row", sharex="col",
            squeeze=False, gridspec_kw=dict(hspace=0, wspace=0),
        )
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]
        for _k, _j in enumerate(_jpts):
            _ai, _asd = _col_axs[_k][0], _col_axs[_k][1]
            for _yobs, _color, _rax in (
                (_var, "blue", _ai),
                (f"sd_{_var}", "red", _asd),
            ):
                _rd = snapshot_state_dict(C["prof_ratio"][(_yobs, _x)][_j], batched=True)
                plot_data_points(_rax, "errorbar", _rd, color=_color, marker="*", linestyle="none")
                _rax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
                # --- old: _rax.set_ylim(*POSTER_PROF_RATIO_YLIM.get(_var, (0.8, 1.2))) ---
                _rax.set_ylim(0.5, 1.5)
            if _x in var_xlim:
                _ai.set_xlim(*var_xlim[_x])
            _asd.set_xlabel(var_xlabel[_x], fontsize="x-large")
            _asd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            # annotations on the top (incl.) ratio panel: tag top-left (headline),
            # context blocks bottom-left (clear region below the data), pt top-right.
            if _k == 0:
                _ai.text(0.03, 0.94, tag, transform=_ai.transAxes, **PRELIM_KW)
                #_ai.text(0.03, 0.06, SYSTEM_INFO_TEXT, transform=_ai.transAxes, ha="left", va="bottom", fontsize="small")
            #if _k == 1:
                #_ai.text(0.03, 0.06, KINEMATIC_CUTS_TEXT, transform=_ai.transAxes, ha="left", va="bottom", fontsize="small")
            _ai.text(0.97, 0.83, pt_label(_j), transform=_ai.transAxes, ha="right", va="top", fontsize="large")
            prune_ratio_panel_yticks(np.array([[_ai], [_asd]]))

        _col_axs[0][0].set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(incl.)}$", fontsize="x-large")
        _col_axs[0][1].set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(SD)}$", fontsize="x-large")

        # mirror the left y-axis (ticks + label) onto the right outer side of the
        # last column.
        _last = _n - 1
        for _r, _rlabel in enumerate((
            r"$\frac{unf.}{truth}\,\mathrm{(incl.)}$",
            r"$\frac{unf.}{truth}\,\mathrm{(SD)}$",
        )):
            _rax = _col_axs[_last][_r]
            _rax.tick_params(axis="y", which="both", right=True, labelright=True)
            _rax.yaxis.set_label_position("right")
            _rax.set_ylabel(_rlabel, fontsize="x-large")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        return fig


    def make_closure_dist_grid(C, tag, fname):
        """2x2 distribution closure grid (4 angularities at CENTER_JPT), mirroring
        the `fig_dist_grid` layout but RATIO-ONLY (incl. + groomed unf/truth ratio
        panels; the top distribution comparison panel is removed)."""
        _dist_vars = ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2", "ch_ang_k2_b0"]
        _j = CENTER_JPT
        fig = plt.figure(figsize=(13, 11))
        _subfigs = fig.subfigures(2, 2, wspace=0.0, hspace=0.0)
        for _idx, _var in enumerate(_dist_vars):
            _is_right = (_idx % 2) == 1
            _sf = _subfigs.flat[_idx]
            _gkw = dict(hspace=0.0, top=0.97, bottom=0.13)
            _gkw.update(dict(left=0.01, right=0.86) if _is_right else dict(left=0.14, right=0.99))
            _axs = _sf.subplots(2, 1, height_ratios=[1, 1], sharex=True, gridspec_kw=_gkw)
            _ai, _asd = _axs[0], _axs[1]
            for _yobs, _color, _rax in (
                (_var, "blue", _ai),
                (f"sd_{_var}", "red", _asd),
            ):
                _rd = closure_state_dict(C["hist_ratio"][_yobs][_j], batched=True)
                plot_data_points(_rax, "errorbar", _rd, color=_color, marker="*", linestyle="none")
                _rax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
                _rax.set_ylim(0.5, 1.5)
            _asd.set_xlabel(var_xlabel[_var], fontsize="x-large")
            _asd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            if _var in var_xlim:
                _ai.set_xlim(*var_xlim[_var])
            # annotations on the top (incl.) ratio panel.
            if _idx == 0:
                _ai.text(0.03, 0.94, tag, transform=_ai.transAxes, **PRELIM_KW)
                #_ai.text(0.03, 0.06, SYSTEM_INFO_TEXT, transform=_ai.transAxes, ha="left", va="bottom", fontsize="small")
            if _idx == 1:
                #_ai.text(0.03, 0.06, KINEMATIC_CUTS_TEXT, transform=_ai.transAxes, ha="left", va="bottom", fontsize="small")
                _ai.text(0.97, 0.88, pt_label(_j), transform=_ai.transAxes, ha="right", va="top", fontsize="large")
            _ai.set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(incl.)}$", fontsize="x-large")
            _asd.set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(SD)}$", fontsize="x-large")
            prune_ratio_panel_yticks(np.array([[_ai], [_asd]]))
            if _is_right:
                for _a in _axs:
                    _a.yaxis.set_label_position("right")
                    _a.tick_params(axis="y", left=True, right=True, labelleft=False, labelright=True)

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        return fig


    def make_closure_dr(C, tag, fname):
        """3-column sd_dR profile closure: kappa=1 angularities (LHA/width/thrust)
        profiled vs the groomed splitting angle dR_g at CENTER_JPT, mirroring the
        `fig_grid` result layout but RATIO-ONLY (incl. + SD unf/truth ratio panels;
        the top <lambda> comparison panel is removed). Columns differ by angularity
        (labelled top-right); jet-pt is shared and stated once (centre column)."""
        from matplotlib.ticker import FixedLocator, FixedFormatter

        _x = "sd_dR"
        _vars = ["ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2"]
        _j = CENTER_JPT
        _n = len(_vars)
        fig = plt.figure(figsize=(6 * _n, 5.5))
        _axs_raw = fig.subplots(
            2, _n, height_ratios=[1, 1], sharey="row", sharex="col",
            squeeze=False, gridspec_kw=dict(hspace=0, wspace=0),
        )
        _col_axs = [_axs_raw[:, _k] for _k in range(_n)]
        for _k, _var in enumerate(_vars):
            _ai, _asd = _col_axs[_k][0], _col_axs[_k][1]
            for _yobs, _color, _rax in (
                (_var, "blue", _ai),
                (f"sd_{_var}", "red", _asd),
            ):
                _rd = snapshot_state_dict(C["prof_ratio"][(_yobs, _x)][_j], batched=True)
                plot_data_points(_rax, "errorbar", _rd, color=_color, marker="*", linestyle="none")
                _rax.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
                _rax.set_ylim(0.5, 1.5)
            if _x in var_xlim:
                _ai.set_xlim(*var_xlim[_x])
            _asd.set_xlabel(var_xlabel[_x], fontsize="x-large")
            _asd.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            # headline (closure type) on col 0; per-column angularity label top-right.
            if _k == 0:
                _ai.text(0.03, 0.94, tag, transform=_ai.transAxes, **PRELIM_KW)
            _ai.text(0.97, 0.90, var_prof_ylabel[_var], transform=_ai.transAxes, ha="right", va="top", fontsize="large")
            # jet-pt is identical across columns -> state once (centre column, bottom-left).
            if _k == 1:
                _ai.text(0.03, 0.07, pt_label(_j), transform=_ai.transAxes, ha="left", va="bottom", fontsize="x-large")
            prune_ratio_panel_yticks(np.array([[_ai], [_asd]]))

        # Columns are flush (wspace=0, shared y): blank the shared-edge x-tick labels
        # so the right tick of one column does not collide with the left tick of the
        # next (drop first label except on col 0, last label except on the final col).
        fig.canvas.draw()
        for _k in range(_n):
            _bx = _col_axs[_k][1]
            _lo, _hi = _bx.get_xlim()
            _ticks = [float(_t) for _t in _bx.get_xticks() if _lo <= _t <= _hi]
            _labels = [f"{_t:g}" for _t in _ticks]
            if _labels and _k != 0:
                _labels[0] = ""
            if _labels and _k != _n - 1:
                _labels[-1] = ""
            _bx.xaxis.set_major_locator(FixedLocator(_ticks))
            _bx.xaxis.set_major_formatter(FixedFormatter(_labels))

        _col_axs[0][0].set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(incl.)}$", fontsize="x-large")
        _col_axs[0][1].set_ylabel(r"$\frac{unf.}{truth}\,\mathrm{(SD)}$", fontsize="x-large")

        _last = _n - 1
        for _r, _rlabel in enumerate((
            r"$\frac{unf.}{truth}\,\mathrm{(incl.)}$",
            r"$\frac{unf.}{truth}\,\mathrm{(SD)}$",
        )):
            _rax = _col_axs[_last][_r]
            _rax.tick_params(axis="y", which="both", right=True, labelright=True)
            _rax.yaxis.set_label_position("right")
            _rax.set_ylabel(_rlabel, fontsize="x-large")

        OUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_DIR / fname, bbox_inches="tight")
        return fig


    return make_closure_dist_grid, make_closure_dr, make_closure_zg


@app.cell
def _(closure_same, make_closure_dr):
    # sd_dR profile closure -- SELF (3-column kappa=1, ratio-only, UNFOLDING_PRIOR_SAME)
    fig_dr_grid_closure_same = make_closure_dr(
        closure_same, "Self Closure", "fig_dr_grid_closure_same.pdf"
    )
    fig_dr_grid_closure_same
    return


@app.cell
def _(closure_like, make_closure_dr):
    # sd_dR profile closure -- DATA-LIKE (3-column kappa=1, ratio-only, UNFOLDING_PRIOR_LIKE_DATA)
    fig_dr_grid_closure_like = make_closure_dr(
        closure_like, "Pseudodata Closure", "fig_dr_grid_closure_like.pdf"
    )
    fig_dr_grid_closure_like
    return


@app.cell
def _(
    CLOSURE_ITER,
    SysVar,
    angularities,
    get_jet_pt_bins,
    histogram,
    json,
    load_config,
    np,
    pa,
    profile,
    ratio_snapshot,
    torch,
):
    # bin_counts-route closure (LIKE_DATA only) -- UNDER-DEVELOPMENT cross-check.
    # OBSERVABLES from the nominal angularities arrows; WEIGHTS + truth from
    # features/bin_counts/embedding/<sysvar>. Mirrors build_closure's LIKE_DATA branch.
    # NEW function: leaves the angularities closure pipeline + its figures untouched.
    def build_closure_bc(sys_var=SysVar.UNFOLDING_PRIOR_LIKE_DATA, iteration=CLOSURE_ITER):
        with open("./runtime-files/bins_p00.02_N100000.json", "rb") as _f:
            _bins = json.load(_f)
        _bins["pt"] = get_jet_pt_bins(SysVar.NONE)
        for _ang in angularities:
            _bins[f"sd_{_ang}"] = _bins[_ang]
        _clo_jpt = _bins["pt"]
        _npt = len(_clo_jpt) - 1

        _cfg = load_config()
        _obs_src = _cfg.dataset_root / "features" / "angularities"
        _w_src = _cfg.dataset_root / "features" / "bin_counts"
        _unf_dir = _w_src / "embedding" / str(sys_var)
        _nom = _obs_src / "embedding" / str(SysVar.NONE)

        _bufs = []

        def _read(p):
            _bufs.append(pa.memory_map(str(p)))
            return pa.ipc.open_file(_bufs[-1]).read_all()

        _gen_table = pa.concat_tables((_read(_nom / "gen-matches.arrow"), _read(_nom / "misses.arrow")))
        _n_gen = len(_gen_table)

        _wz = np.load(_unf_dir / "w_unfolding.npz")
        _max_iter = (len(_wz.files) // 2) - 1
        _it = min(iteration, _max_iter)
        _gen_w = torch.as_tensor(_wz[f"arr_{2 * _it}"], dtype=torch.float32)
        _wz.close()

        _pb = []

        def _readp(p):
            _pb.append(pa.memory_map(str(p)))
            return pa.ipc.open_file(_pb[-1]).read_all()

        _gm = _readp(_unf_dir / "gen-matches.arrow")
        _mi = _readp(_unf_dir / "misses.arrow")
        _truth_w = torch.as_tensor(
            np.concatenate([_gm["weight"].to_numpy(), _mi["weight"].to_numpy()]),
            dtype=torch.float32,
        )
        _unf_w = _gen_w[:, :_n_gen]
        print(
            f"[closure-bc:{sys_var}] iter={_it} n_gen={_n_gen} replicas={_unf_w.shape[0]} (weights=bin_counts, obs=angularities)"
        )

        _dist = list(angularities)
        _hist_obs = list(_dist) + [f"sd_{_v}" for _v in _dist]
        _prof = [("ch_ang_k2_b0", "sd_symmetry"), ("sd_ch_ang_k2_b0", "sd_symmetry")]
        for _dv in ("ch_ang_k1_b0.5", "ch_ang_k1_b1", "ch_ang_k1_b2"):
            _prof += [(_dv, "sd_dR"), (f"sd_{_dv}", "sd_dR")]

        _hu, _ht, _hr = {}, {}, {}
        for _o in _hist_obs:
            _u = histogram(_gen_table, _bins, ("pt", _o), _unf_w).unbind("pt")[1:-1]
            _t = histogram(_gen_table, _bins, ("pt", _o), _truth_w).unbind("pt")[1:-1]
            _hu[_o] = [h.snapshot() for h in _u]
            _ht[_o] = [h.snapshot() for h in _t]
            _hr[_o] = [ratio_snapshot(a, b) for a, b in zip(_hu[_o], _ht[_o])]

        _pu, _ptr, _pr = {}, {}, {}
        for _yo, _x in _prof:
            _u = profile(_gen_table, _bins, ("pt", _x), _yo, _unf_w).unbind("pt")[1:-1]
            _t = profile(_gen_table, _bins, ("pt", _x), _yo, _truth_w).unbind("pt")[1:-1]
            _pu[(_yo, _x)] = [h.snapshot() for h in _u]
            _ptr[(_yo, _x)] = [h.snapshot() for h in _t]
            _pr[(_yo, _x)] = [ratio_snapshot(a, b) for a, b in zip(_pu[(_yo, _x)], _ptr[(_yo, _x)])]

        return dict(
            sys_var=str(sys_var) + " (bin_counts)",
            iteration=_it,
            jpt_bins=_clo_jpt,
            n_pt=_npt,
            hist_unf=_hu,
            hist_truth=_ht,
            hist_ratio=_hr,
            prof_unf=_pu,
            prof_truth=_ptr,
            prof_ratio=_pr,
            _bufs=_bufs,
        )

    return (build_closure_bc,)


@app.cell
def _(SysVar, build_closure_bc):
    # Heavy: build the bin_counts-route LIKE_DATA closure data (obs=angularities,
    # weights=bin_counts). Under-development cross-check; separate from closure_like.
    closure_like_bc = build_closure_bc(SysVar.UNFOLDING_PRIOR_LIKE_DATA)
    return (closure_like_bc,)


@app.cell
def _(closure_like_bc, make_closure_zg):
    # zg profile closure -- DATA-LIKE, bin_counts route (UNDER DEVELOPMENT cross-check)
    fig_zg_123_closure_like_bc = make_closure_zg(
        closure_like_bc,
        "Pseudodata Closure (bin_counts, in dev.)",
        "fig_zg_123_closure_like_bincounts.pdf",
    )
    return


@app.cell
def _(closure_like_bc, make_closure_dist_grid):
    # 2x2 distribution closure -- DATA-LIKE, bin_counts route (UNDER DEVELOPMENT cross-check)
    fig_dist_grid_closure_like_bc = make_closure_dist_grid(
        closure_like_bc,
        "Pseudodata Closure (bin_counts, in dev.)",
        "fig_dist_grid_closure_like_bincounts.pdf",
    )
    return


@app.cell
def _(closure_like_bc, make_closure_dr):
    # sd_dR profile closure -- DATA-LIKE 3-col kappa=1, bin_counts route (UNDER DEVELOPMENT cross-check)
    fig_dr_grid_closure_like_bc = make_closure_dr(
        closure_like_bc,
        "Pseudodata Closure (bin_counts, in dev.)",
        "fig_dr_grid_closure_like_bincounts.pdf",
    )
    return


if __name__ == "__main__":
    app.run()
