import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import json
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    # from adjustText import adjust_text

    import torch

    from systematics import SysVar, get_jet_pt_bins

    with open("./runtime-files/config.json") as _cfg_file:
        _cfg_setup = json.load(_cfg_file)
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

    prefix_dir = Path("./outputs/histograms")

    mc_labels = ("pythia6", "pythia8", "herwig7")
    mc_hist_styles = {
        "pythia6": {"linestyle": "dotted"},
        "pythia8": {"linestyle": "dashdot"},
        "herwig7": {"linestyle": "solid"},
    }

    jpt_bins_to_omit = (0,)

    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols = len(jpt_bins) - 1 - len(jpt_bins_to_omit)
    fig_scale = 5


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

    if plot_type == "errorbar":
        kwargs["xerr"] = hdict["half_bin_width"]
        kwargs["yerr"] = (hdict.get("bin_count_std", hdict["bin_count_err"]),)

    ax_arts = getattr(ax, plot_type)(
        hdict["bin_center"],
        hdict["bin_count"],
        **kwargs,
    )

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
    if sys_err_hdict is None:
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
        points_kwargs = dict(
            linestyle="none",
            marker="*",
            markersize=10,
            markeredgecolor="white",
        )
        sys_err_dict = torch.load(sys_err_path, mmap=True)
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_label: str = rf"$\langle\lambda^{label}\rangle$"
        points_kwargs = dict(linewidth=2)
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
    fig = plt.figure(
        figsize=(num_cols * fig_scale, fig_scale + 0.5),
    )
    axs = fig.subplots(
        1,
        num_cols,
        sharey="row",
        gridspec_kw={"wspace": 0, "right": 0.9, "left": 0.2},
    )

    ax_art_map = {}
    # ax_texts = defaultdict(list)
    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue
        ijpt += 1
        ax_arts = plot_profile_single(
            axs[ijpt],
            "errorbar",
            file_path=prefix_dir
            / str(SysVar.NONE)
            / feature_mode / var_name
            / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
            sys_err_path=prefix_dir
            / "sys_errors"
            / feature_mode / var_name
            / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
            color="blue",
            label="{SD}",
        )
        if var_name != x_var_name:
            ax_arts.update(
                plot_profile_single(
                    axs[ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / str(SysVar.NONE)
                    / feature_mode / var_name
                    / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    sys_err_path=prefix_dir
                    / "sys_errors"
                    / feature_mode / var_name
                    / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                    color="red",
                    label="{incl.}",
                )
            )

        if plot_mc:
            for mc in mc_labels:
                ax_arts.update(
                    plot_profile_single(
                        axs[ijpt],
                        "plot",
                        file_path=prefix_dir
                        / mc
                        / feature_mode / var_name
                        / f"prof_sd_vs_{x_var_name}_jpt{ijpt_true}.pt",
                        label="".join(("{", mc, "}")),
                        color="blue",
                        **(mc_hist_styles[mc]),
                    )
                )

                if var_name != x_var_name:
                    plot_profile_single(
                        axs[ijpt],
                        "plot",
                        file_path=prefix_dir
                        / mc
                        / feature_mode / var_name
                        / f"prof_incl_vs_{x_var_name}_jpt{ijpt_true}.pt",
                        color="red",
                        **(mc_hist_styles[mc]),
                    )
        if ijpt == 0:
            ax_art_map.update(ax_arts)
        # axs[ijpt].set_xlabel(var_xlabel[x_var_name], fontsize="x-large")
        # axs[ijpt].text(
        #    0.05,
        #    0.6,
        #    rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
        #    transform=axs[ijpt].transAxes,
        # )
        axs[ijpt].set_title(
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        )
        if x_var_name in {var_name, "sd_dR"}:
            lims = [
                np.min(
                    [axs[ijpt].get_xlim(), axs[ijpt].get_ylim()]
                ),  # min of both axes
                np.max(
                    [axs[ijpt].get_xlim(), axs[ijpt].get_ylim()]
                ),  # max of both axes
            ]

            axs[ijpt].plot(lims, lims, "--", color="black", alpha=0.3)
            axs[ijpt].set_aspect("equal")
            axs[ijpt].set_xlim(lims)
            axs[ijpt].set_ylim(lims)

        axs[ijpt].set_xlabel(var_xlabel[x_var_name], fontsize="x-large")
        axs[ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[0].set_ylabel(var_prof_ylabel[var_name], fontsize="x-large")
    axs[-1].legend(
        list(ax_art_map.values()),
        list(ax_art_map.keys()),
        frameon=False,
        # loc="upper left",
    )

    # for val in ax_texts.values():
    #    adjust_text(val)

    if save_figs:
        fig_save_dir = prefix_dir / "plots" / feature_mode / var_name
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_save_dir / f"prof_ang_vs_{x_var_name}.pdf"
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


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
    marker = kwargs.pop("marker", "o")
    markersize = kwargs.pop("markersize", 5)
    markeredgecolor = kwargs.pop("markeredgecolor", "white")
    if sys_err_path is not None:
        sys_err_hdict = torch.load(sys_err_path, mmap=True)
        points_kwargs = dict(
            linestyle="none",
            marker=marker,
            markersize=markersize,
            markeredgecolor=markeredgecolor,
        )
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_kwargs = dict(linewidth=2)
        errbar_kwargs = None
        sys_err_hdict = None

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
    fig = plt.figure(
        figsize=(num_cols * fig_scale, fig_scale),
    )
    axs = fig.subplots(
        1,
        num_cols,
        sharey="row",
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

        ax_arts = plot_hist_single(
            axs[ijpt],
            "errorbar",
            file_path=prefix_dir
            / str(SysVar.NONE)
            / feature_mode / var_name
            / f"hist_jpt{ijpt_true}.pt",
            sys_err_path=prefix_dir
            / "sys_errors"
            / feature_mode / var_name
            / f"hist_jpt{ijpt_true}.pt",
            label="inclusive",
            color="red",
        )

        if plot_mc:
            for mc in mc_labels:
                ax_arts.update(
                    plot_hist_single(
                        axs[ijpt],
                        "plot",
                        file_path=prefix_dir
                        / mc
                        / feature_mode / var_name
                        / f"hist_jpt{ijpt_true}.pt",
                        label=mc,
                        color="red",
                        **(mc_hist_styles[mc]),
                    )
                )

        if f"sd_{var_name}" in common_vars:
            ax_arts.update(
                plot_hist_single(
                    axs[ijpt],
                    "errorbar",
                    file_path=prefix_dir
                    / str(SysVar.NONE)
                    / feature_mode / f"sd_{var_name}"
                    / f"hist_jpt{ijpt_true}.pt",
                    sys_err_path=prefix_dir
                    / "sys_errors"
                    / feature_mode / f"sd_{var_name}"
                    / f"hist_jpt{ijpt_true}.pt",
                    label="groomed",
                    color="blue",
                )
            )

            if plot_mc:
                for mc in mc_labels:
                    ax_arts.update(
                        plot_hist_single(
                            axs[ijpt],
                            "plot",
                            file_path=prefix_dir
                            / mc
                            / feature_mode / f"sd_{var_name}"
                            / f"hist_jpt{ijpt_true}.pt",
                            label=mc,
                            color="blue",
                            **(mc_hist_styles[mc]),
                        )
                    )

        if ijpt == 2:
            hist_ax_arts.update(ax_arts)

        # axs[ijpt].text(
        #    0.3,
        #    0.9,
        #    rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
        #    transform=axs[ijpt].transAxes,
        # )
        axs[ijpt].set_title(
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        )
        axs[ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[-1].legend(
        list(hist_ax_arts.values()),
        list(hist_ax_arts.keys()),
        frameon=False,
        # loc="upper left",
        # bbox_to_anchor=(0.45, 0.85),
    )
    # axs[0].set_ylabel(var_hist_ylabel[var_name], fontsize="xx-large")
    # for val in ax_texts.values():
    #    adjust_text(val)

    if save_figs:
        fig_save_dir = prefix_dir / "plots" / feature_mode / var_name
        fig_save_dir.mkdir(parents=True, exist_ok=True)
        fig_save_path = fig_save_dir / f"hist_{var_name}.pdf"
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


@app.function
def plot_ratio_single(
    axs,
    plot_type,
    num_file_paths,
    den_file_paths,
    ratio_file_paths,
    num_label="hist num.",
    den_label="hist den.",
    ratio_label="num./den.",
    **kwargs,
):
    ax_arts = {}
    ax_arts.update(
        plot_hist_single(
            axs[0],
            plot_type,
            file_path=num_file_paths[0],
            sys_err_path=num_file_paths[1],
            color="red",
            label=num_label,
            **kwargs,
        )
    )

    ax_arts.update(
        plot_hist_single(
            axs[0],
            plot_type,
            file_path=den_file_paths[0],
            sys_err_path=den_file_paths[1],
            color="blue",
            marker="^",
            label=den_label,
            **kwargs,
        )
    )

    ax_arts.update(
        plot_hist_single(
            axs[1],
            plot_type,
            file_path=ratio_file_paths[0],
            sys_err_path=ratio_file_paths[1],
            marker="*",
            markersize=10,
            color="magenta" if plot_type == "errorbar" else "black",
            label=ratio_label,
            **kwargs,
        )
    )

    return ax_arts


@app.function
def plot_ratio(
    var_name,
    save_figs=True,
    plot_mc=True,
):
    fig = plt.figure(
        figsize=(num_cols * fig_scale, fig_scale * (1.5)),
    )
    axs = fig.subplots(
        2,
        num_cols,
        height_ratios=[3, 1],
        sharey="row",
        # sharex=True,
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
    # ax_texts = defaultdict(list)

    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue
        ijpt += 1

        num_file_paths = (
            prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"hist_sd_ang_jpt{ijpt_true}.pt",
            prefix_dir / "sys_errors" / feature_mode / var_name / f"hist_sd_ang_jpt{ijpt_true}.pt",
        )
        den_file_paths = (
            prefix_dir / str(SysVar.NONE) / feature_mode / var_name / f"hist_ang_jpt{ijpt_true}.pt",
            prefix_dir / "sys_errors" / feature_mode / var_name / f"hist_ang_jpt{ijpt_true}.pt",
        )
        ratio_file_paths = (
            prefix_dir
            / str(SysVar.NONE)
            / feature_mode / var_name
            / f"ratio_incl_vs_sd_jpt{ijpt_true}.pt",
            prefix_dir
            / "sys_errors"
            / feature_mode / var_name
            / f"ratio_incl_vs_sd_jpt{ijpt_true}.pt",
        )
        ax_arts = plot_ratio_single(
            axs[:, ijpt],
            "errorbar",
            num_file_paths=num_file_paths,
            den_file_paths=den_file_paths,
            ratio_file_paths=ratio_file_paths,
            num_label="groomed",
            den_label="incl.",
            ratio_label=r"\frac{groomed}{incl.}",
        )

        if plot_mc:
            for mc in mc_labels:
                num_file_paths = (
                    prefix_dir / mc / feature_mode / var_name / f"hist_sd_ang_jpt{ijpt_true}.pt",
                    None,
                )
                den_file_paths = (
                    prefix_dir / mc / feature_mode / var_name / f"hist_ang_jpt{ijpt_true}.pt",
                    None,
                )
                ratio_file_paths = (
                    prefix_dir / mc / feature_mode / var_name / f"ratio_incl_vs_sd_jpt{ijpt_true}.pt",
                    None,
                )
                ax_arts.update(
                    plot_ratio_single(
                        axs[:, ijpt],
                        "plot",
                        num_file_paths=num_file_paths,
                        den_file_paths=den_file_paths,
                        ratio_file_paths=ratio_file_paths,
                        num_label=mc,
                        den_label=mc,
                        ratio_label=mc,
                        **(mc_hist_styles[mc]),
                    )
                )

        if ijpt == 2:
            hist_ax_arts.update(ax_arts)

        # axs[0, ijpt].text(
        #    0.3,
        #    0.9,
        #    rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$",
        #    transform=axs[0, ijpt].transAxes,
        # )

        axs[0, ijpt].set_title(
            rf"${jpt_bins[ijpt_true]} < p_{{T, jet}} < {jpt_bins[ijpt_true + 1]}$ GeV/$c$"
        )

        axs[1, ijpt].axhline(y=1, linewidth=2, color="black", linestyle="--", alpha=0.3)
        axs[1, ijpt].set_ylim(0, 2)
        axs[1, ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[1, ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[0, -1].legend(
        list(hist_ax_arts.values()),
        list(hist_ax_arts.keys()),
        frameon=False,
        # loc="upper left",
        # bbox_to_anchor=(0.45, 0.85),
    )
    axs[0, 0].set_ylabel(var_hist_ylabel[var_name], fontsize="xx-large")
    axs[1, 0].set_ylabel(r"$\frac{SD}{incl.}$", fontsize="xx-large")

    # for val in ax_texts.values():
    #    adjust_text(val)

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
