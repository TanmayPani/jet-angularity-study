import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")

with app.setup:
    import os
    from collections import defaultdict

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    # from adjustText import adjust_text

    import torch

    from systematics import SysVar, get_jet_pt_bins

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
        "pt": r"$\frac{1}{N_{jets}}\frac{dN_{jets}{dp_{\rm T, jet}} ((GeV/$c$)^{-1})$",
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

    prefix = "outputs/histograms"

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

    if plot_type in {
        "errorbar",
    }:
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
    htype,
    var_name,
    x_var_name,
    fname_prefix,
    ijpt,
    is_mc=False,
    label="MC",
    **kwargs,
):
    hdict = torch.load(
        os.path.join(
            prefix, htype, var_name, f"{fname_prefix}_vs_{x_var_name}_jpt{ijpt}.pt"
        ),
        mmap=True,
    )
    if not is_mc:
        points_label: str = rf"$\langle\lambda^{label}\rangle\pm\delta_{{sys.}}(\langle \lambda^{label} \rangle)$"
        points_kwargs = dict(
            linestyle="none",
            marker="*",
            markersize=10,
            markeredgecolor="white",
        )
        sys_err_dict = torch.load(
            os.path.join(
                prefix,
                "sys_errors",
                var_name,
                f"{fname_prefix}_vs_{x_var_name}_jpt{ijpt}.pt",
            ),
            mmap=True,
        )
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

    # if not is_mc:
    #     artists[r"$\pm \sigma(\lambda^{SD}) (std. dev.)$"] = plot_error_bars(
    #         ax,
    #         hdict,
    #         hdict["bin_count_err"],
    #         fill=True,
    #         hatch="+++++++",
    #         alpha=0.3,
    #         edgecolor=kwargs.get("color", None),
    #         facecolor="none",
    #     )

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
    ax_texts = defaultdict(list)
    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue
        ijpt += 1

        ax_arts = plot_profile_single(
            axs[ijpt],
            "errorbar",
            str(SysVar.NONE),
            var_name,
            x_var_name,
            "prof_sd",
            ijpt_true,
            color="blue",
            label="{SD}",
        )
        if var_name != x_var_name:
            ax_arts.update(
                plot_profile_single(
                    axs[ijpt],
                    "errorbar",
                    str(SysVar.NONE),
                    var_name,
                    x_var_name,
                    "prof_incl",
                    ijpt_true,
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
                        mc,
                        var_name,
                        x_var_name,
                        "prof_sd",
                        ijpt_true,
                        is_mc=True,
                        label="".join(("{", mc, "}")),
                        color="blue",
                        **(mc_hist_styles[mc]),
                    )
                )

                if var_name != x_var_name:
                    plot_profile_single(
                        axs[ijpt],
                        "plot",
                        mc,
                        var_name,
                        x_var_name,
                        "prof_incl",
                        ijpt_true,
                        is_mc=True,
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
        fig_save_dir = os.path.join(prefix, "plots", var_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, f"prof_ang_vs_{x_var_name}.pdf")
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


@app.function
def plot_ratio_single(
    ax0,
    ax1,
    plot_type,
    htype,
    var_name,
    ijpt,
    is_mc=False,
    mc_label="MC",
    **kwargs,
):
    hist_root_dir = os.path.join(prefix, htype, var_name)
    sys_err_root_dir = os.path.join(prefix, "sys_errors", var_name)
    if not is_mc:
        points_kwargs = dict(linestyle="none", markeredgecolor="white")
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_kwargs = dict(linewidth=2)
        errbar_kwargs = None

    incl_hdict = torch.load(
        os.path.join(hist_root_dir, f"hist_ang_jpt{ijpt}.pt"), mmap=True
    )
    incl_sys_err_hdict = (
        None
        if is_mc
        else torch.load(
            os.path.join(sys_err_root_dir, f"hist_ang_jpt{ijpt}.pt"), mmap=True
        )
    )
    if not is_mc:
        points_kwargs.update(marker="o", markersize=5)
    incl_ax_arts = plot_hist(
        ax0,
        plot_type,
        incl_hdict,
        incl_sys_err_hdict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        color="red",
        **kwargs,
    )

    sd_hdict = torch.load(
        os.path.join(hist_root_dir, f"hist_sd_ang_jpt{ijpt}.pt"), mmap=True
    )
    sd_sys_err_hdict = (
        None
        if is_mc
        else torch.load(
            os.path.join(sys_err_root_dir, f"hist_sd_ang_jpt{ijpt}.pt"), mmap=True
        )
    )
    if not is_mc:
        points_kwargs.update(marker="^", markersize=5)
    sd_ax_arts = plot_hist(
        ax0,
        plot_type,
        sd_hdict,
        sd_sys_err_hdict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        color="blue",
        **kwargs,
    )

    ratio_hdict = torch.load(
        os.path.join(hist_root_dir, f"ratio_incl_vs_sd_jpt{ijpt}.pt"), mmap=True
    )
    ratio_sys_err_hdict = (
        None
        if is_mc
        else torch.load(
            os.path.join(sys_err_root_dir, f"ratio_incl_vs_sd_jpt{ijpt}.pt"),
            mmap=True,
        )
    )
    if not is_mc:
        points_kwargs.update(marker="*", markersize=10)
    ratio_ax_arts = plot_hist(
        ax1,
        plot_type,
        ratio_hdict,
        ratio_sys_err_hdict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        color="magenta" if not is_mc else "black",
        **kwargs,
    )

    if not is_mc:
        return {
            r"$incl. \pm \delta_{sys}$": incl_ax_arts,
            r"$groomed \pm \delta_{sys}$": sd_ax_arts,
            r"$\frac{groomed}{incl.}\pm \delta_{sys}$": ratio_ax_arts,
        }
    else:
        return {mc_label: ratio_ax_arts}


@app.function
def plot_hist_single(
    ax,
    plot_type,
    htype,
    var_name,
    ijpt=None,
    is_mc=False,
    label="hist",
    **kwargs,
):
    hist_root_dir = os.path.join(prefix, htype, var_name)
    sys_err_root_dir = os.path.join(prefix, "sys_errors", var_name)
    if not is_mc:
        points_kwargs = dict(linestyle="none", markeredgecolor="white")
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_kwargs = dict(linewidth=2)
        errbar_kwargs = None

    fname = f"hist_jpt{ijpt}.pt" if ijpt is not None else "hist.pt"
    incl_hdict = torch.load(
        os.path.join(hist_root_dir, fname),
        mmap=True,
    )
    incl_sys_err_hdict = (
        None if is_mc else torch.load(os.path.join(sys_err_root_dir, fname), mmap=True)
    )
    if not is_mc:
        points_kwargs.update(marker="o", markersize=5)
    ax_arts = plot_hist(
        ax,
        plot_type,
        incl_hdict,
        incl_sys_err_hdict,
        points_kwargs=points_kwargs,
        errbar_kwargs=errbar_kwargs,
        **kwargs,
    )

    return {
        rf"${label} \pm \delta_{{sys}}$" if not is_mc else label: ax_arts,
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
    # ax_texts = defaultdict(list)

    ijpt = -1
    for ijpt_true in range(len(jpt_bins) - 1):
        if ijpt_true in jpt_bins_to_omit:
            continue

        ijpt += 1

        ax_arts = plot_hist_single(
            axs[ijpt],
            "errorbar",
            str(SysVar.NONE),
            var_name,
            ijpt,
            label="inclusive",
            color="red",
        )

        if plot_mc:
            for mc in mc_labels:
                ax_arts.update(
                    plot_hist_single(
                        axs[ijpt],
                        "plot",
                        mc,
                        var_name,
                        ijpt_true,
                        is_mc=True,
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
                    str(SysVar.NONE),
                    f"sd_{var_name}",
                    ijpt_true,
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
                            mc,
                            f"sd_{var_name}",
                            ijpt_true,
                            is_mc=True,
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
        fig_save_dir = os.path.join(prefix, "plots", var_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, f"hist_{var_name}.pdf")
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches="tight")


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

        ax_arts = plot_ratio_single(
            axs[0, ijpt],
            axs[1, ijpt],
            "errorbar",
            str(SysVar.NONE),
            var_name,
            ijpt,
        )

        if plot_mc:
            for mc in mc_labels:
                ax_arts.update(
                    plot_ratio_single(
                        axs[0, ijpt],
                        axs[1, ijpt],
                        "plot",
                        mc,
                        var_name,
                        ijpt_true,
                        is_mc=True,
                        mc_label=mc,
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
        fig_save_dir = os.path.join(prefix, "plots", var_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, "ratio_incl_vs_hc.pdf")
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
