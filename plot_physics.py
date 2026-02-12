import os
from copy import deepcopy
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

import torch

from systematics import SysVar, get_jet_pt_bins
from utils.histogram import TorchHist1D, TorchHist2D

jet_columns = [
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1"  ,
    "ch_ang_k1_b2"  ,
    "ch_ang_k2_b0"  ,
]

var_xlabel = {
    "ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1}_{\beta = 0.5}$ (LHA)"          ,
    "ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1}_{\beta = 1}$ (girth)"          , 
    "ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1}_{\beta = 2}$ (thrust)"         ,
    "ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2}_{\beta = 0}$ ($(p_T^D)^2$)"    ,
    "sd_ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 0.5}$ (LHA, SD)"          ,
    "sd_ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 1}$ (girth, SD)"          ,
    "sd_ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1, \rm SD}_{\beta = 2}$ (thrust, SD)"         ,
    "sd_ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2, \rm SD}_{\beta = 0}$ ($(p_T^D)^2$, SD)"    ,
}

var_hist_ylabel = {
    "pt"            :r"$\frac{1}{N_{jets}}\frac{dN_{jets}{dp_{\rm T, jet}} ((GeV/$c$)^{-1})$"     ,
    "ch_ang_k1_b0.5":r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 0.5}}$ " ,
    "ch_ang_k1_b1"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 1}}$"    , 
    "ch_ang_k1_b2"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 2}}$"    ,
    "ch_ang_k2_b0"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 2}_{\beta = 0}}$"    ,
}

var_prof_ylabel = {
    "ch_ang_k1_b0.5":r"$\langle\lambda^{\kappa = 1, SD}_{\beta = 0.5}\rangle$ " ,
    "ch_ang_k1_b1"  :r"$\langle\lambda^{\kappa = 1, SD}_{\beta = 1}\rangle$"    , 
    "ch_ang_k1_b2"  :r"$\langle\lambda^{\kappa = 1, SD}_{\beta = 2}\rangle$"    ,
    "ch_ang_k2_b0"  :r"$\langle\lambda^{\kappa = 2, SD}_{\beta = 0}\rangle$"    ,
}

fname_mods = (
    "h1_prof_incl_vs_sd", 
    "h1_projY_ang", 
    "h1_projY_sd_ang",
    "h1_ratio_incl_vs_sd"
)

def plot_data_points(ax, plot_type, hdict, **kwargs):
    hdict["bin_count"]     = hdict["bin_count"].nan_to_num_(nan=0, posinf=0, neginf=0)
    hdict["bin_count_err"] = hdict["bin_count_err"].nan_to_num_(nan=0, posinf=0, neginf=0)
    if "bin_count_std" in hdict:
        hdict["bin_count_std"] = hdict["bin_count_std"].nan_to_num_(nan=0, posinf=0, neginf=0)
    
    if plot_type in {"errorbar",}:
        kwargs["xerr"] = hdict["half_bin_width"] 
        kwargs["yerr"] = hdict.get("bin_count_std", hdict["bin_count_err"]),
    
    return getattr(ax, plot_type)(
        hdict["bin_center"], 
        hdict["bin_count"], 
        **kwargs,
    )

def plot_error_bars(ax, hdict, sys_err, **kwargs):
    bin_edges = hdict["bin_center"] - hdict["half_bin_width"]
    last_bin_edge = (hdict["bin_center"][-1] + hdict["half_bin_width"][-1]).unsqueeze_(0)
    bin_edges = torch.concatenate((bin_edges, last_bin_edge))
    
    sys_err.nan_to_num_(nan=0, posinf=0, neginf=0)
    return ax.stairs(
        hdict["bin_count"] + sys_err, 
        bin_edges, 
        baseline=(hdict["bin_count"]-sys_err).numpy(),
        **kwargs
    )

def plot_hist(
    ax, plot_type,
    hdict, 
    sys_err_hdict = None,
    points_kwargs = None,
    errbar_kwargs = None,
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
       
def plot_profile_single(
    ax, plot_type,
    path_prefix, 
    htype,
    var_name,
    ijpt, 
    is_mc=False,
    mc_label="MC",
    **kwargs
):
    hdict = torch.load(
        os.path.join(path_prefix, htype, var_name, f"h1_prof_incl_vs_sd_jpt{ijpt}.pt"),
        mmap=True,
    )
    if not is_mc:
        points_label : str = r"$\langle\lambda^{SD}\rangle\pm\delta_{sys.}(\langle \lambda^{SD} \rangle)$"
        points_kwargs = dict(
            linestyle="none", marker = "*", markersize = 10, markeredgecolor="white",
        )
        sys_err_dict = torch.load(
            os.path.join(path_prefix, "sys_errors", var_name, f"h1_prof_incl_vs_sd_jpt{ijpt}.pt"),
            mmap=True,
        )
        errbar_kwargs = dict(fill=True, alpha=0.5) 
        kwargs.update(color = "magenta")
    else:
        points_label : str = fr"$\langle\lambda^{{SD}}_{mc_label}\rangle"
        points_kwargs = dict(linewidth=2)
        sys_err_dict = None
        errbar_kwargs = None
        kwargs.update(color = "black")


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

    if not is_mc:
        artists[r"$\pm \sigma(\lambda^{SD}) (std. dev.)$"] = plot_error_bars(
            ax, hdict, hdict["bin_count_err"],
            fill=True,
            hatch="+++++++", 
            alpha=0.3, 
            edgecolor = "magenta",
            facecolor = "none",
        )

    return artists

def plot_profile(
    var_name, 
    fig_scale=5, 
    path_prefix="outputs/histograms", 
    save_figs=True, 
    plot_mc=True,
):
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols = len(jpt_bins) - 1
    fig = plt.figure(
        figsize=(num_cols * fig_scale, fig_scale+0.5),
    )
    axs = fig.subplots(
        1, num_cols, 
        sharey="row",
        gridspec_kw={"wspace": 0, "right":0.9, "left":0.2}, 
    )
    
    ax_art_map = {}
    for ijpt in range(num_cols):
        ax_arts = plot_profile_single(
            axs[ijpt],
            "errorbar",
            path_prefix, 
            str(SysVar.NONE),
            var_name,
            ijpt,
        )

        if plot_mc:
            ax_arts.update(plot_profile_single(
                axs[ijpt],
                "plot",
                path_prefix,
                "pythia6",
                var_name,
                ijpt,
                is_mc=True,
                mc_label = "PYTHIA-6",
                linestyle="dotted"
            ))

            #if ijpt == 3:
            #    continue

            ax_arts.update(plot_profile_single(
                axs[ijpt], 
                "plot",
                path_prefix,
                htype="pythia8", 
                var_name=var_name,
                ijpt=ijpt,
                is_mc=True,
                mc_label = "PYTHIA-8",
                linestyle="dashdot"
            ))
        if ijpt == 0:
            ax_art_map.update(ax_arts)
        axs[ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        lims = [
            np.min([axs[ijpt].get_xlim(), axs[ijpt].get_ylim()]),  # min of both axes
            np.max([axs[ijpt].get_xlim(), axs[ijpt].get_ylim()]),  # max of both axes
        ]

        axs[ijpt].text(
            0.05, 0.6, 
            fr"${jpt_bins[ijpt]} < p_{{T, jet}} < {jpt_bins[ijpt+1]}$ GeV/$c$", 
            transform=axs[ijpt].transAxes,
        )
        axs[ijpt].plot(lims, lims, '--', color="black", alpha=0.3)
        axs[ijpt].set_aspect('equal')
        axs[ijpt].set_xlim(lims)
        axs[ijpt].set_ylim(lims)
        axs[ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[0].set_ylabel(var_prof_ylabel[var_name], fontsize="x-large")
    axs[-1].legend(list(ax_art_map.values()), list(ax_art_map.keys()), frameon=False, loc="upper left")

    if save_figs:
        fig_save_dir = os.path.join(path_prefix, "plots", var_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, "profile_incl_vs_hc.pdf")
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches='tight')
 
def plot_ratio_single(
    ax0, ax1, plot_type,
    path_prefix, 
    htype, 
    var_name,
    ijpt, 
    is_mc=False,
    mc_label="MC",
    **kwargs,
):
    hist_root_dir = os.path.join(path_prefix, htype, var_name)
    sys_err_root_dir = os.path.join(path_prefix, "sys_errors", var_name)
    if not is_mc:
        points_kwargs = dict(linestyle="none", markeredgecolor="white")
        errbar_kwargs = dict(fill=True, alpha=0.5)
    else:
        points_kwargs = dict(linewidth=2)
        errbar_kwargs = None

    incl_hdict = torch.load(os.path.join(hist_root_dir, f"h1_projY_ang_jpt{ijpt}.pt"), mmap=True)
    incl_sys_err_hdict = None if is_mc else torch.load(os.path.join(sys_err_root_dir, f"h1_projY_ang_jpt{ijpt}.pt"), mmap=True)
    if not is_mc:
        points_kwargs.update(marker = "o", markersize = 5)
    incl_ax_arts = plot_hist(
        ax0, 
        plot_type,
        incl_hdict,  
        incl_sys_err_hdict,
        points_kwargs = points_kwargs,
        errbar_kwargs = errbar_kwargs,
        color="red", 
        **kwargs,
    )
    
    sd_hdict = torch.load(os.path.join(hist_root_dir, f"h1_projY_sd_ang_jpt{ijpt}.pt"), mmap=True)
    sd_sys_err_hdict = None if is_mc else torch.load(os.path.join(sys_err_root_dir, f"h1_projY_sd_ang_jpt{ijpt}.pt"), mmap=True)
    if not is_mc:
        points_kwargs.update(marker = "^", markersize = 5)
    sd_ax_arts = plot_hist(
        ax0, 
        plot_type,
        sd_hdict,  
        sd_sys_err_hdict,   
        points_kwargs = points_kwargs,
        errbar_kwargs = errbar_kwargs,
        color="blue",
        **kwargs,
    )

    ratio_hdict = torch.load(os.path.join(hist_root_dir, f"h1_ratio_incl_vs_sd_jpt{ijpt}.pt"), mmap=True)
    ratio_sys_err_hdict = None if is_mc else torch.load(os.path.join(sys_err_root_dir, f"h1_ratio_incl_vs_sd_jpt{ijpt}.pt"), mmap=True)
    if not is_mc:
        points_kwargs.update(marker = "*", markersize = 10)
    ratio_ax_arts = plot_hist(
        ax1, 
        plot_type,
        ratio_hdict, 
        ratio_sys_err_hdict, 
        points_kwargs = points_kwargs,
        errbar_kwargs = errbar_kwargs,
        color="magenta" if not is_mc else "black", 
        **kwargs,
    )
   
    if not is_mc:
        return {
            r"$incl. \pm \delta_{sys}$" : incl_ax_arts, 
            r"$groomed \pm \delta_{sys}$" : sd_ax_arts, 
            r"$\frac{groomed}{incl.}\pm \delta_{sys}$" : ratio_ax_arts,
        }
    else:
        return {mc_label : ratio_ax_arts[0]}
    
 
def plot_ratio(var_name, fig_scale=5, path_prefix="outputs/histograms", save_figs=True, plot_mc=True):
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols = len(jpt_bins) - 1
    fig = plt.figure(
        figsize=(num_cols * fig_scale, fig_scale * (1.5)),
    )
    axs = fig.subplots(
        2, num_cols, 
        height_ratios=[3, 1],
        sharey="row",
        #sharex=True, 
        sharex="col", 
        gridspec_kw=dict(
            hspace = 0, wspace = 0, 
            right = 0.9, left = 0.2, 
            top = 0.9, bottom = 0.2,
        )
    )
    
    hist_ax_arts = {}
    mc_ax_arts = {}
    for ijpt in range(num_cols):
        ax_arts = plot_ratio_single(
            axs[0, ijpt], axs[1, ijpt], 
            "errorbar", 
            path_prefix,
            str(SysVar.NONE),
            var_name,
            ijpt,
        )
        
        if plot_mc:
            ax_arts.update(plot_ratio_single(
                axs[0, ijpt], axs[1, ijpt],
                "plot",
                path_prefix,
                "pythia6",
                var_name,
                ijpt,
                is_mc=True,
                mc_label = "PYTHIA6",
                linestyle="dotted"
            ))
            
            #if ijpt == 3:
            #    continue
            ax_arts.update(
                plot_ratio_single(
                    axs[0, ijpt], axs[1, ijpt],
                    "plot",
                    path_prefix,
                    "pythia8",
                    var_name,
                    ijpt,
                    is_mc=True,
                    mc_label = "PYTHIA8",
                    linestyle="dashdot"
                )
            )

            #if ijpt == 0:
            #    mc_ax_arts.update( ax_arts_mc)
        if ijpt == 0:
            hist_ax_arts.update(ax_arts) 

        axs[0,ijpt].text(
            0.3, 0.9, 
            fr"${jpt_bins[ijpt]} < p_{{T, jet}} < {jpt_bins[ijpt+1]}$ GeV/$c$", 
            transform=axs[0,ijpt].transAxes,
        )

        #axs[0, ijpt].set_yscale("log")
        
        axs[1, ijpt].axhline(y=1, linewidth=2, color="black",linestyle="--", alpha=0.3)
        axs[1, ijpt].set_ylim(0, 2)
        axs[1, ijpt].set_xlabel(var_xlabel[var_name], fontsize="x-large")
        axs[1, ijpt].xaxis.set_major_formatter(FormatStrFormatter("%g"))

    axs[0, 3].legend(list(hist_ax_arts.values()), list(hist_ax_arts.keys()), frameon=False, loc="upper left", bbox_to_anchor=(0.45, 0.85))
    #axs[0, 3].legend(list(mc_ax_arts.values()), list(mc_ax_arts.keys()), frameon=False, loc="upper left", bbox_to_anchor=(0.45, 0.85))
    axs[0,0].set_ylabel(var_hist_ylabel[var_name], fontsize="xx-large")
    axs[1,0].set_ylabel(r"$\frac{SD}{incl.}$", fontsize="xx-large")
    
    if save_figs:
        fig_save_dir = os.path.join(path_prefix, "plots", var_name)
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, "ratio_incl_vs_hc.pdf")
        print("Saving figure to:", fig_save_path)
        fig.savefig(fig_save_path, bbox_inches='tight')

   

if __name__ == "__main__":
    save_figs = True
    plot_mc = True
    for var_name in jet_columns:
        plot_ratio(var_name, save_figs=save_figs, plot_mc=plot_mc)
        plot_profile(var_name, save_figs=save_figs, plot_mc=plot_mc)
    plt.show() 



