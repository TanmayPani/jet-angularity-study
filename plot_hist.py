import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt 

import torch

from systematics import SysVar, get_jet_pt_bins
from utils.histogram import TorchHist1D, TorchHist2D

jet_columns = [
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1"  ,
    "ch_ang_k1_b2"  ,
    "ch_ang_k2_b0"  ,
]

var_axlabel = {
    "pt"            :r"$p_{\rm T, jet} (GeV/$c$)$"                          ,
    "nef"           :r"$(\sum p_{\rm T, jet}^{\rm neutral})/p_{\rm T, jet}$",
    "ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1}_{\beta = 0.5}$ (LHA)"          ,
    "ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1}_{\beta = 1}$ (girth)"          , 
    "ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1}_{\beta = 2}$ (thrust)"         ,
    "ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2}_{\beta = 0}$ ($(p_T^D)^2$)"    ,
    "leading_constit_pt"   : r"$p_{\rm T, constit.}^{\rm leading} (GeV/$c$)$",
    "subleading_constit_pt " :r"$p_{\rm T, constit.}^{\rm sub-leading} (GeV/$c$)$",
    "hc_pt"            :r"$p_{\rm T, jet}^{\rm h.c.} (GeV/$c$)$"                          ,
    "hc_ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 0.5}$ (LHA, h.c.)"          ,
    "hc_ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 1}$ (girth, h.c.)"          ,
    "hc_ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 2}$ (thrust, h.c.)"         ,
    "hc_ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2, \rm h.c.}_{\beta = 0}$ ($(p_T^D)^2$, h.c.)"    ,
}                      

var_ax_ylabel = {
    "pt"            :r"$\frac{1}{N_{jets}}\frac{dN_{jets}{dp_{\rm T, jet}} ((GeV/$c$)^{-1})$"     ,
    "ch_ang_k1_b0.5":r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 0.5}}$ " ,
    "ch_ang_k1_b1"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 1}}$"    , 
    "ch_ang_k1_b2"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 1}_{\beta = 2}}$"    ,
    "ch_ang_k2_b0"  :r"$\frac{1}{N_{jets}}\frac{dN_{jets}}{\lambda^{\kappa = 2}_{\beta = 0}}$"    ,
}

var_prof_ax_ylabel = {
    "ch_ang_k1_b0.5":r"$\langle\lambda^{\kappa = 1, h.c.}_{\beta = 0.5}\rangle$ " ,
    "ch_ang_k1_b1"  :r"$\langle\lambda^{\kappa = 1, h.c.}_{\beta = 1}\rangle$"    , 
    "ch_ang_k1_b2"  :r"$\langle\lambda^{\kappa = 1, h.c.}_{\beta = 2}\rangle$"    ,
    "ch_ang_k2_b0"  :r"$\langle\lambda^{\kappa = 2, h.c.}_{\beta = 0}\rangle$"    ,
}


def plot_mean_with_std_band(ax, hist_file, color="", **kwargs):
    h = torch.load(hist_file, mmap=True)
    #print(
    #    h["bin_center"].shape, 
    #    h["bin_count"].shape, 
    #    h["half_bin_width"].shape, 
    #    h["bin_count_err"].shape,
    #)
    ax_points = ax.errorbar(
        h["bin_center"], 
        h["bin_count"], 
        xerr=h["half_bin_width"], 
        yerr=h["bin_count_err"],
        linestyle="none", color=color, 
        marker="o", markeredgecolor="white", markerfacecolor=color,
    )
   
    if "bin_count_std" in h:
        ax_errband = ax.fill_between(
            h["bin_center"], 
            h["bin_count"]-h["bin_count_std"], 
            h["bin_count"]+h["bin_count_std"], 
            alpha=0.5, 
            color=color,
        )
        return (ax_points, ax_errband)
    else:
        return ax_points
    del h

def main_closure_like(sys_var, load_prefix=None):
    jpt_bins = get_jet_pt_bins(sys_var)
    jpt_bin_labels =[
        fr"${pt_min} < p_{{T, jet}} < {pt_max}$ GeV/$c$" 
        for pt_min, pt_max in zip(jpt_bins[:-1], jpt_bins[1:])
    ]

    if load_prefix is None:
        load_prefix = "outputs/histograms"
    load_prefix = os.path.join(load_prefix, str(sys_var))

    num_cols =len(jpt_bin_labels) 
    fig_scale = 4

    fig = defaultdict(dict)
    for var_name in jet_columns:
        load_dir = os.path.join(load_prefix, var_name)
        print("Plotting from", load_dir, "...")
        
        for mod in ("h1_prof_incl_vs_hc", "h1_projY_ang", "h1_projY_hc_ang"):
            is_prof = "prof" in mod
            y_label = var_prof_ax_ylabel[var_name] if is_prof else var_ax_ylabel[var_name] 
            fig[var_name][mod] = plt.figure(
                figsize=(num_cols * fig_scale, fig_scale),
            )
            #fig[var_name].suptitle("i vs hard-core")
                    
            ax = fig[var_name][mod].subplots(
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
            ax_arts = []
            ax_labels = []

            for ipt_bin in range(num_cols):
                print("Plotting ratios jet-pt bin:", ipt_bin, ", for:", var_name )
                #print(x[var_name].shape, h1_count_mean[f"projY_{var_name}"][ipt_bin].shape)
                unf_plot, unf_errband = plot_mean_with_std_band(
                    ax[0, ipt_bin],
                    os.path.join(load_dir, "unfolded", f"{mod}_jpt{ipt_bin}.pt"), 
                    color="red",
                ) 
                if ipt_bin == 0:
                    ax_arts.append((unf_plot, unf_errband))
                    ax_labels.append("unfolded (unf.)")

                #print("--- h.c.", x[var_name].shape, h1_count_stacked[f"projY_hc_{var_name}"][ipt_bin].shape)
                truth_plot = plot_mean_with_std_band(
                    ax[0, ipt_bin],
                    os.path.join(load_dir, "truth", f"{mod}_jpt{ipt_bin}.pt"),
                    color="blue",
                )
                
                if ipt_bin == 0:
                    ax_arts.append((truth_plot,))
                    ax_labels.append("truth")

                ax[0, ipt_bin].text(0.05, 0.3, jpt_bin_labels[ipt_bin], transform=ax[0, ipt_bin].transAxes)
                if not is_prof:
                    ax[0, ipt_bin].set_yscale("log")
                ratio_plot, ratio_errband = plot_mean_with_std_band(
                    ax[1, ipt_bin],
                    os.path.join(load_dir, "ratio", f"{mod}_jpt{ipt_bin}.pt"), 
                    color="magenta",
                )
                ax[1, ipt_bin].axhline(y=1, linewidth=2, color="black", alpha=0.5)
                ax[1, ipt_bin].set_ylim(0, 2)
                ax[1, ipt_bin].set_xlabel(var_axlabel[var_name], fontsize="x-large")
            ax[1,0].set_ylabel("unf./truth", fontsize="x-large")
            ax[0,0].set_ylabel(y_label, fontsize="x-large")
            ax[0,num_cols-1].legend(ax_arts, ax_labels)
            

def main_nominal_like(sys_var, load_prefix=None):
    jpt_bins = get_jet_pt_bins(sys_var)
    jpt_bin_labels =[
        fr"${pt_min} < p_{{T, jet}} < {pt_max}$ GeV/$c$" 
        for pt_min, pt_max in zip(jpt_bins[:-1], jpt_bins[1:])
    ]

    if load_prefix is None:
        load_prefix = "outputs/histograms"
    load_prefix = os.path.join(load_prefix, str(sys_var))

    num_cols =len(jpt_bin_labels) 
    fig_scale = 4

    prof_fig = {}
    fig = {}
    
    for var_name in jet_columns:
        load_dir = os.path.join(load_prefix, var_name)
        print("Plotting from", load_dir, "...")

        prof_fig[var_name] = plt.figure(
            figsize=(num_cols * fig_scale, fig_scale),
        )
        prof_fig[var_name].suptitle("inclusive vs hard-core")

        axes = prof_fig[var_name].subplots(
            1, num_cols, 
            sharey=True, 
            gridspec_kw={"wspace": 0, "right":0.9, "left":0.2}, 
        )
        
        for ipt_bin, ax in enumerate(axes):
            print("Plotting profiles for jet-pt bin:", ipt_bin, ", for:", var_name )
            
            _, _ = plot_mean_with_std_band(
                ax, 
                os.path.join(load_dir, f"h1_prof_incl_vs_hc_jpt{ipt_bin}.pt"),
                color = "magenta",
            )
            
            lims = [
                np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
            ax.plot(lims, lims, '--', alpha=0.5)
            ax.set_aspect('equal')
            ax.set_xlim(lims)
            ax.set_ylim(lims)
            
            ax.text(0.1, 0.8, jpt_bin_labels[ipt_bin], transform=ax.transAxes)
            # now plot both limits against eachother
            ax.set_xlabel(var_axlabel[var_name], fontsize="x-large")
        axes[0].set_ylabel(var_axlabel[f"hc_{var_name}"], fontsize="x-large")
        #prof_fig[var_name].savefig(f"slides/unfolded_hist/profile_niter{iteration}.pdf")

        fig[var_name] = plt.figure(
            figsize=(num_cols * fig_scale, fig_scale),
        )
        fig[var_name].suptitle("inclusive vs hard-core")
                
        ax = fig[var_name].subplots(
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
        ax_arts = []
        ax_labels = []
        for ipt_bin in range(num_cols):
            print("Plotting ratios jet-pt bin:", ipt_bin, ", for:", var_name )
            #print(x[var_name].shape, h1_count_mean[f"projY_{var_name}"][ipt_bin].shape)
            incl_plot, incl_errband = plot_mean_with_std_band(
                ax[0, ipt_bin],
                os.path.join(load_dir, f"h1_projY_ang_jpt{ipt_bin}.pt"), 
                color="red",
            ) 
            if ipt_bin == 0:
                ax_arts.append((incl_plot, incl_errband))
                ax_labels.append("inclusive (incl.)")

            #print("--- h.c.", x[var_name].shape, h1_count_stacked[f"projY_hc_{var_name}"][ipt_bin].shape)
            hc_plot, hc_errband = plot_mean_with_std_band(
                ax[0, ipt_bin],
                os.path.join(load_dir, f"h1_projY_hc_ang_jpt{ipt_bin}.pt"), 
                color="blue",
            )
            
            if ipt_bin == 0:
                ax_arts.append((hc_plot, hc_errband))
                ax_labels.append("hard-core (h.c.)")

            ax[0, ipt_bin].text(0.05, 0.3, jpt_bin_labels[ipt_bin], transform=ax[0, ipt_bin].transAxes)
            ax[0, ipt_bin].set_yscale("log")

            ratio_plot, ratio_errband = plot_mean_with_std_band(
                ax[1, ipt_bin],
                os.path.join(load_dir, f"h1_ratio_incl_vs_hc_jpt{ipt_bin}.pt"), 
                color="magenta",
            )
            ax[1, ipt_bin].axhline(y=1, linewidth=2, color="black", alpha=0.5)
            ax[1, ipt_bin].set_ylim(0, 2)
            ax[1, ipt_bin].set_xlabel(var_axlabel[var_name], fontsize="x-large")
        ax[1,0].set_ylabel("h.c./incl.", fontsize="x-large")
        ax[0,0].set_ylabel(var_ax_ylabel[var_name], fontsize="x-large")
        ax[0,num_cols-1].legend(ax_arts, ax_labels)
        #fig[var_name].savefig(f"slides/unfolded_hist/ratio_{var_name}_niter{iteration}.pdf")

if __name__ == "__main__":
    sys_var = SysVar.UNFOLDING_PRIOR
    if sys_var == SysVar.UNFOLDING_PRIOR:
        main_closure_like(sys_var)
    else:
        main_nominal_like(sys_var)

    plt.show()




