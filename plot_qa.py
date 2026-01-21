import os
from math import ceil
from collections import defaultdict
import json
import pickle

import numpy as np
import matplotlib.pyplot as plt 

import pyarrow as pa

import torch

from process_arrow import add_extra_columns
from utils.histogram import TorchHist1D, TorchHist2D

var_xlabel = {
    "pt"            :r"$p_{\rm T, jet} (GeV/$c$)$"                          ,
    "nef"           :r"$(\sum p_{\rm T, jet}^{\rm neutral})/p_{\rm T, jet}$",
    "ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1}_{\beta = 0.5}$ (LHA)"          ,
    "ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1}_{\beta = 1}$ (girth)"          , 
    "ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1}_{\beta = 2}$ (thrust)"         ,
    "ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2}_{\beta = 0}$ ($(p_T^D)^2$)"    ,
    "leading_constit_pt"   : r"$p_{\rm T, constit.}^{\rm leading} (GeV/$c$)$",
    "subleading_constit_pt " :r"$p_{\rm T, constit.}^{\rm sub-leading} (GeV/$c$)$",
    "sd_pt"            :r"$p_{\rm T, jet}^{\rm h.c.} (GeV/$c$)$"                          ,
    "sd_ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 0.5}$ (LHA, h.c.)"          ,
    "sd_ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 1}$ (girth, h.c.)"          ,
    "sd_ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 2}$ (thrust, h.c.)"         ,
    "sd_ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2, \rm h.c.}_{\beta = 0}$ ($(p_T^D)^2$, h.c.)"    ,
} 

if __name__ == "__main__":
    source_dir : str = "/home/tanmaypani/star-workspace/jet-angularity-study/datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding/nominal"
    
    buffers = {}
    tables = {}
    
    buffers["gen_matched"] = pa.memory_map(os.path.join(source_dir, "gen-matches.arrow"), "rb")
    tables["gen_matched"] = pa.ipc.open_file(buffers["gen_matched"]).read_all()
    buffers["reco_matched"] = pa.memory_map(os.path.join(source_dir, "reco-matches.arrow"), "rb")
    tables["reco_matched"] = pa.ipc.open_file(buffers["reco_matched"]).read_all()


    with open('outputs/omnisequential_1/omniseq-bins10.pkl', 'rb') as file:
        _bins = pickle.load(file)
    bins = {} 
    x = {}
    x_err = {}
    #hc_jet_columns = [f"hc_{col}" for col in jet_columns]
    for var_name, binning in _bins.items():
        bins[var_name] = torch.as_tensor(binning)
        x[var_name] = 0.5*(bins[var_name][1:]+bins[var_name][:-1])
        x_err[var_name] = 0.5*(bins[var_name][1:]-bins[var_name][:-1])
 
    weights = {}
    weights_sq = {}

    resp2d   = {} 
    resp2d_bin_counts = {} 
    resp2d_bin_errors = {} 
    
    resp_prof   = {} 
    resp_prof_bin_counts = {} 
    resp_prof_bin_errors = {} 

    reso2d = {}
    reso2d_bin_counts = {}
    reso2d_bin_errors = {}

    reso1d = {}
    reso1d_bin_counts = {}
    reso1d_bin_errors = {}

    reso_bin_counts = {}
    reso_bin_errors = {}
 
    for key, pa_table in tables.items():
        weights[key] = torch.as_tensor(
            pa_table["weight"].to_numpy(), 
            dtype=torch.float32
        )
        weights_sq[key] = weights[key]*weights[key]
       
    
    var_name = "sd_pt"
    x_arr = torch.as_tensor(tables["gen_matched"][var_name].to_numpy(), dtype=torch.float32)
    y_arr = torch.as_tensor(tables["reco_matched"][var_name].to_numpy(), dtype=torch.float32)
    x_bins = torch.linspace(5, 70, 14)
    y_bins = torch.linspace(10, 70, 13)
    resp2d[var_name] = TorchHist2D(x_arr, y_arr, x_bins, y_bins, overflow=False)
    resp2d_bin_counts[var_name], resp2d_bin_errors[var_name] = resp2d[var_name].histogram(
        weights["gen_matched"], weights_sq["gen_matched"],
    )
    resp_prof_bin_counts, resp_prof_bin_errors = resp2d[var_name].profileX(weights["gen_matched"])
    fig_resp, ax_resp = plt.subplots(figsize=(7, 7))
    ax_resp.pcolormesh(
        resp2d[var_name].xbins[:-1], 
        resp2d[var_name].ybins[:-1], 
        resp2d_bin_counts[var_name][:-1, :-1], 
        cmap="jet", norm="log",
    )
    ax_resp.errorbar(
        resp2d[var_name].h_xbin_centers[2:-1],
        resp_prof_bin_counts[2:-1],
        xerr=resp2d[var_name].h_xbin_widths[2:-1]/2., 
        yerr=resp_prof_bin_errors[2:-1],
        color="magenta", linestyle="none", 
        marker="*", markersize=15, 
        markeredgecolor="white", markerfacecolor="magenta",
        label = r"$\langle p_{\rm T, jet}^{\rm det. lvl.} \rangle \pm \sigma(p_{\rm T, jet}^{\rm det. lvl.})$",
    )
    lims = (
        np.min([ax_resp.get_xlim(), ax_resp.get_ylim()]),  # min of both axes
        np.max([ax_resp.get_xlim(), ax_resp.get_ylim()]),  # max of both axes
    )
    ax_resp.plot(lims, lims, '--', alpha=0.5, color="black")
    ax_resp.set_aspect('equal')
    ax_resp.set_xlim(lims)
    ax_resp.set_ylim(lims)
    ax_resp.set_xlabel(r"$p_{\rm T, jet}^{\rm part. lvl.}$ (GeV/$c$)", fontsize="x-large")
    ax_resp.set_ylabel(r"$p_{\rm T, jet}^{\rm det. lvl.}$ (GeV/$c$)", fontsize="x-large")
    ax_resp.legend()
    fig_resp.savefig(f"./outputs/histograms/plots/jet_{var_name}_response.pdf" ,bbox_inches='tight')

    dyx_arr = (y_arr - x_arr).div_(x_arr)
    dyx_bins = torch.linspace(-1, 1, 20)
    jpt_bins = [10, 15, 20, 40, 80] 
    
    reso2d[var_name] = TorchHist2D(x_arr, dyx_arr, jpt_bins, dyx_bins)
    reso2d_bin_counts[var_name], reso2d_bin_errors[var_name] = reso2d[var_name].histogram(
        weights["gen_matched"], weights_sq["gen_matched"],
    )
    reso_bin_counts[var_name], reso_bin_errors[var_name] = reso2d[var_name].profileX(weights["gen_matched"])      
    
    reso_projY = defaultdict(list)
        
    for i, (pt_min, pt_max) in enumerate(zip(jpt_bins[:-1], jpt_bins[1:])):
        bin_counts, bin_errors= reso2d[var_name].projY(xrange = (pt_min+0.5, pt_max-0.5), is_batched=False)
        reso_projY["bin_counts"].append(bin_counts)
        reso_projY["bin_errors"].append(bin_errors)
        reso_projY["bin_centers"].append(reso2d[var_name].h_ybin_centers) 
        reso_projY["bin_xerrors"].append(reso2d[var_name].h_ybin_widths/2.) 
        reso_projY["labels"].append(rf"${pt_min} < p_{{T, jet}} < {pt_max}$ GeV/$c$")
    
    fig_reso, ax_reso = plt.subplots(1, 2, figsize=(10, 7))
    ax_reso[0].pcolormesh(
        reso2d[var_name].xbins[1:-1], 
        reso2d[var_name].ybins[1:-1], 
        reso2d_bin_counts[var_name][1:-1, 1:-1], 
        cmap="jet", norm="log",
    )

    ax_reso[0].errorbar(
        reso2d[var_name].h_xbin_centers[1:-1], 
        reso_bin_counts[var_name][1:-1], 
        xerr=reso2d[var_name].h_xbin_widths[1:-1]/2.,
        yerr=reso_bin_errors[var_name][1:-1], 
    )
    
    for i in range(len(reso_projY["bin_counts"])):
        ax_reso[1].errorbar(
            reso_projY["bin_centers"][i][1:-1], 
            reso_projY["bin_counts"][i][1:-1], 
            xerr=reso_projY["bin_xerrors"][i][1:-1],
            yerr=reso_projY["bin_errors"][i][1:-1],
            label=reso_projY["labels"][i],
        )

    ax_reso[1].legend()
    fig_reso.savefig(f"./outputs/histograms/plots/jet_{var_name}_resolution.pdf" ,bbox_inches='tight')

    num_cols = len(jpt_bins) - 1
    fig_reso, ax_reso = plt.subplots(
        1, num_cols, figsize=(5*num_cols, 7), sharey=True,
        gridspec_kw={"wspace": 0, "right":0.9, "left":0.2}, 
    )
    
    for ijpt in range(num_cols):
        ax_reso[ijpt].stairs(
            reso_projY["bin_counts"][ijpt][:-1] + reso_projY["bin_errors"][ijpt][:-1],
            dyx_bins,
            baseline=(reso_projY["bin_counts"][ijpt][:-1] - reso_projY["bin_errors"][ijpt][:-1]).numpy(),
            fill=True,
            color="magenta",
            alpha=0.3,
        )
        ax_reso[ijpt].stairs(
            reso_projY["bin_counts"][ijpt][:-1],
            dyx_bins,
            color="magenta",
        )
        ax_reso[ijpt].axvline(x=0, linewidth=1.5, linestyle="--", color="gray", alpha=0.3)
        ax_reso[ijpt].text(0.02, 0.95, reso_projY["labels"][ijpt], transform=ax_reso[ijpt].transAxes,)
        if ijpt == 2:
            ax_reso[ijpt].text(
                0.02, 0.8, 
                r"$ R(p_{\rm T, jet}) = \frac{p_{\rm T, jet}^{\rm det. lvl.} - p_{\rm T, jet}^{\rm part. lvl.}}{p_{\rm T, jet}^{\rm part. lvl.}}$",
                fontsize="x-large", 
                transform=ax_reso[ijpt].transAxes,
            )
        ax_reso[ijpt].text(0.55, 0.95, rf"$\langle R(p_{{\rm T, jet}}) \rangle = {reso_bin_counts[var_name][ijpt+1].abs()*100:.1f}$%", transform=ax_reso[ijpt].transAxes,)
        ax_reso[ijpt].text(0.55, 0.9, rf"$\sigma(R(p_{{\rm T, jet}})) = {reso_bin_errors[var_name][ijpt+1].abs()*100:.1f}$%", transform=ax_reso[ijpt].transAxes,)
        ax_reso[ijpt].set_xlim(-1.2, 1.2)
        ax_reso[ijpt].set_xlabel(
            r"$ R(p_{\rm T, jet})$",
            fontsize="x-large",
        ) 

    ax_reso[0].set_ylabel(r"$\frac{1}{N_{\rm jets}}\frac{dN_{\rm jets}}{dR(p_{\rm T, jet})}$", fontsize="xx-large")
    fig_reso.savefig(f"./outputs/histograms/plots/jet_{var_name}_res.pdf", bbox_inches='tight')

    plt.show()
