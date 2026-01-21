import os
from copy import deepcopy
from collections import defaultdict
from enum import StrEnum


import numpy as np
import numba as nb
import torch
import awkward as ak

import matplotlib.pyplot as plt 

from utils.histogram import TorchHist1D, TorchHist2D

class SysVar(StrEnum):
    NONE="nominal"
    TOWER_ET_CORRECTION="tower_et_corr_sys"
    TRACK_EFFICIENCY="track_pt_sys"
    JET_PT_RESOLUTION_0 = "jet_pt_res_sys_0"
    JET_PT_RESOLUTION_1 = "jet_pt_res_sys_1"
    UNFOLDING_PRIOR = "unf_prior_sys"
    #UNFOLDING_BOOTSTRAP = "unf_bootstrap_sys"
    UNFOLDING_ITERATION_0 = "unf_iter_sys_0"
    UNFOLDING_ITERATION_1 = "unf_iter_sys_1"

jet_columns = [
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1"  ,
    "ch_ang_k1_b2"  ,
    "ch_ang_k2_b0"  ,
]

var_label = {
    "ch_ang_k1_b0.5"   :r"\lambda^{\kappa = 1}_{\beta = 0.5}",
    "ch_ang_k1_b1"     :r"\lambda^{\kappa = 1}_{\beta = 1}",
    "ch_ang_k1_b2"     :r"\lambda^{\kappa = 1}_{\beta = 2}",
    "ch_ang_k2_b0"     :r"\lambda^{\kappa = 2}_{\beta = 0}",
    "sd_ch_ang_k1_b0.5":r"\lambda^{\kappa = 1, \rm SD}_{\beta = 0.5}",
    "sd_ch_ang_k1_b1"  :r"\lambda^{\kappa = 1, \rm SD}_{\beta = 1}",
    "sd_ch_ang_k1_b2"  :r"\lambda^{\kappa = 1, \rm SD}_{\beta = 2}",
    "sd_ch_ang_k2_b0"  :r"\lambda^{\kappa = 2, \rm SD}_{\beta = 0}",
}

var_unit = {
    "ch_ang_k1_b0.5"   :r"(LHA)",
    "ch_ang_k1_b1"     :r"(girth)", 
    "ch_ang_k1_b2"     :r"(thrust)",
    "ch_ang_k2_b0"     :r"((p_T^D)^2)",
    "sd_ch_ang_k1_b0.5":r"(LHA, groomed)",
    "sd_ch_ang_k1_b1"  :r"(girth, groomed)",
    "sd_ch_ang_k1_b2"  :r"(thrust, groomed)",
    "sd_ch_ang_k2_b0"  :r"((p_T^D)^2, groomed)",
}

filename_mods = (
    "h1_prof_incl_vs_sd", 
    "h1_projY_ang", 
    "h1_projY_sd_ang",
    "h1_ratio_incl_vs_sd"
)
def apply_hadronic_correction_sys_var(events, hadr_corr_frac=0.5):
    tower_dE = events["towers._RawE"] - events["towers._E"]
    events["towers._E"] = events["towers._E"] - hadr_corr_frac*tower_dE
    mass_array = ak.full_like(events["towers._E"], 0.13957)
    tower_p2 = events["towers._E"]**2 - mass_array**2
    tower_p2 = ak.fill_none(ak.mask(tower_p2, tower_p2 > 0), value=0)
    tower_p = np.sqrt(tower_p2)
    events["towers._Pt"] = tower_p/np.cosh(events["towers._Eta"])
    return events

@nb.jit
def apply_flat_track_pt_factors(builder, event_track_pt, flat_rel_factors):
    i_trk = 0 
    for track_pt in event_track_pt:
        builder.begin_list()
        for pt in track_pt:
            if flat_rel_factors[i_trk] > 0.04:
                builder.append(True)
            else:
                builder.append(False)
            i_trk += 1
        builder.end_list()
    return builder

def get_tracking_efficiency_sys_var_mask(events, seed=None):
    n_tot_trk = ak.sum(ak.count(events["tracks._Pt"], axis=0))
    #flat_factors = np.random.default_rng().uniform(-0.04, 0.04, n_tot_trk)
    flat_factors = np.random.default_rng(seed).random(n_tot_trk)
    return apply_flat_track_pt_factors(ak.ArrayBuilder(), events["tracks._Pt"], flat_factors).snapshot()

def get_jet_pt_bins(sys_var):
    match sys_var:
        case SysVar.JET_PT_RESOLUTION_0:
            return (11., 14., 21., 38., 82.)
        case SysVar.JET_PT_RESOLUTION_1:
            return (9., 16., 19., 42., 78.)
        case _:
            return (10., 15., 20., 40., 80.)

def get_unfolding_iter(sys_var, nom_iter = 5):
    match sys_var:
        case SysVar.UNFOLDING_ITERATION_0:
            return nom_iter-1
        case SysVar.UNFOLDING_ITERATION_1:
            return nom_iter+2
        case _:
            return nom_iter

def _unc_less_preferred_sys_var(h_nominal : dict[str, torch.Tensor], *h_sys_vars : dict[str, torch.Tensor]):
    h_sys_vars_stacked = torch.stack([h["bin_count"] for h in h_sys_vars])
    h_sys_vars_mean = h_sys_vars_stacked.mean(0)
    sys_var_unc = (h_nominal["bin_count"] - h_sys_vars_mean).abs_()
    return sys_var_unc

def _unc_equal_preferred_sys_var(h_nominal : dict[str, torch.Tensor], *h_sys_vars : dict[str, torch.Tensor]):
    hist_list = [h_nominal["bin_count"]]
    hist_list.extend(h["bin_count"] for h in h_sys_vars)
    hist_stacked = torch.stack(hist_list)
    hist_mean = hist_stacked.mean(0)
    hist_std = hist_stacked.std(0)
    return hist_mean, hist_std 

def _unc_prior_var(h_nominal : dict[str, torch.Tensor], ratio : dict[str, torch.Tensor]):
    h_sys_var = h_nominal["bin_count"] / ratio["bin_count"]
    #h_sys_var_sum = h_sys_var.nansum()
    #h_sys_var_scale = (h_nominal["half_bin_width"] * 2.).mul_(h_sys_var_sum)
    #h_sys_var.div_(h_sys_var_scale)
    prior_var_unc = (h_sys_var - h_nominal["bin_count"]).abs_()
    return prior_var_unc

def calculate_sys_uncertainty(
    hist_nominal : dict[str, torch.Tensor], 
    var_name : str, 
    hist_file_name : str, 
    *sys_var_names : str,
    path_prefix : str = "",
    is_equal_pref : bool = False,
):
    sys_var_paths = [
        os.path.join(path_prefix, sys_var, var_name, hist_file_name) 
        for sys_var in sys_var_names
    ] 

    sys_var_hists = [
        torch.load(file_path) for file_path in sys_var_paths
    ]

    if is_equal_pref:
        return _unc_equal_preferred_sys_var(hist_nominal, *sys_var_hists)
    else:
        return _unc_less_preferred_sys_var(hist_nominal, *sys_var_hists)

def calculate_prior_uncertainty(
    hist_nominal : dict[str, torch.Tensor], 
    var_name : str, 
    hist_file_name : str, 
    path_prefix : str,
):
    ratio_path = os.path.join(
        path_prefix,
        str(SysVar.UNFOLDING_PRIOR), 
        var_name, 
        "ratio", 
        hist_file_name
    )
    hist_ratio = torch.load(ratio_path, mmap=True)
    sys_var_prior =  _unc_prior_var(hist_nominal, hist_ratio)
    #print(var_name, hist_file_name, sys_var_prior, hist_ratio)
    return sys_var_prior

def calculate_uncertainties(var_name : str, path_prefix : str):
    nominal_path = os.path.join(path_prefix, "nominal", var_name)
    sys_vars = defaultdict(dict)
    hist_nominal = {} 
    for file_name in os.listdir(nominal_path):
        hist_name = file_name.removesuffix(".pt")
        hist_nominal[hist_name] = torch.load(
            os.path.join(nominal_path, file_name), mmap=True,
        )
        sys_vars[hist_name]["unf_iter_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "unf_iter_sys_0",
            "unf_iter_sys_1",
            path_prefix=path_prefix
        )

        sys_vars[hist_name]["jet_pt_res_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "jet_pt_res_sys_0",
            "jet_pt_res_sys_1",
            path_prefix=path_prefix
        ) 

        sys_vars[hist_name]["track_pt_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "track_pt_sys",
            path_prefix=path_prefix
        )
        
        sys_vars[hist_name]["tower_et_corr_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "tower_et_corr_sys",
            path_prefix=path_prefix
        )

        sys_vars[hist_name]["unf_prior_sys"] = calculate_prior_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            path_prefix=path_prefix
        )

        squared_sum = torch.zeros_like(hist_nominal[hist_name]["bin_count"])
        for sys_var_unc in sys_vars[hist_name].values():
            squared_sum.add_(sys_var_unc.square())
        sys_vars[hist_name]["total_sys"] = squared_sum.sqrt_()
        #sys_vars[hist_name]["bootstrap_stat"] = hist_nominal[hist_name]["bin_count_std"]
    return sys_vars, hist_nominal

def save_sys_uncertainties(sys_vars, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for hname, sys_var_dict in sys_vars.items():
        save_path = os.path.join(save_dir, f"{hname}.pt")
        torch.save(sys_var_dict, save_path)

def plot_uncertainties(var_name, hnominal, sys_vars, fig_save_dir):
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols =len(jpt_bins)-1 
    fig_scale = 5
    
    fig  = {}
    for mod in filename_mods:
        if mod == "h1_prof_incl_vs_sd":
            ylabel = rf"$\frac{{|\Delta \langle {var_label[f"sd_{var_name}"]} \rangle|}}{{ \langle {var_label[f"sd_{var_name}"]} \rangle}} \times 100$"
        else:
            ylabel = r"$\frac{|\Delta h|}{h} \times 100$"

        if mod == "h1_projY_sd_ang":
            hc_var_name = f"sd_{var_name}"
            xlabel = rf"${var_label[hc_var_name]} {var_unit[hc_var_name]}$"
        else:
            xlabel = rf"${var_label[var_name]} {var_unit[var_name]}$"

        fig[mod] = plt.figure(
            figsize=(num_cols * fig_scale, fig_scale + 1),
        )
        axs = fig[mod].subplots(
            1, num_cols, sharey=True, 
            gridspec_kw={"wspace": 0, "right":0.9, "left":0.2},
        )
        axs[0].set_ylabel(ylabel, labelpad=10, size="xx-large" ) 

        _fig_single_sys = {} 
        _axs_single_sys = {}
        for sys_var_name in sys_vars[f"{mod}_jpt0"].keys():
            _fig_single_sys[sys_var_name] = plt.figure(figsize=(num_cols * fig_scale, fig_scale + 1))
            _axs_single_sys[sys_var_name] = _fig_single_sys[sys_var_name].subplots(
                1, num_cols, sharey=True, gridspec_kw={"wspace": 0, "right":0.9, "left":0.2},
            )
            _axs_single_sys[sys_var_name][0].set_ylabel(ylabel, labelpad=10, size="xx-large" ) 

        for ijpt, (jpt_min, jpt_max) in enumerate(zip(jpt_bins[:-1], jpt_bins[1:])):
            hist_name = f"{mod}_jpt{ijpt}"
            
            x = hnominal[hist_name]["bin_center"]
            x_err = hnominal[hist_name]["half_bin_width"]

            bin_edges_low = x - x_err 
            bin_edge_high = (x[-1] + x_err[-1]).unsqueeze_(0)
            bins = torch.concatenate((bin_edges_low, bin_edge_high))

            y = hnominal[hist_name]["bin_count"]
            rel_stat_unc = (hnominal[hist_name]["bin_count_std"] / y).mul_(100)
            dy_stat_low = torch.zeros_like(y)

            axs[ijpt].set_xlabel(xlabel, labelpad=10, size="x-large")
            axs[ijpt].text(
                0.15, 0.55, 
                fr"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$", 
                transform=axs[ijpt].transAxes,
            )
            axs[ijpt].stairs(
                rel_stat_unc, bins, baseline=dy_stat_low.numpy(), 
                fill=True, color="magenta", alpha=0.3, label="stat. (bootstrap)",
            )
             
            total_rel_unc = (sys_vars[hist_name]["total_sys"]/y).mul_(100)
            axs[ijpt].stairs(total_rel_unc,  bins, label="total sys", linewidth=2., linestyle="dotted")
            for sys_var_name, sys_var in sys_vars[hist_name].items():
                if sys_var_name == "total_sys":
                    continue
                rel_unc = (sys_var / y).mul_(100)
                 
                _axs_single_sys[sys_var_name][ijpt].stairs(rel_unc,  bins, label=sys_var_name, linewidth=2.) 
                _axs_single_sys[sys_var_name][ijpt].stairs(total_rel_unc,  bins, label="total sys", linewidth=2., linestyle="dotted") 
                _axs_single_sys[sys_var_name][ijpt].stairs(
                    rel_stat_unc, bins, baseline=dy_stat_low.numpy(), 
                    fill=True, color="magenta", alpha=0.3, label="stat. (bootstrap)",
                )
                _axs_single_sys[sys_var_name][ijpt].set_xlabel(xlabel, labelpad=10, size="x-large")
                _axs_single_sys[sys_var_name][ijpt].text(
                    0.2, 0.75, fr"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$", 
                    transform=_axs_single_sys[sys_var_name][ijpt].transAxes,
                )
                axs[ijpt].stairs(rel_unc,  bins, label=sys_var_name, linewidth=2.)
         
        axs[-1].legend()
        
        for sys_var_name in sys_vars[f"{mod}_jpt0"].keys():
            _axs_single_sys[sys_var_name][-1].legend()
            _fig_single_sys[sys_var_name].savefig(
                os.path.join(fig_save_dir, f"{mod}-{sys_var_name}-sysQA.pdf"), 
                bbox_inches='tight'
            )
            plt.close(_fig_single_sys[sys_var_name])
        
        os.makedirs(fig_save_dir, exist_ok=True)
        fig_save_path = os.path.join(fig_save_dir, f"{mod}-sysQA.pdf")
        print("Saving figure to:", fig_save_path)
        fig[mod].savefig(fig_save_path ,bbox_inches='tight')

def main():
    path_prefix="outputs/histograms"
    for var_name in jet_columns:
        sys_vars, hnominal = calculate_uncertainties(var_name, path_prefix=path_prefix)
        save_dir = os.path.join(path_prefix, "sys_errors", var_name)
        save_sys_uncertainties(sys_vars, save_dir)
        fig_save_dir = os.path.join(path_prefix, "plots", var_name)
        plot_uncertainties(var_name, hnominal, sys_vars, fig_save_dir)
        
                     
if __name__ == "__main__":
    main()
    plt.show()


