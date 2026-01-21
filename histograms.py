import os
import json

import numpy as np
import pyarrow as pa
import torch

from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
from utils.histogram import TorchHist2D

jet_columns = [
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1"  ,
    "ch_ang_k1_b2"  ,
    "ch_ang_k2_b0"  ,
]
pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
#binning_pkl_file = "outputs/omnisequential_1/omniseq-bins10.pkl"
def make_stacked_hist1d(
    x_bins, 
    y_bins, 
    x_arr, 
    y_arr, 
    y_hc_arr, 
    weights, 
    weights_sq,
    is_batched=True,
):
    #print(is_batched, x_arr.shape, y_arr.shape, y_hc_arr.shape, weights.shape, weights_sq.shape)
    h2_pt_vs_ang = TorchHist2D(x_arr, y_arr, x_bins, y_bins, overflow=False)
    h2_pt_vs_ang_count_stacked, h2_pt_vs_ang_count_err_stacked = h2_pt_vs_ang.histogram(
        weights, weights_sq, is_batched=is_batched,
    )
    h2_prof_incl_vs_hc_count_stacked, h2_prof_incl_vs_hc_count_err_stacked = h2_pt_vs_ang.profileZ(
        y_hc_arr, weights, is_batched=is_batched,
    )

    h2_pt_vs_hc_ang = TorchHist2D(x_arr, y_hc_arr, x_bins, y_bins, overflow=False)
    h2_pt_vs_hc_ang_count_stacked, h2_pt_vs_hc_ang_count_err_stacked = h2_pt_vs_hc_ang.histogram(
        weights, weights_sq, is_batched=is_batched,
    )
    
    count_stacked    = {} 
    count_err_stacked= {}

    for ijpt, (jpt_min, jpt_max) in enumerate(zip(x_bins[:-1], x_bins[1:])):
        count_stacked[f"h1_projY_ang_jpt{ijpt}"], count_err_stacked[f"h1_projY_ang_jpt{ijpt}"] = h2_pt_vs_ang.projY(
            xrange=(jpt_min+0.5, jpt_max-0.5), is_batched=is_batched
        )
        count_stacked[f"h1_projY_sd_ang_jpt{ijpt}"], count_err_stacked[f"h1_projY_sd_ang_jpt{ijpt}"] = h2_pt_vs_hc_ang.projY(
            xrange = (jpt_min+0.5, jpt_max-0.5), is_batched=is_batched
        )
        count_stacked[f"h1_prof_incl_vs_sd_jpt{ijpt}"] = h2_prof_incl_vs_hc_count_stacked[..., ijpt]
        count_err_stacked[f"h1_prof_incl_vs_sd_jpt{ijpt}"] = h2_prof_incl_vs_hc_count_err_stacked[..., ijpt]
        
        count_stacked[f"h1_ratio_incl_vs_sd_jpt{ijpt}"] = count_stacked[f"h1_projY_sd_ang_jpt{ijpt}"].div(count_stacked[f"h1_projY_ang_jpt{ijpt}"])
        count_err_stacked[f"h1_ratio_incl_vs_sd_jpt{ijpt}"] = (
            count_err_stacked[f"h1_projY_sd_ang_jpt{ijpt}"].div(count_stacked[f"h1_projY_sd_ang_jpt{ijpt}"]).pow_(2) + \
            count_err_stacked[f"h1_projY_ang_jpt{ijpt}"].div(count_stacked[f"h1_projY_ang_jpt{ijpt}"]).pow_(2)
        ).sqrt_().mul_(count_stacked[f"h1_ratio_incl_vs_sd_jpt{ijpt}"])

    return count_stacked, count_err_stacked

def save_hist(prefix, filename=None, **kwargs):
    os.makedirs(prefix, exist_ok=True)
    filename = "hist.pt" if filename is None else f"{filename}.pt"
    torch.save(kwargs, os.path.join(prefix, filename))

def write_histograms_for_closure_sys(
    save_path, 
    x_bins, 
    y_bins, 
    x_arr, 
    y_arr, 
    y_hc_arr, 
    weights, weights_sq,
    truth_weights, truth_weights_sq,
):
    os.makedirs(save_path, exist_ok=True)
    print("Histograms will be saved to directory:", save_path)
           
    unf_count_stacked, unf_count_err_stacked = make_stacked_hist1d(
        x_bins,
        y_bins,
        x_arr,
        y_arr,
        y_hc_arr,
        weights,
        weights_sq,
    )

    truth_count, truth_count_err = make_stacked_hist1d(
        x_bins,
        y_bins,
        x_arr,
        y_arr,
        y_hc_arr,
        truth_weights,
        truth_weights_sq,
        is_batched=False,
    )

    hkeys = list(truth_count.keys())
    bin_center = 0.5*(y_bins[1:]+y_bins[:-1])
    half_bin_width = 0.5*(y_bins[1:]-y_bins[:-1])
    for key in hkeys:
        print(f"Saving {key}...")
        save_hist(
            os.path.join(save_path, "truth"),
            filename=key,
            bin_center = bin_center,
            half_bin_width = half_bin_width,
            bin_count     = truth_count[key],
            bin_count_err = truth_count_err[key], 
        )
    
        save_hist(
            os.path.join(save_path, "unfolded"),
            filename=key,
            bin_center = bin_center,
            half_bin_width = half_bin_width,
            bin_count     = unf_count_stacked[key].mean(0),
            bin_count_std = unf_count_stacked[key].std(0),
            bin_count_err = unf_count_err_stacked[key].mean(0), 
        )

        ratio_stacked = unf_count_stacked[key].div(truth_count[key])
        ratio_err_stacked = (
                unf_count_err_stacked[key].div(unf_count_stacked[key]).pow_(2) + \
                truth_count_err[key].div(truth_count[key]).pow_(2)
            ).sqrt_().mul_(ratio_stacked) 
        
        save_hist(
            os.path.join(save_path, "ratio"),
            filename=key,
            bin_center = bin_center,
            half_bin_width = half_bin_width,
            bin_count     = ratio_stacked.mean(0),
            bin_count_std = ratio_stacked.std(0),
            bin_count_err = ratio_err_stacked.mean(0), 
        )

def write_histograms_nominal_like(
    save_path, 
    x_bins, 
    y_bins, 
    x_arr, 
    y_arr, y_hc_arr, 
    weights, weights_sq,
    is_batched=True,
):
    os.makedirs(save_path, exist_ok=True)
    print("Histograms will be saved to directory:", save_path)
            
    count_stacked, count_err_stacked = make_stacked_hist1d(
        x_bins,
        y_bins,
        x_arr,
        y_arr,
        y_hc_arr,
        weights,
        weights_sq,
        is_batched,
    )

    bin_center = 0.5*(y_bins[1:]+y_bins[:-1])
    half_bin_width = 0.5*(y_bins[1:]-y_bins[:-1]) 
    for key in count_stacked.keys():
        print(f"Saving {key}...")
        if is_batched:
            save_hist(
                save_path,
                filename=key,
                bin_center = bin_center,
                half_bin_width = half_bin_width,
                bin_count = count_stacked[key].mean(0),
                bin_count_std = count_stacked[key].std(0),
                bin_count_err = count_err_stacked[key].mean(0),
            )
        else:
            save_hist(
                save_path,
                filename=key,
                bin_center = bin_center,
                half_bin_width = half_bin_width,
                bin_count = count_stacked[key],
                bin_count_err = count_err_stacked[key],
            )
def main(sys_var : SysVar, source_dir : str, save_prefix : str, binning_json_file : str):
    buffers = []
    if sys_var in {SysVar.NONE, SysVar.TOWER_ET_CORRECTION, SysVar.TRACK_EFFICIENCY}:
        input_root_dir = os.path.join(source_dir, str(sys_var))
    else:
        input_root_dir = os.path.join(source_dir, str(SysVar.NONE))
    
    buffers.append(pa.memory_map(os.path.join(input_root_dir, "gen-matches.arrow")))
    gen_match_table = pa.ipc.open_file(buffers[-1]).read_all()
    buffers.append(pa.memory_map(os.path.join(input_root_dir, "misses.arrow")))
    gen_misses_table = pa.ipc.open_file(buffers[-1]).read_all()
    gen_table = pa.concat_tables((gen_match_table, gen_misses_table))
   
    if sys_var in {SysVar.TOWER_ET_CORRECTION, SysVar.TRACK_EFFICIENCY, SysVar.UNFOLDING_PRIOR}:
        unf_wts_filename : str =f"outputs/unfolding_{str(sys_var)}/w_unfolding.npz"
    else:
        unf_wts_filename : str =f"outputs/unfolding_{str(SysVar.NONE)}/w_unfolding.npz"

    
    print(f"Getting unfolded weights from {unf_wts_filename}")
    weights = np.load(unf_wts_filename)

    jpt_bins = get_jet_pt_bins(sys_var)
    iteration = get_unfolding_iter(sys_var, 5) 
    unf_weights = torch.as_tensor(weights[f"arr_{2*iteration}"], dtype=torch.float32)
    unf_weights_sq = unf_weights*unf_weights
    x_arr = torch.as_tensor(gen_table["pt"].to_numpy(), dtype=torch.float32)
    
    print("Reading histogram bins from:", binning_json_file)
    with open(binning_json_file, 'rb') as file:
        bins = json.load(file)

    if sys_var == SysVar.UNFOLDING_PRIOR:
        truth_weight_file = "outputs/omnisequential/omniseq-wts-iter2.npz"
        print("Reading truth weights from:", truth_weight_file)
        closure_wts = np.load(truth_weight_file)
        truth_wt_key =list(closure_wts.keys())[-1]
        truth_weights = torch.as_tensor(closure_wts[truth_wt_key], dtype=torch.float32)
        truth_weights_sq = truth_weights * truth_weights
        for var_name in jet_columns:
            print(f"Setting up 2D histograms for {var_name}...")
            save_path = os.path.join(save_prefix, var_name)
            write_histograms_for_closure_sys(
                save_path, 
                jpt_bins, 
                torch.as_tensor(bins[var_name], dtype=torch.float32), 
                x_arr, 
                torch.as_tensor(gen_table[var_name].to_numpy(), dtype=torch.float32), 
                torch.as_tensor(gen_table[f"sd_{var_name}"].to_numpy(), dtype=torch.float32), 
                unf_weights, 
                unf_weights_sq,
                truth_weights,
                truth_weights_sq,
            )
    else:
        for var_name in jet_columns:
            print(f"Setting up 2D histograms for {var_name}...")
            save_path = os.path.join(save_prefix, var_name)
            write_histograms_nominal_like(
                save_path, 
                jpt_bins, 
                torch.as_tensor(bins[var_name], dtype=torch.float32), 
                x_arr, 
                torch.as_tensor(gen_table[var_name].to_numpy(), dtype=torch.float32), 
                torch.as_tensor(gen_table[f"sd_{var_name}"].to_numpy(), dtype=torch.float32), 
                unf_weights, 
                unf_weights_sq,
            )
        
                    
if __name__ == "__main__":
    source_dir = "./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding"
    save_prefix = "outputs/histograms"
    binning_json_file = "./runtime-files/bins_p00.02_N100000.json"
    for sys_var in SysVar:
        print("Computing histograms for systematic variation:", str(sys_var))
        save_path = os.path.join(save_prefix, str(sys_var))
        main(sys_var, source_dir, save_path, binning_json_file)
        print("Done!")
    

