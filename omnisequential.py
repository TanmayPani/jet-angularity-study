import os
import json
import copy
from collections import defaultdict

import numba as nb
import numpy as np
import pyarrow as pa
from matplotlib import pyplot as plt

import torch
from scipy.stats import chisquare
from sklearn.gaussian_process import GaussianProcessRegressor

from utils import bayesian_blocks
from utils.histogram import TorchHist1D

from systematics import SysVar

jet_columns = [
    "pt",
    "nef",
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1",
    "ch_ang_k1_b2",
    "ch_ang_k2_b0",
    "leading_constit_pt",
    "subleading_constit_pt",
    "sd_pt",
    "sd_dR", 
    "sd_symmetry",
    "sd_ch_ang_k1_b0.5",
    "sd_ch_ang_k1_b1",
    "sd_ch_ang_k1_b2",
    "sd_ch_ang_k2_b0",
]

col_hist_args = {
    "pt" : {
        "range" : (None, 60.),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "nef" : {
        "range" : (0.01, 0.9),
        "chi2_left_edge" : 1, 
        "chi2_right_edge" : None,
    },
    "ch_ang_k1_b0.5" : {
        "range" : (None, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "ch_ang_k1_b1" : {
        "range" : (None, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "ch_ang_k1_b2" : {
        "range" : (None, None),
        "chi2_left_edge" : 1, 
        "chi2_right_edge" : None,
    },
    "ch_ang_k2_b0" : {
        "range" : (0.01, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "leading_constit_pt" : {
        "range" : (None, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "subleading_constit_pt" : {
        "range" : (None, 12.5),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "sd_pt" : {
        "range" : (5., 50.),
        "chi2_left_edge" : 1, 
        "chi2_right_edge" : None,
    },
    
    "sd_dR" : {
        "range" : (0.06, None),
        "chi2_left_edge" : 1, 
        "chi2_right_edge" : None,
    },
    
    "sd_symmetry" : {
        "range" : (0, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },

    "sd_ch_ang_k1_b0.5" : {
        "range" : (0.001, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "sd_ch_ang_k1_b1" : {
        "range" : (0.025, None),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
    "sd_ch_ang_k1_b2" : {
        "range" : (0.006, None),
        "chi2_left_edge" : 1, 
        "chi2_right_edge" : None,
    },
    "sd_ch_ang_k2_b0" : {
        "range" : (0.01, 0.8),
        "chi2_left_edge" : 0, 
        "chi2_right_edge" : None,
    },
}

def propagate_values(x1, y1, x2, estimator=None, num_prediction_batches=None):
    if estimator is None:
        estimator = GaussianProcessRegressor(
            normalize_y=True, n_restarts_optimizer=10, copy_X_train=False
        )
    if len(x1.shape) == 1:
        x1 = x1.reshape(-1, 1)
    estimator.fit(x1, y1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(-1, 1)
    if num_prediction_batches is None:
        return estimator.predict(x2)
    else:
        predictions = []
        for ibatch, batch in enumerate(np.array_split(x2, num_prediction_batches)):
            if ibatch > 0 and ibatch % 10 == 0:
                print(f"------Predicting for batch [{ibatch}/{num_prediction_batches}]")
            predictions.append(estimator.predict(batch))
        return np.concatenate(predictions)

def main(
    n_iterations : int,
    do_use_gen_misses : bool = True,
    recalculate_bins_for : list[str] | None = None,
    do_diff_gen_bins : bool = False,
):
    source_dir = "./datasets/STAR_pp200GeV_production_2012/clustered_jets"
    emb_dir = os.path.join(source_dir, "embedding", str(SysVar.NONE))
    gen_match_buffer = pa.memory_map(os.path.join(emb_dir, "gen-matches.arrow"), "rb")
    gen_match_table = pa.ipc.open_file(gen_match_buffer).read_all()

    gen_miss_buffer = pa.memory_map(os.path.join(emb_dir, "misses.arrow"), "rb")
    gen_miss_table = pa.ipc.open_file(gen_miss_buffer).read_all()

    reco_match_buffer = pa.memory_map(os.path.join(emb_dir, "reco-matches.arrow"), "rb")
    reco_match_table = pa.ipc.open_file(reco_match_buffer).read_all()

    reco_fake_buffer = pa.memory_map(os.path.join(emb_dir, "fakes.arrow"), "rb")
    reco_fake_table = pa.ipc.open_file(reco_fake_buffer).read_all()
    
    data_buffer = pa.memory_map(os.path.join(source_dir, "preproc_data.arrow"), "rb")
    data_table = pa.ipc.open_file(data_buffer).read_all()


    if do_use_gen_misses:
        gen_table = pa.concat_tables([gen_match_table, gen_miss_table])
    else:
        gen_table = gen_match_table

    reco_table =pa.concat_tables([reco_match_table, reco_fake_table])
 
    n_data = len(data_table)

    n_gen_matches = len(gen_match_table)
    n_gen_misses = len(gen_miss_table)
    n_reco_matches = len(reco_match_table)
    n_reco_fakes = len(reco_fake_table)

    assert n_gen_matches == n_reco_matches

    n_matches = n_gen_matches 
    n_gen = n_matches + n_gen_misses
    n_reco = n_matches + n_reco_fakes

    print("Number of matched gen jets, matched reco jets:", n_gen_matches, n_reco_matches, n_matches)
    print("Number of missed gen jets, fake reco jets:", n_gen_misses, n_reco_fakes)
    print("Number of data jets:", n_data)

    bins = {}
    data_arr = {}
    data_hist = {}
    data_hist_count = {}
    data_hist_count_err = {}

    p0=0.02
    undersample=100000
    bin_file_path = f"./runtime-files/bins_p0{p0:g}_N{undersample:g}.json"
    try:
        with open(bin_file_path, "rb") as bin_file:
            bins = json.load(bin_file)
        if recalculate_bins_for is not None:
            for key in recalculate_bins_for:
                del bins[key]
        print("Read bins from :", bin_file_path)
    except Exception as exc:
        print(exc)
        bins = {}
        print("Binning will be saved to :", bin_file_path)

    bins_updated = False
    for col in jet_columns:
        data_arr[col] = torch.as_tensor(data_table[col].to_numpy(), dtype=torch.float64)
        if not col in bins: 
            bins_updated = True
            print("Calculating binning for", col)
            bins[col] = bayesian_blocks(
                data_arr[col], 
                p0=p0, 
                ranges=col_hist_args[col]["range"], 
                undersample=undersample,
                device="cuda", 
            )
            print(col, ":", bins[col], len(bins[col]))
        
        data_hist[col] = TorchHist1D(data_arr[col], bins[col], overflow=False)
        data_hist_count[col], data_hist_count_err[col] = data_hist[col].histogram()
    
    if bins_updated:
        with open(bin_file_path, "w") as bin_file:
            json.dump(
                {k : v.tolist() for k, v in bins.items()}, 
                bin_file, 
                indent=4,
            )

    reco_arr = {}
    gen_arr = {}
    gen_match_arr = {}
 
    reco_hist = {}
    gen_hist = {}
    gen_match_hist = {}
    
    reco_weights = torch.as_tensor(reco_table["weight"].to_numpy(), dtype=torch.float32)
    gen_weights = torch.as_tensor(gen_table["weight"].to_numpy(), dtype=torch.float32)

    num_data_samples = torch.as_tensor(float(n_data), dtype=torch.float32)
    reco_weights_sum = reco_weights.sum()
    gen_weights_sum = gen_weights.sum()

    reco_weights = reco_weights.mul_(num_data_samples).div_(reco_weights_sum)
    gen_weights = gen_weights.mul_(num_data_samples).div_(gen_weights_sum)
 
    for col in jet_columns:
        reco_arr[col] = torch.as_tensor(reco_table[col].to_numpy(), dtype=torch.float64)
        gen_arr[col] = torch.as_tensor(gen_table[col].to_numpy(), dtype=torch.float64)
        gen_match_arr[col] = torch.as_tensor(gen_match_table[col].to_numpy(), dtype=torch.float64)

        reco_hist[col] = TorchHist1D(reco_arr[col], bins[col], overflow=False)
        gen_hist[col] = TorchHist1D(gen_arr[col], bins[col], overflow=False)
        gen_match_hist[col] = TorchHist1D(gen_match_arr[col], bins[col], overflow=False)
   
    iter_hist_list = []
    hists = defaultdict(dict)
    last_iteration = -1 
    max_chi2_var = ""

    output_folder=f"outputs/omnisequential"
    os.makedirs(output_folder, exist_ok=True)

    w_unfolding = [reco_weights, gen_weights]
    for iteration in range(n_iterations):
        max_chi2 = 0
        max_chi2_var = ""
        print(f"Iteration: {iteration}")
        for icol, col in enumerate(jet_columns):
            hists[col]["data"] = data_hist_count[col]
            hists[col]["data_err"] = data_hist_count_err[col]
            #print(reco_weights.shape, gen_weights.shape, reco_hist[col].x.shape, gen_arr[col].shape)
            hists[col]["reco"], hists[col]["reco_err"] = reco_hist[col].histogram(weights=reco_weights)
            hists[col]["gen"], hists[col]["gen_err"] = gen_hist[col].histogram(weights=gen_weights)

            hists[col]["ratio"] = hists[col]["data"] / hists[col]["reco"]
            hists[col]["ratio_err"] =  (
                hists[col]["data_err"].div(hists[col]["data"]).pow_(2) + hists[col]["reco_err"].div(hists[col]["reco"]).pow_(2)
            ).sqrt_().mul_(hists[col]["ratio"])

            left_edge = col_hist_args[col]["chi2_left_edge"]
            right_edge  = col_hist_args[col]["chi2_right_edge"]
            
            chi2_slice = slice(left_edge, right_edge)
            hists[col]["chi2_slice"] = chi2_slice
            h_data = hists[col]["data"][chi2_slice]
            h_reco = hists[col]["reco"][chi2_slice]
            h_reco = h_reco.div(h_reco.sum()).mul_(h_data.sum())
            chi2, _ = chisquare(h_reco, h_data)
            hists[col]["chi2"] = chi2

            if chi2 > max_chi2:
                max_chi2 = chi2
                max_chi2_var = col

        iter_hist_list.append(copy.deepcopy(hists))
        
        if max_chi2 < 1:
            break
        
        print(f"---Max chi2 = {max_chi2} from {max_chi2_var}")
        features = reco_hist[max_chi2_var].h_xbin_centers
        nbins = len(features)
        print("---Obtaining reco lvl reweight factors...")
        reco_reweights = torch.as_tensor(
            propagate_values(
                features[hists[max_chi2_var]["chi2_slice"]], 
                hists[max_chi2_var]["ratio"][hists[max_chi2_var]["chi2_slice"]], 
                reco_arr[max_chi2_var]
            ),
            dtype=torch.float32,
        )
        reco_weights = reco_weights.mul_(reco_reweights)
        reco_weights_sum = reco_weights.sum()
        reco_weights = reco_weights.mul_(num_data_samples).div_(reco_weights_sum)

        print("---Obtaining gen lvl reweight factors...")
        #bins = hists[max_chi2_var]["gen_bins"]
        gen_features = gen_match_hist[max_chi2_var].h_xbin_centers
        nbins = len(gen_features)
        gen_match_reweights = reco_reweights[: n_gen_matches]
        gen_reweight_profile, _ = gen_match_hist[max_chi2_var].profileY(gen_match_reweights, weights=gen_weights[: n_gen_matches])

        selection = torch.isnan(gen_reweight_profile).logical_not_()
        gen_features = gen_features[selection]
        print(f"---{len(gen_features)} good bins out of {nbins} gen bins...")
        gen_reweight_profile = gen_reweight_profile[selection]
        gen_reweights = torch.as_tensor(
            propagate_values(
                gen_features, 
                gen_reweight_profile, 
                gen_arr[max_chi2_var], 
                num_prediction_batches=40
            ),
            dtype=torch.float32,
        )
        gen_weights = gen_weights.mul_(gen_reweights)
        gen_weights_sum = gen_weights.sum()
        gen_weights = gen_weights.mul_(num_data_samples).div_(gen_weights_sum)

        w_unfolding.extend([reco_weights, gen_weights])

    print("Iterations done...")
    last_iteration = len(iter_hist_list)
    with open(f"{output_folder}/omniseq-wts-iter{last_iteration}.npz", "wb") as f:
        np.savez(f, *w_unfolding)
    return iter_hist_list, bin_file_path

    #col_bins = {col:h["bins"] for col, h in hists.items()}
def plot_hist(ax, bins, counts, errors=None, bin_range=(0, None), **kwargs):
    binWidths = bins[1:] - bins[:-1]
    binCenters = (bins[1:] + bins[:-1]) * 0.5

    if bin_range[0] > 0 or bin_range[1] is not None:
        if bin_range[1] is None:
            binWidths = binWidths[bin_range[0] :]
            binCenters = binCenters[bin_range[0] :]
            counts = counts[bin_range[0] :]
            if errors is not None:
                errors = errors[bin_range[0] :]
        else:
            binCenters = binCenters[bin_range[0] : bin_range[1]]
            binWidths = binWidths[bin_range[0] : bin_range[1]]
            counts = counts[bin_range[0] : bin_range[1]]
            if errors:
                errors = errors[bin_range[0] : bin_range[1]]

    # print(len(binCenters), len(counts), len(binWidths), len(errors))
    # print(binCenters, counts, binWidths, errors)
    ax.errorbar(binCenters, counts, xerr=binWidths * 0.5, yerr=errors, **kwargs)
    return binCenters, binWidths, counts, errors


def plot_ratios(
    fig,
    bins,
    counts1,
    counts2,
    ratios,
    errors1,
    errors2,
    ratio_errs,
    labels1=None,
    labels2=None,
    fill_style="none",
    line_style="none",
    markersize=10,
    markers1=None,
    markers2=None,
):
    axs = fig.subplots(
        2,
        1,
        sharex=True,
        height_ratios=[3, 1],
        gridspec_kw={"hspace": 0, "top": 0.85, "bottom": 0.1},
    )

    for iden, (count, error) in enumerate(zip(counts2, errors2)):
        label = str(iden) if labels2 is None else labels2[iden]
        marker = "o" if markers2 is None else markers2[iden]
        #print(iden, label, marker)
        plot_hist(
            axs[0],
            bins,
            count,
            errors=error,
            label=label,
            marker=marker,
            fillstyle=fill_style,
            linestyle=line_style,
            markersize=markersize,
        )

    for inum, (count1, error1, ratios1, ratio_errs1) in enumerate(
        zip(counts1, errors1, ratios, ratio_errs)
    ):
        label1 = str(inum) if labels1 is None else labels1[inum]
        marker1 = "o" if markers1 is None else markers1[inum]
        plot_hist(
            axs[0],
            bins,
            count1,
            errors=error1,
            label=label1,
            marker=marker1,
            fillstyle=fill_style,
            linestyle=line_style,
            markersize=markersize,
        )

        for iden, (count2, error2, ratio, ratio_err) in enumerate(
            zip(counts2, errors2, ratios1, ratio_errs1)
        ):
            label2 = str(iden) if labels2 is None else labels2[iden]
            marker2 = "o" if markers2 is None else markers2[iden]
            #print(inum, iden, label1, label2, marker1, marker2)
            if ratio is None:
                ratio = count1 / count2
                if error1 is not None and error2 is not None:
                    ratio_err = ratio * np.sqrt(
                        (error1 / count1) ** 2 + (error2 / count2) ** 2
                    )
            #print(ratio, ratio_err)
            plot_hist(
                axs[1],
                bins,
                ratio,
                errors=ratio_err,
                label=f"{label1}/{label2}",
                marker=marker2,
                fillstyle=fill_style,
                linestyle=line_style,
                markersize=markersize,
            )
        axs[0].set_yscale("log")
        axs[0].tick_params(axis="x", direction="in")
        axs[1].set_ylim(0, 2)
        axs[1].axhline(y=1, color="red", linestyle="--")
        axs[1].axhspan(0.8, 1.2, color="darkgrey")
    return axs

def plot(iter_hist_list, bin_file_path):
    last_iteration = len(iter_hist_list)
    
    with open(bin_file_path, "rb") as bin_file:
        bins = json.load(bin_file)

    fig_scale = 6
    nrows = 3
    ncols = int(np.ceil(len(jet_columns) / nrows))

    fig = plt.figure(figsize=(ncols * fig_scale, nrows * fig_scale))  # , layout="constrained")
    fig.suptitle(f"Iteration {last_iteration}", fontsize=30)
    subfigs = fig.subfigures(nrows, ncols)
    for ivar, var in enumerate(jet_columns):
        irow = int(np.floor(ivar / ncols))
        icol = ivar % ncols
        subfig = subfigs[irow, icol]

        h = iter_hist_list[-1][var]
        h0 = iter_hist_list[0][var]

        print(ivar, var)
        axs = plot_ratios(
            subfig,
            torch.as_tensor(bins[var]),
            [h["data"]],
            [h["reco"], h0["reco"]],
            [[h["ratio"], h0["ratio"]]],
            [h["data_err"]],
            [h["reco_err"], h0["reco_err"]],
            [[h["ratio_err"], h0["ratio_err"]]],
            labels1=["data"],
            labels2=[f"reco (iter = {last_iteration})", "reco (iter = 0)"],
            markers1=["o"],
            markers2=["^", "v"],
        )
        plot_hist(
            axs[0],
            torch.as_tensor(bins[var]),
            h["gen"],
            errors=h["gen_err"],
            label=f"gen, iter = {last_iteration}",
            marker="^",
            fillstyle="none",
            markersize=10,
        )
        plot_hist(
            axs[0],
            torch.as_tensor(bins[var]),
            h0["gen"],
            errors=h0["gen_err"],
            label="gen, iter = 0",
            marker="v",
            fillstyle="none",
            markersize=10,
        )
        axs[0].set_title(f"chi2 = {h['chi2']:.2f}")
        axs[0].legend()
        axs[1].set_xlabel(var)
        axs[1].legend()
    fig.savefig(f"outputs/plot_output/omniseq_new.pdf")
    plt.show(block=True)

if __name__ == "__main__":
    torch.manual_seed(0)
    out_list, bin_path = main(
        10,
        #recalculate_bins_for=jet_columns
    )
    plot(out_list, bin_path)
