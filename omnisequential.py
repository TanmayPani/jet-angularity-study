import os
from collections.abc import Sequence, Sized, Iterable
from datetime import datetime
from typing import Optional, Union, List, Any, Generic
import pickle

import numpy as np
from numpy import typing as npt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split

import pyarrow as pa
from pyarrow import compute as pc 

from scipy.stats import chisquare

from matplotlib import pyplot as plt
import plotting
import histograms

import copy

def add_constit_slice_column(jet_table, consit_col_name, new_col_name, start, stop=None):
    if stop is None:
        stop = start+1
    carr = pc.list_slice(jet_table[consit_col_name], start, stop=stop).combine_chunks().flatten()
    return jet_table.append_column(new_col_name, carr)

def add_leading_constit_column(jet_table): 
    constit_pt_arr = jet_table["constit_pt"].combine_chunks().values
    constit_jet_indices = jet_table["constit_pt"].combine_chunks().value_parent_indices()
    constit_table = pa.table({"constit_pt":constit_pt_arr, "jet_index":constit_jet_indices})
    agg = constit_table.group_by("jet_index").aggregate([("constit_pt", "max")])

    return jet_table.append_column("leading_constit_pt", agg["constit_pt_max"])

def pa_table(source : str , label : Optional[npt.ArrayLike] = None):
    buffer = pa.memory_map(source, "rb")
    table = pa.ipc.open_file(buffer).read_all()
    _len = len(table)
    label_arr = None
    if isinstance(label, (int, np.number)):
        label_arr = np.full(_len, label, dtype=np.int_)
    elif isinstance(label, np.ndarray):
        assert len(label) == _len
        label_arr = np.asarray(label, dtype=np.int_)
    else: 
        label_arr = np.empty(0)

    return buffer, add_extra_columns(table), label_arr

def pa_concated_table(source : Sequence[str], label : Optional[Sequence[npt.ArrayLike]] = None):
    n_tables = len(source)
    label_iter = label if label is not None else [None]*n_tables 
    buffer_list = []
    table_list = []
    label_list = []
    for _source, _label in zip(source, label_iter):
        if not isinstance(_source, str):
            raise TypeError("Can't use sources other than path strings for pa.Table!")
        buffer, table, label_arr = pa_table(_source, label=_label)
        buffer_list.append(buffer)
        table_list.append(table)
        label_list.append(label_arr)

    return buffer_list, pa.concat_tables(table_list), np.concatenate(label_list)

def add_extra_columns(table : pa.Table) -> pa.Table:
    table = add_constit_slice_column(table, "constit_pt", "leading_constit_pt", 0)
    table = add_constit_slice_column(table, "constit_eta", "leading_constit_eta", 0)
    table = add_constit_slice_column(table, "constit_phi", "leading_constit_phi", 0)

    table = add_constit_slice_column(table, "constit_pt", "subleading_constit_pt", 1)
    table = add_constit_slice_column(table, "constit_eta", "subleading_constit_eta", 1)
    table = add_constit_slice_column(table, "constit_phi", "subleading_constit_phi", 1)

    return table

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

def main():
    jet_columns = [
        "pt",
        "nef",
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
        "leading_constit_pt",
        "subleading_constit_pt",
        "hc_pt",
        "hc_ch_ang_k1_b0.5",
        "hc_ch_ang_k1_b1",
        "hc_ch_ang_k1_b2",
        "hc_ch_ang_k2_b0",
        ]

    col_ranges = {
        "pt" : (None, 40),
        "nef" : (0.01, None),
        "ch_ang_k1_b0.5" : (None, None),
        "ch_ang_k1_b1" : (None, None),
        "ch_ang_k1_b2" : (None, None),
        "ch_ang_k2_b0" : (None, None),
        "leading_constit_pt" : (None, None),
        "subleading_constit_pt" : (None, 12.5),
        "hc_pt" : (5, 40),
        "hc_ch_ang_k1_b0.5" : (0.001, None),
        "hc_ch_ang_k1_b1" : (0.01, None),
        "hc_ch_ang_k1_b2" : (0.001, None),
        "hc_ch_ang_k2_b0" : (0.001, None)
    }

    emb_data_folder = "outputs/30May25-1147"
    pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    pth_bin_folders = [f"{emb_data_folder}/ptHat{pth_low}to{pth_high}" for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])]
    n_pth_bins = len(pth_bin_folders)
    pth_label = list(range(1, n_pth_bins+1))

    gen_match_buffers, gen_match_table, gen_match_stratify_label = pa_concated_table([f"{folder}/gen-matches.arrow" for folder in pth_bin_folders], label=pth_label)
    gen_miss_buffers, gen_miss_table, gen_miss_stratify_label = pa_concated_table([f"{folder}/misses.arrow" for folder in pth_bin_folders], label=pth_label) 
    reco_match_buffers, reco_match_table, reco_match_stratify_label = pa_concated_table([f"{folder}/reco-matches.arrow" for folder in pth_bin_folders], label=pth_label)
    reco_fake_buffers, reco_fake_table, reco_fake_stratify_label = pa_concated_table([f"{folder}/fakes.arrow" for folder in pth_bin_folders], label=pth_label)

    data_buffer, data_table, data_stratify_label = pa_table("outputs/jets-conPtMin0.2.arrow", label=0)
    
    do_use_gen_misses = False

    if do_use_gen_misses:
        gen_table = pa.concat_tables([gen_match_table, gen_miss_table])
        gen_stratify_label = np.concatenate([gen_match_stratify_label, gen_miss_stratify_label])
    else:
        gen_table = gen_match_table
        gen_stratify_label = gen_match_stratify_label

    reco_table =pa.concat_tables([reco_match_table, reco_fake_table])
    reco_stratify_label = np.concatenate([reco_match_stratify_label, reco_fake_stratify_label])
 
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

    output_folder=f"{emb_data_folder}/omnisequential_1"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    do_diff_gen_bins = False

    reco_weights = reco_table["weight"].to_numpy()
    gen_weights = gen_table["weight"].to_numpy()

    reco_weights = reco_weights*(float(n_data)/np.sum(reco_weights))
    gen_weights = gen_weights*(float(n_data)/np.sum(gen_weights))
   
    w_unfolding = [gen_weights, reco_weights]

    iter_hist_list = []
    hists = {}
    n_iterations = 20
    last_iteration = -1 
    max_chi2_var = ""

    for iteration in range(0, n_iterations):
        max_chi2 = 0
        max_chi2_var = ""
        print(f"Iteration: {iteration}")
        for icol, col in enumerate(jet_columns):
            data_array = data_table[col]
            gen_array = gen_table[col]
            reco_array = reco_table[col]

            if iteration == 0:
                print(f"---Calculating bin edges using Baysian blocks for {col} histograms...")
                h = {}
                h["bins"], h["data"], h["data_err"] = histograms.make_hist(data_array.to_numpy(), name=col, p0=0.06, range=col_ranges[col])
                if do_diff_gen_bins:
                    h["gen_bins"], h["gen"], h["gen_err"] = histograms.make_hist(gen_array.to_numpy(), weight=gen_weights, name=col, size_for_bins=0.1, p0=0.05, shuffle=True, stratify=gen_match_stratify_label)
                else:
                    h["gen_bins"] = h["bins"]
            else:
                h = hists[col]

            _, h["reco"], h["reco_err"] = histograms.make_hist(reco_array.to_numpy(), weight=reco_weights, name=col, bins=h["bins"])
            _, h["gen"], h["gen_err"] = histograms.make_hist(gen_array.to_numpy(), weight=gen_weights, name=col,bins=h["gen_bins"])

            h["ratio"] = h["data"] / h["reco"]
            h["ratio_err"] = h["ratio"] * np.sqrt((h["data_err"] / h["data"]) ** 2 + (h["reco_err"] / h["reco"]) ** 2)

            h_data = h["data"][1:]
            h_reco = h["reco"][1:]
            h_reco = (h_reco / np.sum(h_reco)) * np.sum(h_data)
            dof = len(h_data) - 1
            # chi2 = np.sum((hReco-hData)**2/hData)
            chi2, _ = chisquare(h_reco, h_data)
            # chi2 = (chi2 - dof)/np.sqrt(2*dof)
            h["chi2"] = chi2
            hists[col] = h

            if chi2 > max_chi2:
                max_chi2 = chi2
                max_chi2_var = col

        iter_hist_list.append(copy.deepcopy(hists))
        if max_chi2 < 1:
            break
        
        last_iteration = iteration

        print(f"---Max chi2 = {max_chi2} from {max_chi2_var}")
        bins = hists[max_chi2_var]["bins"]
        nbins = len(bins) - 1
        scale = np.mean(bins[1:] - bins[:-1])
        features = (bins[1:] + bins[:-1]) / 2.0
        print("---Obtaining reco lvl reweight factors...")
        reco_reweights = propagate_values(features[1:], hists[max_chi2_var]["ratio"][1:], reco_table[max_chi2_var].to_numpy())
        reco_weights = reco_weights * reco_reweights
        reco_weights = reco_weights * (float(n_data)/np.sum(reco_weights))

        print("---Obtaining gen lvl reweight factors...")
        bins = hists[max_chi2_var]["gen_bins"]
        nbins = len(bins) - 1
        scale = np.mean(bins[1:] - bins[:-1])
        gen_features = (bins[1:] + bins[:-1]) / 2.0
        gen_matches = gen_match_table[max_chi2_var].to_numpy()
        gen_match_reweights = reco_reweights[0 : len(gen_matches)]
        _, gen_reweight_profile, _ = histograms.make_profile(gen_matches, gen_match_reweights, weight=gen_weights[0 : len(gen_matches)], bins=bins)

        selection = np.isnan(gen_reweight_profile)
        gen_features = gen_features[~selection]
        print(f"---{len(gen_features)} good bins out of {nbins} gen bins...")
        gen_reweight_profile = gen_reweight_profile[~selection]
        gen_reweights = propagate_values(gen_features, gen_reweight_profile, gen_table[max_chi2_var].to_numpy(), num_prediction_batches=40)
        gen_weights = gen_weights * gen_reweights
        gen_weights = gen_weights * (float(n_data)/np.sum(gen_weights))

        w_unfolding.extend([gen_weights, reco_weights])
    print("Iterations done...")
    with open(f"{output_folder}/omniseq-wts-iter{iteration+1}.npz", "wb") as f:
        np.savez(f, *w_unfolding)

    col_bins = {col:h["bins"] for col, h in hists.items()}

    with open(f"{output_folder}/omniseq-bins{iteration+1}.pkl", "wb") as f:
        pickle.dump(col_bins, f)

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

        axs = plotting.plot_ratios(
            subfig,
            h["bins"],
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
        if var == max_chi2_var:
            subfig.set_facecolor("red")
        plotting.plot_hist(
            axs[0],
            h["gen_bins"],
            h["gen"],
            errors=h["gen_err"],
            label=f"gen, iter = {last_iteration}",
            marker="^",
            fillstyle="none",
            markersize=10,
        )
        plotting.plot_hist(
            axs[0],
            h0["gen_bins"],
            h0["gen"],
            errors=h0["gen_err"],
            label="gen, iter = 0",
            marker="v",
            fillstyle="none",
            markersize=10,
        )
        # if var == max_chi2_var:
        #    axs[1].plot(
        #        reco_table[f"reco_{max_chi2_var}"].to_numpy(),
        #        reco_reweights,
        #        "o",
        #        label=f"data/reco, iter={last_iteration} fitted",
        #    )
        axs[0].set_title(f"chi2 = {h['chi2']:.2f}")
        axs[0].legend()
        axs[1].set_xlabel(var)
        axs[1].legend()
    fig.savefig(f"plot_output/omniseq.pdf")
    plt.show(block=True)

if __name__ == "__main__":
    main()
