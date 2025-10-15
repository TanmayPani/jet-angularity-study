import os
import pickle

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

import pyarrow as pa

from scipy.stats import chisquare

from matplotlib import pyplot as plt
from utils import plotting
from utils import histogram

import copy

from utils.arrow_table import pa_concated_table, pa_table

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
    pth_bin_folders = [
        f"{emb_data_folder}/ptHat{pth_low}to{pth_high}" 
        for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])
    ]
    n_pth_bins = len(pth_bin_folders)
    pth_label = list(range(1, n_pth_bins+1))

    gen_match_buffers, gen_match_table = pa_concated_table([
        f"{folder}/gen-matches.arrow" for folder in pth_bin_folders
    ], label=pth_label, label_key="stratify_labels")
    gen_miss_buffers, gen_miss_table = pa_concated_table([f"{folder}/misses.arrow" for folder in pth_bin_folders], label=pth_label, label_key="stratify_labels") 
    reco_match_buffers, reco_match_table = pa_concated_table([f"{folder}/reco-matches.arrow" for folder in pth_bin_folders], label=pth_label, label_key="stratify_labels")
    reco_fake_buffers, reco_fake_table = pa_concated_table([f"{folder}/fakes.arrow" for folder in pth_bin_folders], label=pth_label, label_key="stratify_labels")

    data_buffer, data_table = pa_table("outputs/jets-conPtMin0.2.arrow", label=0, label_key="stratify_labels")
    
    do_use_gen_misses = False

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
            data_array = data_table[col].to_numpy()
            gen_array = gen_table[col].to_numpy()
            reco_array = reco_table[col].to_numpy()

            if iteration == 0:
                print(f"---Calculating bin edges using Baysian blocks for {col} histograms...")
                h = {}
                h["bins"], h["data"], h["data_err"] = histogram.make_hist(data_array, p0=0.06, range=col_ranges[col])
                if do_diff_gen_bins:
                    h["gen_bins"], h["gen"], h["gen_err"] = histogram.make_hist(
                        gen_array, 
                        weight=gen_weights, 
                        size_for_bins=0.1, 
                        p0=0.05, 
                        shuffle=True, 
                        #stratify=gen_match_stratify_label
                    )
                else:
                    h["gen_bins"] = h["bins"]
            else:
                h = hists[col]

            _, h["reco"], h["reco_err"] = histogram.make_hist(reco_array, weight=reco_weights, bins=h["bins"])
            _, h["gen"], h["gen_err"] = histogram.make_hist(gen_array, weight=gen_weights,bins=h["gen_bins"])

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
        _, gen_reweight_profile, _ = histogram.make_profile(gen_matches, gen_match_reweights, weight=gen_weights[0 : len(gen_matches)], bins=bins)

        selection = np.isnan(gen_reweight_profile)
        gen_features = gen_features[~selection]
        print(f"---{len(gen_features)} good bins out of {nbins} gen bins...")
        gen_reweight_profile = gen_reweight_profile[~selection]
        gen_reweights = propagate_values(gen_features, gen_reweight_profile, gen_table[max_chi2_var].to_numpy(), num_prediction_batches=40)
        gen_weights = gen_weights * gen_reweights
        gen_weights = gen_weights * (float(n_data)/np.sum(gen_weights))

        w_unfolding.extend([gen_weights, reco_weights])

    print("Iterations done...")
    last_iteration = len(iter_hist_list)
    with open(f"{output_folder}/omniseq-wts-iter{last_iteration}.npz", "wb") as f:
        np.savez(f, *w_unfolding)

    col_bins = {col:h["bins"] for col, h in hists.items()}

    with open(f"{output_folder}/omniseq-bins{last_iteration}.pkl", "wb") as f:
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
