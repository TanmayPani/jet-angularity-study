from functools import partial
from math import ceil
import matplotlib.pyplot as plt 
from matplotlib import colors

import numpy as np
import pyarrow as pa 

from arrow_to_tensordict import add_extra_columns
from utils.histogram import make_hist

if __name__ == "__main__": 
    num_replicas = 10
    num_data_subsample = 500000
    batch_size = 10000
    train_size = 0.6
    num_iterations = 10 
    num_epochs = 20
    dataseed = 42
    modelseed = 0

    filename_stub_fmt = "nreplicas{}_sub{}_batch_size{}_niter{}_nepochs{}_trainsplit{}_dataseed{}_modelseed{}"
   
    filename_stub = filename_stub_fmt.format(
        num_replicas, num_data_subsample, batch_size, num_iterations, num_epochs, 
        f"{int(train_size*100)}{int((1-train_size)*100)}", dataseed, modelseed,
    )

    source_dir = "partitioned_datasets/nominal"
    pth_bins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
    pth_bin_dirs = [
        f"{source_dir}/arrow_data/ptHat{pth_low}to{pth_high}" 
        for pth_low, pth_high in zip(pth_bins[:-1], pth_bins[1:])
    ]
    
    jet_columns = [
        "pt"            ,
       # "nef"           ,
       # "ch_ang_k1_b0.5",
        "ch_ang_k1_b1"  ,
        "ch_ang_k1_b2"  ,
        "ch_ang_k2_b0"  ,
       # "leading_constit_pt",
       # "subleading_constit_pt",
       # "hc_pt",
       # "hc_ch_ang_k1_b0.5",
       # "hc_ch_ang_k1_b1",
       # "hc_ch_ang_k1_b2",
       # "hc_ch_ang_k2_b0",
        ]

    var_xlabel = {
        "pt"            :r"$p_{\rm T, jet} (GeV/$c$)$"                          ,
        "nef"           :r"$(\sum p_{\rm T, jet}^{\rm neutral})/p_{\rm T, jet}$",
        "ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1}_{\beta = 0.5}$ (LHA)"          ,
        "ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1}_{\beta = 1}$ (girth)"          , 
        "ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1}_{\beta = 2}$ (thrust)"         ,
        "ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2}_{\beta = 0}$ ($(p_T^D)^2$)"    ,
        "leading_constit_pt"   : r"$p_{\rm T, constit.}^{\rm leading} (GeV/$c$)$",
        "subleading_constit_pt " :r"$p_{\rm T, constit.}^{\rm sub-leading} (GeV/$c$)$",
        "hc_pt"            :r"$p_{\rm T, jet}^{\rm h.c.} (GeV/$c$)$"                          ,
        "hc_ch_ang_k1_b0.5":r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 0.5}$ (LHA, hard-core jets)"          ,
        "hc_ch_ang_k1_b1"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 1}$ (girth, hard-core jets)"          ,
        "hc_ch_ang_k1_b2"  :r"$\lambda^{\kappa = 1, \rm h.c.}_{\beta = 2}$ (thrust, hard-core jets)"         ,
        "hc_ch_ang_k2_b0"  :r"$\lambda^{\kappa = 2, \rm h.c.}_{\beta = 0}$ ($(p_T^D)^2$, hard-core jets)"    ,
    }                      

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

    data_infile = f"{source_dir}/arrow_data/jets-conPtMin0.2.arrow"
    reco_files = [f"{dir}/reco-matches.arrow" for dir in pth_bin_dirs]+[f"{dir}/fakes.arrow" for dir in pth_bin_dirs] 
    gen_files = [f"{dir}/gen-matches.arrow" for dir in pth_bin_dirs]+[f"{dir}/misses.arrow" for dir in pth_bin_dirs]

    data_buffer = pa.memory_map(data_infile, "rb")
    data_table = add_extra_columns(pa.ipc.open_file(data_buffer).read_all())

    reco_buffers = [pa.memory_map(path, "rb") for path in reco_files]
    reco_tables = [pa.ipc.open_file(buff).read_all() for buff in reco_buffers]
    reco_table = add_extra_columns(pa.concat_tables(reco_tables))

    gen_buffers = [pa.memory_map(path, "rb") for path in gen_files]
    gen_tables = [pa.ipc.open_file(buff).read_all() for buff in gen_buffers]
    gen_table = add_extra_columns(pa.concat_tables(gen_tables))

    weights = np.load(f"outputs/unfolding_{filename_stub}/w_unfolding_{num_iterations}.npz")

    data_array = {}
    data_count = {}
    data_count_err = {}

    reco_array = {}
    reco_count = {}
    reco_count_err = {}

    gen_array = {}
    gen_count = {}
    gen_count_err = {}

    bins = {}
    x = {}
    x_err = {}

    for var_name in jet_columns:
        data_array[var_name] = data_table[var_name].to_numpy()
        reco_array[var_name] = reco_table[var_name].to_numpy()
        gen_array[var_name] = gen_table[var_name].to_numpy()

        bins[var_name], data_count[var_name], data_count_err[var_name] = make_hist(
            data_array[var_name], p0=0.06, range=col_ranges[var_name]
        )

        _, reco_count[var_name], reco_count_err[var_name] = make_hist(
            reco_array[var_name], weight = weights["arr_1"][0], bins=bins[var_name],
        )
        _, gen_count[var_name], gen_count_err[var_name] = make_hist(
            gen_array[var_name], weight = weights["arr_0"][0], bins=bins[var_name],
        )
        
        x[var_name] = 0.5*(bins[var_name][1:]+bins[var_name][:-1])
        x_err[var_name] = 0.5*(bins[var_name][1:]-bins[var_name][:-1])

    num_cols = 4 
    num_rows = ceil(len(jet_columns)/num_cols)
    fig_scale = 4
    for iteration in range(6):
        print(f"___processing iteration {iteration}...")

        fig, ax = plt.subplots(
            num_rows, num_cols, 
            figsize=(fig_scale*num_cols, fig_scale*num_rows),
            constrained_layout=True,
        )
        axs_flat = ax.flatten()

        fig_wt, ax_wt = plt.subplots(
            num_rows, num_cols, 
            figsize=(fig_scale*num_cols, fig_scale*num_rows),
            sharey=True,
            width_ratios=(1,1,1,1),
            gridspec_kw={"wspace": 0, "right": 0.99, "left": 0.05, "top": 0.9, "bottom": 0.15},
        )
        fig_wt.suptitle(f"Iteration : {iteration}")
        ax_wt_flat = ax_wt.flatten()
       
        wt_iter_gen = weights[f"arr_{2*iteration}"]
        wt_iter_reco = weights[f"arr_{2*iteration+1}"]
        wt_mean_iter = np.mean(weights[f"arr_{2*iteration}"], axis=0)
        for ivar, var_name in enumerate(jet_columns):
            print(f"plotting {var_name}...")
            unf_counts = []
            unf_count_errs = []
            for ireplica in range(num_replicas):
                _, unf_count, unf_count_err = make_hist(
                    gen_array[var_name], weight = wt_iter_gen[ireplica], bins=bins[var_name]
                )
                unf_counts.append(unf_count)
                unf_count_errs.append(unf_count_err)

            unf_count_stacked = np.stack(unf_counts, axis=1)
            unf_count_mean = unf_count_stacked.mean(axis=1)
            unf_count_std  = unf_count_stacked.std(axis=1)
            unf_count_low  = unf_count_mean - unf_count_std
            unf_count_high = unf_count_mean + unf_count_std
            
            axs_flat[ivar].fill_between(
                x[var_name], unf_count_low, unf_count_high, 
                label=r"unfolded $\pm \Delta$ unfolded", alpha=0.3, color="magenta"
            )

            axs_flat[ivar].errorbar(
                x[var_name], data_count[var_name], 
                xerr=x_err[var_name], yerr=data_count_err[var_name], 
                label="data", linestyle="none", color="blue",
                marker="o", markeredgecolor="white", markerfacecolor="blue",
            )

            axs_flat[ivar].errorbar(
                x[var_name], reco_count[var_name], 
                xerr=x_err[var_name], yerr=reco_count_err[var_name], 
                label="reco", linestyle="none", color="red", 
                marker="v", markeredgecolor="white", markerfacecolor="red",
            )

            axs_flat[ivar].errorbar(
                x[var_name], gen_count[var_name], 
                xerr=x_err[var_name], yerr=gen_count_err[var_name],
                label="gen", linestyle="none", color="orange", 
                marker="^", markeredgecolor="white", markerfacecolor="orange",
            )

            ax_wt_flat[ivar].hist2d(
                gen_array[var_name], np.log10(wt_mean_iter), 
                density=True, norm="log",bins=20,
            )
            ax_wt_flat[ivar].set_xlabel(var_xlabel[var_name], fontsize="xx-large")
            ax_wt_flat[ivar].set_ylim(-5, 3)

            axs_flat[ivar].set_xlabel(var_xlabel[var_name], fontsize="xx-large")
            axs_flat[ivar].set_yscale("log")
            #axs_flat[ivar].set_title(f"iteration {iteration}")
            if ivar % num_cols == 0:
                axs_flat[ivar].set_ylabel("Arbitrary Units", fontsize="xx-large")
                ax_wt_flat[ivar].set_ylabel(r"$\log(\langle w_{\rm sample}\rangle)$", fontsize="xx-large")
            if ivar % num_cols == 3:
                axs_flat[ivar].legend()

        fig.savefig(f"slides/unfolded_hist/iteration_{iteration}.pdf")
        fig_wt.savefig(f"slides/unfolded_hist/wts_iteration_{iteration}.pdf")
        
    try:
        plt.show()
    except KeyboardInterrupt:
        exit()










