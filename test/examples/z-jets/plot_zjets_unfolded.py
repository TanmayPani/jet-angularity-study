import os
import json
from math import ceil

import matplotlib.pyplot as plt 

import numpy as np

import torch
from tensordict import TensorDict

from utils.histogram import make_hist
from utils import zjets

keys_to_stack = ["ms", "mults", "tau21s", "widths", "lnrhos", "zgs"]

key_ranges ={
    "ms"     : (0, 60), 
    "mults"  : (0, 80), 
    "widths" : (0, 0.6), 
    "lnrhos" : (-14, -2), 
    "tau21s" : (0, 1.2), 
    "zgs"    : (0.1, 0.5),
}
key_xlabels ={
    "ms"     :r"Mass $m$ (GeV/$c^2$)", 
    "mults"  :r"Multiplicity $M$", 
    "widths" :r"Width $w$",   
    "lnrhos" :r"Groomed Mass  $\ln \rho = \ln(m^2_{SD}/p^2)$",
    "tau21s" :r"N-subjetty ratio $\tau_{21}$",
    "zgs"    :r"Groomed Mom. Fraction $z_g$",
}

def add_columns(td):
    td["lnrhos"]= torch.log(td["sdms"]**2/td["pts"]**2)
    td["tau21s"] = td["tau2s"]/td["widths"]
    return td


if __name__ == "__main__":
    cfg = {}
    with open('config.json', 'r') as config_file:
        cfg.update(json.load(config_file))
    filename_stub = cfg["filename_stub_fmt"].format(
        cfg["num_replicas"], 
        cfg["num_data_subsample"], 
        cfg["batch_size"], 
        cfg["num_iterations"], 
        cfg["num_epochs"], 
        f"{int(cfg["train_size"]*100)}{int((1-cfg["train_size"])*100)}", 
        cfg["dataseed"], 
        cfg["modelseed"],
    )

    num_iters_to_plot = cfg["num_iterations"] + 1

    data_prefix = os.path.join('datasets', 'ZjetsDelphes')
    datasets = ("herwig", "pythia26")
    levels = ("sim", "gen")

    data_td = TensorDict.load_memmap(
        os.path.join(
            data_prefix, zjets.FILENAME_PREFIX[datasets[0]], "sim",
        )
    )
    data_sel = data_td["zgs"] > 0
    data = add_columns(data_td[data_sel].unlock_())

    truth_td = TensorDict.load_memmap(
        os.path.join(
            data_prefix, zjets.FILENAME_PREFIX[datasets[0]], "gen",
        )
    )
    truth_sel = truth_td["zgs"] > 0
    truth = add_columns(truth_td[data_sel].unlock_())

    reco_td = TensorDict.load_memmap(
        os.path.join(
            data_prefix, zjets.FILENAME_PREFIX[datasets[1]], "sim",
        )
    )

    gen_td = TensorDict.load_memmap(
        os.path.join(
            data_prefix, zjets.FILENAME_PREFIX[datasets[1]], "gen",
        )
    )

    reco_gen_sel = (reco_td["zgs"] > 0) & (gen_td["zgs"] > 0)
    reco = add_columns(reco_td[reco_gen_sel].unlock_())
    gen  = add_columns(gen_td[reco_gen_sel].unlock_())

    dir_prefix = f"./outputs/cms_z_jets/unfolding_{filename_stub}"
    wt_unf = []
    wt_reco = []
    with np.load(f"{dir_prefix}/w_unfolding_{cfg["num_iterations"]}.npz") as f:
        for iter in range(num_iters_to_plot):
            if iter == 0:
                wt_reco.append(f[f"arr_0"])
                wt_unf.append(f[f"arr_0"])
                continue

            wt_reco.append(f[f"arr_{2*iter-1}"])
            wt_unf.append(f[f"arr_{2*iter}"])
    
    data_array = {}
    data_count = {}
    data_count_err = {}

    reco_array = {}
    reco_count = {}
    reco_count_err = {}

    gen_array = {}
    gen_count = {}
    gen_count_err = {}

    truth_array = {}
    truth_count = {}
    truth_count_err = {}

    bins = {}
    x = {}
    x_err = {}


    for var_name in keys_to_stack:
        data_array[var_name] = data[var_name].numpy()
        reco_array[var_name] = reco[var_name].numpy()
        gen_array[var_name] = gen[var_name].numpy()
        truth_array[var_name] = truth[var_name].numpy()

        bins[var_name], data_count[var_name], data_count_err[var_name] = make_hist(
            data_array[var_name], p0=0.06, range=key_ranges[var_name]
        )

        _, reco_count[var_name], reco_count_err[var_name] = make_hist(
            reco_array[var_name], bins=bins[var_name],
        )

        _, gen_count[var_name], gen_count_err[var_name] = make_hist(
            gen_array[var_name], bins=bins[var_name],
        )

        _, truth_count[var_name], truth_count_err[var_name] = make_hist(
            truth_array[var_name], bins=bins[var_name],
        )

        x[var_name] = 0.5*(bins[var_name][1:]+bins[var_name][:-1])
        x_err[var_name] = 0.5*(bins[var_name][1:]-bins[var_name][:-1])

    #figs = {}
    #axs = {}

    replicas = list(range(cfg["num_replicas"])) 
        
    for iter in range(num_iters_to_plot):
        print("Iteration:", iter)
        fig, ax = plt.subplots(2, 3, figsize=(12, 8))
        fig.suptitle(f"Iteration: {iter}")
        ax_flat = ax.flatten()
        
        fig_wt, ax_wt = plt.subplots(
            2, 3, 
            figsize=(12, 8),
            sharey=True,
            #width_ratios=(1,1,1,1),
            gridspec_kw={"wspace": 0, "right": 0.99, "left": 0.05, "top": 0.9, "bottom": 0.15},
        )
        fig_wt.suptitle(f"Iteration : {iter}")
        ax_wt_flat = ax_wt.flatten()
        for ikey, key in enumerate(keys_to_stack):
            ax_flat[ikey].errorbar(
                x[key], truth_count[key], xerr=x_err[key], yerr=truth_count_err[key],
                label="truth", color="red", markerfacecolor="red", 
                linestyle="none", marker="o", markeredgecolor="white",
            )
           
            ax_flat[ikey].errorbar(
                x[key], gen_count[key], xerr=x_err[key], yerr=gen_count_err[key],
                label="gen", color="brown", markerfacecolor="brown", 
                linestyle="none", marker="o", markeredgecolor="white",
            )

            unf_counts = []
            unf_count_errs = []

            for ireplica in replicas:

                _, unf_count, unf_count_err = make_hist(
                    gen_array[key], weight = wt_unf[iter][ireplica], bins=bins[key], 
                )

                unf_counts.append(unf_count)
                unf_count_errs.append(unf_count_err)

                ax_flat[ikey].plot(
                    x[key], unf_count, alpha=0.45, #label=f"replica: {ireplica}"
                )
            
            unf_count_stacked = np.stack(unf_counts, axis=1)
            unf_count_mean = np.mean(unf_count_stacked, axis=1, keepdims=False)
            unf_count_median = np.median(unf_count_stacked, axis=1, keepdims=False)
            unf_count_std  = unf_count_stacked.std(axis=1, keepdims=False)
            
            #if iter > 0:
            #    delta_unf_count_stacked = np.abs(unf_count_stacked - np.expand_dims(unf_count_mean, axis=1))
            #    is_bin_good = delta_unf_count_stacked <= 3.0*np.expand_dims(unf_count_std, axis=1)
            #    is_replica_good = np.all(is_bin_good, axis=0, keepdims=False)
            #    #print(delta_unf_count_stacked, is_bin_good)
            #    #print(is_replica_good)
            #    replicas = np.asarray(replicas)[is_replica_good].tolist()
            #    print("replicas remaining:", replicas)
            #    unf_count_mean = np.mean(unf_count_stacked, axis=1, keepdims=False, where=is_replica_good)
            #    unf_count_median = np.mean(unf_count_stacked, axis=1, keepdims=False, where=is_replica_good)
            #    unf_count_std  = unf_count_stacked.std(axis=1, keepdims=False)
            
            unf_count_mid = unf_count_median
            
            wt_unf_mean = np.mean(wt_unf[iter], axis=0)

            ax_wt_flat[ikey].hist2d(
                gen_array[key], np.log10(wt_unf_mean), 
                density=True, norm="log",bins=20, range=(key_ranges[key],(-4,4)),
            )
            ax_wt_flat[ikey].set_xlabel(key_xlabels[key], fontsize="x-large")
            ax_wt_flat[ikey].set_ylim(-3, 3)

            unf_count_high = unf_count_mid + unf_count_std
            unf_count_low = unf_count_mid - unf_count_std
                        
            ax_flat[ikey].errorbar(
                x[key], unf_count_mid.squeeze(), xerr=x_err[key], yerr=unf_count_std,
                label="unfolded", color="magenta", markerfacecolor="magenta", 
                linestyle="none", marker="o", markeredgecolor="white",
            )

            ax_flat[ikey].fill_between(
                x[key], unf_count_low, unf_count_high, 
                label=r"unfolded $\pm \Delta$ unfolded", alpha=0.3, color="magenta"
            )

            ax_flat[ikey].set_xlabel(key_xlabels[key], fontsize="x-large")
            ax_flat[ikey].set_yscale("log")
            if ikey % 3 == 0:
                ax_flat[ikey].set_ylabel("Arbitrary Units", fontsize="xx-large")
                ax_wt_flat[ikey].set_ylabel(r"$\log(\langle w_{\rm sample}\rangle)$", fontsize="xx-large")
        ax_flat[-1].legend()
    
        fig.savefig(f"slides/unfolded_hist/_iteration_{iter}.pdf")
        fig_wt.savefig(f"slides/unfolded_hist/_wts_iteration_{iter}.pdf")
        
    try:
        plt.show()
    except KeyboardInterrupt:
        pass



