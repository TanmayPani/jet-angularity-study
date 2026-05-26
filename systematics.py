import os
from copy import deepcopy
from collections import defaultdict
from enum import StrEnum


import numpy as np
import numba as nb
import torch
import awkward as ak

import matplotlib.pyplot as plt


class SysVar(StrEnum):
    NONE = "nominal"
    TOWER_ET_CORRECTION = "tower_et_corr_sys"
    TRACK_EFFICIENCY = "track_pt_sys"
    JET_PT_RESOLUTION_0 = "jet_pt_res_sys_0"
    JET_PT_RESOLUTION_1 = "jet_pt_res_sys_1"
    UNFOLDING_PRIOR_SAME = "unf_prior_same"
    UNFOLDING_PRIOR_LIKE_DATA = "unf_prior_like_data"
    UNFOLDING_PRIOR_HERWIG7 = "unf_prior_herwig7"
    UNFOLDING_PRIOR_PYTHIA8 = "unf_prior_pythia8"
    # UNFOLDING_BOOTSTRAP = "unf_bootstrap_sys"
    UNFOLDING_ITERATION_0 = "unf_iter_sys_0"
    UNFOLDING_ITERATION_1 = "unf_iter_sys_1"


common_vars = (
    "m",
    "sd_m",
    "sd_dR",
    "sd_symmetry",
)

angularities = (
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1",
    "ch_ang_k1_b2",
    "ch_ang_k2_b0",
)

var_label = {
    "ch_ang_k1_b0.5": r"\lambda^{\kappa = 1}_{\beta = 0.5}",
    "ch_ang_k1_b1": r"\lambda^{\kappa = 1}_{\beta = 1}",
    "ch_ang_k1_b2": r"\lambda^{\kappa = 1}_{\beta = 2}",
    "ch_ang_k2_b0": r"\lambda^{\kappa = 2}_{\beta = 0}",
    "sd_ch_ang_k1_b0.5": r"\lambda^{\kappa = 1, \rm SD}_{\beta = 0.5}",
    "sd_ch_ang_k1_b1": r"\lambda^{\kappa = 1, \rm SD}_{\beta = 1}",
    "sd_ch_ang_k1_b2": r"\lambda^{\kappa = 1, \rm SD}_{\beta = 2}",
    "sd_ch_ang_k2_b0": r"\lambda^{\kappa = 2, \rm SD}_{\beta = 0}",
}

var_unit = {
    "ch_ang_k1_b0.5": r"(LHA)",
    "ch_ang_k1_b1": r"(girth)",
    "ch_ang_k1_b2": r"(thrust)",
    "ch_ang_k2_b0": r"((p_T^D)^2)",
    "sd_ch_ang_k1_b0.5": r"(LHA, groomed)",
    "sd_ch_ang_k1_b1": r"(girth, groomed)",
    "sd_ch_ang_k1_b2": r"(thrust, groomed)",
    "sd_ch_ang_k2_b0": r"((p_T^D)^2, groomed)",
}


def apply_hadronic_correction_sys_var(events, hadr_corr_frac=0.5):
    tower_dE = events["towers._RawE"] - events["towers._E"]
    events["towers._E"] = events["towers._E"] - hadr_corr_frac * tower_dE
    mass_array = ak.full_like(events["towers._E"], 0.13957)
    tower_p2 = events["towers._E"] ** 2 - mass_array**2
    tower_p2 = ak.fill_none(ak.mask(tower_p2, tower_p2 > 0), value=0)
    tower_p = np.sqrt(tower_p2)
    events["towers._Pt"] = tower_p / np.cosh(events["towers._Eta"])
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
    # flat_factors = np.random.default_rng().uniform(-0.04, 0.04, n_tot_trk)
    flat_factors = np.random.default_rng(seed).random(n_tot_trk)
    return apply_flat_track_pt_factors(
        ak.ArrayBuilder(), events["tracks._Pt"], flat_factors
    ).snapshot()


# def get_jet_pt_bins(sys_var):
#    match sys_var:
#        case SysVar.JET_PT_RESOLUTION_0:
#            return (11.0, 14.0, 21.0, 38.0, 82.0)
#        case SysVar.JET_PT_RESOLUTION_1:
#            return (9.0, 16.0, 19.0, 42.0, 78.0)
#        case _:
#            return (10.0, 15.0, 20.0, 40.0, 80.0)


def get_jet_pt_bins(sys_var):
    match sys_var:
        case SysVar.JET_PT_RESOLUTION_0:
            return (11.0, 14.0, 21.0, 28.0, 62.0)
        case SysVar.JET_PT_RESOLUTION_1:
            return (9.0, 16.0, 19.0, 32.0, 58.0)
        case _:
            return (10.0, 15.0, 20.0, 30.0, 60.0)


def get_unfolding_iter(sys_var, nom_iter=5):
    match sys_var:
        case SysVar.UNFOLDING_ITERATION_0:
            return nom_iter - 1
        case SysVar.UNFOLDING_ITERATION_1:
            return nom_iter + 2
        case _:
            return nom_iter


def _unc_less_preferred_sys_var(
    h_nominal: dict[str, torch.Tensor], *h_sys_vars: dict[str, torch.Tensor]
):
    h_sys_vars_stacked = torch.stack([h["bin_count"] for h in h_sys_vars])
    h_sys_vars_mean = h_sys_vars_stacked.mean(0)
    sys_var_unc = (h_nominal["bin_count"] - h_sys_vars_mean).abs_()
    return sys_var_unc


def _unc_equal_preferred_sys_var(
    h_nominal: dict[str, torch.Tensor], *h_sys_vars: dict[str, torch.Tensor]
):
    hist_list = [h_nominal["bin_count"]]
    hist_list.extend(h["bin_count"] for h in h_sys_vars)
    hist_stacked = torch.stack(hist_list)
    hist_mean = hist_stacked.mean(0)
    hist_std = hist_stacked.std(0)
    return hist_mean, hist_std


def _unc_prior_var(h_nominal: dict[str, torch.Tensor], ratio: dict[str, torch.Tensor]):
    h_sys_var = h_nominal["bin_count"] / ratio["bin_count"]
    # h_sys_var_sum = h_sys_var.nansum()
    # h_sys_var_scale = (h_nominal["half_bin_width"] * 2.).mul_(h_sys_var_sum)
    # h_sys_var.div_(h_sys_var_scale)
    prior_var_unc = (h_sys_var - h_nominal["bin_count"]).abs_()
    return prior_var_unc


def calculate_sys_uncertainty(
    hist_nominal: dict[str, torch.Tensor],
    var_name: str,
    hist_file_name: str,
    *sys_var_names: str,
    path_prefix: str = "",
    feature_mode: str,
    is_equal_pref: bool = False,
):
    sys_var_paths = [
        os.path.join(path_prefix, sys_var, feature_mode, var_name, hist_file_name)
        for sys_var in sys_var_names
    ]

    sys_var_hists = [torch.load(file_path) for file_path in sys_var_paths]

    if is_equal_pref:
        return _unc_equal_preferred_sys_var(hist_nominal, *sys_var_hists)
    else:
        return _unc_less_preferred_sys_var(hist_nominal, *sys_var_hists)


def calculate_prior_uncertainty(
    hist_nominal: dict[str, torch.Tensor],
    var_name: str,
    hist_file_name: str,
    path_prefix: str,
    feature_mode: str,
    prior_sysvar: SysVar = SysVar.UNFOLDING_PRIOR_LIKE_DATA,
):
    ratio_path = os.path.join(
        path_prefix,
        str(prior_sysvar),
        feature_mode,
        var_name,
        "ratio",
        hist_file_name,
    )
    hist_ratio = torch.load(ratio_path, mmap=True)
    sys_var_prior = _unc_prior_var(hist_nominal, hist_ratio)
    return sys_var_prior


def calculate_uncertainties(var_name: str, path_prefix: str, feature_mode: str):
    nominal_path = os.path.join(path_prefix, "nominal", feature_mode, var_name)
    sys_vars = defaultdict(dict)
    hist_nominal = {}
    for file_name in os.listdir(nominal_path):
        hist_name = file_name.removesuffix(".pt")
        hist_nominal[hist_name] = torch.load(
            os.path.join(nominal_path, file_name),
            mmap=True,
        )
        sys_vars[hist_name]["unf_iter_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "unf_iter_sys_0",
            "unf_iter_sys_1",
            path_prefix=path_prefix,
            feature_mode=feature_mode,
        )

        sys_vars[hist_name]["jet_pt_res_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "jet_pt_res_sys_0",
            "jet_pt_res_sys_1",
            path_prefix=path_prefix,
            feature_mode=feature_mode,
        )

        sys_vars[hist_name]["track_pt_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "track_pt_sys",
            path_prefix=path_prefix,
            feature_mode=feature_mode,
        )

        sys_vars[hist_name]["tower_et_corr_sys"] = calculate_sys_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            "tower_et_corr_sys",
            path_prefix=path_prefix,
            feature_mode=feature_mode,
        )

        sys_vars[hist_name]["unf_prior_like_data"] = calculate_prior_uncertainty(
            hist_nominal[hist_name],
            var_name,
            file_name,
            path_prefix=path_prefix,
            feature_mode=feature_mode,
        )

        _h7_ratio_path = os.path.join(
            path_prefix,
            str(SysVar.UNFOLDING_PRIOR_HERWIG7),
            feature_mode,
            var_name,
            "ratio",
            file_name,
        )
        if os.path.exists(_h7_ratio_path):
            sys_vars[hist_name]["unf_prior_herwig7"] = calculate_prior_uncertainty(
                hist_nominal[hist_name],
                var_name,
                file_name,
                path_prefix=path_prefix,
                feature_mode=feature_mode,
                prior_sysvar=SysVar.UNFOLDING_PRIOR_HERWIG7,
            )

        squared_sum = torch.zeros_like(hist_nominal[hist_name]["bin_count"])
        for sys_var_unc in sys_vars[hist_name].values():
            squared_sum.add_(sys_var_unc.square())
        sys_vars[hist_name]["total_sys"] = squared_sum.sqrt_()
        # sys_vars[hist_name]["bootstrap_stat"] = hist_nominal[hist_name]["bin_count_std"]
    return sys_vars, hist_nominal


def save_sys_uncertainties(sys_vars, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for hname, sys_var_dict in sys_vars.items():
        save_path = os.path.join(save_dir, f"{hname}.pt")
        torch.save(sys_var_dict, save_path)


def _safe_rel_unc(unc, y):
    """Return |unc / y| * 100 with NaN/inf scrubbed and a defensive copy."""
    safe_y = torch.where(y != 0, y, torch.full_like(y, float("nan")))
    return (
        (unc.clone() / safe_y)
        .abs_()
        .mul_(100)
        .nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    )


def plot_uncertainties(var_name, hnominal, sys_vars, fig_save_dir):
    import re

    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    num_cols = len(jpt_bins) - 1
    fig_scale = 5
    os.makedirs(fig_save_dir, exist_ok=True)

    # Group hist names by mod (everything before "_jpt<i>").
    mods = defaultdict(list)
    for hist_name in sys_vars.keys():
        m = re.match(r"^(.*)_jpt(\d+)$", hist_name)
        if m is None:
            continue
        mods[m.group(1)].append(int(m.group(2)))

    ylabel = r"$\frac{|\Delta h|}{h} \times 100$"

    for mod, ijpts in mods.items():
        ijpts = sorted(set(ijpts))
        first_hname = f"{mod}_jpt{ijpts[0]}"
        sv_keys = [k for k in sys_vars[first_hname].keys() if k != "total_sys"]

        fig = plt.figure(figsize=(num_cols * fig_scale, fig_scale + 1))
        axs = fig.subplots(
            1, num_cols, sharey=True,
            gridspec_kw={"wspace": 0, "right": 0.9, "left": 0.2},
        )
        if num_cols == 1:
            axs = [axs]
        axs[0].set_ylabel(ylabel, labelpad=10, size="xx-large")

        single_figs, single_axs = {}, {}
        for sv in sv_keys:
            single_figs[sv] = plt.figure(figsize=(num_cols * fig_scale, fig_scale + 1))
            ax_arr = single_figs[sv].subplots(
                1, num_cols, sharey=True,
                gridspec_kw={"wspace": 0, "right": 0.9, "left": 0.2},
            )
            if num_cols == 1:
                ax_arr = [ax_arr]
            ax_arr[0].set_ylabel(ylabel, labelpad=10, size="xx-large")
            single_axs[sv] = ax_arr

        for ijpt in ijpts:
            if ijpt >= num_cols:
                continue
            hist_name = f"{mod}_jpt{ijpt}"
            hist = hnominal[hist_name]
            x = hist["bin_center"]
            x_err = hist["half_bin_width"]
            bin_edges_low = x - x_err
            bin_edge_high = (x[-1] + x_err[-1]).unsqueeze(0)
            bins = torch.concatenate((bin_edges_low, bin_edge_high))

            y = hist["bin_count"]
            jpt_min, jpt_max = jpt_bins[ijpt], jpt_bins[ijpt + 1]

            stat_unc = hist.get("bin_count_std", torch.zeros_like(y))
            rel_stat = _safe_rel_unc(stat_unc, y)
            total_rel = _safe_rel_unc(sys_vars[hist_name]["total_sys"], y)

            ax = axs[ijpt]
            ax.text(
                0.15, 0.55,
                rf"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$",
                transform=ax.transAxes,
            )
            zero_base = torch.zeros_like(y).numpy()
            ax.stairs(rel_stat, bins, baseline=zero_base,
                      fill=True, color="magenta", alpha=0.3,
                      label="stat. (bootstrap)")
            ax.stairs(total_rel, bins, label="total sys",
                      linewidth=2.0, linestyle="dotted")

            for sv in sv_keys:
                rel = _safe_rel_unc(sys_vars[hist_name][sv], y)
                ax.stairs(rel, bins, label=sv, linewidth=2.0)

                sax = single_axs[sv][ijpt]
                sax.stairs(rel_stat, bins, baseline=zero_base,
                           fill=True, color="magenta", alpha=0.3,
                           label="stat. (bootstrap)")
                sax.stairs(total_rel, bins, label="total sys",
                           linewidth=2.0, linestyle="dotted")
                sax.stairs(rel, bins, label=sv, linewidth=2.0)
                sax.text(
                    0.2, 0.75,
                    rf"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$",
                    transform=sax.transAxes,
                )

        axs[-1].legend()
        combined_path = os.path.join(fig_save_dir, f"{mod}-sysQA.pdf")
        print("Saving figure to:", combined_path)
        fig.savefig(combined_path, bbox_inches="tight")
        plt.close(fig)

        for sv in sv_keys:
            single_axs[sv][-1].legend()
            sv_path = os.path.join(fig_save_dir, f"{mod}-{sv}-sysQA.pdf")
            single_figs[sv].savefig(sv_path, bbox_inches="tight")
            plt.close(single_figs[sv])


def main():
    import json

    path_prefix = "outputs/histograms"
    with open("runtime-files/config.json") as _cfg_file:
        feature_mode = json.load(_cfg_file)["feature_mode"]

    for var_name in common_vars + angularities:
        sys_vars, hnominal = calculate_uncertainties(
            var_name, path_prefix=path_prefix, feature_mode=feature_mode
        )
        save_dir = os.path.join(path_prefix, "sys_errors", feature_mode, var_name)
        save_sys_uncertainties(sys_vars, save_dir)
        fig_save_dir = os.path.join(path_prefix, "plots", feature_mode, var_name)
        plot_uncertainties(var_name, hnominal, sys_vars, fig_save_dir)
    print("Done!")


if __name__ == "__main__":
    main()
    plt.show()
