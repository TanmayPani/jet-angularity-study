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
        case SysVar.UNFOLDING_PRIOR_LIKE_DATA:
            # Non-closure (LIKE_DATA) closure test keeps its early-iteration choice.
            return 1
        # --- old: HERWIG7 returned 3 because the old closure run stopped at
        # iter 3. HERWIG7/PYTHIA8 are now real-data unfoldings and should use
        # the nominal iteration (override here only if a re-run is truncated). ---
        # case SysVar.UNFOLDING_PRIOR_HERWIG7:
        #     return 3
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


def _stat_sigma(
    h: dict[str, torch.Tensor], stat_key: str = "bin_count_std"
) -> torch.Tensor:
    sig = h.get(stat_key)
    if sig is None:
        sig = h["bin_count_err"]
    return sig


def _barlow_prune_unc(
    unc: torch.Tensor,
    h_nominal: dict[str, torch.Tensor],
    *h_sys_vars: dict[str, torch.Tensor],
    threshold: float = 1.0,
    stat_key: str = "bin_count_std",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the Barlow check to a per-source absolute uncertainty.

    σ²_uncorr = |mean(σ²_var) − σ²_nom|
    significance = |unc| / √σ²_uncorr
    Returns (pruned_unc, significance). Bins with significance < threshold
    are set to 0; bins where σ²_uncorr ≈ 0 get significance 0 and are pruned.
    """
    sigma2_nom = _stat_sigma(h_nominal, stat_key).square()
    sigma2_var = torch.stack(
        [_stat_sigma(h, stat_key).square() for h in h_sys_vars]
    ).mean(0)
    sigma2_uncorr = (sigma2_var - sigma2_nom).abs_()

    eps = torch.finfo(unc.dtype).eps
    denom = sigma2_uncorr.sqrt().clamp_min(eps)
    significance = (unc.abs() / denom).nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    significance[sigma2_uncorr <= 0.0] = 0.0
    pruned = torch.where(significance >= threshold, unc, torch.zeros_like(unc))
    return pruned, significance


def calculate_sys_uncertainty(
    hist_nominal: dict[str, torch.Tensor],
    var_name: str,
    hist_file_name: str,
    *sys_var_names: str,
    path_prefix: str = "",
    feature_mode: str,
    is_equal_pref: bool = False,
    barlow_threshold: float | None = 1.0,
) -> dict[str, torch.Tensor]:
    sys_var_paths = [
        os.path.join(path_prefix, sys_var, feature_mode, var_name, hist_file_name)
        for sys_var in sys_var_names
    ]

    sys_var_hists = [torch.load(file_path) for file_path in sys_var_paths]

    if is_equal_pref:
        # Active analysis never sets is_equal_pref=True; keep returning the
        # legacy std so the helper is still callable but skip Barlow.
        _, raw = _unc_equal_preferred_sys_var(hist_nominal, *sys_var_hists)
    else:
        raw = _unc_less_preferred_sys_var(hist_nominal, *sys_var_hists)

    if barlow_threshold is None or is_equal_pref:
        return {"raw": raw, "pruned": raw, "barlow_sig": torch.zeros_like(raw)}

    pruned, sig = _barlow_prune_unc(
        raw, hist_nominal, *sys_var_hists, threshold=barlow_threshold
    )
    return {"raw": raw, "pruned": pruned, "barlow_sig": sig}


def calculate_prior_uncertainty(
    hist_nominal: dict[str, torch.Tensor],
    var_name: str,
    hist_file_name: str,
    path_prefix: str,
    feature_mode: str,
    prior_sysvar: SysVar = SysVar.UNFOLDING_PRIOR_LIKE_DATA,
    barlow_threshold: float | None = 1.0,
) -> dict[str, torch.Tensor]:
    ratio_path = os.path.join(
        path_prefix,
        str(prior_sysvar),
        feature_mode,
        var_name,
        "ratio",
        hist_file_name,
    )
    hist_ratio = torch.load(ratio_path, mmap=True)
    raw = _unc_prior_var(hist_nominal, hist_ratio)

    if barlow_threshold is None:
        return {"raw": raw, "pruned": raw, "barlow_sig": torch.zeros_like(raw)}

    # Barlow for priors uses the variation's own unfolded hist for σ²_var.
    # Prior sysvars share most events with nominal, so |σ²_var − σ²_nom| is a
    # weaker uncorrelated-stat proxy than for detector vars — treat the
    # pruning as a sanity check rather than a strict statistical test.
    unfolded_path = os.path.join(
        path_prefix,
        str(prior_sysvar),
        feature_mode,
        var_name,
        "unfolded",
        hist_file_name,
    )
    if not os.path.exists(unfolded_path):
        return {"raw": raw, "pruned": raw, "barlow_sig": torch.zeros_like(raw)}

    hist_unfolded = torch.load(unfolded_path, mmap=True)
    pruned, sig = _barlow_prune_unc(
        raw, hist_nominal, hist_unfolded, threshold=barlow_threshold
    )
    return {"raw": raw, "pruned": pruned, "barlow_sig": sig}


def calculate_prior_uncertainty_combined(
    hist_nominal: dict[str, torch.Tensor],
    var_name: str,
    hist_file_name: str,
    path_prefix: str,
    feature_mode: str,
    prior_sysvars: tuple[SysVar, ...] = (
        SysVar.UNFOLDING_PRIOR_LIKE_DATA,
        SysVar.UNFOLDING_PRIOR_HERWIG7,
    ),
    barlow_threshold: float | None = 1.0,
) -> dict[str, torch.Tensor]:
    """Single *equal-status* prior systematic from all prior variations.

    Each prior contributes a pseudo-varied spectrum ``nominal / closure_ratio``
    (``closure_ratio`` = its own unfolded/truth). The priors are treated as one
    uncertainty of equal status: ``unc = |nominal − mean_p(pseudo_p)|`` (a
    priors-only mean, deviation measured vs nominal — nominal is NOT a member),
    replacing the old per-prior quadrature sum. Priors whose closure ratio is
    absent on disk are skipped; if none exist the source is zero.
    """
    pseudo_hists: list[dict[str, torch.Tensor]] = []
    unfolded_hists: list[dict[str, torch.Tensor]] = []
    for prior_sysvar in prior_sysvars:
        ratio_path = os.path.join(
            path_prefix, str(prior_sysvar), feature_mode, var_name, "ratio",
            hist_file_name,
        )
        if not os.path.exists(ratio_path):
            continue
        hist_ratio = torch.load(ratio_path, mmap=True)
        pseudo_hists.append(
            {"bin_count": hist_nominal["bin_count"] / hist_ratio["bin_count"]}
        )
        # Barlow σ²_var proxy uses the prior's own unfolded hist (shared-event,
        # so this is a sanity check rather than a strict statistical test).
        unfolded_path = os.path.join(
            path_prefix, str(prior_sysvar), feature_mode, var_name, "unfolded",
            hist_file_name,
        )
        if os.path.exists(unfolded_path):
            unfolded_hists.append(torch.load(unfolded_path, mmap=True))

    if not pseudo_hists:
        zero = torch.zeros_like(hist_nominal["bin_count"])
        return {"raw": zero, "pruned": zero, "barlow_sig": zero}

    # |nominal − mean(pseudo_p)|, the same "less preferred" form as detector
    # sources but over the prior pseudo-spectra.
    raw = _unc_less_preferred_sys_var(hist_nominal, *pseudo_hists)

    if barlow_threshold is None or not unfolded_hists:
        return {"raw": raw, "pruned": raw, "barlow_sig": torch.zeros_like(raw)}

    pruned, sig = _barlow_prune_unc(
        raw, hist_nominal, *unfolded_hists, threshold=barlow_threshold
    )
    return {"raw": raw, "pruned": pruned, "barlow_sig": sig}


# Display/diagnostic keys: every per-source uncertainty drawn in the QA plots and
# tabulated individually. The four unfolding-group components (iter, JER, the two
# priors) are kept for diagnostics AND collapsed into the `unfolding_sys` envelope
# (max), which is what actually feeds the quadrature total (see TOTAL_KEYS).
SOURCE_KEYS = (
    "unf_iter_sys",
    "jet_pt_res_sys",
    "track_pt_sys",
    "tower_et_corr_sys",
    # --- old: two independent prior sources summed in quadrature ---
    # "unf_prior_like_data",
    # "unf_prior_herwig7",
    # --- older: single equal-status combined prior source (mean of priors) ---
    # "unf_prior",
    # --- old: all three priors treated as closure-ratio envelope members ---
    # "unf_prior_like_data",
    # "unf_prior_herwig7",
    # --- new: HERWIG7/PYTHIA8 are model-dependence variations (|nominal - alt|,
    # alternate unfolded real data); LIKE_DATA is the method non-closure. All
    # collapse into the single unfolding envelope. ---
    "unf_prior_herwig7",
    "unf_prior_pythia8",
    "unf_nonclosure",
    "unfolding_sys",
)

# Sources that actually enter the quadrature `total_sys`. The four unfolding-group
# components are NOT here individually — they are enveloped (per-bin max) into the
# single `unfolding_sys` source, which stands in for all of them.
TOTAL_KEYS = (
    "unfolding_sys",
    "track_pt_sys",
    "tower_et_corr_sys",
)

# The non-closure source shares most events with the nominal (it is a closure
# test on the reweighted sim), so the Barlow check's uncorrelated-stat proxy
# |σ²_var − σ²_nom| is weak — it gets its own (typically different) significance
# threshold from the detector vars. The model priors (HERWIG7/PYTHIA8) now
# unfold real data and are treated like detector sources (default threshold).
PRIOR_SOURCE_KEYS = ("unf_nonclosure",)
# --- old: ("unf_prior_like_data", "unf_prior_herwig7") ---


def _is_source_key(k: str) -> bool:
    return k in SOURCE_KEYS


def calculate_uncertainties(
    var_name: str,
    path_prefix: str,
    feature_mode: str,
    barlow_threshold: float | None = 1.0,
    prior_barlow_threshold: float | None = None,
):
    # Shared-event (prior) sources use their own threshold; default to the
    # detector one when unset so old callers behave as before.
    if prior_barlow_threshold is None:
        prior_barlow_threshold = barlow_threshold

    nominal_path = os.path.join(path_prefix, "nominal", feature_mode, var_name)
    sys_vars = defaultdict(dict)
    hist_nominal = {}

    detector_sources = {
        "unf_iter_sys": ("unf_iter_sys_0", "unf_iter_sys_1"),
        "jet_pt_res_sys": ("jet_pt_res_sys_0", "jet_pt_res_sys_1"),
        "track_pt_sys": ("track_pt_sys",),
        "tower_et_corr_sys": ("tower_et_corr_sys",),
    }

    for file_name in os.listdir(nominal_path):
        hist_name = file_name.removesuffix(".pt")
        hist_nominal[hist_name] = torch.load(
            os.path.join(nominal_path, file_name),
            mmap=True,
        )
        h_nom = hist_nominal[hist_name]

        for src_key, var_names in detector_sources.items():
            res = calculate_sys_uncertainty(
                h_nom,
                var_name,
                file_name,
                *var_names,
                path_prefix=path_prefix,
                feature_mode=feature_mode,
                barlow_threshold=barlow_threshold,
            )
            sys_vars[hist_name][src_key] = res["pruned"]
            sys_vars[hist_name][f"{src_key}_raw"] = res["raw"]
            sys_vars[hist_name][f"{src_key}_barlow_sig"] = res["barlow_sig"]

        # --- old: per-prior quadrature sources (LIKE-DATA + HERWIG7 separate) ---
        # prior_sources = [("unf_prior_like_data", SysVar.UNFOLDING_PRIOR_LIKE_DATA)]
        # ...
        # --- older: single equal-status combined prior source (mean of priors) ---
        # res = calculate_prior_uncertainty_combined(...)
        # --- old: ALL priors treated as closure-ratio sources via
        # calculate_prior_uncertainty (nominal/ratio), incl. HERWIG7 ---
        # prior_sources = (
        #     ("unf_prior_like_data", SysVar.UNFOLDING_PRIOR_LIKE_DATA),
        #     ("unf_prior_herwig7", SysVar.UNFOLDING_PRIOR_HERWIG7),
        # )
        # for src_key, prior_sysvar in prior_sources: ... calculate_prior_uncertainty(...)
        #
        # --- new: model-dependence priors (HERWIG7/PYTHIA8) unfold REAL data
        # with their reweighted embedding as prior, so they are detector-style
        # "less preferred" variations |nominal - alt_unfolded| read from the
        # top-level snapshot (NOT the closure-ratio trick). Guard on the
        # snapshot existing (PYTHIA8 may be absent). ---
        model_prior_sources = {
            "unf_prior_herwig7": (str(SysVar.UNFOLDING_PRIOR_HERWIG7),),
            "unf_prior_pythia8": (str(SysVar.UNFOLDING_PRIOR_PYTHIA8),),
        }
        for src_key, var_names in model_prior_sources.items():
            snap_path = os.path.join(
                path_prefix, var_names[0], feature_mode, var_name, file_name
            )
            if not os.path.exists(snap_path):
                continue
            res = calculate_sys_uncertainty(
                h_nom,
                var_name,
                file_name,
                *var_names,
                path_prefix=path_prefix,
                feature_mode=feature_mode,
                barlow_threshold=barlow_threshold,
            )
            sys_vars[hist_name][src_key] = res["pruned"]
            sys_vars[hist_name][f"{src_key}_raw"] = res["raw"]
            sys_vars[hist_name][f"{src_key}_barlow_sig"] = res["barlow_sig"]

        # --- new: LIKE_DATA is now the method NON-CLOSURE source. It stays a
        # closure test (reweighted reco pseudo-data vs nominal sim); the
        # uncertainty form is unchanged (nominal x |1/ratio - 1| via
        # calculate_prior_uncertainty), only relabeled `unf_nonclosure`. Guard
        # on the closure ratio existing on disk. ---
        _nc_ratio_path = os.path.join(
            path_prefix, str(SysVar.UNFOLDING_PRIOR_LIKE_DATA), feature_mode,
            var_name, "ratio", file_name,
        )
        if os.path.exists(_nc_ratio_path):
            res = calculate_prior_uncertainty(
                h_nom,
                var_name,
                file_name,
                path_prefix=path_prefix,
                feature_mode=feature_mode,
                prior_sysvar=SysVar.UNFOLDING_PRIOR_LIKE_DATA,
                barlow_threshold=prior_barlow_threshold,
            )
            sys_vars[hist_name]["unf_nonclosure"] = res["pruned"]
            sys_vars[hist_name]["unf_nonclosure_raw"] = res["raw"]
            sys_vars[hist_name]["unf_nonclosure_barlow_sig"] = res["barlow_sig"]

        # --- unfolding envelope: per-bin MAX over the raw unfolding-group
        # components (iter, JER, model-prior HERWIG7/PYTHIA8, non-closure). Raw,
        # no Barlow gate within the group (user choice). Absent members are
        # simply excluded from the max. This single source stands in for all of
        # them in the total. ---
        _env_members = [
            sys_vars[hist_name]["unf_iter_sys_raw"],
            sys_vars[hist_name]["jet_pt_res_sys_raw"],
        ]
        for _pk in (
            "unf_prior_herwig7_raw",
            "unf_prior_pythia8_raw",
            "unf_nonclosure_raw",
        ):
            if _pk in sys_vars[hist_name]:
                _env_members.append(sys_vars[hist_name][_pk])
        unfolding_sys = torch.stack(_env_members).amax(0)
        sys_vars[hist_name]["unfolding_sys"] = unfolding_sys
        # raw == pruned for the envelope (it is never Barlow-pruned)
        sys_vars[hist_name]["unfolding_sys_raw"] = unfolding_sys

        zero = torch.zeros_like(h_nom["bin_count"])

        # Quadrature over TOTAL_KEYS only (envelope + the two standalone detector
        # sources); the per-component unfolding sources live in the dict for
        # diagnostics but must NOT be summed in again (that would double-count).
        sq_sum_pruned = zero.clone()
        sq_sum_raw = zero.clone()
        for tk in TOTAL_KEYS:
            v = sys_vars[hist_name][tk]
            sq_sum_pruned.add_(v.square())
            v_raw = sys_vars[hist_name].get(f"{tk}_raw", v)
            sq_sum_raw.add_(v_raw.square())
        sys_vars[hist_name]["total_sys"] = sq_sum_pruned.sqrt_()
        sys_vars[hist_name]["total_sys_raw"] = sq_sum_raw.sqrt_()
        # sys_vars[hist_name]["bootstrap_stat"] = h_nom["bin_count_std"]
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


def plot_uncertainties(
    var_name,
    hnominal,
    sys_vars,
    fig_save_dir,
    barlow_threshold=None,
    prior_barlow_threshold=None,
):
    import re

    if prior_barlow_threshold is None:
        prior_barlow_threshold = barlow_threshold

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
        # Canonical source keys: skip totals, *_raw, *_barlow_sig.
        sv_keys = [k for k in sys_vars[first_hname].keys() if _is_source_key(k)]

        fig = plt.figure(figsize=(num_cols * fig_scale, fig_scale + 1))
        axs = fig.subplots(
            1,
            num_cols,
            sharey=True,
            gridspec_kw={"wspace": 0, "right": 0.9, "left": 0.2},
        )
        if num_cols == 1:
            axs = [axs]
        axs[0].set_ylabel(ylabel, labelpad=10, size="xx-large")

        # Per-source single figure: 2 rows — top: pre/post Barlow rel. unc, bottom: significance.
        single_figs, single_axs_top, single_axs_bot = {}, {}, {}
        for sv in sv_keys:
            single_figs[sv] = plt.figure(
                figsize=(num_cols * fig_scale, fig_scale * 1.5)
            )
            ax_grid = single_figs[sv].subplots(
                2,
                num_cols,
                sharey="row",
                sharex="col",
                height_ratios=[2, 1],
                gridspec_kw={"hspace": 0, "wspace": 0, "right": 0.9, "left": 0.2},
            )
            if num_cols == 1:
                ax_grid = ax_grid.reshape(2, 1)
            ax_grid[0, 0].set_ylabel(ylabel, labelpad=10, size="xx-large")
            ax_grid[1, 0].set_ylabel("Barlow sig.", labelpad=10, size="x-large")
            single_axs_top[sv] = ax_grid[0]
            single_axs_bot[sv] = ax_grid[1]

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
            total_rel_raw = _safe_rel_unc(
                sys_vars[hist_name].get(
                    "total_sys_raw", sys_vars[hist_name]["total_sys"]
                ),
                y,
            )

            ax = axs[ijpt]
            ax.text(
                0.15,
                0.55,
                rf"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$",
                transform=ax.transAxes,
            )
            zero_base = torch.zeros_like(y).numpy()
            ax.stairs(
                rel_stat,
                bins,
                baseline=zero_base,
                fill=True,
                color="magenta",
                alpha=0.3,
                label="stat. (bootstrap)",
            )
            ax.stairs(
                total_rel_raw,
                bins,
                label="total sys (raw)",
                linewidth=1.5,
                linestyle="dashed",
                color="grey",
            )
            ax.stairs(
                total_rel,
                bins,
                label="total sys (Barlow)",
                linewidth=2.0,
                linestyle="dotted",
                color="black",
            )

            for sv in sv_keys:
                rel = _safe_rel_unc(sys_vars[hist_name][sv], y)
                rel_raw = _safe_rel_unc(
                    sys_vars[hist_name].get(f"{sv}_raw", sys_vars[hist_name][sv]), y
                )
                bsig = sys_vars[hist_name].get(f"{sv}_barlow_sig")

                line = ax.stairs(rel, bins, label=sv, linewidth=2.0)
                ax.stairs(
                    rel_raw,
                    bins,
                    linewidth=1.0,
                    linestyle="dashed",
                    color=line.get_edgecolor(),
                    alpha=0.6,
                )

                sax = single_axs_top[sv][ijpt]
                sax.stairs(
                    rel_stat,
                    bins,
                    baseline=zero_base,
                    fill=True,
                    color="magenta",
                    alpha=0.3,
                    label="stat. (bootstrap)",
                )
                sax.stairs(
                    total_rel_raw,
                    bins,
                    label="total sys (raw)",
                    linewidth=1.5,
                    linestyle="dashed",
                    color="grey",
                )
                sax.stairs(
                    total_rel,
                    bins,
                    label="total sys (Barlow)",
                    linewidth=2.0,
                    linestyle="dotted",
                    color="black",
                )
                sline = sax.stairs(rel, bins, label=sv, linewidth=2.0)
                sax.stairs(
                    rel_raw,
                    bins,
                    linewidth=1.0,
                    linestyle="dashed",
                    color=sline.get_edgecolor(),
                    alpha=0.6,
                    label=f"{sv} (raw)",
                )
                sax.text(
                    0.2,
                    0.75,
                    rf"${jpt_min} < p_{{T, jet}} < {jpt_max}$ GeV/$c$",
                    transform=sax.transAxes,
                )

                if bsig is not None:
                    bax = single_axs_bot[sv][ijpt]
                    bsig_clean = bsig.clone().nan_to_num_(
                        nan=0.0, posinf=0.0, neginf=0.0
                    )
                    bax.stairs(
                        bsig_clean, bins, linewidth=2.0, color=sline.get_edgecolor()
                    )
                    sv_threshold = (
                        prior_barlow_threshold
                        if sv in PRIOR_SOURCE_KEYS
                        else barlow_threshold
                    )
                    if sv_threshold is not None:
                        bax.axhline(
                            sv_threshold, color="black", linestyle="--", linewidth=1.0
                        )

        axs[-1].legend(fontsize="small")
        combined_path = os.path.join(fig_save_dir, f"{mod}-sysQA.pdf")
        print("Saving figure to:", combined_path)
        fig.savefig(combined_path, bbox_inches="tight")
        plt.close(fig)

        for sv in sv_keys:
            single_axs_top[sv][-1].legend(fontsize="small")
            sv_path = os.path.join(fig_save_dir, f"{mod}-{sv}-sysQA.pdf")
            single_figs[sv].savefig(sv_path, bbox_inches="tight")
            plt.close(single_figs[sv])


def main():
    import json

    path_prefix = "outputs/histograms"
    with open("runtime-files/config.json") as _cfg_file:
        _cfg = json.load(_cfg_file)
    feature_mode = _cfg["feature_mode"]
    barlow_threshold = _cfg.get("barlow_threshold", 1.0)
    # Shared-event (prior) systematics get their own threshold; fall back to the
    # detector one when the key is absent.
    prior_barlow_threshold = _cfg.get("barlow_threshold_prior", barlow_threshold)
    print(
        f"Barlow check: detector "
        f"{'disabled' if barlow_threshold is None else f'threshold = {barlow_threshold}'}"
        f", prior (shared-event) "
        f"{'disabled' if prior_barlow_threshold is None else f'threshold = {prior_barlow_threshold}'}"
    )

    for var_name in common_vars + angularities:
        sys_vars, hnominal = calculate_uncertainties(
            var_name,
            path_prefix=path_prefix,
            feature_mode=feature_mode,
            barlow_threshold=barlow_threshold,
            prior_barlow_threshold=prior_barlow_threshold,
        )
        save_dir = os.path.join(path_prefix, "sys_errors", feature_mode, var_name)
        save_sys_uncertainties(sys_vars, save_dir)
        fig_save_dir = os.path.join(path_prefix, "plots", feature_mode, var_name)
        plot_uncertainties(
            var_name,
            hnominal,
            sys_vars,
            fig_save_dir,
            barlow_threshold=barlow_threshold,
            prior_barlow_threshold=prior_barlow_threshold,
        )

    # Regenerate the LaTeX per-source uncertainty tables from the freshly saved
    # sys_errors files. Lazy import: make_sys_var_tables imports from this module.
    from make_sys_var_tables import generate as _make_sys_var_tables

    _make_sys_var_tables(
        feature_mode=feature_mode,
        threshold=barlow_threshold,
        prior_threshold=prior_barlow_threshold,
    )

    print("Done!")


if __name__ == "__main__":
    main()
    plt.show()
