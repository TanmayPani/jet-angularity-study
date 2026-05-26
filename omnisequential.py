import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    import os
    import json
    import copy
    from collections import defaultdict

    import numpy as np
    import pyarrow as pa
    from matplotlib import pyplot as plt

    import torch
    from scipy.stats import chisquare
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

    from thoda import Histogram, Profile, bayesian_blocks

    from systematics import SysVar

    with open("./runtime-files/config.json") as _cfg_file:
        _cfg_setup = json.load(_cfg_file)
    feature_mode = _cfg_setup["feature_mode"]

    GP_SEED = 0

    jet_columns = [
        "pt",
        "m",
        "nef",
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
        #    "leading_constit_pt",
        #    "subleading_constit_pt",
        "sd_pt",
        "sd_m",
        "sd_dR",
        "sd_symmetry",
        "sd_ch_ang_k1_b0.5",
        "sd_ch_ang_k1_b1",
        "sd_ch_ang_k1_b2",
        "sd_ch_ang_k2_b0",
    ]

    col_hist_args = {
        "pt": {
            "range": (None, 60.0),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "m": {
            "range": (None, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "nef": {
            "range": (0.01, 0.9),
            "chi2_left_edge": 1,
            "chi2_right_edge": None,
        },
        "ch_ang_k1_b0.5": {
            "range": (None, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "ch_ang_k1_b1": {
            "range": (None, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "ch_ang_k1_b2": {
            "range": (None, None),
            "chi2_left_edge": 1,
            "chi2_right_edge": None,
        },
        "ch_ang_k2_b0": {
            "range": (0.01, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "leading_constit_pt": {
            "range": (None, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "subleading_constit_pt": {
            "range": (None, 12.5),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "sd_pt": {
            "range": (5.0, 50.0),
            "chi2_left_edge": 1,
            "chi2_right_edge": None,
        },
        "sd_m": {
            "range": (0.1405, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "sd_dR": {
            "range": (0.06, None),
            "chi2_left_edge": 1,
            "chi2_right_edge": None,
        },
        "sd_symmetry": {
            "range": (0, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "sd_ch_ang_k1_b0.5": {
            "range": (0.001, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "sd_ch_ang_k1_b1": {
            "range": (0.025, None),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
        "sd_ch_ang_k1_b2": {
            "range": (0.006, None),
            "chi2_left_edge": 1,
            "chi2_right_edge": None,
        },
        "sd_ch_ang_k2_b0": {
            "range": (0.01, 0.8),
            "chi2_left_edge": 0,
            "chi2_right_edge": None,
        },
    }


@app.function
def weight_summary(w):
    w = w.detach().cpu().to(torch.float64)
    s = float(w.sum())
    return {
        "mean": float(w.mean()),
        "std": float(w.std()),
        "min": float(w.min()),
        "max": float(w.max()),
        "median": float(w.median()),
        "neg": int((w < 0).sum()),
        "ess": (s * s) / float((w * w).sum()) if s > 0 else float("nan"),
    }


@app.function
def propagate_values(
    x1, y1, x2, estimator=None, num_prediction_batches=None, y1_err=None
):
    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # Min-max normalise X to [0, 1] using x1's range so length_scale is a
    # fraction of the data range — interpretable across variables and
    # well-conditioned for the L-BFGS hyperparameter optimiser.
    _x_min = float(x1.min())
    _x_max = float(x1.max())
    _x_span = max(_x_max - _x_min, 1e-12)
    x1 = (x1 - _x_min) / _x_span
    x2 = (x2 - _x_min) / _x_span

    if estimator is None:
        if y1_err is None:
            _alpha = 1e-10
        else:
            _alpha = np.asarray(y1_err, dtype=np.float64) ** 2
            _alpha = np.clip(_alpha, 1e-10, None)
        # WhiteKernel soaks up residual noise so the RBF length scale
        # doesn't collapse to fit per-bin scatter (previous run hit
        # length_scale → 1e-5 bound + L-BFGS ABNORMAL termination).
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=0.3, length_scale_bounds=(1e-3, 1e1)
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
        estimator = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=10,
            copy_X_train=False,
            random_state=GP_SEED,
            alpha=_alpha,
        )

    estimator.fit(x1, y1)

    if num_prediction_batches is None:
        return estimator.predict(x2)
    else:
        predictions = []
        for ibatch, batch in enumerate(np.array_split(x2, num_prediction_batches)):
            if ibatch > 0 and ibatch % 10 == 0:
                print(f"------Predicting for batch [{ibatch}/{num_prediction_batches}]")
            predictions.append(estimator.predict(batch))
        return np.concatenate(predictions)


@app.function
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


@app.function
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
        # print(iden, label, marker)
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
            # print(inum, iden, label1, label2, marker1, marker2)
            if ratio is None:
                ratio = count1 / count2
                if error1 is not None and error2 is not None:
                    ratio_err = ratio * np.sqrt(
                        (error1 / count1) ** 2 + (error2 / count2) ** 2
                    )
            # print(ratio, ratio_err)
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


@app.cell
def _():
    _do_use_gen_misses: bool = True
    _source_dir = os.path.join(
        "./datasets/STAR_pp200GeV_production_2012/features", feature_mode
    )
    _emb_dir = os.path.join(_source_dir, "embedding", str(SysVar.NONE))

    _gen_match_buffer = pa.memory_map(os.path.join(_emb_dir, "gen-matches.arrow"), "rb")
    gen_match_table = pa.ipc.open_file(_gen_match_buffer).read_all()

    _gen_miss_buffer = pa.memory_map(os.path.join(_emb_dir, "misses.arrow"), "rb")
    gen_miss_table = pa.ipc.open_file(_gen_miss_buffer).read_all()

    _reco_match_buffer = pa.memory_map(
        os.path.join(_emb_dir, "reco-matches.arrow"), "rb"
    )
    reco_match_table = pa.ipc.open_file(_reco_match_buffer).read_all()

    _reco_fake_buffer = pa.memory_map(os.path.join(_emb_dir, "fakes.arrow"), "rb")
    reco_fake_table = pa.ipc.open_file(_reco_fake_buffer).read_all()

    _data_buffer = pa.memory_map(os.path.join(_source_dir, "data.arrow"), "rb")
    data_table = pa.ipc.open_file(_data_buffer).read_all()

    if _do_use_gen_misses:
        gen_table = pa.concat_tables([gen_match_table, gen_miss_table])
    else:
        gen_table = gen_match_table

    reco_table = pa.concat_tables([reco_match_table, reco_fake_table])

    n_data = len(data_table)
    n_gen_matches = len(gen_match_table)
    n_gen_misses = len(gen_miss_table)
    n_reco_matches = len(reco_match_table)
    n_reco_fakes = len(reco_fake_table)

    assert n_gen_matches == n_reco_matches

    n_matches = n_gen_matches
    n_gen = n_matches + n_gen_misses
    n_reco = n_matches + n_reco_fakes
    print(
        "Number of matched gen jets, matched reco jets:",
        n_gen_matches,
        n_reco_matches,
        n_matches,
    )
    print("Number of missed gen jets, fake reco jets:", n_gen_misses, n_reco_fakes)
    print("Number of data jets, gen jets, reco jets:", n_data, n_gen, n_reco)

    # Capture the pre-iteration P6 weight sums so the arrow-writing cell can
    # renormalise the post-loop weights back to the original per-side totals
    # (mirrors reverse_omnisequential.py:369-374).
    _w_gm_orig = gen_match_table["weight"].to_numpy().astype(np.float32)
    _w_mi_orig = gen_miss_table["weight"].to_numpy().astype(np.float32)
    _w_rm_orig = reco_match_table["weight"].to_numpy().astype(np.float32)
    _w_fk_orig = reco_fake_table["weight"].to_numpy().astype(np.float32)
    sum_w_gen_orig = float(_w_gm_orig.sum() + _w_mi_orig.sum())
    sum_w_reco_orig = float(_w_rm_orig.sum() + _w_fk_orig.sum())

    print(gen_table.column_names)
    return (
        data_table,
        gen_match_table,
        gen_miss_table,
        gen_table,
        n_data,
        n_gen_matches,
        n_reco_matches,
        reco_fake_table,
        reco_match_table,
        reco_table,
        sum_w_gen_orig,
        sum_w_reco_orig,
    )


@app.cell
def _(data_table):
    _p0 = 0.02
    _undersample = 55000
    _bin_file_path = f"./runtime-files/bins_p0{_p0:g}_N{_undersample:g}.json"
    _recalculate_bins_for: list[str] | None = None
    _do_diff_gen_bins: bool = False

    bins = {}

    try:
        with open(_bin_file_path, "rb") as _bin_file:
            bins = json.load(_bin_file)
        if _recalculate_bins_for is not None:
            for _key in _recalculate_bins_for:
                del bins[_key]
        print("Read bins from :", _bin_file_path)
    except Exception as _exc:
        print(_exc)
        bins = {}
        print("Binning will be saved to :", _bin_file_path)

    _bins_updated = False
    for _col in jet_columns:
        if _col not in bins:
            _bins_updated = True
            print("Calculating binning for", _col)
            bins[_col] = bayesian_blocks(
                torch.as_tensor(
                    data_table[_col].to_numpy(), dtype=torch.float64, device="cuda"
                ),
                p0=_p0,
                ranges=col_hist_args[_col]["range"],
                undersample=_undersample,
                device="cuda",
            )
            print(_col, ":", bins[_col], len(bins[_col]))

    if _bins_updated:
        with open(_bin_file_path, "w") as _bin_file:
            json.dump(
                {k: v.tolist() for k, v in bins.items()},
                _bin_file,
                indent=4,
            )
    return (bins,)


@app.cell
def _(bins, data_table, gen_table, n_data, reco_table):
    _bin_tensors = (bins[k] for k in jet_columns)

    data_hist = {}
    binned_data = {}
    data_hist_count = {}
    data_hist_count_err = {}

    reco_weights = torch.as_tensor(reco_table["weight"].to_numpy(), dtype=torch.float64)
    reco_weights = reco_weights.div_(reco_weights.sum()).mul_(float(n_data))
    reco_arr = {}
    reco_hist = {}
    binned_reco = {}

    gen_weights = torch.as_tensor(gen_table["weight"].to_numpy(), dtype=torch.float64)
    gen_weights = gen_weights.div_(gen_weights.sum()).mul_(float(n_data))
    gen_arr = {}
    gen_hist = {}
    binned_gen = {}

    axes = {}
    print(n_data, reco_weights.sum().item(), gen_weights.sum().item())

    for _col in jet_columns:
        _data_hist, binned_data[_col] = Histogram.create(
            (data_table[_col].to_numpy(),), bins=(bins[_col],), return_binned_data=True
        )
        data_hist[_col] = _data_hist.snapshot()
        data_hist_count[_col] = data_hist[_col].values[1:-1]
        data_hist_count_err[_col] = data_hist[_col].variances.sqrt_()[1:-1]

        gen_arr[_col] = torch.as_tensor(gen_table[_col].to_numpy(), dtype=torch.float64)

        _gen_hist, binned_gen[_col] = Histogram.create(
            (gen_arr[_col],),
            bins=(bins[_col],),
            weights=gen_weights,
            return_binned_data=True,
        )

        reco_arr[_col] = torch.as_tensor(
            reco_table[_col].to_numpy(), dtype=torch.float64
        )

        _reco_hist, binned_reco[_col] = Histogram.create(
            (reco_arr[_col],),
            bins=(bins[_col],),
            weights=reco_weights,
            return_binned_data=True,
        )
        gen_hist[_col] = _gen_hist.snapshot()
        reco_hist[_col] = _reco_hist.snapshot()
        axes[_col] = _data_hist.axes
        del _data_hist, _reco_hist, _gen_hist

    reco_pth_bin = torch.as_tensor(reco_table["pth_bin"].to_numpy(), dtype=torch.int64)
    gen_pth_bin = torch.as_tensor(gen_table["pth_bin"].to_numpy(), dtype=torch.int64)
    return (
        axes,
        binned_gen,
        binned_reco,
        data_hist_count,
        data_hist_count_err,
        gen_arr,
        gen_hist,
        gen_weights,
        reco_arr,
        reco_hist,
        reco_pth_bin,
        reco_weights,
    )


@app.cell
def _(
    axes,
    binned_gen,
    binned_reco,
    data_hist_count,
    data_hist_count_err,
    gen_arr,
    gen_hist,
    gen_weights,
    n_data,
    n_gen_matches,
    reco_arr,
    reco_hist,
    reco_pth_bin,
    reco_weights,
):
    n_iterations: int = 10
    _damping_alpha: float = 0.5  # 0.5 = sqrt-damping; 1.0 = no damping

    iter_hist_list = []

    hists = defaultdict(dict)

    gen_match_reweight_profile = {}

    w_unfolding = [reco_weights, gen_weights]

    max_chi2_var = ""

    num_data_samples = torch.as_tensor(float(n_data), dtype=torch.float32)

    iter_diag = {
        "chi2": [],
        "max_chi2_var": [],
        "max_chi2": [],
        "reco_weight_stats": [],
        "gen_weight_stats": [],
        "reco_reweight_stats": [],
        "gen_reweight_stats": [],
        "gen_profile_nan_frac": [],
        "gp_reco_train_xy": [],
        "gp_gen_train_xy": [],
        "reco_topk": [],
        "reco_reweight_quantiles": [],
        "gen_reweight_pre_clamp_min": [],
        "gen_reweight_n_clamped": [],
        "reco_reweight_pre_clamp_min": [],
        "reco_reweight_n_clamped": [],
    }

    iter_diag["reco_weight_stats"].append(weight_summary(reco_weights))
    iter_diag["gen_weight_stats"].append(weight_summary(gen_weights))

    _traj_gen = torch.Generator().manual_seed(0)
    _n_traj = 2000
    _sample_reco_idx = torch.randperm(len(reco_weights), generator=_traj_gen)[:_n_traj]
    _sample_gen_idx = torch.randperm(len(gen_weights), generator=_traj_gen)[:_n_traj]
    _reco_traj = [reco_weights[_sample_reco_idx].detach().cpu().clone()]
    _gen_traj = [gen_weights[_sample_gen_idx].detach().cpu().clone()]

    for iteration in range(n_iterations):
        max_chi2 = 0
        max_chi2_var = ""
        print(f"Iteration: {iteration}")
        for _col in jet_columns:
            hists[_col]["data"] = data_hist_count[_col]
            hists[_col]["data_err"] = data_hist_count_err[_col]
            # print(reco_weights.shape, gen_weights.shape, reco_hist[_col].x.shape, gen_arr[_col].shape)

            del reco_hist[_col]
            reco_hist[_col] = Histogram.from_binned(
                axes[_col], binned_reco[_col], weights=reco_weights
            ).snapshot()
            hists[_col]["reco"] = reco_hist[_col].values[1:-1]
            hists[_col]["reco_err"] = reco_hist[_col].variances.sqrt_()[1:-1]

            del gen_hist[_col]
            gen_hist[_col] = Histogram.from_binned(
                axes[_col], binned_gen[_col], weights=gen_weights
            ).snapshot()
            hists[_col]["gen"] = gen_hist[_col].values[1:-1]
            hists[_col]["gen_err"] = gen_hist[_col].variances.sqrt_()[1:-1]

            hists[_col]["ratio"] = hists[_col]["data"] / hists[_col]["reco"]
            hists[_col]["ratio_err"] = (
                (
                    hists[_col]["data_err"].div(hists[_col]["data"]).pow_(2)
                    + hists[_col]["reco_err"].div(hists[_col]["reco"]).pow_(2)
                )
                .sqrt_()
                .mul_(hists[_col]["ratio"])
            )

            left_edge = col_hist_args[_col]["chi2_left_edge"]
            right_edge = col_hist_args[_col]["chi2_right_edge"]
            chi2_slice = slice(left_edge, right_edge)
            hists[_col]["chi2_slice"] = chi2_slice
            h_data = hists[_col]["data"][chi2_slice]
            h_reco = hists[_col]["reco"][chi2_slice]
            h_reco = h_reco.div(h_reco.sum()).mul_(h_data.sum())
            chi2, _ = chisquare(h_reco, h_data)
            hists[_col]["chi2"] = chi2

            if chi2 > max_chi2:
                max_chi2 = chi2
                max_chi2_var = _col
        iter_diag["chi2"].append({c: float(hists[c]["chi2"]) for c in jet_columns})
        iter_diag["max_chi2_var"].append(max_chi2_var)
        iter_diag["max_chi2"].append(float(max_chi2))
        iter_hist_list.append(copy.deepcopy(hists))
        if max_chi2 < 0.0:
            break

        print(f"---Max chi2 = {max_chi2} from {max_chi2_var}")
        features = reco_hist[max_chi2_var].bin_centers[1:-1]
        nbins = len(features)
        print("---Obtaining reco lvl reweight factors...")
        reco_reweights = torch.as_tensor(
            propagate_values(
                features[hists[max_chi2_var]["chi2_slice"]],
                hists[max_chi2_var]["ratio"][hists[max_chi2_var]["chi2_slice"]],
                # features, hists[max_chi2_var]["ratio"],
                reco_arr[max_chi2_var],
                num_prediction_batches=40,
                y1_err=hists[max_chi2_var]["ratio_err"][
                    hists[max_chi2_var]["chi2_slice"]
                ]
                .detach()
                .cpu()
                .numpy(),
            ),
            dtype=torch.float32,
        )
        _reco_pre_clamp_min = float(reco_reweights.min())
        _reco_n_clamped = int((reco_reweights < 0).sum())
        reco_reweights.clamp_min_(0.0)
        reco_reweights.pow_(_damping_alpha)
        iter_diag["reco_reweight_pre_clamp_min"].append(_reco_pre_clamp_min)
        iter_diag["reco_reweight_n_clamped"].append(_reco_n_clamped)
        iter_diag["reco_reweight_stats"].append(weight_summary(reco_reweights))
        iter_diag["gp_reco_train_xy"].append(
            (
                features[hists[max_chi2_var]["chi2_slice"]]
                .detach()
                .cpu()
                .numpy()
                .copy(),
                hists[max_chi2_var]["ratio"][hists[max_chi2_var]["chi2_slice"]]
                .detach()
                .cpu()
                .numpy()
                .copy(),
            )
        )

        _K = 1000
        _rw_np = reco_reweights.detach().cpu().numpy()
        _pre_np = reco_weights.detach().cpu().numpy()
        _topk_part = np.argpartition(-_rw_np, _K)[:_K]
        _topk_idx = _topk_part[np.argsort(-_rw_np[_topk_part])]
        iter_diag["reco_topk"].append(
            {
                "indices": _topk_idx.astype(np.int64),
                "reweights": _rw_np[_topk_idx].astype(np.float64),
                "pre_weights": _pre_np[_topk_idx].astype(np.float64),
                "winner_var": max_chi2_var,
                "winner_values": reco_arr[max_chi2_var][_topk_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                "pth_bin": reco_pth_bin[_topk_idx].numpy().astype(np.int64),
            }
        )
        iter_diag["reco_reweight_quantiles"].append(
            np.quantile(_rw_np, [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0])
        )

        reco_weights.mul_(reco_reweights)
        reco_weights.div_(reco_weights.sum()).mul_(num_data_samples)

        print("---Obtaining gen lvl reweight factors...")
        # bins = hists[max_chi2_var]["gen_bins"]
        gen_features = gen_hist[max_chi2_var].bin_centers[1:-1]
        nbins = len(gen_features)

        if max_chi2_var in gen_match_reweight_profile:
            del gen_match_reweight_profile[max_chi2_var]
        gen_match_reweight_profile[max_chi2_var] = Profile.from_binned(
            axes[max_chi2_var],
            binned_gen[max_chi2_var][:n_gen_matches],
            reco_reweights[:n_gen_matches],
            weights=gen_weights[:n_gen_matches],
        ).snapshot()
        gen_reweight_profile = gen_match_reweight_profile[max_chi2_var].values[1:-1]
        gen_reweight_variance = (
            gen_match_reweight_profile[max_chi2_var].variances[1:-1].clamp_min(0)
        )
        gen_effective_counts = gen_match_reweight_profile[
            max_chi2_var
        ].effective_counts[1:-1]
        selection = torch.isnan(gen_reweight_profile).logical_not_()
        gen_features = gen_features[selection]
        print(f"---{len(gen_features)} good bins out of {nbins} gen bins...")
        gen_reweight_profile = gen_reweight_profile[selection]
        gen_reweight_variance = gen_reweight_variance[selection]
        gen_effective_counts = gen_effective_counts[selection]
        gen_y1_err = (
            (gen_reweight_variance / gen_effective_counts.clamp_min(1e-12))
            .sqrt_()
            .detach()
            .cpu()
            .numpy()
        )
        iter_diag["gen_profile_nan_frac"].append(
            1.0 - float(selection.sum()) / float(selection.numel())
        )
        iter_diag["gp_gen_train_xy"].append(
            (
                gen_features.detach().cpu().numpy().copy(),
                gen_reweight_profile.detach().cpu().numpy().copy(),
            )
        )
        gen_reweights = torch.as_tensor(
            propagate_values(
                gen_features,
                gen_reweight_profile,
                gen_arr[max_chi2_var],
                num_prediction_batches=40,
                y1_err=gen_y1_err,
            ),
            dtype=torch.float32,
        )
        _gen_pre_clamp_min = float(gen_reweights.min())
        _gen_n_clamped = int((gen_reweights < 0).sum())
        gen_reweights.clamp_min_(0.0)
        gen_reweights.pow_(_damping_alpha)
        iter_diag["gen_reweight_pre_clamp_min"].append(_gen_pre_clamp_min)
        iter_diag["gen_reweight_n_clamped"].append(_gen_n_clamped)
        iter_diag["gen_reweight_stats"].append(weight_summary(gen_reweights))

        gen_weights.mul_(gen_reweights)
        gen_weights.div_(gen_weights.sum()).mul_(num_data_samples)

        iter_diag["reco_weight_stats"].append(weight_summary(reco_weights))
        iter_diag["gen_weight_stats"].append(weight_summary(gen_weights))

        _reco_traj.append(reco_weights[_sample_reco_idx].detach().cpu().clone())
        _gen_traj.append(gen_weights[_sample_gen_idx].detach().cpu().clone())

        w_unfolding.extend([reco_weights, gen_weights])

    # Post-loop snapshot reflecting reco/gen weights AFTER the final
    # reweight, so iter_hist_list[-1] matches the on-disk weight state
    # downstream cells compare against.
    for _col in jet_columns:
        hists[_col]["data"] = data_hist_count[_col]
        hists[_col]["data_err"] = data_hist_count_err[_col]

        del reco_hist[_col]
        reco_hist[_col] = Histogram.from_binned(
            axes[_col], binned_reco[_col], weights=reco_weights
        ).snapshot()
        hists[_col]["reco"] = reco_hist[_col].values[1:-1]
        hists[_col]["reco_err"] = reco_hist[_col].variances.sqrt_()[1:-1]

        del gen_hist[_col]
        gen_hist[_col] = Histogram.from_binned(
            axes[_col], binned_gen[_col], weights=gen_weights
        ).snapshot()
        hists[_col]["gen"] = gen_hist[_col].values[1:-1]
        hists[_col]["gen_err"] = gen_hist[_col].variances.sqrt_()[1:-1]
    iter_hist_list.append(copy.deepcopy(hists))

    reco_traj_arr = torch.stack(_reco_traj).numpy()
    gen_traj_arr = torch.stack(_gen_traj).numpy()
    print("Iterations done...")

    output_folder = f"outputs/omnisequential/{feature_mode}"
    os.makedirs(output_folder, exist_ok=True)
    last_iteration = len(iter_hist_list)
    with open(f"{output_folder}/omniseq-wts-iter{last_iteration}.npz", "wb") as f:
        np.savez(f, *w_unfolding)
    with open(f"{output_folder}/omniseq-diag-iter{last_iteration}.npz", "wb") as f:
        np.savez(
            f,
            chi2=np.array(
                [[d.get(c, np.nan) for c in jet_columns] for d in iter_diag["chi2"]]
            ),
            jet_columns=np.array(jet_columns),
            max_chi2_var=np.array(iter_diag["max_chi2_var"]),
            max_chi2=np.array(iter_diag["max_chi2"]),
            ess_reco=np.array([s["ess"] for s in iter_diag["reco_weight_stats"]]),
            ess_gen=np.array([s["ess"] for s in iter_diag["gen_weight_stats"]]),
            max_over_median_reco=np.array(
                [s["max"] / s["median"] for s in iter_diag["reco_weight_stats"]]
            ),
            max_over_median_gen=np.array(
                [s["max"] / s["median"] for s in iter_diag["gen_weight_stats"]]
            ),
            reco_reweight_max=np.array(
                [s["max"] for s in iter_diag["reco_reweight_stats"]]
            ),
            gen_reweight_max=np.array(
                [s["max"] for s in iter_diag["gen_reweight_stats"]]
            ),
            gen_profile_nan_frac=np.array(iter_diag["gen_profile_nan_frac"]),
            reco_traj=reco_traj_arr,
            gen_traj=gen_traj_arr,
            gp_seed=np.int64(GP_SEED),
            reco_topk=np.array(iter_diag["reco_topk"], dtype=object),
            reco_reweight_quantiles=np.array(iter_diag["reco_reweight_quantiles"]),
            gen_reweight_pre_clamp_min=np.array(
                iter_diag["gen_reweight_pre_clamp_min"]
            ),
            gen_reweight_n_clamped=np.array(iter_diag["gen_reweight_n_clamped"]),
            reco_reweight_pre_clamp_min=np.array(
                iter_diag["reco_reweight_pre_clamp_min"]
            ),
            reco_reweight_n_clamped=np.array(iter_diag["reco_reweight_n_clamped"]),
        )

    # Expose the post-loop in-place-mutated weight tensors under explicit names
    # so the arrow-writing cell's marimo dependency on them fires after this
    # cell completes. (Marimo tracks assignments, not in-place mutations.)
    gen_weights_final = gen_weights
    reco_weights_final = reco_weights
    return (
        gen_traj_arr,
        gen_weights_final,
        iter_diag,
        iter_hist_list,
        last_iteration,
        reco_traj_arr,
        reco_weights_final,
    )


@app.cell
def _(
    gen_match_table,
    gen_miss_table,
    gen_weights_final,
    last_iteration,
    n_gen_matches,
    n_reco_matches,
    reco_fake_table,
    reco_match_table,
    reco_weights_final,
    sum_w_gen_orig,
    sum_w_reco_orig,
):
    from preprocessing import replace_table_column

    # Bake the post-loop reweighted P6 weights into the 4 nominal embedding
    # arrows so downstream code (preprocessing.make_datasets_for_unfolding,
    # plot_closure.py, histograms.py) can treat UNFOLDING_PRIOR_LIKE_DATA the
    # same way it treats UNFOLDING_PRIOR_HERWIG7 / _PYTHIA8. Renormalise each
    # side back to the original P6 weight total so the sample-weight scale on
    # disk is invariant under reweighting. Mirrors
    # reverse_omnisequential.py:944-1010.
    _out_dir = os.path.join(
        "./datasets/STAR_pp200GeV_production_2012/features",
        feature_mode,
        "embedding",
        str(SysVar.UNFOLDING_PRIOR_LIKE_DATA),
    )
    os.makedirs(_out_dir, exist_ok=True)

    _w_gen_final = gen_weights_final.detach().cpu().numpy().astype(np.float64)
    _w_reco_final = reco_weights_final.detach().cpu().numpy().astype(np.float64)
    _w_gen_out = (_w_gen_final * sum_w_gen_orig / _w_gen_final.sum()).astype(np.float32)
    _w_reco_out = (_w_reco_final * sum_w_reco_orig / _w_reco_final.sum()).astype(
        np.float32
    )

    _gm_out = replace_table_column(
        gen_match_table, "weight", _w_gen_out[:n_gen_matches]
    )
    _mi_out = replace_table_column(
        gen_miss_table, "weight", _w_gen_out[n_gen_matches:]
    )
    _rm_out = replace_table_column(
        reco_match_table, "weight", _w_reco_out[:n_reco_matches]
    )
    _fk_out = replace_table_column(
        reco_fake_table, "weight", _w_reco_out[n_reco_matches:]
    )

    for _name, _t in (
        ("gen-matches", _gm_out),
        ("misses", _mi_out),
        ("reco-matches", _rm_out),
        ("fakes", _fk_out),
    ):
        _path = os.path.join(_out_dir, f"{_name}.arrow")
        with pa.OSFile(_path, "wb") as _sink:
            with pa.ipc.new_file(_sink, _t.schema) as _writer:
                for _batch in _t.to_batches():
                    _writer.write_batch(_batch)
        print(f"  wrote {_path} ({len(_t)} rows)")

    print(
        f"[omniseq] last_iteration={last_iteration}; "
        f"4 arrows in {_out_dir}"
    )
    return


@app.cell
def _(bins, iter_hist_list, last_iteration):
    fig_scale = 6
    nrows = 3
    ncols = int(np.ceil(len(jet_columns) / nrows))

    fig = plt.figure(
        figsize=(ncols * fig_scale, nrows * fig_scale)
    )  # , layout="constrained")
    fig.suptitle(f"Iteration {last_iteration}", fontsize=30)
    subfigs = fig.subfigures(nrows, ncols)
    for ivar, var in enumerate(jet_columns):
        irow = int(np.floor(ivar / ncols))
        icol = ivar % ncols
        subfig = subfigs[irow, icol]
        h = iter_hist_list[-1][var]
        h0 = iter_hist_list[0][var]

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

    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig, _ax = plt.subplots(figsize=(10, 6))
    _chi2_table = iter_diag["chi2"]
    _n_iter = len(_chi2_table)
    _iters = np.arange(_n_iter)
    _cmap = plt.get_cmap("tab20")
    for _ivar, _var in enumerate(jet_columns):
        _vals = np.array([d.get(_var, np.nan) for d in _chi2_table])
        _color = _cmap(_ivar % 20)
        _ax.plot(_iters, _vals, marker=".", color=_color, label=_var, linewidth=1.2)
        _winner_mask = np.array(
            [iter_diag["max_chi2_var"][i] == _var for i in range(_n_iter)]
        )
        if _winner_mask.any():
            _ax.scatter(
                _iters[_winner_mask],
                _vals[_winner_mask],
                s=120,
                facecolors="none",
                edgecolors=_color,
                linewidths=2,
            )
    _ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    _ax.set_yscale("log")
    _ax.set_xlabel("iteration")
    _ax.set_ylabel(r"$\chi^2$")
    _ax.set_title(
        r"Per-variable $\chi^2$ trajectory (open circles = max-$\chi^2$ winner)"
    )
    _ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig, _ax = plt.subplots(figsize=(10, 4))
    _n_iter = len(iter_diag["max_chi2"])
    _iters = np.arange(_n_iter)
    _winners = iter_diag["max_chi2_var"]
    _heights = iter_diag["max_chi2"]
    _cmap = plt.get_cmap("tab20")
    _var_to_color = {v: _cmap(i % 20) for i, v in enumerate(jet_columns)}
    _colors = [_var_to_color.get(w, "0.5") for w in _winners]
    _ax.bar(_iters, _heights, color=_colors)
    for _i, _w in enumerate(_winners):
        _ax.text(
            _i,
            _heights[_i],
            _w,
            rotation=90,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    _ax.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
    _ax.set_xlabel("iteration")
    _ax.set_ylabel(r"max $\chi^2$")
    _ax.set_title(r"Winner variable per iteration (label) and its $\chi^2$")
    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig, _axes = plt.subplots(2, 2, figsize=(12, 8))
    _n_full = len(iter_diag["reco_weight_stats"])
    _n_iter = len(iter_diag["reco_reweight_stats"])
    _iters_full = np.arange(_n_full)
    _iters_step = np.arange(1, _n_iter + 1)

    _ess_reco = np.array([s["ess"] for s in iter_diag["reco_weight_stats"]])
    _ess_gen = np.array([s["ess"] for s in iter_diag["gen_weight_stats"]])
    _axes[0, 0].plot(_iters_full, _ess_reco, marker="o", label="reco")
    _axes[0, 0].plot(_iters_full, _ess_gen, marker="s", label="gen")
    _axes[0, 0].set_xlabel("iteration (0 = initial)")
    _axes[0, 0].set_ylabel("ESS")
    _axes[0, 0].set_title("Effective sample size")
    _axes[0, 0].legend()

    _mom_reco = np.array(
        [s["max"] / s["median"] for s in iter_diag["reco_weight_stats"]]
    )
    _mom_gen = np.array([s["max"] / s["median"] for s in iter_diag["gen_weight_stats"]])
    _axes[0, 1].plot(_iters_full, _mom_reco, marker="o", label="reco")
    _axes[0, 1].plot(_iters_full, _mom_gen, marker="s", label="gen")
    _axes[0, 1].set_xlabel("iteration (0 = initial)")
    _axes[0, 1].set_ylabel("max / median")
    _axes[0, 1].set_yscale("log")
    _axes[0, 1].set_title("Per-event weight max/median")
    _axes[0, 1].legend()

    _gp_reco_n = np.array([len(xy[0]) for xy in iter_diag["gp_reco_train_xy"]])
    _gp_gen_n = np.array([len(xy[0]) for xy in iter_diag["gp_gen_train_xy"]])
    _axes[1, 0].plot(_iters_step, _gp_reco_n, marker="o", label="reco")
    _axes[1, 0].plot(_iters_step, _gp_gen_n, marker="s", label="gen (post-NaN drop)")
    _axes[1, 0].set_xlabel("iteration")
    _axes[1, 0].set_ylabel("# training points")
    _axes[1, 0].set_title("GP training-set size")
    _axes[1, 0].legend()

    _axes[1, 1].plot(
        _iters_step,
        np.array(iter_diag["gen_profile_nan_frac"]),
        marker="o",
        color="C2",
    )
    _axes[1, 1].set_xlabel("iteration")
    _axes[1, 1].set_ylabel("NaN bin fraction")
    _axes[1, 1].set_title("Gen profile NaN bin fraction")

    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(gen_traj_arr, reco_traj_arr):
    _fig, _axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    _rng = np.random.default_rng(0)
    _n_show = min(200, reco_traj_arr.shape[1])
    _pick_reco = _rng.choice(reco_traj_arr.shape[1], size=_n_show, replace=False)
    _pick_gen = _rng.choice(gen_traj_arr.shape[1], size=_n_show, replace=False)
    _iters_reco = np.arange(reco_traj_arr.shape[0])
    _iters_gen = np.arange(gen_traj_arr.shape[0])

    for _j in _pick_reco:
        _axes[0].plot(
            _iters_reco, reco_traj_arr[:, _j], color="C0", alpha=0.15, linewidth=0.7
        )
    _axes[0].set_yscale("log")
    _axes[0].set_xlabel("iteration (0 = initial)")
    _axes[0].set_ylabel("per-event weight")
    _axes[0].set_title(f"reco weight trajectories (n={_n_show})")

    for _j in _pick_gen:
        _axes[1].plot(
            _iters_gen, gen_traj_arr[:, _j], color="C1", alpha=0.15, linewidth=0.7
        )
    _axes[1].set_yscale("log")
    _axes[1].set_xlabel("iteration (0 = initial)")
    _axes[1].set_title(f"gen weight trajectories (n={_n_show})")

    _fig.tight_layout()
    plt.gcf()
    return


@app.cell
def _(bins, iter_diag):
    _topks = iter_diag["reco_topk"]
    _gps = iter_diag["gp_reco_train_xy"]
    _n_iter_e = len(_topks)
    if _n_iter_e == 0:
        _fig_e = None
    else:
        _ncols_e = min(3, _n_iter_e)
        _nrows_e = int(np.ceil(_n_iter_e / _ncols_e))
        _fig_e, _axes_e = plt.subplots(
            _nrows_e,
            _ncols_e,
            figsize=(5 * _ncols_e, 4 * _nrows_e),
            squeeze=False,
        )
        _sc_e = None
        for _i in range(_n_iter_e):
            _ax_e = _axes_e[_i // _ncols_e, _i % _ncols_e]
            _entry_e = _topks[_i]
            _wvar_e = _entry_e["winner_var"]
            _xvals_e = _entry_e["winner_values"]
            _rw_e = _entry_e["reweights"]
            _pth_e = _entry_e["pth_bin"]
            _sc_e = _ax_e.scatter(
                _xvals_e, _rw_e, c=_pth_e, cmap="tab20", s=8, alpha=0.5
            )
            _gp_x_e, _gp_y_e = _gps[_i]
            _ax_e.scatter(
                _gp_x_e,
                _gp_y_e,
                marker="+",
                color="k",
                s=70,
                linewidths=1.2,
                label="GP train",
            )
            _bins_np_e = np.asarray(bins[_wvar_e])
            _left_e = col_hist_args[_wvar_e]["chi2_left_edge"]
            _right_e = col_hist_args[_wvar_e]["chi2_right_edge"]
            if _left_e is not None and _left_e > 0:
                _ax_e.axvline(_bins_np_e[_left_e], color="k", linestyle="--", lw=0.7)
            if _right_e is not None:
                _ax_e.axvline(_bins_np_e[_right_e], color="k", linestyle="--", lw=0.7)
            _ax_e.set_yscale("log")
            _ax_e.set_xlabel(_wvar_e)
            _ax_e.set_ylabel("reweight factor")
            _ax_e.set_title(f"iter {_i} (winner = {_wvar_e})")
            _ax_e.legend(fontsize=7, loc="upper right")
        for _j in range(_n_iter_e, _nrows_e * _ncols_e):
            _axes_e[_j // _ncols_e, _j % _ncols_e].set_visible(False)
        if _sc_e is not None:
            _fig_e.colorbar(_sc_e, ax=_axes_e.ravel().tolist(), label="pth_bin")
        _fig_e.suptitle(
            "Top-K reco reweights per iteration (dashed = chi2_slice; + = GP train)"
        )
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _q_arr = np.asarray(iter_diag["reco_reweight_quantiles"])
    if _q_arr.size == 0:
        _fig_f = None
    else:
        _n_iter_f = _q_arr.shape[0]
        _iters_f = np.arange(1, _n_iter_f + 1)
        _q_labels = ["min", "median", "p90", "p99", "p99.9", "p99.99", "max"]
        _fig_f, _ax_f = plt.subplots(figsize=(8, 5))
        _cmap_f = plt.get_cmap("viridis")
        for _j in range(len(_q_labels)):
            _ax_f.plot(
                _iters_f,
                _q_arr[:, _j],
                marker="o",
                color=_cmap_f(_j / max(1, len(_q_labels) - 1)),
                label=_q_labels[_j],
            )
        _ax_f.axhline(1.0, color="k", linestyle="--", lw=0.7, alpha=0.6)
        _ax_f.set_yscale("log")
        _ax_f.set_xlabel("iteration")
        _ax_f.set_ylabel("reco reweight factor")
        _ax_f.set_title("Reco reweight factor quantiles per iteration")
        _ax_f.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        _fig_f.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _topks_g = iter_diag["reco_topk"]
    _n_iter_g = len(_topks_g)
    if _n_iter_g == 0:
        _fig_g = None
    else:
        _ncols_g = min(3, _n_iter_g)
        _nrows_g = int(np.ceil(_n_iter_g / _ncols_g))
        _fig_g, _axes_g = plt.subplots(
            _nrows_g,
            _ncols_g,
            figsize=(5 * _ncols_g, 3.5 * _nrows_g),
            squeeze=False,
        )
        for _i in range(_n_iter_g):
            _ax_g = _axes_g[_i // _ncols_g, _i % _ncols_g]
            _entry_g = _topks_g[_i]
            _pth_g = _entry_g["pth_bin"]
            _rw_g = _entry_g["reweights"]
            _pre_g = _entry_g["pre_weights"]
            _unique_g, _counts_g = np.unique(_pth_g, return_counts=True)
            _ax_g.bar(_unique_g, _counts_g, color="C0", alpha=0.6, label="count")
            _ax_g.set_xlabel("pth_bin")
            _ax_g.set_ylabel("# top-K events")
            _ax_g2 = _ax_g.twinx()
            _mean_post_g = np.array(
                [
                    float((_pre_g[_pth_g == _b] * _rw_g[_pth_g == _b]).mean())
                    for _b in _unique_g
                ]
            )
            _ax_g2.plot(
                _unique_g,
                _mean_post_g,
                marker="o",
                color="C3",
                label="mean(pre*reweight)",
            )
            _ax_g2.set_yscale("log")
            _ax_g2.set_ylabel("mean(pre*reweight) [log]")
            _ax_g.set_title(f"iter {_i} (winner = {_entry_g['winner_var']})")
        for _j in range(_n_iter_g, _nrows_g * _ncols_g):
            _axes_g[_j // _ncols_g, _j % _ncols_g].set_visible(False)
        _fig_g.suptitle("pT-hat bin attribution of top-K reweighted reco events")
        _fig_g.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _topks_h = iter_diag["reco_topk"]
    _n_iter_h = len(_topks_h)
    if _n_iter_h == 0:
        _fig_h = None
    else:
        _fig_h, _ax_h = plt.subplots(figsize=(8, 6))
        _cmap_h = plt.get_cmap("plasma")
        for _i, _entry_h in enumerate(_topks_h):
            _color_h = _cmap_h(_i / max(1, _n_iter_h - 1))
            _ax_h.scatter(
                np.log10(np.clip(_entry_h["pre_weights"], 1e-300, None)),
                np.log10(np.clip(_entry_h["reweights"], 1e-300, None)),
                s=6,
                alpha=0.3,
                color=_color_h,
                label=f"iter {_i}",
            )
        _ax_h.set_xlabel(r"$\log_{10}(\mathrm{pre\_weight})$")
        _ax_h.set_ylabel(r"$\log_{10}(\mathrm{reweight})$")
        _ax_h.set_title("Top-K reco events: pre-existing weight vs reweight factor")
        _ax_h.legend(loc="best", fontsize=8)
        _fig_h.tight_layout()
    plt.gcf()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
