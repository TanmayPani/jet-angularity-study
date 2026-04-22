import marimo

__generated_with = "0.20.4"
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

    from thoda import Histogram, Profile, bayesian_blocks

    from systematics import SysVar

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
    _source_dir = "./datasets/STAR_pp200GeV_production_2012/clustered_jets"
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

    _data_buffer = pa.memory_map(os.path.join(_source_dir, "preproc_data.arrow"), "rb")
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

    print(gen_table.column_names)
    return data_table, gen_table, n_data, n_gen_matches, reco_table


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
        if not _col in bins:
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
    reco_weights,
):
    n_iterations: int = 10

    iter_hist_list = []

    hists = defaultdict(dict)

    gen_match_reweight_profile = {}

    w_unfolding = [reco_weights, gen_weights]

    max_chi2_var = ""

    num_data_samples = torch.as_tensor(float(n_data), dtype=torch.float32)

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
        iter_hist_list.append(copy.deepcopy(hists))
        if max_chi2 < 1.0:
            break

        print(f"---Max chi2 = {max_chi2} from {max_chi2_var}")
        features = reco_hist[max_chi2_var].bin_centers[1:-1]
        nbins = len(features)
        print("---Obtaining reco lvl reweight factors...")
        reco_reweights = torch.as_tensor(
            propagate_values(
                features[hists[max_chi2_var]["chi2_slice"]], hists[max_chi2_var]["ratio"][hists[max_chi2_var]["chi2_slice"]],
                #features, hists[max_chi2_var]["ratio"],
                reco_arr[max_chi2_var],
            ),
            dtype=torch.float32,
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
        selection = torch.isnan(gen_reweight_profile).logical_not_()
        gen_features = gen_features[selection]
        print(f"---{len(gen_features)} good bins out of {nbins} gen bins...")
        gen_reweight_profile = gen_reweight_profile[selection]
        gen_reweights = torch.as_tensor(
            propagate_values(
                gen_features,
                gen_reweight_profile,
                gen_arr[max_chi2_var],
                num_prediction_batches=40,
            ),
            dtype=torch.float32,
        )
        gen_weights.mul_(gen_reweights)
        gen_weights.div_(gen_weights.sum()).mul_(num_data_samples)

        w_unfolding.extend([reco_weights, gen_weights])
    print("Iterations done...")

    output_folder = "outputs/omnisequential"
    os.makedirs(output_folder, exist_ok=True)
    last_iteration = len(iter_hist_list)
    with open(f"{output_folder}/omniseq-wts-iter{last_iteration}.npz", "wb") as f:
        np.savez(f, *w_unfolding)
    return iter_hist_list, last_iteration


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
def _():
    return


if __name__ == "__main__":
    app.run()
