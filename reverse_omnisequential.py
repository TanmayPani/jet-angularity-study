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

    from preprocessing import replace_table_column
    from systematics import SysVar

    from config import load_config

    _cfg_setup = load_config()
    feature_mode = _cfg_setup["feature_mode"]
    _dataset_root = str(_cfg_setup.dataset_root)

    GP_SEED = 0

    # User toggle: which alt generator drives the gen-level reweighting.
    GENERATOR: str = "herwig7"  # "herwig7" | "pythia8"

    GENERATOR_OUT_SYSVAR = {
        "herwig7": SysVar.UNFOLDING_PRIOR_HERWIG7,
        "pythia8": SysVar.UNFOLDING_PRIOR_PYTHIA8,
    }

    ALT_GEN_ARROW = {
        "herwig7": _dataset_root + "/jets/alt_gen/herwig7.arrow",
        "pythia8": _dataset_root + "/jets/alt_gen/pythia8.arrow",
    }

    jet_columns = [
        "pt",
        "m",
        "nef",
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
        #"leading_constit_pt",
        #"subleading_constit_pt",
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
    y1 = np.asarray(y1, dtype=np.float64)
    x2 = np.asarray(x2)
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
    if x2.ndim == 1:
        x2 = x2.reshape(-1, 1)

    # Drop training points where y1 (or its error, if provided) is non-finite
    # — empty histogram bins produce inf/nan ratios that sklearn rejects.
    _finite_mask = np.isfinite(y1)
    if y1_err is not None:
        y1_err = np.asarray(y1_err, dtype=np.float64)
        _finite_mask &= np.isfinite(y1_err)
    if not _finite_mask.all():
        x1 = x1[_finite_mask]
        y1 = y1[_finite_mask]
        if y1_err is not None:
            y1_err = y1_err[_finite_mask]

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
            _alpha = y1_err ** 2
            _alpha = np.clip(_alpha, 1e-10, None)
        # WhiteKernel soaks up residual noise so the RBF length scale
        # doesn't collapse to fit per-bin scatter (previous run hit
        # length_scale → 1e-5 bound + L-BFGS ABNORMAL termination).
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=0.3, length_scale_bounds=(1e-3, 1e2)
        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1.0))
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
            if ratio is None:
                ratio = count1 / count2
                if error1 is not None and error2 is not None:
                    ratio_err = ratio * np.sqrt(
                        (error1 / count1) ** 2 + (error2 / count2) ** 2
                    )
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
    _source_dir = _dataset_root + "/features"
    _emb_dir = os.path.join(_source_dir, feature_mode, "embedding", str(SysVar.NONE))

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

    if _do_use_gen_misses:
        gen_table = pa.concat_tables([gen_match_table, gen_miss_table])
    else:
        gen_table = gen_match_table

    reco_table = pa.concat_tables([reco_match_table, reco_fake_table])

    n_gen_matches = len(gen_match_table)
    n_gen_misses = len(gen_miss_table)
    n_reco_matches = len(reco_match_table)
    n_reco_fakes = len(reco_fake_table)

    assert n_gen_matches == n_reco_matches

    n_gen = n_gen_matches + n_gen_misses
    n_reco = n_reco_matches + n_reco_fakes

    _w_gm_orig = gen_match_table["weight"].to_numpy().astype(np.float32)
    _w_mi_orig = gen_miss_table["weight"].to_numpy().astype(np.float32)
    _w_rm_orig = reco_match_table["weight"].to_numpy().astype(np.float32)
    _w_fk_orig = reco_fake_table["weight"].to_numpy().astype(np.float32)
    sum_w_gen_orig = float(_w_gm_orig.sum() + _w_mi_orig.sum())
    sum_w_reco_orig = float(_w_rm_orig.sum() + _w_fk_orig.sum())

    print(
        "Number of gen matches/misses:",
        n_gen_matches,
        n_gen_misses,
        " ; reco matches/fakes:",
        n_reco_matches,
        n_reco_fakes,
    )
    print(
        f"sum(w_gen_p6_orig)={sum_w_gen_orig:.4g}  sum(w_reco_p6_orig)={sum_w_reco_orig:.4g}"
    )
    print("Columns:", gen_table.column_names)
    return (
        gen_match_table,
        gen_miss_table,
        gen_table,
        n_gen_matches,
        n_reco_matches,
        reco_fake_table,
        reco_match_table,
        reco_table,
        sum_w_gen_orig,
        sum_w_reco_orig,
    )


@app.cell
def _(sum_w_gen_orig):
    _path = ALT_GEN_ARROW[GENERATOR]
    if not os.path.exists(_path):
        raise FileNotFoundError(
            f"Alt-gen arrow not found at {_path!r}. Build it once via a "
            f"sidecar script (read raw alt-gen arrows, run process_table from "
            f"preprocessing, concat to a single arrow at this path) before "
            f"running the reverse-omniseq notebook."
        )
    _alt_buf = pa.memory_map(_path, "rb")
    alt_table = pa.ipc.open_file(_alt_buf).read_all()

    _missing = [
        c
        for c in jet_columns + ["pth_bin", "weight"]
        if c not in alt_table.column_names
    ]
    if _missing:
        raise RuntimeError(
            f"alt-gen arrow at {_path!r} missing required columns: {_missing}"
        )

    alt_weights = torch.as_tensor(alt_table["weight"].to_numpy(), dtype=torch.float64)
    _alt_sum_raw = float(alt_weights.sum())
    if _alt_sum_raw <= 0:
        raise RuntimeError(
            f"alt-gen sum(w)={_alt_sum_raw}; cannot equalize against P6 gen."
        )
    alt_weights.mul_(sum_w_gen_orig / _alt_sum_raw)

    n_alt = len(alt_table)
    print(
        f"[reverse-omniseq] generator={GENERATOR}  n_alt={n_alt}\n"
        f"  sum(w_alt)_raw={_alt_sum_raw:.4g}  "
        f"sum(w_alt)_equalized={float(alt_weights.sum()):.4g}  "
        f"sum(w_gen_p6_orig)={sum_w_gen_orig:.4g}"
    )
    return alt_table, alt_weights, n_alt


@app.cell
def _(alt_table):
    _p0 = 0.02
    _undersample = 55000
    _bin_file_path = f"./runtime-files/bins_p0{_p0:g}_N{_undersample:g}.json"
    _recalculate_bins_for: list[str] | None = None

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
    _device = "cpu"
    for _col in jet_columns:
        if _col not in bins:
            _bins_updated = True
            print("Calculating binning for", _col, "(from alt-gen distribution)")
            bins[_col] = bayesian_blocks(
                torch.as_tensor(
                    alt_table[_col].to_numpy(), dtype=torch.float64, device=_device
                ),
                p0=_p0,
                ranges=col_hist_args[_col]["range"],
                undersample=_undersample,
                device=_device,
            ).tolist()
            print(_col, ":", bins[_col], len(bins[_col]))

    if _bins_updated:
        with open(_bin_file_path, "w") as _bin_file:
            json.dump(
                {
                    k: v#.tolist() if hasattr(v, "tolist") else list(v)
                    for k, v in bins.items()
                },
                _bin_file,
                indent=4,
            )
    return (bins,)


@app.cell
def _(alt_table, alt_weights, bins, gen_table, n_alt, reco_table):
    num_samples = torch.as_tensor(float(n_alt), dtype=torch.float64)

    # Initial P6 gen weights, normalized to num_samples scale.
    gen_weights = torch.as_tensor(gen_table["weight"].to_numpy(), dtype=torch.float64)
    gen_weights = gen_weights.div_(gen_weights.sum()).mul_(num_samples)

    # Initial P6 reco weights, normalized to num_samples scale.
    reco_weights = torch.as_tensor(reco_table["weight"].to_numpy(), dtype=torch.float64)
    reco_weights = reco_weights.div_(reco_weights.sum()).mul_(num_samples)

    # Frozen clones of the initial weights — cell D mutates gen_weights /
    # reco_weights in-place and the mutation is invisible to marimo's
    # dependency tracking, so we capture an untouched copy here for cell Q's
    # per-jet ratio diagnostic.
    gen_weights_initial = gen_weights.clone()
    reco_weights_initial = reco_weights.clone()

    # Alt weights rescaled to num_samples too (chi² scan compares shapes).
    _alt_weights_norm = alt_weights.clone()
    _alt_weights_norm.div_(_alt_weights_norm.sum()).mul_(num_samples)

    alt_hist = {}
    alt_hist_count = {}
    alt_hist_count_err = {}
    gen_arr = {}
    gen_hist = {}
    binned_gen = {}
    reco_arr = {}
    reco_hist = {}
    binned_reco = {}
    axes = {}

    print(
        f"n_alt={n_alt}  num_samples={float(num_samples):.4g}  "
        f"sum(alt)={float(_alt_weights_norm.sum()):.4g}  "
        f"sum(gen)={float(gen_weights.sum()):.4g}  "
        f"sum(reco)={float(reco_weights.sum()):.4g}"
    )

    for _col in jet_columns:
        _alt_hist, _ = Histogram.create(
            (torch.as_tensor(alt_table[_col].to_numpy(), dtype=torch.float64),),
            bins=(bins[_col],),
            weights=_alt_weights_norm,
            return_binned_data=True,
        )
        alt_hist[_col] = _alt_hist.snapshot()
        alt_hist_count[_col] = alt_hist[_col].values[1:-1]
        alt_hist_count_err[_col] = alt_hist[_col].variances.sqrt_()[1:-1]

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
        axes[_col] = _alt_hist.axes
        del _alt_hist, _gen_hist, _reco_hist

    gen_pth_bin = torch.as_tensor(gen_table["pth_bin"].to_numpy(), dtype=torch.int64)
    reco_pth_bin = torch.as_tensor(reco_table["pth_bin"].to_numpy(), dtype=torch.int64)
    return (
        alt_hist_count,
        alt_hist_count_err,
        axes,
        binned_gen,
        binned_reco,
        gen_arr,
        gen_hist,
        gen_pth_bin,
        gen_weights,
        gen_weights_initial,
        num_samples,
        reco_arr,
        reco_hist,
        reco_pth_bin,
        reco_weights,
        reco_weights_initial,
    )


@app.cell
def _(
    alt_hist_count,
    alt_hist_count_err,
    axes,
    binned_gen,
    binned_reco,
    gen_arr,
    gen_hist,
    gen_pth_bin,
    gen_weights,
    n_gen_matches,
    n_reco_matches,
    num_samples,
    reco_arr,
    reco_hist,
    reco_pth_bin,
    reco_weights,
):
    n_iterations: int = 10
    _damping_alpha: float = 0.5  # 0.5 = sqrt-damping; 1.0 = no damping

    iter_hist_list = []

    hists = defaultdict(dict)

    reco_match_reweight_profile = {}

    w_unfolding = [reco_weights.clone(), gen_weights.clone()]

    max_chi2_var = ""

    iter_diag = {
        "chi2": [],
        "max_chi2_var": [],
        "max_chi2": [],
        "gen_weight_stats": [],
        "reco_weight_stats": [],
        "gen_reweight_stats": [],
        "reco_reweight_stats": [],
        "reco_profile_nan_frac": [],
        "gp_gen_train_xy": [],
        "gp_reco_train_xy": [],
        "gen_topk": [],
        "reco_topk": [],
        "gen_reweight_quantiles": [],
        "reco_reweight_quantiles": [],
        "gen_reweight_pre_clamp_min": [],
        "gen_reweight_n_clamped": [],
        "reco_reweight_pre_clamp_min": [],
        "reco_reweight_n_clamped": [],
    }

    iter_diag["gen_weight_stats"].append(weight_summary(gen_weights))
    iter_diag["reco_weight_stats"].append(weight_summary(reco_weights))

    _traj_rng = torch.Generator().manual_seed(0)
    _n_traj = 2000
    _sample_gen_idx = torch.randperm(len(gen_weights), generator=_traj_rng)[:_n_traj]
    _sample_reco_idx = torch.randperm(len(reco_weights), generator=_traj_rng)[:_n_traj]
    _gen_traj = [gen_weights[_sample_gen_idx].detach().cpu().clone()]
    _reco_traj = [reco_weights[_sample_reco_idx].detach().cpu().clone()]

    for iteration in range(n_iterations):
        max_chi2 = 0
        max_chi2_var = ""
        print(f"Iteration: {iteration}")
        for _col in jet_columns:
            hists[_col]["alt"] = alt_hist_count[_col]
            hists[_col]["alt_err"] = alt_hist_count_err[_col]

            del gen_hist[_col]
            gen_hist[_col] = Histogram.from_binned(
                axes[_col], binned_gen[_col], weights=gen_weights
            ).snapshot()
            hists[_col]["gen"] = gen_hist[_col].values[1:-1]
            hists[_col]["gen_err"] = gen_hist[_col].variances.sqrt_()[1:-1]

            del reco_hist[_col]
            reco_hist[_col] = Histogram.from_binned(
                axes[_col], binned_reco[_col], weights=reco_weights
            ).snapshot()
            hists[_col]["reco"] = reco_hist[_col].values[1:-1]
            hists[_col]["reco_err"] = reco_hist[_col].variances.sqrt_()[1:-1]

            hists[_col]["ratio"] = hists[_col]["alt"] / hists[_col]["gen"]
            hists[_col]["ratio_err"] = (
                (
                    hists[_col]["alt_err"].div(hists[_col]["alt"]).pow_(2)
                    + hists[_col]["gen_err"].div(hists[_col]["gen"]).pow_(2)
                )
                .sqrt_()
                .mul_(hists[_col]["ratio"])
            )

            left_edge = col_hist_args[_col]["chi2_left_edge"]
            right_edge = col_hist_args[_col]["chi2_right_edge"]
            chi2_slice = slice(left_edge, right_edge)
            hists[_col]["chi2_slice"] = chi2_slice
            h_alt = hists[_col]["alt"][chi2_slice]
            h_gen = hists[_col]["gen"][chi2_slice]
            h_gen = h_gen.div(h_gen.sum()).mul_(h_alt.sum())
            chi2, _ = chisquare(h_gen, h_alt)
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

        # ----- GEN-level GP fit (driver: alt / gen ratio) -----
        features = gen_hist[max_chi2_var].bin_centers[1:-1]
        print("---Obtaining gen lvl reweight factors...")
        gen_reweights = torch.as_tensor(
            propagate_values(
                features[hists[max_chi2_var]["chi2_slice"]],
                hists[max_chi2_var]["ratio"][hists[max_chi2_var]["chi2_slice"]],
                gen_arr[max_chi2_var],
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
        _gen_pre_clamp_min = float(gen_reweights.min())
        _gen_n_clamped = int((gen_reweights < 0).sum())
        gen_reweights.clamp_min_(0.0)
        gen_reweights.pow_(_damping_alpha)
        iter_diag["gen_reweight_pre_clamp_min"].append(_gen_pre_clamp_min)
        iter_diag["gen_reweight_n_clamped"].append(_gen_n_clamped)
        iter_diag["gen_reweight_stats"].append(weight_summary(gen_reweights))
        iter_diag["gp_gen_train_xy"].append(
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
        _grw_np = gen_reweights.detach().cpu().numpy()
        _gpre_np = gen_weights.detach().cpu().numpy()
        _g_topk_part = np.argpartition(-_grw_np, _K)[:_K]
        _g_topk_idx = _g_topk_part[np.argsort(-_grw_np[_g_topk_part])]
        iter_diag["gen_topk"].append(
            {
                "indices": _g_topk_idx.astype(np.int64),
                "reweights": _grw_np[_g_topk_idx].astype(np.float64),
                "pre_weights": _gpre_np[_g_topk_idx].astype(np.float64),
                "winner_var": max_chi2_var,
                "winner_values": gen_arr[max_chi2_var][_g_topk_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                "pth_bin": gen_pth_bin[_g_topk_idx].numpy().astype(np.int64),
            }
        )
        iter_diag["gen_reweight_quantiles"].append(
            np.quantile(_grw_np, [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0])
        )

        gen_weights.mul_(gen_reweights)
        gen_weights.div_(gen_weights.sum()).mul_(num_samples)

        # ----- Propagate GEN→RECO via matched-pair profile -----
        print("---Propagating gen reweights to reco level (matched pairs)...")
        reco_features = reco_hist[max_chi2_var].bin_centers[1:-1]
        nbins_reco = len(reco_features)

        if max_chi2_var in reco_match_reweight_profile:
            del reco_match_reweight_profile[max_chi2_var]
        reco_match_reweight_profile[max_chi2_var] = Profile.from_binned(
            axes[max_chi2_var],
            binned_reco[max_chi2_var][:n_reco_matches],
            gen_reweights[:n_gen_matches],
            weights=reco_weights[:n_reco_matches],
        ).snapshot()
        reco_reweight_profile = reco_match_reweight_profile[max_chi2_var].values[1:-1]
        reco_reweight_variance = (
            reco_match_reweight_profile[max_chi2_var].variances[1:-1].clamp_min(0)
        )
        reco_effective_counts = reco_match_reweight_profile[
            max_chi2_var
        ].effective_counts[1:-1]
        selection = torch.isnan(reco_reweight_profile).logical_not_()
        reco_features = reco_features[selection]
        print(f"---{len(reco_features)} good bins out of {nbins_reco} reco bins...")
        reco_reweight_profile = reco_reweight_profile[selection]
        reco_reweight_variance = reco_reweight_variance[selection]
        reco_effective_counts = reco_effective_counts[selection]
        reco_y1_err = (
            (reco_reweight_variance / reco_effective_counts.clamp_min(1e-12))
            .sqrt_()
            .detach()
            .cpu()
            .numpy()
        )
        iter_diag["reco_profile_nan_frac"].append(
            1.0 - float(selection.sum()) / float(selection.numel())
        )
        iter_diag["gp_reco_train_xy"].append(
            (
                reco_features.detach().cpu().numpy().copy(),
                reco_reweight_profile.detach().cpu().numpy().copy(),
            )
        )

        reco_reweights = torch.as_tensor(
            propagate_values(
                reco_features,
                reco_reweight_profile,
                reco_arr[max_chi2_var],
                num_prediction_batches=40,
                y1_err=reco_y1_err,
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

        _rrw_np = reco_reweights.detach().cpu().numpy()
        _rpre_np = reco_weights.detach().cpu().numpy()
        _r_topk_part = np.argpartition(-_rrw_np, _K)[:_K]
        _r_topk_idx = _r_topk_part[np.argsort(-_rrw_np[_r_topk_part])]
        iter_diag["reco_topk"].append(
            {
                "indices": _r_topk_idx.astype(np.int64),
                "reweights": _rrw_np[_r_topk_idx].astype(np.float64),
                "pre_weights": _rpre_np[_r_topk_idx].astype(np.float64),
                "winner_var": max_chi2_var,
                "winner_values": reco_arr[max_chi2_var][_r_topk_idx]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64),
                "pth_bin": reco_pth_bin[_r_topk_idx].numpy().astype(np.int64),
            }
        )
        iter_diag["reco_reweight_quantiles"].append(
            np.quantile(_rrw_np, [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 1.0])
        )

        reco_weights.mul_(reco_reweights)
        reco_weights.div_(reco_weights.sum()).mul_(num_samples)

        iter_diag["gen_weight_stats"].append(weight_summary(gen_weights))
        iter_diag["reco_weight_stats"].append(weight_summary(reco_weights))

        _gen_traj.append(gen_weights[_sample_gen_idx].detach().cpu().clone())
        _reco_traj.append(reco_weights[_sample_reco_idx].detach().cpu().clone())

        w_unfolding.extend([reco_weights, gen_weights])

    # Post-loop snapshot reflecting gen/reco weights AFTER the final
    # reweight, so iter_hist_list[-1] matches the on-disk weight state
    # cell P compares against.
    for _col in jet_columns:
        hists[_col]["alt"] = alt_hist_count[_col]
        hists[_col]["alt_err"] = alt_hist_count_err[_col]

        del gen_hist[_col]
        gen_hist[_col] = Histogram.from_binned(
            axes[_col], binned_gen[_col], weights=gen_weights
        ).snapshot()
        hists[_col]["gen"] = gen_hist[_col].values[1:-1]
        hists[_col]["gen_err"] = gen_hist[_col].variances.sqrt_()[1:-1]

        del reco_hist[_col]
        reco_hist[_col] = Histogram.from_binned(
            axes[_col], binned_reco[_col], weights=reco_weights
        ).snapshot()
        hists[_col]["reco"] = reco_hist[_col].values[1:-1]
        hists[_col]["reco_err"] = reco_hist[_col].variances.sqrt_()[1:-1]
    iter_hist_list.append(copy.deepcopy(hists))

    gen_traj_arr = torch.stack(_gen_traj).numpy()
    reco_traj_arr = torch.stack(_reco_traj).numpy()
    print("Iterations done...")

    output_folder = f"outputs/reverse_omnisequential/{GENERATOR}/{feature_mode}"
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
            ess_gen=np.array([s["ess"] for s in iter_diag["gen_weight_stats"]]),
            ess_reco=np.array([s["ess"] for s in iter_diag["reco_weight_stats"]]),
            max_over_median_gen=np.array(
                [s["max"] / s["median"] for s in iter_diag["gen_weight_stats"]]
            ),
            max_over_median_reco=np.array(
                [s["max"] / s["median"] for s in iter_diag["reco_weight_stats"]]
            ),
            gen_reweight_max=np.array(
                [s["max"] for s in iter_diag["gen_reweight_stats"]]
            ),
            reco_reweight_max=np.array(
                [s["max"] for s in iter_diag["reco_reweight_stats"]]
            ),
            reco_profile_nan_frac=np.array(iter_diag["reco_profile_nan_frac"]),
            gen_traj=gen_traj_arr,
            reco_traj=reco_traj_arr,
            gp_seed=np.int64(GP_SEED),
            gen_topk=np.array(iter_diag["gen_topk"], dtype=object),
            reco_topk=np.array(iter_diag["reco_topk"], dtype=object),
            gen_reweight_quantiles=np.array(iter_diag["gen_reweight_quantiles"]),
            reco_reweight_quantiles=np.array(iter_diag["reco_reweight_quantiles"]),
        )

    # Expose post-mutation tensors under explicit names so cell E's dependency
    # on them runs after this cell. (Marimo tracks assignments, not in-place
    # mutations.)
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
    _do_write_arrows: bool = True
    arrows_written = False
    out_dir = ""

    if not _do_write_arrows:
        print("[reverse-omniseq] _do_write_arrows = False; skipping output write")
    else:
        _out_sysvar = GENERATOR_OUT_SYSVAR[GENERATOR]
        out_dir = os.path.join(
            _dataset_root + "/features",
            feature_mode,
            "embedding",
            str(_out_sysvar),
        )
        os.makedirs(out_dir, exist_ok=True)

        _w_gen_final = gen_weights_final.detach().cpu().numpy().astype(np.float64)
        _w_reco_final = reco_weights_final.detach().cpu().numpy().astype(np.float64)
        _w_gen_out = (_w_gen_final * sum_w_gen_orig / _w_gen_final.sum()).astype(
            np.float32
        )
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
            _path = os.path.join(out_dir, f"{_name}.arrow")
            with pa.OSFile(_path, "wb") as _sink:
                with pa.ipc.new_file(_sink, _t.schema) as _writer:
                    for _batch in _t.to_batches():
                        _writer.write_batch(_batch)
            print(f"  wrote {_path} ({len(_t)} rows)")

        arrows_written = True
        print(
            f"[reverse-omniseq] last_iteration={last_iteration}; 4 arrows in {out_dir}"
        )
    return arrows_written, out_dir


@app.cell
def _(bins, iter_hist_list, last_iteration):
    if len(iter_hist_list) != 0:
        _fig_scale = 5
        _nrows = 3
        _ncols = int(np.ceil(len(jet_columns) / _nrows))
        _fig_f = plt.figure(figsize=(_ncols * _fig_scale, _nrows * _fig_scale))
        _fig_f.suptitle(
            f"[{GENERATOR}] gen-driven closure, iteration {last_iteration}",
            fontsize=24,
        )
        _subfigs = _fig_f.subfigures(_nrows, _ncols)
        for _ivar, _var in enumerate(jet_columns):
            _irow = int(np.floor(_ivar / _ncols))
            _icol = _ivar % _ncols
            _subfig = _subfigs[_irow, _icol]
            _h = iter_hist_list[-1][_var]
            _h0 = iter_hist_list[0][_var]

            _axs = plot_ratios(
                _subfig,
                torch.as_tensor(bins[_var]),
                [_h["alt"]],
                [_h["gen"], _h0["gen"]],
                [[_h["ratio"], _h0["ratio"]]],
                [_h["alt_err"]],
                [_h["gen_err"], _h0["gen_err"]],
                [[_h["ratio_err"], _h0["ratio_err"]]],
                labels1=["alt gen (target)"],
                labels2=[f"P6 gen (iter={last_iteration})", "P6 gen (iter=0)"],
                markers1=["o"],
                markers2=["s", "v"],
            )
            plot_hist(
                _axs[0],
                torch.as_tensor(bins[_var]),
                _h["reco"],
                errors=_h["reco_err"],
                label=f"P6 reco (iter={last_iteration})",
                marker="^",
                fillstyle="none",
                markersize=10,
            )
            plot_hist(
                _axs[0],
                torch.as_tensor(bins[_var]),
                _h0["reco"],
                errors=_h0["reco_err"],
                label="P6 reco (iter=0)",
                marker="x",
                fillstyle="none",
                markersize=10,
            )
            _axs[0].set_title(f"chi2 = {_h['chi2']:.2f}")
            _axs[0].legend(fontsize=7)
            _axs[1].set_xlabel(_var)
            _axs[1].legend(fontsize=7)
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig_g, _ax_g = plt.subplots(figsize=(10, 6))
    _chi2_table = iter_diag["chi2"]
    _n_iter = len(_chi2_table)
    if _n_iter == 0:
        plt.gcf()
    else:
        _iters = np.arange(_n_iter)
        _cmap = plt.get_cmap("tab20")
        for _ivar, _var in enumerate(jet_columns):
            _vals = np.array([d.get(_var, np.nan) for d in _chi2_table])
            _color = _cmap(_ivar % 20)
            _ax_g.plot(
                _iters, _vals, marker=".", color=_color, label=_var, linewidth=1.2
            )
            _winner_mask = np.array(
                [iter_diag["max_chi2_var"][i] == _var for i in range(_n_iter)]
            )
            if _winner_mask.any():
                _ax_g.scatter(
                    _iters[_winner_mask],
                    _vals[_winner_mask],
                    s=120,
                    facecolors="none",
                    edgecolors=_color,
                    linewidths=2,
                )
        _ax_g.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        _ax_g.set_yscale("log")
        _ax_g.set_xlabel("iteration")
        _ax_g.set_ylabel(r"$\chi^2$ (alt vs P6 gen)")
        _ax_g.set_title(
            r"Per-variable $\chi^2$ trajectory (open circles = max-$\chi^2$ winner)"
        )
        _ax_g.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        _fig_g.tight_layout()

    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig_h, _ax_h = plt.subplots(figsize=(10, 4))
    _n_iter = len(iter_diag["max_chi2"])
    if _n_iter != 0:
        _iters = np.arange(_n_iter)
        _winners = iter_diag["max_chi2_var"]
        _heights = iter_diag["max_chi2"]
        _cmap = plt.get_cmap("tab20")
        _var_to_color = {v: _cmap(i % 20) for i, v in enumerate(jet_columns)}
        _colors = [_var_to_color.get(w, "0.5") for w in _winners]
        _ax_h.bar(_iters, _heights, color=_colors)
        for _i, _w in enumerate(_winners):
            _ax_h.text(
                _i,
                _heights[_i],
                _w,
                rotation=90,
                ha="center",
                va="bottom",
                fontsize=8,
            )
        _ax_h.axhline(1.0, color="k", linestyle="--", linewidth=0.8, alpha=0.6)
        _ax_h.set_xlabel("iteration")
        _ax_h.set_ylabel(r"max $\chi^2$ (alt vs P6 gen)")
        _ax_h.set_title(r"Winner variable per iteration (label) and its $\chi^2$")
        _fig_h.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _fig_i, _axes_i = plt.subplots(2, 2, figsize=(12, 8))
    _n_full = len(iter_diag["gen_weight_stats"])
    _n_step = len(iter_diag["gen_reweight_stats"])
    if _n_full != 0:
        _iters_full = np.arange(_n_full)
        _iters_step = np.arange(1, _n_step + 1)

        _ess_gen = np.array([s["ess"] for s in iter_diag["gen_weight_stats"]])
        _ess_reco = np.array([s["ess"] for s in iter_diag["reco_weight_stats"]])
        _axes_i[0, 0].plot(_iters_full, _ess_gen, marker="o", label="gen")
        _axes_i[0, 0].plot(_iters_full, _ess_reco, marker="s", label="reco")
        _axes_i[0, 0].set_xlabel("iteration (0 = initial)")
        _axes_i[0, 0].set_ylabel("ESS")
        _axes_i[0, 0].set_title("Effective sample size")
        _axes_i[0, 0].legend()

        _mom_gen = np.array(
            [s["max"] / s["median"] for s in iter_diag["gen_weight_stats"]]
        )
        _mom_reco = np.array(
            [s["max"] / s["median"] for s in iter_diag["reco_weight_stats"]]
        )
        _axes_i[0, 1].plot(_iters_full, _mom_gen, marker="o", label="gen")
        _axes_i[0, 1].plot(_iters_full, _mom_reco, marker="s", label="reco")
        _axes_i[0, 1].set_xlabel("iteration (0 = initial)")
        _axes_i[0, 1].set_ylabel("max / median")
        _axes_i[0, 1].set_yscale("log")
        _axes_i[0, 1].set_title("Per-event weight max/median")
        _axes_i[0, 1].legend()

        _gp_gen_n = np.array([len(xy[0]) for xy in iter_diag["gp_gen_train_xy"]])
        _gp_reco_n = np.array([len(xy[0]) for xy in iter_diag["gp_reco_train_xy"]])
        _axes_i[1, 0].plot(
            _iters_step, _gp_gen_n, marker="o", label="gen (bins in chi2_slice)"
        )
        _axes_i[1, 0].plot(
            _iters_step, _gp_reco_n, marker="s", label="reco (post-NaN drop)"
        )
        _axes_i[1, 0].set_xlabel("iteration")
        _axes_i[1, 0].set_ylabel("# training points")
        _axes_i[1, 0].set_title("GP training-set size")
        _axes_i[1, 0].legend()

        _axes_i[1, 1].plot(
            _iters_step,
            np.array(iter_diag["reco_profile_nan_frac"]),
            marker="o",
            color="C2",
        )
        _axes_i[1, 1].set_xlabel("iteration")
        _axes_i[1, 1].set_ylabel("NaN bin fraction")
        _axes_i[1, 1].set_title("Reco profile NaN bin fraction")

        _fig_i.tight_layout()
    plt.gcf()
    return


@app.cell
def _(gen_traj_arr, reco_traj_arr):
    _fig_j, _axes_j = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    if gen_traj_arr.shape[0] != 0 and reco_traj_arr.shape[0] != 0:
        _rng = np.random.default_rng(0)
        _n_show = min(200, gen_traj_arr.shape[1])
        _pick_gen = _rng.choice(gen_traj_arr.shape[1], size=_n_show, replace=False)
        _pick_reco = _rng.choice(reco_traj_arr.shape[1], size=_n_show, replace=False)
        _iters_gen = np.arange(gen_traj_arr.shape[0])
        _iters_reco = np.arange(reco_traj_arr.shape[0])

        for _j in _pick_gen:
            _axes_j[0].plot(
                _iters_gen,
                gen_traj_arr[:, _j],
                color="C0",
                alpha=0.15,
                linewidth=0.7,
            )
        _axes_j[0].set_yscale("log")
        _axes_j[0].set_xlabel("iteration (0 = initial)")
        _axes_j[0].set_ylabel("per-event weight")
        _axes_j[0].set_title(f"GEN weight trajectories (n={_n_show})")

        for _j in _pick_reco:
            _axes_j[1].plot(
                _iters_reco,
                reco_traj_arr[:, _j],
                color="C1",
                alpha=0.15,
                linewidth=0.7,
            )
        _axes_j[1].set_yscale("log")
        _axes_j[1].set_xlabel("iteration (0 = initial)")
        _axes_j[1].set_title(f"RECO weight trajectories (n={_n_show})")

        _fig_j.tight_layout()
    plt.gcf()
    return


@app.cell
def _(bins, iter_diag):
    _topks = iter_diag["gen_topk"]
    _gps = iter_diag["gp_gen_train_xy"]
    _n_iter_k = len(_topks)
    if _n_iter_k == 0:
        _fig_k = None
    else:
        _ncols_k = min(3, _n_iter_k)
        _nrows_k = int(np.ceil(_n_iter_k / _ncols_k))
        _fig_k, _axes_k = plt.subplots(
            _nrows_k,
            _ncols_k,
            figsize=(5 * _ncols_k, 4 * _nrows_k),
            squeeze=False,
        )
        _sc_k = None
        for _i in range(_n_iter_k):
            _ax_k = _axes_k[_i // _ncols_k, _i % _ncols_k]
            _entry_k = _topks[_i]
            _wvar_k = _entry_k["winner_var"]
            _xvals_k = _entry_k["winner_values"]
            _rw_k = _entry_k["reweights"]
            _pth_k = _entry_k["pth_bin"]
            _sc_k = _ax_k.scatter(
                _xvals_k, _rw_k, c=_pth_k, cmap="tab20", s=8, alpha=0.5
            )
            _gp_x_k, _gp_y_k = _gps[_i]
            _ax_k.scatter(
                _gp_x_k,
                _gp_y_k,
                marker="+",
                color="k",
                s=70,
                linewidths=1.2,
                label="GP train",
            )
            _bins_np_k = np.asarray(bins[_wvar_k])
            _left_k = col_hist_args[_wvar_k]["chi2_left_edge"]
            _right_k = col_hist_args[_wvar_k]["chi2_right_edge"]
            if _left_k is not None and _left_k > 0:
                _ax_k.axvline(_bins_np_k[_left_k], color="k", linestyle="--", lw=0.7)
            if _right_k is not None:
                _ax_k.axvline(_bins_np_k[_right_k], color="k", linestyle="--", lw=0.7)
            _ax_k.set_yscale("log")
            _ax_k.set_xlabel(_wvar_k)
            _ax_k.set_ylabel("gen reweight factor")
            _ax_k.set_title(f"iter {_i} (winner = {_wvar_k})")
            _ax_k.legend(fontsize=7, loc="upper right")
        for _j in range(_n_iter_k, _nrows_k * _ncols_k):
            _axes_k[_j // _ncols_k, _j % _ncols_k].set_visible(False)
        if _sc_k is not None:
            _fig_k.colorbar(_sc_k, ax=_axes_k.ravel().tolist(), label="gen_pth_bin")
        _fig_k.suptitle(
            "GEN top-K reweights per iteration (dashed = chi2_slice; + = GP train)"
        )
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _topks = iter_diag["reco_topk"]
    _gps = iter_diag["gp_reco_train_xy"]
    _n_iter_l = len(_topks)
    if _n_iter_l == 0:
        _fig_l = None
    else:
        _ncols_l = min(3, _n_iter_l)
        _nrows_l = int(np.ceil(_n_iter_l / _ncols_l))
        _fig_l, _axes_l = plt.subplots(
            _nrows_l,
            _ncols_l,
            figsize=(5 * _ncols_l, 4 * _nrows_l),
            squeeze=False,
        )
        _sc_l = None
        for _i in range(_n_iter_l):
            _ax_l = _axes_l[_i // _ncols_l, _i % _ncols_l]
            _entry_l = _topks[_i]
            _wvar_l = _entry_l["winner_var"]
            _xvals_l = _entry_l["winner_values"]
            _rw_l = _entry_l["reweights"]
            _pth_l = _entry_l["pth_bin"]
            _sc_l = _ax_l.scatter(
                _xvals_l, _rw_l, c=_pth_l, cmap="tab20", s=8, alpha=0.5
            )
            _gp_x_l, _gp_y_l = _gps[_i]
            _ax_l.scatter(
                _gp_x_l,
                _gp_y_l,
                marker="+",
                color="k",
                s=70,
                linewidths=1.2,
                label="profile train",
            )
            _ax_l.set_yscale("log")
            _ax_l.set_xlabel(_wvar_l)
            _ax_l.set_ylabel("reco reweight factor")
            _ax_l.set_title(f"iter {_i} (winner = {_wvar_l})")
            _ax_l.legend(fontsize=7, loc="upper right")
        for _j in range(_n_iter_l, _nrows_l * _ncols_l):
            _axes_l[_j // _ncols_l, _j % _ncols_l].set_visible(False)
        if _sc_l is not None:
            _fig_l.colorbar(_sc_l, ax=_axes_l.ravel().tolist(), label="reco_pth_bin")
        _fig_l.suptitle("RECO top-K reweights per iteration (+ = profile train pts)")
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _q_gen = np.asarray(iter_diag["gen_reweight_quantiles"])
    _q_reco = np.asarray(iter_diag["reco_reweight_quantiles"])
    if _q_gen.size == 0 or _q_reco.size == 0:
        _fig_m = None
    else:
        _q_labels = ["min", "median", "p90", "p99", "p99.9", "p99.99", "max"]
        _fig_m, _axes_m = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        _cmap_m = plt.get_cmap("viridis")
        for _side, _q_arr, _ax_m, _title in (
            ("gen", _q_gen, _axes_m[0], "GEN"),
            ("reco", _q_reco, _axes_m[1], "RECO"),
        ):
            _n_iter_m = _q_arr.shape[0]
            _iters_m = np.arange(1, _n_iter_m + 1)
            for _j in range(len(_q_labels)):
                _ax_m.plot(
                    _iters_m,
                    _q_arr[:, _j],
                    marker="o",
                    color=_cmap_m(_j / max(1, len(_q_labels) - 1)),
                    label=_q_labels[_j],
                )
            _ax_m.axhline(1.0, color="k", linestyle="--", lw=0.7, alpha=0.6)
            _ax_m.set_yscale("log")
            _ax_m.set_xlabel("iteration")
            _ax_m.set_ylabel("reweight factor")
            _ax_m.set_title(f"{_title} reweight quantiles per iteration")
        _axes_m[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
        _fig_m.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _topks_gen = iter_diag["gen_topk"]
    _topks_reco = iter_diag["reco_topk"]
    _n_iter_n = max(len(_topks_gen), len(_topks_reco))
    if _n_iter_n == 0:
        _fig_n = None
    else:
        _ncols_n = min(3, _n_iter_n)
        _nrows_n = 2 * int(np.ceil(_n_iter_n / _ncols_n))
        _fig_n, _axes_n = plt.subplots(
            _nrows_n,
            _ncols_n,
            figsize=(5 * _ncols_n, 3.5 * _nrows_n),
            squeeze=False,
        )
        for _side_idx, (_label, _topks) in enumerate(
            (("GEN", _topks_gen), ("RECO", _topks_reco))
        ):
            for _i in range(len(_topks)):
                _row = _side_idx * int(np.ceil(_n_iter_n / _ncols_n)) + _i // _ncols_n
                _col = _i % _ncols_n
                _ax_n = _axes_n[_row, _col]
                _entry_n = _topks[_i]
                _pth_n = _entry_n["pth_bin"]
                _rw_n = _entry_n["reweights"]
                _pre_n = _entry_n["pre_weights"]
                _unique_n, _counts_n = np.unique(_pth_n, return_counts=True)
                _ax_n.bar(_unique_n, _counts_n, color="C0", alpha=0.6, label="count")
                _ax_n.set_xlabel("pth_bin")
                _ax_n.set_ylabel(f"# top-K {_label} events")
                _ax_n2 = _ax_n.twinx()
                _mean_post_n = np.array(
                    [
                        float((_pre_n[_pth_n == _b] * _rw_n[_pth_n == _b]).mean())
                        for _b in _unique_n
                    ]
                )
                _ax_n2.plot(
                    _unique_n,
                    _mean_post_n,
                    marker="o",
                    color="C3",
                    label="mean(pre*reweight)",
                )
                _ax_n2.set_yscale("log")
                _ax_n2.set_ylabel("mean(pre*reweight) [log]")
                _ax_n.set_title(
                    f"{_label} iter {_i} (winner = {_entry_n['winner_var']})"
                )
        # Hide any unused subplots.
        for _row in range(_nrows_n):
            for _col in range(_ncols_n):
                _slot_idx_in_side = (
                    _row % int(np.ceil(_n_iter_n / _ncols_n))
                ) * _ncols_n + _col
                _side_idx = _row // int(np.ceil(_n_iter_n / _ncols_n))
                _topks_side = _topks_gen if _side_idx == 0 else _topks_reco
                if _slot_idx_in_side >= len(_topks_side):
                    _axes_n[_row, _col].set_visible(False)
        _fig_n.suptitle("pT-hat bin attribution of top-K reweighted events")
        _fig_n.tight_layout()
    plt.gcf()
    return


@app.cell
def _(iter_diag):
    _topks_gen = iter_diag["gen_topk"]
    _topks_reco = iter_diag["reco_topk"]
    if len(_topks_gen) == 0 and len(_topks_reco) == 0:
        _fig_o = None
    else:
        _fig_o, _axes_o = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        _cmap_o = plt.get_cmap("plasma")
        for _ax_o, _topks, _title in (
            (_axes_o[0], _topks_gen, "GEN"),
            (_axes_o[1], _topks_reco, "RECO"),
        ):
            _n_o = len(_topks)
            for _i, _entry_o in enumerate(_topks):
                _color_o = _cmap_o(_i / max(1, _n_o - 1))
                _ax_o.scatter(
                    np.log10(np.clip(_entry_o["pre_weights"], 1e-300, None)),
                    np.log10(np.clip(_entry_o["reweights"], 1e-300, None)),
                    s=6,
                    alpha=0.3,
                    color=_color_o,
                    label=f"iter {_i}",
                )
            _ax_o.set_xlabel(r"$\log_{10}(\mathrm{pre\_weight})$")
            _ax_o.set_ylabel(r"$\log_{10}(\mathrm{reweight})$")
            _ax_o.set_title(f"{_title} top-K: pre-weight vs reweight")
            _ax_o.legend(loc="best", fontsize=7)
        _fig_o.tight_layout()
    plt.gcf()
    return


@app.cell
def _(
    alt_hist_count,
    alt_hist_count_err,
    arrows_written,
    bins,
    iter_hist_list,
    out_dir,
):
    if not arrows_written or len(iter_hist_list) == 0:
        print("[reverse-omniseq] cell P skipped (no arrows written or no iterations).")
        _fig_p = None
    else:
        # Re-read the four arrows, rebuild gen and reco histograms, compare to alt.
        _buf_gm = pa.memory_map(os.path.join(out_dir, "gen-matches.arrow"), "rb")
        _t_gm = pa.ipc.open_file(_buf_gm).read_all()
        _buf_mi = pa.memory_map(os.path.join(out_dir, "misses.arrow"), "rb")
        _t_mi = pa.ipc.open_file(_buf_mi).read_all()
        _buf_rm = pa.memory_map(os.path.join(out_dir, "reco-matches.arrow"), "rb")
        _t_rm = pa.ipc.open_file(_buf_rm).read_all()
        _buf_fk = pa.memory_map(os.path.join(out_dir, "fakes.arrow"), "rb")
        _t_fk = pa.ipc.open_file(_buf_fk).read_all()

        _gen_disk = pa.concat_tables([_t_gm, _t_mi])
        _reco_disk = pa.concat_tables([_t_rm, _t_fk])
        _w_gen_disk = torch.as_tensor(
            _gen_disk["weight"].to_numpy(), dtype=torch.float64
        )
        _w_reco_disk = torch.as_tensor(
            _reco_disk["weight"].to_numpy(), dtype=torch.float64
        )
        print(
            f"[cell P] disk-read sums: gen={float(_w_gen_disk.sum()):.4g}  "
            f"reco={float(_w_reco_disk.sum()):.4g}"
        )

        _nrows_p = 3
        _ncols_p = int(np.ceil(len(jet_columns) / _nrows_p))
        _fig_p = plt.figure(figsize=(_ncols_p * 5, _nrows_p * 4))
        _fig_p.suptitle(
            f"[{GENERATOR}] post-write closure (gen-from-disk vs alt target)",
            fontsize=18,
        )
        _subfigs_p = _fig_p.subfigures(_nrows_p, _ncols_p)
        _chi2_pre = []
        _chi2_post_mem = []
        _chi2_post_disk = []
        for _ivar, _var in enumerate(jet_columns):
            _irow = int(np.floor(_ivar / _ncols_p))
            _icol = _ivar % _ncols_p
            _subfig_p = _subfigs_p[_irow, _icol]
            _hist_disk, _ = Histogram.create(
                (torch.as_tensor(_gen_disk[_var].to_numpy(), dtype=torch.float64),),
                bins=(bins[_var],),
                weights=_w_gen_disk,
                return_binned_data=True,
            )
            _snap_disk = _hist_disk.snapshot()
            _gen_disk_count = _snap_disk.values[1:-1]
            _gen_disk_err = _snap_disk.variances.sqrt_()[1:-1]

            _h_last = iter_hist_list[-1][_var]
            _h_first = iter_hist_list[0][_var]

            _left = col_hist_args[_var]["chi2_left_edge"]
            _right = col_hist_args[_var]["chi2_right_edge"]
            _sl = slice(_left, _right)
            _alt_sliced = alt_hist_count[_var][_sl]
            _gen_pre_sliced = _h_first["gen"][_sl]
            _gen_post_mem_sliced = _h_last["gen"][_sl]
            _gen_post_disk_sliced = _gen_disk_count[_sl]
            # Rescale shape-only as in the loop.
            _gen_pre_resc = _gen_pre_sliced.div(_gen_pre_sliced.sum()).mul_(
                _alt_sliced.sum()
            )
            _gen_post_mem_resc = _gen_post_mem_sliced.div(
                _gen_post_mem_sliced.sum()
            ).mul_(_alt_sliced.sum())
            _gen_post_disk_resc = _gen_post_disk_sliced.div(
                _gen_post_disk_sliced.sum()
            ).mul_(_alt_sliced.sum())
            _c_pre, _ = chisquare(_gen_pre_resc, _alt_sliced)
            _c_post_mem, _ = chisquare(_gen_post_mem_resc, _alt_sliced)
            _c_post_disk, _ = chisquare(_gen_post_disk_resc, _alt_sliced)
            _chi2_pre.append(float(_c_pre))
            _chi2_post_mem.append(float(_c_post_mem))
            _chi2_post_disk.append(float(_c_post_disk))

            _axs_p = plot_ratios(
                _subfig_p,
                torch.as_tensor(bins[_var]),
                [alt_hist_count[_var]],
                [_gen_disk_count, _h_first["gen"]],
                [
                    [
                        (alt_hist_count[_var] / _gen_disk_count),
                        (alt_hist_count[_var] / _h_first["gen"]),
                    ]
                ],
                [alt_hist_count_err[_var]],
                [_gen_disk_err, _h_first["gen_err"]],
                [
                    [None, None],
                ],
                labels1=["alt (target)"],
                labels2=["gen (from disk)", "gen (iter=0)"],
                markers1=["o"],
                markers2=["s", "v"],
            )
            _axs_p[0].set_title(
                f"chi2: pre={_c_pre:.2f}  post_mem={_c_post_mem:.2f}  "
                f"post_disk={_c_post_disk:.2f}"
            )
            _axs_p[0].legend(fontsize=7)
            _axs_p[1].set_xlabel(_var)
        print(
            "[cell P] summary chi² (mean across vars):\n"
            f"  pre        = {np.nanmean(_chi2_pre):.4g}\n"
            f"  post_mem   = {np.nanmean(_chi2_post_mem):.4g}\n"
            f"  post_disk  = {np.nanmean(_chi2_post_disk):.4g}\n"
            f"  max |post_mem - post_disk| (disk-IO sanity) = "
            f"{np.nanmax(np.abs(np.array(_chi2_post_mem) - np.array(_chi2_post_disk))):.4g}"
        )
    plt.gcf()
    return


@app.cell
def _(
    arrows_written,
    gen_pth_bin,
    gen_weights_final,
    gen_weights_initial,
    reco_pth_bin,
    reco_weights_final,
    reco_weights_initial,
):
    if not arrows_written:
        print("[reverse-omniseq] cell Q skipped (arrows not written).")
        _fig_q = None
    else:
        # Per-jet ratios as ratio_of_normalized_weights. Both initial and final
        # tensors are normalized to num_samples, so the ratio is the multiplicative
        # cumulative reweight (without the global scale).
        _g_init = gen_weights_initial.detach().cpu().numpy().astype(np.float64)
        _r_init = reco_weights_initial.detach().cpu().numpy().astype(np.float64)
        _g_fin = gen_weights_final.detach().cpu().numpy().astype(np.float64)
        _r_fin = reco_weights_final.detach().cpu().numpy().astype(np.float64)
        # Avoid division by zero; mark zero-initial entries as r=1 (no movement)
        # purely for the diagnostic — the actual splice keeps zero rows at zero.
        _r_gen = np.where(_g_init > 0, _g_fin / np.maximum(_g_init, 1e-300), 1.0)
        _r_reco = np.where(_r_init > 0, _r_fin / np.maximum(_r_init, 1e-300), 1.0)

        _ess_pre_gen = float(_g_init.sum() ** 2 / (_g_init**2).sum())
        _ess_post_gen = float(_g_fin.sum() ** 2 / (_g_fin**2).sum())
        _ess_pre_reco = float(_r_init.sum() ** 2 / (_r_init**2).sum())
        _ess_post_reco = float(_r_fin.sum() ** 2 / (_r_fin**2).sum())
        print(
            f"[cell Q] ESS / n  gen pre={_ess_pre_gen / len(_g_init):.3f}  "
            f"post={_ess_post_gen / len(_g_fin):.3f}\n"
            f"           reco pre={_ess_pre_reco / len(_r_init):.3f}  "
            f"post={_ess_post_reco / len(_r_fin):.3f}  (target >= 0.50)"
        )

        _pth_g = gen_pth_bin.numpy()
        _pth_r = reco_pth_bin.numpy()
        _unique_g = np.unique(_pth_g)
        _unique_r = np.unique(_pth_r)

        print("[cell Q] per-pth_bin gen-side  r̄  (unweighted / weighted by init)")
        print(f"    {'pth_bin':>8}  {'n':>10}  {'<r>_unwt':>10}  {'<r>_wt':>10}")
        _gen_unwt = []
        _gen_wt = []
        for _b in _unique_g:
            _sel = _pth_g == _b
            _n_sel = int(_sel.sum())
            _r_unwt = float(_r_gen[_sel].mean()) if _n_sel > 0 else float("nan")
            _w_init_sel = _g_init[_sel]
            _r_wt = (
                float((_r_gen[_sel] * _w_init_sel).sum() / _w_init_sel.sum())
                if float(_w_init_sel.sum()) > 0
                else float("nan")
            )
            _gen_unwt.append(_r_unwt)
            _gen_wt.append(_r_wt)
            print(f"    {int(_b):>8d}  {_n_sel:>10d}  {_r_unwt:>10.4f}  {_r_wt:>10.4f}")

        print("[cell Q] per-pth_bin reco-side r̄  (unweighted / weighted by init)")
        print(f"    {'pth_bin':>8}  {'n':>10}  {'<r>_unwt':>10}  {'<r>_wt':>10}")
        _reco_unwt = []
        _reco_wt = []
        for _b in _unique_r:
            _sel = _pth_r == _b
            _n_sel = int(_sel.sum())
            _r_unwt = float(_r_reco[_sel].mean()) if _n_sel > 0 else float("nan")
            _w_init_sel = _r_init[_sel]
            _r_wt = (
                float((_r_reco[_sel] * _w_init_sel).sum() / _w_init_sel.sum())
                if float(_w_init_sel.sum()) > 0
                else float("nan")
            )
            _reco_unwt.append(_r_unwt)
            _reco_wt.append(_r_wt)
            print(f"    {int(_b):>8d}  {_n_sel:>10d}  {_r_unwt:>10.4f}  {_r_wt:>10.4f}")

        _fig_q, _axes_q = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        _axes_q[0].bar(
            np.arange(len(_unique_g)) - 0.2, _gen_unwt, width=0.4, label="unwt"
        )
        _axes_q[0].bar(
            np.arange(len(_unique_g)) + 0.2, _gen_wt, width=0.4, label="wt by init"
        )
        _axes_q[0].set_xticks(np.arange(len(_unique_g)))
        _axes_q[0].set_xticklabels([str(int(b)) for b in _unique_g])
        _axes_q[0].axhline(1.0, color="k", linestyle="--", lw=0.7)
        _axes_q[0].set_xlabel("gen_pth_bin")
        _axes_q[0].set_ylabel("r̄_gen")
        _axes_q[0].set_title("GEN per-pth_bin reweight")
        _axes_q[0].legend()

        _axes_q[1].bar(
            np.arange(len(_unique_r)) - 0.2, _reco_unwt, width=0.4, label="unwt"
        )
        _axes_q[1].bar(
            np.arange(len(_unique_r)) + 0.2, _reco_wt, width=0.4, label="wt by init"
        )
        _axes_q[1].set_xticks(np.arange(len(_unique_r)))
        _axes_q[1].set_xticklabels([str(int(b)) for b in _unique_r])
        _axes_q[1].axhline(1.0, color="k", linestyle="--", lw=0.7)
        _axes_q[1].set_xlabel("reco_pth_bin")
        _axes_q[1].set_title("RECO per-pth_bin reweight")
        _axes_q[1].legend()
        _fig_q.tight_layout()
    plt.gcf()
    return


if __name__ == "__main__":
    app.run()
