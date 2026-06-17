import marimo

__generated_with = "0.23.9"
app = marimo.App(width="columns")

with app.setup:
    import json
    from pathlib import Path

    import numpy as np
    import pyarrow as pa
    import torch
    from tensordict import TensorDict
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    plt.style.use("default")
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
    from histograms import (
        closure_state_dict,
        histogram,
        profile,
        ratio_snapshot,
        snapshot_state_dict,
    )
    from plot_physics import (
        plot_data_points,
        var_xlabel,
        var_hist_ylabel,
        var_prof_ylabel,
        var_xlim,
    )

    # Switch between SAME (AB-split) and LIKE_DATA (omniseq-reweighted) closure.
    sys_var = SysVar.UNFOLDING_PRIOR_LIKE_DATA

    common_vars = ("m", "sd_m", "sd_dR", "sd_symmetry")
    angularities = (
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
    )

    from config import load_config

    _cfg_setup = load_config()
    # Observables (angularity scalars, jet mass, z_g, ...) only exist in the
    # `angularities` arrows, so the closure's physics distributions are ALWAYS
    # histogrammed from the angularities embedding.
    feature_mode = "angularities"  # observable source (gen/reco tables)
    # The unfolding output (w_unfolding.npz / index_split.npz) and the baked
    # truth weights come from the RUN's feature_mode — e.g. "bin_counts" for the
    # CNN closure. The jets are the same in the same row order across feature
    # modes, so the per-jet weights are row-aligned with the angularities
    # observables. When run_feature_mode == "angularities" this is the old path.
    run_feature_mode = _cfg_setup["feature_mode"]

    source_dir = _cfg_setup.dataset_root / "features" / feature_mode
    run_source_dir = _cfg_setup.dataset_root / "features" / run_feature_mode
    # multifold.py writes w_unfolding.npz, index_split.npz and config.json
    # alongside the embedding arrows for each sysvar (under the run feature_mode).
    unf_dir = run_source_dir / "embedding" / str(sys_var)
    fig_dir = Path("./outputs/closure_plots") / str(sys_var) / run_feature_mode

    fig_scale = 5
    best_iter = 2


@app.function
def take_table(table, indices):
    return table.take(pa.array(indices, type=pa.int64()))


@app.function
def gen_table_weight(table):
    return torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)


@app.function
def unfolded_hist_2d(table, bins, obs, weights_2d, indices):
    """Per-replica (pt, obs) histogram of a subset of `table`."""
    sub = take_table(table, indices)
    return histogram(sub, bins, ("pt", obs), weights_2d)


@app.function
def truth_hist_2d(table, bins, obs, indices, override_weights=None):
    """(pt, obs) histogram of the truth side; defaults to the table's `weight` column."""
    sub = take_table(table, indices)
    if override_weights is None:
        w = gen_table_weight(sub)
    else:
        w = override_weights[indices]
    return histogram(sub, bins, ("pt", obs), w)


@app.function
def unfolded_prof_2d(table, bins, x_var, y_obs, weights_2d, indices):
    """Per-replica profile of <y_obs> vs (pt, x_var) for a subset of `table`."""
    sub = take_table(table, indices)
    return profile(sub, bins, ("pt", x_var), y_obs, weights_2d)


@app.function
def truth_prof_2d(table, bins, x_var, y_obs, indices, override_weights=None):
    """Profile of <y_obs> vs (pt, x_var) for the truth side; defaults to `weight`."""
    sub = take_table(table, indices)
    if override_weights is None:
        w = gen_table_weight(sub)
    else:
        w = override_weights[indices]
    return profile(sub, bins, ("pt", x_var), y_obs, w)


@app.function
def pull_values(num_snap, den_snap):
    """Per-bin (n - d) / sqrt(var_n + var_d); reduces per-replica mean for batched snapshots."""
    n_vals = num_snap.values
    d_vals = den_snap.values
    n_var = num_snap.variances
    d_var = den_snap.variances
    if n_vals.dim() > 1:
        n_vals = n_vals.mean(0)
        n_var = n_var.mean(0)
    diff = n_vals - d_vals
    err = (n_var + d_var).clamp_min_(1e-30).sqrt()
    return diff / err


@app.function
def save_fig(fig, fname):
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(fig_dir / fname), bbox_inches="tight")
    print("Saved", fig_dir / fname)


@app.function
def mark_result_range(axs, var):
    """Draw vertical dashed lines on every panel of `axs` at the x-range that
    the published result plots actually use for `var` (``plot_physics.var_xlim``).
    No-op when `var` has no entry, so it is safe to call on any closure figure."""
    if var not in var_xlim:
        return
    _lo, _hi = var_xlim[var]
    for _ax in np.ravel(axs):
        for _x in (_lo, _hi):
            _ax.axvline(
                _x, color="0.35", linestyle=(0, (5, 3)), linewidth=1.0,
                alpha=0.8, zorder=0.5,
            )


@app.cell
def _():
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as _f:
        bins = json.load(_f)
    bins["pt"] = get_jet_pt_bins(SysVar.NONE)
    for _ang in angularities:
        bins[f"sd_{_ang}"] = bins[_ang]
    bins["m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    bins["sd_m"] = torch.linspace(0, 12, 13, dtype=torch.float32)

    jpt_bins = bins["pt"]
    n_pt = len(jpt_bins) - 1
    return bins, jpt_bins, n_pt


@app.cell
def _():
    # with open(unf_dir / "config.json") as _f:
    #    cfg = json.load(_f)
    num_iter = 4  # cfg["num_iterations"]
    final_iter = 2  # get_unfolding_iter(sys_var, num_iter)

    w_unf = np.load(unf_dir / "w_unfolding.npz")
    n_arrays = len(w_unf.files)
    # arr_0..arr_{2N+1}: even = gen weights, odd = reco weights.
    max_iter = (n_arrays // 2) - 1
    if final_iter > max_iter:
        print(
            f"[warn] final_iter={final_iter} > available iterations ({max_iter}); "
            f"clipping to {max_iter}."
        )
        final_iter = max_iter

    gen_unf_iter = [
        torch.as_tensor(w_unf[f"arr_{2 * i}"], dtype=torch.float32) for i in range(max_iter + 1)
    ]
    reco_unf_iter = [
        torch.as_tensor(w_unf[f"arr_{2 * i + 1}"], dtype=torch.float32) for i in range(max_iter + 1)
    ]

    print(
        f"Loaded w_unfolding.npz: {n_arrays} arrays => {max_iter} iterations. "
        f"gen weights shape = {tuple(gen_unf_iter[0].shape)}, "
        f"reco weights shape = {tuple(reco_unf_iter[0].shape)}."
    )
    return final_iter, gen_unf_iter, max_iter, reco_unf_iter


@app.cell
def _():
    # Both closure variants reuse the nominal embedding arrow files. LIKE_DATA's
    # data-side reweighting affects weights only, not the raw observables.
    _input_root = source_dir / "embedding" / str(SysVar.NONE)

    _buffers = []
    _buffers.append(pa.memory_map(str(_input_root / "gen-matches.arrow")))
    _gen_match = pa.ipc.open_file(_buffers[-1]).read_all()
    _buffers.append(pa.memory_map(str(_input_root / "misses.arrow")))
    _gen_miss = pa.ipc.open_file(_buffers[-1]).read_all()
    gen_table = pa.concat_tables((_gen_match, _gen_miss))
    n_gen_match = len(_gen_match)

    _buffers.append(pa.memory_map(str(_input_root / "reco-matches.arrow")))
    _reco_match = pa.ipc.open_file(_buffers[-1]).read_all()
    _buffers.append(pa.memory_map(str(_input_root / "fakes.arrow")))
    _reco_fake = pa.ipc.open_file(_buffers[-1]).read_all()
    reco_table = pa.concat_tables((_reco_match, _reco_fake))
    n_reco_match = len(_reco_match)

    # Buffers held in cell scope so the memmaps stay alive.
    return gen_table, reco_table


@app.cell
def _(gen_table, reco_table):
    # Build the "ordering" arrays that align arrow rows to w_unfolding rows.
    #
    # multifold packs gen weights as [w_matched | w_miss] and reco weights as
    # [w_matched | w_fake]. For the SAME closure the matched/miss/fake indices
    # come from `read_datasets(mode="ab_closure")`, recorded in TENSORDICT
    # space. The nominal tensordict concatenates [data-like | sim-like], and
    # only the sim-like half carries the original is_matched values, so every
    # recorded index falls in the sim-like half = the LAST n_sim rows. The
    # data-like half length differs from n_sim at detector level (real STAR
    # data count != embedding reco count), so the tensordict->arrow offset is
    # (len(td) - n_sim), NOT n_sim itself.
    n_gen = len(gen_table)
    n_reco = len(reco_table)

    if sys_var == SysVar.UNFOLDING_PRIOR_SAME:
        _td_root = run_source_dir / "tensordicts" / str(SysVar.NONE)
        _gen_offset = len(TensorDict.load_memmap(_td_root / "part_lvl")) - n_gen
        _reco_offset = len(TensorDict.load_memmap(_td_root / "det_lvl")) - n_reco

        _idx = np.load(unf_dir / "index_split.npz")
        # B-side indices in tensordict space -> arrow space.
        b_gen_match_arrow = (
            _idx["partlvl_matched_indices"].astype(np.int64) - _gen_offset
        )
        b_gen_miss_arrow = _idx["partlvl_missed_indices"].astype(np.int64) - _gen_offset
        b_reco_match_arrow = (
            _idx["detlvl_matched_indices"].astype(np.int64) - _reco_offset
        )
        b_reco_fake_arrow = _idx["detlvl_fake_indices"].astype(np.int64) - _reco_offset

        gen_unf_order = np.concatenate([b_gen_match_arrow, b_gen_miss_arrow])
        reco_unf_order = np.concatenate([b_reco_match_arrow, b_reco_fake_arrow])

        gen_truth_order = np.setdiff1d(
            np.arange(n_gen, dtype=np.int64), gen_unf_order, assume_unique=False
        )
        reco_truth_order = np.setdiff1d(
            np.arange(n_reco, dtype=np.int64), reco_unf_order, assume_unique=False
        )

        truth_gen_weights = None  # use the table's own `weight` column
        truth_reco_weights = None
    else:
        # LIKE_DATA / HERWIG7 / PYTHIA8: no AB split. Unfolded weights run
        # over the full nominal tables in their natural
        # [matches | misses] / [matches | fakes] order. The truth-side
        # reweighted P6 weights are baked into the prior-variant arrows by
        # omnisequential.py (LIKE_DATA) and reverse_omnisequential.py
        # (HERWIG7 / PYTHIA8), so all three variants travel the same path.
        gen_unf_order = np.arange(n_gen, dtype=np.int64)
        reco_unf_order = np.arange(n_reco, dtype=np.int64)
        gen_truth_order = gen_unf_order
        reco_truth_order = reco_unf_order

        _prior_root = run_source_dir / "embedding" / str(sys_var)
        # Buffers held in a local list to keep the memmaps alive only as long
        # as the to_numpy + np.concatenate calls run; the concatenated arrays
        # are fresh allocations, so the buffers can be released at end-of-cell.
        _prior_buffers = []
        for _fname in (
            "gen-matches.arrow",
            "misses.arrow",
            "reco-matches.arrow",
            "fakes.arrow",
        ):
            _prior_buffers.append(pa.memory_map(str(_prior_root / _fname)))
        _gm_prior = pa.ipc.open_file(_prior_buffers[0]).read_all()
        _mi_prior = pa.ipc.open_file(_prior_buffers[1]).read_all()
        _rm_prior = pa.ipc.open_file(_prior_buffers[2]).read_all()
        _fk_prior = pa.ipc.open_file(_prior_buffers[3]).read_all()

        truth_gen_weights = torch.as_tensor(
            np.concatenate(
                [_gm_prior["weight"].to_numpy(), _mi_prior["weight"].to_numpy()]
            ),
            dtype=torch.float32,
        )
        truth_reco_weights = torch.as_tensor(
            np.concatenate(
                [_rm_prior["weight"].to_numpy(), _fk_prior["weight"].to_numpy()]
            ),
            dtype=torch.float32,
        )

        # Row-alignment guard: the run feature_mode's truth weights must line up
        # 1:1 with the angularities observable rows (same jets, same order).
        assert len(truth_gen_weights) == n_gen, (
            f"gen truth-weight rows ({len(truth_gen_weights)}) != angularities "
            f"gen rows ({n_gen}); feature modes are misaligned."
        )
        assert len(truth_reco_weights) == n_reco, (
            f"reco truth-weight rows ({len(truth_reco_weights)}) != angularities "
            f"reco rows ({n_reco}); feature modes are misaligned."
        )

    print(
        f"gen: n_unfolded={len(gen_unf_order)}, n_truth={len(gen_truth_order)} "
        f"(of {n_gen}); reco: n_unfolded={len(reco_unf_order)}, "
        f"n_truth={len(reco_truth_order)} (of {n_reco})."
    )
    return (
        gen_truth_order,
        gen_unf_order,
        reco_truth_order,
        reco_unf_order,
        truth_gen_weights,
        truth_reco_weights,
    )


@app.cell
def _(
    bins,
    final_iter,
    gen_table,
    gen_truth_order,
    gen_unf_iter,
    gen_unf_order,
    truth_gen_weights,
):
    # Build per-(observable) 2-D histograms once for the final iteration.
    # Slicing into [1:-1] of the pt-unbind drops over/underflow.
    all_obs = (
        tuple(common_vars)
        + tuple(angularities)
        + tuple(f"sd_{_a}" for _a in angularities)
    )
    print(f"Building (pt, obs) histograms for {len(all_obs)} observables.")

    final_gen_unf_weights = gen_unf_iter[final_iter][:, : len(gen_unf_order)]

    unfolded_snaps = {}  # obs -> list of per-pt-bin batched snapshots
    truth_snaps = {}
    ratio_snaps = {}
    for _obs in all_obs:
        _h_unf = unfolded_hist_2d(
            gen_table, bins, _obs, final_gen_unf_weights, gen_unf_order
        )
        _h_truth = truth_hist_2d(
            gen_table, bins, _obs, gen_truth_order, truth_gen_weights
        )
        _unf_per_pt = _h_unf.unbind("pt")[1:-1]
        _truth_per_pt = _h_truth.unbind("pt")[1:-1]
        unfolded_snaps[_obs] = [_h.snapshot() for _h in _unf_per_pt]
        truth_snaps[_obs] = [_h.snapshot() for _h in _truth_per_pt]
        ratio_snaps[_obs] = [
            ratio_snapshot(_un, _tr)
            for _un, _tr in zip(unfolded_snaps[_obs], truth_snaps[_obs])
        ]
        del _h_unf, _h_truth
    return all_obs, ratio_snaps, truth_snaps, unfolded_snaps


@app.cell
def _(
    all_obs,
    bins,
    gen_table,
    gen_unf_iter,
    gen_unf_order,
    max_iter,
    n_pt,
    truth_snaps,
):
    # Closure-quality metric per iteration:
    #   chi2_i = sum_{obs, pT, bin} (mean_unf - truth)^2 / (var_unf + var_truth)
    # var_unf combines per-replica Poisson variance and the replica spread; we
    # exclude over/underflow bins to match the displayed histograms.
    iter_chi2 = []
    for _it in range(max_iter + 1):
        _w = gen_unf_iter[_it][:, : len(gen_unf_order)]
        _chi2_total = 0.0
        _ndof = 0
        for _obs in all_obs:
            _h_unf = unfolded_hist_2d(gen_table, bins, _obs, _w, gen_unf_order)
            _unf_per_pt = _h_unf.unbind("pt")[1:-1]
            for _j in range(n_pt):
                _u_snap = _unf_per_pt[_j].snapshot()
                _t_snap = truth_snaps[_obs][_j]
                _u_vals = _u_snap.values[:, 1:-1]
                _u_vars = _u_snap.variances[:, 1:-1]
                _t_vals = _t_snap.values[1:-1]
                _t_vars = _t_snap.variances[1:-1]
                _u_mean = _u_vals.mean(0)
                _u_var = _u_vars.mean(0) + _u_vals.var(0)
                _denom = (_u_var + _t_vars).clamp_min(1e-30)
                _terms = ((_u_mean - _t_vals) ** 2 / _denom).nan_to_num_(
                    nan=0.0, posinf=0.0, neginf=0.0
                )
                _chi2_total += float(_terms.sum())
                _ndof += int(((_u_var + _t_vars) > 0).sum())
            del _h_unf
        iter_chi2.append((_it, _chi2_total, _ndof))
    print(f"{'iter':>4} | {'chi2':>14} | {'ndof':>6} | {'chi2/ndof':>10}")
    print("-" * 48)
    for _it, _c, _n in iter_chi2:
        _norm = _c / _n if _n > 0 else float("nan")
        print(f"{_it:>4} | {_c:>14.4f} | {_n:>6} | {_norm:>10.5f}")
    return


@app.cell
def _(all_obs, jpt_bins, n_pt, ratio_snaps, truth_snaps, unfolded_snaps):
    # Cell A: headline figures — final-iter unfolded vs truth, with ratio
    # panel. One figure per observable; columns = pT bins.
    figs_A = {}
    for _obs in all_obs:
        _fig, _axs = plt.subplots(
            2,
            n_pt,
            figsize=(n_pt * fig_scale, fig_scale * 1.5),
            height_ratios=[3, 1],
            sharex="col",
            sharey="row",
            squeeze=False,
            gridspec_kw=dict(hspace=0, wspace=0),
        )
        for _j in range(n_pt):
            _ax_top = _axs[0, _j]
            _ax_bot = _axs[1, _j]
            _un_dict = closure_state_dict(unfolded_snaps[_obs][_j], batched=True)
            _tr_dict = snapshot_state_dict(truth_snaps[_obs][_j], batched=False)
            _ra_dict = closure_state_dict(ratio_snaps[_obs][_j], batched=True)

            plot_data_points(
                _ax_top,
                "errorbar",
                _un_dict,
                color="C0",
                marker="o",
                linestyle="none",
                label="unfolded",
            )
            plot_data_points(
                _ax_top,
                "errorbar",
                _tr_dict,
                color="C3",
                marker="^",
                linestyle="none",
                label="truth",
            )
            plot_data_points(
                _ax_bot,
                "errorbar",
                _ra_dict,
                color="black",
                marker="*",
                linestyle="none",
            )
            _ax_bot.axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
            _ax_bot.set_ylim(0.0, 2.0)
            _ax_bot.set_xlabel(var_xlabel.get(_obs, _obs), fontsize="large")
            _ax_bot.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            _ax_top.set_title(
                rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$"
            )
        _axs[0, 0].set_ylabel(var_hist_ylabel.get(_obs, _obs), fontsize="large")
        _axs[1, 0].set_ylabel(r"$\frac{unf.}{truth}$", fontsize="large")
        _axs[0, -1].legend(frameon=False, loc="upper right")
        mark_result_range(_axs, _obs)
        save_fig(_fig, f"{_obs}_ratio.pdf")
        figs_A[_obs] = _fig
    figs_A
    return


@app.cell
def _(
    all_obs,
    bins,
    gen_table,
    gen_unf_iter,
    gen_unf_order,
    jpt_bins,
    max_iter,
    n_pt,
    truth_snaps,
):
    # Cell B: ratio at the chosen single iteration (best_iter).
    figs_B = {}
    _iter_to_plot = min(best_iter, max_iter)
    for _obs in all_obs:
        _fig, _axs = plt.subplots(
            1,
            n_pt,
            figsize=(n_pt * fig_scale, fig_scale),
            sharex="col",
            sharey=True,
            squeeze=False,
            gridspec_kw=dict(wspace=0),
        )
        _w = gen_unf_iter[_iter_to_plot][:, : len(gen_unf_order)]
        _h_unf = unfolded_hist_2d(gen_table, bins, _obs, _w, gen_unf_order)
        _per_pt = _h_unf.unbind("pt")[1:-1]
        for _j in range(n_pt):
            _ratio_snap = ratio_snapshot(_per_pt[_j].snapshot(), truth_snaps[_obs][_j])
            _ra_dict = closure_state_dict(_ratio_snap, batched=True)
            plot_data_points(
                _axs[0, _j],
                "plot",
                _ra_dict,
                color="C0",
                marker=".",
                linestyle="-",
                label=f"iter {_iter_to_plot}" if _j == n_pt - 1 else None,
            )
            # stat (+) replica band around the ratio line
            _axs[0, _j].fill_between(
                _ra_dict["bin_center"],
                _ra_dict["bin_count"] - _ra_dict["bin_count_std"],
                _ra_dict["bin_count"] + _ra_dict["bin_count_std"],
                color="C0",
                alpha=0.2,
                linewidth=0,
            )
        del _h_unf
        for _j in range(n_pt):
            _axs[0, _j].axhline(
                1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5
            )
            _axs[0, _j].set_ylim(0.25, 1.75)
            _axs[0, _j].set_xlabel(var_xlabel.get(_obs, _obs), fontsize="large")
            _axs[0, _j].xaxis.set_major_formatter(FormatStrFormatter("%g"))
            _axs[0, _j].set_title(
                rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$"
            )
        _axs[0, 0].set_ylabel(r"$\frac{unf.}{truth}$", fontsize="large")
        #_axs[0, -1].legend(frameon=False, loc="upper right", fontsize="small")
        mark_result_range(_axs, _obs)
        save_fig(_fig, f"{_obs}_iter{_iter_to_plot}.pdf")
        figs_B[_obs] = _fig
    figs_B
    return


@app.cell
def _(all_obs, jpt_bins, n_pt, truth_snaps, unfolded_snaps):
    # Cell C: pull plots — (unf - truth) / sqrt(var_unf + var_truth) per bin.
    figs_C = {}
    for _obs in all_obs:
        _fig, _axs = plt.subplots(
            1,
            n_pt,
            figsize=(n_pt * fig_scale, fig_scale),
            sharex="col",
            sharey=True,
            squeeze=False,
            gridspec_kw=dict(wspace=0),
        )
        for _j in range(n_pt):
            _pulls = pull_values(unfolded_snaps[_obs][_j], truth_snaps[_obs][_j])
            _centers = unfolded_snaps[_obs][_j].bin_centers[1:-1]
            _widths = unfolded_snaps[_obs][_j].bin_widths[1:-1]
            _pulls_inner = _pulls[1:-1]
            _ax = _axs[0, _j]
            _ax.bar(
                _centers.numpy(),
                _pulls_inner.numpy(),
                width=_widths.numpy(),
                color="C2",
                edgecolor="black",
                linewidth=0.5,
                alpha=0.7,
            )
            _ax.axhspan(-1, 1, color="grey", alpha=0.2, zorder=0)
            _ax.axhspan(-2, 2, color="grey", alpha=0.1, zorder=0)
            _ax.axhline(0.0, color="black", linewidth=0.8)
            _ax.set_ylim(-5, 5)
            _ax.set_xlabel(var_xlabel.get(_obs, _obs), fontsize="large")
            _ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            _ax.set_title(rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$")
        _axs[0, 0].set_ylabel(
            r"$(\mathrm{unf.} - \mathrm{truth})/\sigma$", fontsize="large"
        )
        mark_result_range(_axs, _obs)
        save_fig(_fig, f"{_obs}_pulls.pdf")
        figs_C[_obs] = _fig
    figs_C
    return


@app.cell
def _(
    bins,
    final_iter,
    jpt_bins,
    n_pt,
    reco_table,
    reco_truth_order,
    reco_unf_iter,
    reco_unf_order,
    truth_reco_weights,
):
    # Cell D: detector-level closure — unfolded reco vs data-target reco for
    # the jet-pt observable only.
    _obs_det = "pt"
    final_reco_unf_weights = reco_unf_iter[final_iter][:, : len(reco_unf_order)]
    _h_unf_reco = unfolded_hist_2d(
        reco_table, bins, _obs_det, final_reco_unf_weights, reco_unf_order
    )
    _h_truth_reco = truth_hist_2d(
        reco_table, bins, _obs_det, reco_truth_order, truth_reco_weights
    )
    _unf_per_pt = _h_unf_reco.unbind("pt")[1:-1]
    _truth_per_pt = _h_truth_reco.unbind("pt")[1:-1]

    _fig, _axs = plt.subplots(
        2,
        n_pt,
        figsize=(n_pt * fig_scale, fig_scale * 1.5),
        height_ratios=[3, 1],
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw=dict(hspace=0, wspace=0),
    )
    for _j in range(n_pt):
        _un_snap = _unf_per_pt[_j].snapshot()
        _tr_snap = _truth_per_pt[_j].snapshot()
        _ra_snap = ratio_snapshot(_un_snap, _tr_snap)
        _un_dict = closure_state_dict(_un_snap, batched=True)
        _tr_dict = snapshot_state_dict(_tr_snap, batched=False)
        _ra_dict = closure_state_dict(_ra_snap, batched=True)

        plot_data_points(
            _axs[0, _j],
            "errorbar",
            _un_dict,
            color="C0",
            marker="o",
            linestyle="none",
            label="unfolded reco",
        )
        plot_data_points(
            _axs[0, _j],
            "errorbar",
            _tr_dict,
            color="C3",
            marker="^",
            linestyle="none",
            label="data target",
        )
        plot_data_points(
            _axs[1, _j],
            "errorbar",
            _ra_dict,
            color="black",
            marker="*",
            linestyle="none",
        )
        _axs[1, _j].axhline(1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        _axs[1, _j].set_ylim(0.0, 2.0)
        _axs[1, _j].set_xlabel(var_xlabel.get(_obs_det, _obs_det), fontsize="large")
        _axs[1, _j].xaxis.set_major_formatter(FormatStrFormatter("%g"))
        _axs[0, _j].set_title(
            rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$"
        )
    _axs[0, 0].set_ylabel(var_hist_ylabel.get(_obs_det, _obs_det), fontsize="large")
    _axs[1, 0].set_ylabel(r"$\frac{unf. reco}{data}$", fontsize="large")
    _axs[0, -1].legend(frameon=False, loc="upper right")
    mark_result_range(_axs, _obs_det)
    save_fig(_fig, "det_level_closure.pdf")
    return


@app.cell
def _(
    bins,
    final_iter,
    gen_table,
    gen_truth_order,
    gen_unf_iter,
    gen_unf_order,
    truth_gen_weights,
):
    # Cell E: profile closure — <angularity> profiled vs an x-variable, in pT
    # bins, comparing the final-iter unfolded (reweighted) profile against the
    # truth profile. Mirrors plot_physics.plot_profile but unfolded-vs-truth.
    # For each angularity we profile both the inclusive (`<ang>`) and softdrop
    # (`sd_<ang>`) observable against each x-variable.
    prof_pairs = [
        (_ang, _x) for _ang in angularities for _x in (*common_vars, _ang)
    ]
    _final_w = gen_unf_iter[final_iter][:, : len(gen_unf_order)]

    # keyed by (y_obs, x_var) -> list of per-pt-bin snapshots
    prof_unf_snaps = {}
    prof_truth_snaps = {}
    prof_ratio_snaps = {}

    def _build_prof(y_obs, x_var):
        if (y_obs, x_var) in prof_unf_snaps:
            return
        _h_unf = unfolded_prof_2d(
            gen_table, bins, x_var, y_obs, _final_w, gen_unf_order
        )
        _h_tru = truth_prof_2d(
            gen_table, bins, x_var, y_obs, gen_truth_order, truth_gen_weights
        )
        _u = [_h.snapshot() for _h in _h_unf.unbind("pt")[1:-1]]
        _t = [_h.snapshot() for _h in _h_tru.unbind("pt")[1:-1]]
        prof_unf_snaps[(y_obs, x_var)] = _u
        prof_truth_snaps[(y_obs, x_var)] = _t
        prof_ratio_snaps[(y_obs, x_var)] = [
            ratio_snapshot(_un, _tr) for _un, _tr in zip(_u, _t)
        ]
        del _h_unf, _h_tru

    print(f"Building profile-closure snapshots for {len(prof_pairs)} (ang, x) pairs.")
    for _ang, _x in prof_pairs:
        _build_prof(_ang, _x)  # inclusive angularity profile
        _build_prof(f"sd_{_ang}", _x)  # softdrop angularity profile
    return prof_pairs, prof_ratio_snaps, prof_truth_snaps, prof_unf_snaps


@app.cell
def _(
    jpt_bins,
    n_pt,
    prof_pairs,
    prof_ratio_snaps,
    prof_truth_snaps,
    prof_unf_snaps,
):
    # Cell F: profile-closure figures. Top row = inclusive (red) and softdrop
    # (blue) angularity profiles, unfolded (filled) vs truth (open); rows 2/3 =
    # unfolded/truth ratios for the inclusive and softdrop profiles. Unfolded /
    # ratio error bars are the bootstrap (replica-ensemble spread) only — for
    # profiles we do NOT fold in the per-replica statistical error on the mean,
    # so these use snapshot_state_dict (bin_count_std = replica std), not
    # closure_state_dict.
    figs_E = {}
    for _ang, _x in prof_pairs:
        _fig, _axs = plt.subplots(
            3,
            n_pt,
            figsize=(n_pt * fig_scale, fig_scale * 2),
            height_ratios=[3, 1, 1],
            sharex="col",
            sharey="row",
            squeeze=False,
            gridspec_kw=dict(hspace=0, wspace=0),
        )
        for _j in range(n_pt):
            for _pref, _color, _lbl, _rrow in (
                ("", "red", "incl.", 1),
                ("sd_", "blue", "SD", 2),
            ):
                _yobs = f"{_pref}{_ang}"
                _un_dict = snapshot_state_dict(prof_unf_snaps[(_yobs, _x)][_j], batched=True)
                _tr_dict = snapshot_state_dict(
                    prof_truth_snaps[(_yobs, _x)][_j], batched=False
                )
                _ra_dict = snapshot_state_dict(
                    prof_ratio_snaps[(_yobs, _x)][_j], batched=True
                )
                plot_data_points(
                    _axs[0, _j], "errorbar", _un_dict,
                    color=_color, marker="o", linestyle="none",
                    label=f"unfolded ({_lbl})" if _j == n_pt - 1 else None,
                )
                plot_data_points(
                    _axs[0, _j], "errorbar", _tr_dict,
                    color=_color, marker="^", markerfacecolor="none", linestyle="none",
                    label=f"truth ({_lbl})" if _j == n_pt - 1 else None,
                )
                plot_data_points(
                    _axs[_rrow, _j], "errorbar", _ra_dict,
                    color=_color, marker="*", linestyle="none",
                )
                _axs[_rrow, _j].axhline(
                    1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5
                )
                _axs[_rrow, _j].set_ylim(0.5, 1.5)
            _axs[2, _j].set_xlabel(var_xlabel.get(_x, _x), fontsize="large")
            _axs[2, _j].xaxis.set_major_formatter(FormatStrFormatter("%g"))
            _axs[0, _j].set_title(
                rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$"
            )
        _axs[0, 0].set_ylabel(var_prof_ylabel.get(_ang, _ang), fontsize="large")
        _axs[1, 0].set_ylabel(r"$\frac{unf.}{truth}$ (incl.)", fontsize="large")
        _axs[2, 0].set_ylabel(r"$\frac{unf.}{truth}$ (SD)", fontsize="large")
        _axs[0, -1].legend(frameon=False, loc="upper right", fontsize="small")
        mark_result_range(_axs, _x)
        save_fig(_fig, f"prof_{_ang}_vs_{_x}_closure.pdf")
        figs_E[(_ang, _x)] = _fig
    figs_E
    return


@app.cell
def _(
    bins,
    gen_table,
    gen_unf_iter,
    gen_unf_order,
    jpt_bins,
    max_iter,
    n_pt,
    prof_pairs,
    prof_truth_snaps,
):
    # Cell G: profile-closure ratio-only at the chosen single iteration
    # (best_iter). One single-row figure per (y_obs, x_var) — matching the 1-D
    # histogram ratio-only style (e.g. sd_ch_ang_k1_b2_iter1.pdf) — emitted for
    # both the inclusive (`<ang>`) and softdrop (`sd_<ang>`) angularity. The
    # ratio band is the bootstrap (replica-ensemble spread) only — no per-replica
    # statistical error folded in — so this uses snapshot_state_dict.
    figs_G = {}
    _iter_to_plot = min(best_iter, max_iter)
    _w = gen_unf_iter[_iter_to_plot][:, : len(gen_unf_order)]
    # expand each (ang, x) pair into its inclusive and softdrop y-observable
    _yobs_pairs = []
    for _ang, _x in prof_pairs:
        _yobs_pairs.append((_ang, _x))
        _yobs_pairs.append((f"sd_{_ang}", _x))
    for _yobs, _x in _yobs_pairs:
        _fig, _axs = plt.subplots(
            1,
            n_pt,
            figsize=(n_pt * fig_scale, fig_scale),
            sharex="col",
            sharey=True,
            squeeze=False,
            gridspec_kw=dict(wspace=0),
        )
        _h_unf = unfolded_prof_2d(gen_table, bins, _x, _yobs, _w, gen_unf_order)
        _per_pt = _h_unf.unbind("pt")[1:-1]
        for _j in range(n_pt):
            _ratio_snap = ratio_snapshot(
                _per_pt[_j].snapshot(), prof_truth_snaps[(_yobs, _x)][_j]
            )
            _ra_dict = snapshot_state_dict(_ratio_snap, batched=True)
            plot_data_points(
                _axs[0, _j], "plot", _ra_dict,
                color="C0", marker=".", linestyle="-",
                label=f"iter {_iter_to_plot}" if _j == n_pt - 1 else None,
            )
            _axs[0, _j].fill_between(
                _ra_dict["bin_center"],
                _ra_dict["bin_count"] - _ra_dict["bin_count_std"],
                _ra_dict["bin_count"] + _ra_dict["bin_count_std"],
                color="C0", alpha=0.2, linewidth=0,
            )
        del _h_unf
        for _j in range(n_pt):
            _axs[0, _j].axhline(
                1.0, color="grey", linestyle="--", linewidth=1, alpha=0.5
            )
            _axs[0, _j].set_ylim(0.25, 1.75)
            _axs[0, _j].set_xlabel(var_xlabel.get(_x, _x), fontsize="large")
            _axs[0, _j].xaxis.set_major_formatter(FormatStrFormatter("%g"))
            _axs[0, _j].set_title(
                rf"${jpt_bins[_j]} < p_{{T}} < {jpt_bins[_j + 1]}$ GeV/$c$"
            )
        _axs[0, 0].set_ylabel(r"$\frac{unf.}{truth}$", fontsize="large")
        mark_result_range(_axs, _x)
        save_fig(_fig, f"prof_{_yobs}_vs_{_x}_iter{_iter_to_plot}.pdf")
        figs_G[(_yobs, _x)] = _fig
    figs_G
    return


if __name__ == "__main__":
    app.run()
