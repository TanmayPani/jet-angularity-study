import marimo

__generated_with = "0.20.2"
app = marimo.App(width="columns")

with app.setup:
    import os
    import json

    import numpy as np
    import pyarrow as pa
    import torch

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
    from thoda import Profile, Histogram, Snapshot

    common_vars = (
        "pt",
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

    sys_var = SysVar.NONE


@app.function
def histogram(table, bins, cols, weights):
    hist, _ = Histogram.create(
        [table[col].to_numpy() for col in cols],
        bins=[bins[key] for key in cols],
        weights=weights,
        axis_names=cols,
    )
    return hist


@app.function
def profile(table, bins, cols, xcol, weights):
    prof, _ = Profile.create(
        [table[col].to_numpy() for col in cols],
        bins=[bins[key] for key in cols],
        y=table[xcol].to_numpy(),
        weights=weights,
        axis_names=cols,
    )
    return prof


@app.function
def snapshot_state_dict(hsnap, batched=False):
    sdict = dict(
        bin_center=hsnap.bin_centers[1:-1],
        half_bin_width=hsnap.bin_widths[1:-1] / 2.0,
        bin_count=hsnap.values[1:-1] if not batched else hsnap.values[:, 1:-1].mean(0),
        bin_count_err=hsnap.variances[1:-1].sqrt()
        if not batched
        else hsnap.variances[:, 1:-1].sqrt().mean(0),
    )

    if batched:
        sdict["bin_count_std"] = hsnap.values[:, 1:-1].std(0)
    return sdict


@app.function
def save_snapshot(hsnap, prefix=".", fname="hist", batched=False):
    os.makedirs(prefix, exist_ok=True)
    sdict = snapshot_state_dict(hsnap, batched=batched)
    torch.save(sdict, os.path.join(prefix, f"{fname}.pt"))


@app.function
def ratio_snapshot(hn_snap, hd_snap):
    bin_centers = hn_snap.bin_centers
    bin_widths = hn_snap.bin_widths
    ratio = hn_snap.values / hd_snap.values
    rel_err2_num = hn_snap.variances / (hn_snap.values) ** 2
    rel_err2_den = hd_snap.variances / (hd_snap.values) ** 2
    ratio_err2 = ratio**2 * (rel_err2_den + rel_err2_num)

    return Snapshot(bin_centers, bin_widths, ratio, ratio_err2)


@app.function
def save_ratio_2d(
    hist_num, hist_den, prefix=".", fname_prefix="ratio", batched=False, true_hists=None
):
    num_h1ds = hist_num.unbind("pt")[1:-1]
    den_h1ds = hist_den.unbind("pt")[1:-1]
    if true_hists is not None:
        true_num_h1ds = true_hists[0].unbind("pt")[1:-1]
        true_den_h1ds = true_hists[1].unbind("pt")[1:-1]
    else:
        true_num_h1ds = (None,) * len(num_h1ds)
        true_den_h1ds = (None,) * len(den_h1ds)

    for ijpt, (num_h1d, den_h1d, true_num_h1d, true_den_h1d) in enumerate(
        zip(num_h1ds, den_h1ds, true_num_h1ds, true_den_h1ds)
    ):
        fname = f"{fname_prefix}_jpt{ijpt}"
        ratio_snap = ratio_snapshot(num_h1d.snapshot(), den_h1d.snapshot())
        if true_num_h1d is not None and true_den_h1d is not None:
            save_snapshot(
                ratio_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )
            true_ratio_snap = ratio_snapshot(
                true_num_h1d.snapshot(), true_den_h1d.snapshot()
            )
            save_snapshot(
                true_ratio_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )
            ratio_rsnap = ratio_snapshot(ratio_snap, true_ratio_snap)
            save_snapshot(
                ratio_rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )

        else:
            save_snapshot(ratio_snap, prefix=prefix, fname=fname, batched=batched)


@app.function
def save_hist_2d(hist, prefix=".", fname_prefix="hist", batched=False, true_hist=None):
    h1ds = hist.unbind("pt")[1:-1]
    if true_hist is not None:
        true_h1ds = true_hist.unbind("pt")[1:-1]
    else:
        true_h1ds = (None,) * len(h1ds)

    for ijpt, (h1d, true_h1d) in enumerate(zip(h1ds, true_h1ds)):
        # print(h1d.shape)
        fname = f"{fname_prefix}_jpt{ijpt}"
        h1d_snap = h1d.snapshot()
        if true_h1d is not None:
            # print(true_h1d.shape)
            save_snapshot(
                h1d_snap,
                prefix=os.path.join(prefix, "unfolded"),
                fname=fname,
                batched=batched,
            )

            true_h1d_snap = true_h1d.snapshot()
            save_snapshot(
                true_h1d_snap,
                prefix=os.path.join(prefix, "truth"),
                fname=fname,
                batched=False,
            )

            rsnap = ratio_snapshot(h1d_snap, true_h1d_snap)
            save_snapshot(
                rsnap,
                prefix=os.path.join(prefix, "ratio"),
                fname=fname,
                batched=batched,
            )
        else:
            save_snapshot(h1d_snap, prefix=prefix, fname=fname, batched=batched)


@app.cell
def _():
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as file:
        bins = json.load(file)
    bins["pt"] = get_jet_pt_bins(sys_var)
    for _ang in angularities:
        bins[f"sd_{_ang}"] = bins[_ang]

    bins["m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    bins["sd_m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    return (bins,)


@app.cell
def _():
    _source_dir = "./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding"

    if sys_var in {SysVar.NONE, SysVar.TOWER_ET_CORRECTION, SysVar.TRACK_EFFICIENCY}:
        _input_root_dir = os.path.join(_source_dir, str(sys_var))
    else:
        _input_root_dir = os.path.join(_source_dir, str(SysVar.NONE))

    _buffers = []
    _buffers.append(pa.memory_map(os.path.join(_input_root_dir, "gen-matches.arrow")))
    _gen_match_table = pa.ipc.open_file(_buffers[-1]).read_all()
    _buffers.append(pa.memory_map(os.path.join(_input_root_dir, "misses.arrow")))
    _gen_misses_table = pa.ipc.open_file(_buffers[-1]).read_all()
    gen_table = pa.concat_tables((_gen_match_table, _gen_misses_table))
    return (gen_table,)


@app.cell
def _(gen_table):
    mc_tables = []

    if sys_var == SysVar.NONE:
        _py6_weights = torch.as_tensor(
            gen_table["weight"].to_numpy(), dtype=torch.float32
        )

        mc_tables.append(("pythia6", gen_table, _py6_weights))

        _py8_dir = "./datasets/STAR_pp200GeV_production_2012"
        _py8_file = "Pythia8_pp200GeV.arrow"

        _py8_path = os.path.join(_py8_dir, f"preproc_{_py8_file}")
        _py8_buffer = pa.memory_map(_py8_path, "rb")
        _py8_table = pa.ipc.open_file(_py8_buffer).read_all()
        _py8_weights = torch.as_tensor(
            _py8_table["weight"].to_numpy(), dtype=torch.float32
        )

        mc_tables.append(("pythia8", _py8_table, _py8_weights))

        _hw_dir = "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs"
        _hw_file = "combined_HwJets_nEv500000.arrow"

        _hw_path = os.path.join(_hw_dir, _hw_file)
        _hw_buffer = pa.memory_map(_hw_path, "rb")
        _hw_table = pa.ipc.open_file(_hw_buffer).read_all()
        _hw_weights = torch.as_tensor(
            _hw_table["weight"].to_numpy(), dtype=torch.float32
        )

        mc_tables.append(("herwig7", _hw_table, _hw_weights))
    return (mc_tables,)


@app.cell
def _():
    if sys_var in {
        SysVar.TOWER_ET_CORRECTION,
        SysVar.TRACK_EFFICIENCY,
        SysVar.UNFOLDING_PRIOR,
    }:
        _unf_wts_filename: str = f"outputs/unfolding_{str(sys_var)}/w_unfolding.npz"
    else:
        _unf_wts_filename: str = f"outputs/unfolding_{str(SysVar.NONE)}/w_unfolding.npz"
    print(f"Getting unfolded weights from {_unf_wts_filename}")
    _unf_weights_dict = np.load(_unf_wts_filename)
    _iteration = get_unfolding_iter(sys_var, 5)
    unf_weights = torch.as_tensor(
        _unf_weights_dict[f"arr_{2 * _iteration}"], dtype=torch.float32
    )

    if sys_var == SysVar.UNFOLDING_PRIOR:
        _truth_weight_file = "outputs/omnisequential/omniseq-wts-iter2.npz"
        print("Reading truth weights from:", _truth_weight_file)
        _closure_wts = np.load(_truth_weight_file)
        _truth_wt_key = list(_closure_wts.keys())[-1]
        truth_weights = torch.as_tensor(
            _closure_wts[_truth_wt_key], dtype=torch.float32
        )
    else:
        truth_weights = None
    return truth_weights, unf_weights


@app.cell
def _(bins, gen_table, truth_weights, unf_weights, mc_tables):
    prefix = os.path.join("outputs", "histograms", str(sys_var))

    print("Data histograms will be saved to:", prefix)

    print("Calculating histograms for:", common_vars[1:])

    for _var in common_vars[1:]:
        # print(_var)
        _hist = histogram(gen_table, bins, ("pt", _var), unf_weights)
        _var_prefix = os.path.join(prefix, _var)

        _truth_hist = (
            histogram(gen_table, bins, ("pt", _var), truth_weights)
            if truth_weights is not None
            else None
        )
        save_hist_2d(
            _hist,
            prefix=_var_prefix,
            fname_prefix="hist",
            batched=True,
            true_hist=_truth_hist,
        )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name)
            print(f"{_mc_name} histograms will be saved to:", _mc_prefix)

            _mc_hist = histogram(_mc_table, bins, ("pt", _var), _mc_weights)
            save_hist_2d(
                _mc_hist,
                prefix=os.path.join(_mc_prefix, _var),
                fname_prefix="hist",
                batched=False,
            )
            save_ratio_2d(
                _mc_hist,
                _hist,
                prefix=os.path.join(_mc_prefix, _var),
                fname_prefix=f"ratio_data_vs_{_mc_name}",
                batched=True,
            )
            del _mc_hist

        del _hist, _truth_hist
        torch.cuda.empty_cache()

    return (prefix,)


@app.cell
def _(bins, gen_table, prefix, truth_weights, unf_weights, mc_tables):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)

        print("Calculating histograms for:", ("pt", _ang, f"sd_{_ang}"))

        _hist = histogram(gen_table, bins, ("pt", _ang, f"sd_{_ang}"), unf_weights)
        _truth_hist = (
            histogram(gen_table, bins, ("pt", _ang, f"sd_{_ang}"), truth_weights)
            if truth_weights is not None
            else None
        )

        _hist_incl = _hist.project("pt", _ang)
        _hist_sd = _hist.project("pt", f"sd_{_ang}")
        _truth_hist_incl = (
            _truth_hist.project("pt", _ang) if _truth_hist is not None else None
        )
        _truth_hist_sd = (
            _truth_hist.project("pt", f"sd_{_ang}") if _truth_hist is not None else None
        )

        save_hist_2d(
            _hist_incl,
            prefix=_ang_prefix,
            fname_prefix="hist_ang",
            batched=True,
            true_hist=_truth_hist_incl,
        )

        save_hist_2d(
            _hist_sd,
            prefix=_ang_prefix,
            fname_prefix="hist_sd_ang",
            batched=True,
            true_hist=_truth_hist_sd,
        )

        save_ratio_2d(
            _hist_sd,
            _hist_incl,
            prefix=_ang_prefix,
            fname_prefix="ratio_incl_vs_sd",
            batched=True,
            true_hists=(_truth_hist_sd, _truth_hist_incl)
            if _truth_hist_sd is not None and _truth_hist_incl is not None
            else None,
        )

        del _truth_hist_incl, _truth_hist_sd, _truth_hist

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name)
            _mc_hist = histogram(
                _mc_table, bins, ("pt", _ang, f"sd_{_ang}"), _mc_weights
            )

            _mc_hist_incl = _mc_hist.project("pt", _ang)
            save_hist_2d(
                _mc_hist_incl,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="hist_ang",
                batched=False,
            )
            save_ratio_2d(
                _mc_hist_incl,
                _hist_incl,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_incl_vs_{_mc_name}",
                batched=True,
            )

            _mc_hist_sd = _mc_hist.project("pt", f"sd_{_ang}")
            save_hist_2d(
                _mc_hist_sd,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="hist_sd_ang",
                batched=False,
            )
            save_ratio_2d(
                _mc_hist_sd,
                _hist_sd,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix=f"ratio_sd_vs_{_mc_name}",
                batched=True,
            )

            save_ratio_2d(
                _mc_hist_sd,
                _mc_hist_incl,
                prefix=os.path.join(_mc_prefix, _ang),
                fname_prefix="ratio_incl_vs_sd",
            )
            del _mc_hist_incl, _mc_hist_sd, _mc_hist

        del (
            _hist_sd,
            _hist_incl,
            _hist,
        )
        torch.cuda.empty_cache()
    return


@app.cell
def _(bins, gen_table, prefix, truth_weights, unf_weights, mc_tables):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)
        _bin_cols = (*common_vars, f"sd_{_ang}")
        print("Calculating profile histograms for", _ang, "in bins of", _bin_cols)

        _prof = profile(gen_table, bins, _bin_cols, _ang, unf_weights)
        _truth_prof = (
            profile(gen_table, bins, _bin_cols, _ang, truth_weights)
            if truth_weights is not None
            else None
        )

        for _var in _bin_cols[1:]:
            save_hist_2d(
                _prof.project("pt", _var),
                prefix=_ang_prefix,
                fname_prefix=f"prof_incl_vs_{_var}",
                batched=True,
                true_hist=_truth_prof.project("pt", _var)
                if _truth_prof is not None
                else None,
            )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name)
            _mc_prof = profile(_mc_table, bins, _bin_cols, _ang, _mc_weights)
            for _var in _bin_cols[1:]:
                save_hist_2d(
                    _mc_prof.project("pt", _var),
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"prof_incl_vs_{_var}",
                    batched=False,
                )

            del _mc_prof

        del _prof, _truth_prof
        torch.cuda.empty_cache()
    return


@app.cell
def _(bins, gen_table, prefix, truth_weights, unf_weights, mc_tables):
    for _ang in angularities:
        _ang_prefix = os.path.join(prefix, _ang)
        _bin_cols = (*common_vars, _ang)

        print(
            "Calculating profile histograms for", f"sd_{_ang}", "in bins of", _bin_cols
        )
        _prof = profile(gen_table, bins, _bin_cols, f"sd_{_ang}", unf_weights)
        _truth_prof = (
            profile(gen_table, bins, _bin_cols, f"sd_{_ang}", truth_weights)
            if truth_weights is not None
            else None
        )

        for _var in _bin_cols[1:]:
            save_hist_2d(
                _prof.project("pt", _var),
                prefix=_ang_prefix,
                fname_prefix=f"prof_sd_vs_{_var}",
                batched=True,
                true_hist=_truth_prof.project("pt", _var)
                if _truth_prof is not None
                else None,
            )

        for _mc_name, _mc_table, _mc_weights in mc_tables:
            _mc_prefix = os.path.join("outputs", "histograms", _mc_name)
            _mc_prof = profile(_mc_table, bins, _bin_cols, f"sd_{_ang}", _mc_weights)

            for _var in _bin_cols[1:]:
                save_hist_2d(
                    _mc_prof.project("pt", _var),
                    prefix=os.path.join(_mc_prefix, _ang),
                    fname_prefix=f"prof_sd_vs_{_var}",
                    batched=False,
                )

            del _mc_prof

        del _prof, _truth_prof
        torch.cuda.empty_cache()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
