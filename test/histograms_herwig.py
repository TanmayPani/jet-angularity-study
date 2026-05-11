import marimo

__generated_with = "0.20.4"
app = marimo.App()

with app.setup:
    import os
    import json

    import pyarrow as pa
    import torch

    from systematics import SysVar, get_jet_pt_bins

    from histograms import (
        histogram,
        profile,
        save_snapshot,
        save_ratio_2d,
        save_hist_2d,
    )

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


@app.cell
def _():
    with open("./runtime-files/bins_p00.02_N100000.json", "rb") as file:
        bins = json.load(file)
    for _ang in angularities:
        bins[f"sd_{_ang}"] = bins[_ang]

    bins["m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    bins["sd_m"] = torch.linspace(0, 12, 13, dtype=torch.float32)
    return (bins,)


@app.cell
def _():
    _hw_dir = (
        "/home/tanmaypani/star-workspace/mc_generators/herwig7/JobScripts/RHIC/outputs"
    )
    _hw_file = "combined_HwJets_nEv500000.arrow"

    _hw_path = os.path.join(_hw_dir, _hw_file)
    _hw_buffer = pa.memory_map(_hw_path, "rb")
    hw_table = pa.ipc.open_file(_hw_buffer).read_all()
    hw_weights = torch.as_tensor(hw_table["weight"].to_numpy(), dtype=torch.float32)
    return hw_table, hw_weights


@app.cell
def _(
    bins,
    hw_table,
    hw_weights,
):
    hw_prefix = os.path.join("outputs", "histograms", "herwig")

    print("PYTHIA8 histograms will be saved to:", hw_prefix)

    print("Calculating histograms for:", common_vars[1:])

    print(bins["pt"])
    _hw_hist_pt = histogram(hw_table, bins, ("pt",), hw_weights)
    save_snapshot(
        _hw_hist_pt.snapshot(),
        prefix=os.path.join(hw_prefix, "pt"),
        batched=False,
    )

    bins["pt"] = get_jet_pt_bins(sys_var)
    for _var in common_vars[1:]:
        # print(_var)
        if hw_table is not None:
            _hw_hist = histogram(hw_table, bins, ("pt", _var), hw_weights)
            save_hist_2d(
                _hw_hist,
                prefix=os.path.join(hw_prefix, _var),
                fname_prefix="hist",
                batched=False,
            )
            del _hw_hist

        torch.cuda.empty_cache()

    return hw_prefix


@app.cell
def _(
    bins,
    hw_prefix,
    hw_table,
    hw_weights,
):
    for _ang in angularities:
        if hw_table is not None:
            _hw_hist = histogram(hw_table, bins, ("pt", _ang, f"sd_{_ang}"), hw_weights)
            _hw_hist_incl = _hw_hist.project("pt", _ang)
            save_hist_2d(
                _hw_hist_incl,
                prefix=os.path.join(hw_prefix, _ang),
                fname_prefix="hist_ang",
                batched=False,
            )

            _hw_hist_sd = _hw_hist.project("pt", f"sd_{_ang}")
            save_hist_2d(
                _hw_hist_sd,
                prefix=os.path.join(hw_prefix, _ang),
                fname_prefix="hist_sd_ang",
                batched=False,
            )
            save_ratio_2d(
                _hw_hist_sd,
                _hw_hist_incl,
                prefix=os.path.join(hw_prefix, _ang),
                fname_prefix="ratio_incl_vs_sd",
            )

            del _hw_hist_incl, _hw_hist_sd, _hw_hist

        torch.cuda.empty_cache()
    return


@app.cell
def _(
    bins,
    hw_prefix,
    hw_table,
    hw_weights,
):
    for _ang in angularities:
        _bin_cols = (*common_vars, f"sd_{_ang}")
        print("Calculating profile histograms for", _ang, "in bins of", _bin_cols)

        _hw_prof = (
            profile(hw_table, bins, _bin_cols, _ang, hw_weights)
            if hw_table is not None
            else None
        )

        for _var in _bin_cols[1:]:
            if _hw_prof is not None:
                save_hist_2d(
                    _hw_prof.project("pt", _var),
                    prefix=os.path.join(hw_prefix, _ang),
                    fname_prefix=f"prof_incl_vs_{_var}",
                    batched=False,
                )
        del _hw_prof
        torch.cuda.empty_cache()
    return


@app.cell
def _(
    bins,
    hw_prefix,
    hw_table,
    hw_weights,
):
    for _ang in angularities:
        _bin_cols = (*common_vars, _ang)

        _hw_prof = (
            profile(hw_table, bins, _bin_cols, f"sd_{_ang}", hw_weights)
            if sys_var == SysVar.NONE
            else None
        )

        for _var in _bin_cols[1:]:
            if _hw_prof is not None:
                save_hist_2d(
                    _hw_prof.project("pt", _var),
                    prefix=os.path.join(hw_prefix, _ang),
                    fname_prefix=f"prof_sd_vs_{_var}",
                    batched=False,
                )

        del _hw_prof
        torch.cuda.empty_cache()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
