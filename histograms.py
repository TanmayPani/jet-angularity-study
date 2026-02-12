import marimo

__generated_with = "0.19.11"
app = marimo.App(width="columns")

with app.setup:
    import os
    import json

    import numpy as np
    import pyarrow as pa
    import torch
    import matplotlib.pyplot as plt

    from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter
    from utils.accumulator import Accumulator

    common_vars = (
        "pt",
        # "m",
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
def get_accumulator(table : pa.Table, cols : tuple[str, ...], bins : dict[str, torch.Tensor], weights : torch.Tensor | np.ndarray,) -> Accumulator:
    if not isinstance(weights, torch.Tensor):
        weights = torch.as_tensor(weights, dtype=torch.float32)

    bin_tensors = (*(bins[key] for key in cols),)
    acc = Accumulator.create(
        *bin_tensors, num_weights=(weights.shape[0] if weights.ndim > 1 else 1)
    )

    data = (
            *(
                torch.as_tensor(table[col].to_numpy(), dtype=torch.float32)
                for col in cols
            ),
        )

    acc.fill(data, weights=weights)
    return acc


@app.function
def plot_2d_from_Nd_profiles(
    ijpt,
    ix,
    *profiles,
    labels=None,
    xlabel=None,

):
    profiles_2d = (*(prof.project(ijpt, ix) for prof in profiles),)
    if not isinstance(labels, (tuple, list)):
        labels = (labels,) * len(profiles)

    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey="row")
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    for iax, ax in enumerate(axs):
        for iprof, prof in enumerate(profiles_2d):
            prof_1d = prof[iax + 1]
            ax.errorbar(
                prof_1d.axes[0].bin_centers[1:-1],
                prof_1d.values[:, 1:-1].mean(0),
                label=labels[iprof],
            )
        ax.set_title(f"{jpt_bins[iax]}-{jpt_bins[iax + 1]}")
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        ax.legend()
    return fig, axs


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
    unf_weights = torch.as_tensor(_unf_weights_dict[f"arr_{2 * _iteration}"], dtype=torch.float32)
    return (unf_weights,)


@app.cell
def _():
    _binning_json_file = "./runtime-files/bins_p00.02_N100000.json"
    print("Reading histogram bins from:", _binning_json_file)
    with open(_binning_json_file, "rb") as file:
        bins = json.load(file)
    bins["pt"] = get_jet_pt_bins(sys_var)
    return (bins,)


@app.cell
def _(bins, gen_table, unf_weights):
    ang = "ch_ang_k1_b1"
    variables = (*common_vars, ang, f"sd_{ang}")
    acc = get_accumulator(gen_table, variables, bins, unf_weights)
    hist_nd = acc.histogram()
    profile_incl = acc.profile(len(acc.axes) - 2)
    profile_sd = acc.profile(len(acc.axes) - 1)
    return profile_incl, profile_sd


@app.cell
def _(profile_incl, profile_sd):
    _fig, _axs = plot_2d_from_Nd_profiles(
        0,
        1,
        profile_incl,
        profile_sd,
        labels=("incl.", "sd"),
        xlabel=common_vars[1],
    )
    plt.gcf()
    return


@app.cell
def _(profile_incl, profile_sd):
    _fig, _axs = plot_2d_from_Nd_profiles(
        0,
        2,
        profile_incl,
        profile_sd,
        labels=("incl.", "sd"),
        xlabel=common_vars[2],
    )
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
