import marimo

__generated_with = "0.19.9"
app = marimo.App(width="columns")

with app.setup:
    import os
    import json

    import pyarrow as pa
    import torch

    import matplotlib.pyplot as plt

    from systematics import SysVar, get_jet_pt_bins
    from utils.accumulator import Accumulator

    jet_columns = (
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
    )


@app.function
def main(table, prefix, binning_json_file):
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    weights = torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)

    with open(binning_json_file, "rb") as file:
        bins = json.load(file)
    hvars = ("pt", "ch_ang_k1_b1", "sd_ch_ang_k1_b1")
    data = torch.stack(
        [torch.as_tensor(table[var].to_numpy(), dtype=torch.float32) for var in hvars],
        dim=0,
    )

    hbins = (jpt_bins, bins["ch_ang_k1_b1"], bins["ch_ang_k1_b1"])
    hist = Accumulator.create(*hbins)
    hist.fill(data, weights=weights)
    return hist.histogram(), hist.profile(2)


@app.cell
def _():
    input_root_dir = os.path.join(
        "./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding",
        str(SysVar.NONE),
    )
    binning_json_file = "./runtime-files/bins_p00.02_N100000.json"

    py6_buffers = []
    py6_buffers.append(pa.memory_map(os.path.join(input_root_dir, "gen-matches.arrow")))
    py6_match_table = pa.ipc.open_file(py6_buffers[-1]).read_all()
    py6_buffers.append(pa.memory_map(os.path.join(input_root_dir, "misses.arrow")))
    py6_misses_table = pa.ipc.open_file(py6_buffers[-1]).read_all()
    py6_table = pa.concat_tables((py6_match_table, py6_misses_table))

    hist, profile = main(py6_table, "pythia6", binning_json_file)

    hist_2d = hist.project(0, 1)
    hist_2d_sd = hist.project(0, 2)
    fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey="row")
    pfig, paxs = plt.subplots(1, 4, figsize=(20, 5), sharey="row")
    jpt_bins = (10, 15, 20, 40, 80)

    for iax, (ax, pax) in enumerate(zip(axs, paxs)):
        ax.errorbar(
            hist_2d.axes[1].bin_centers[1:-1],
            hist_2d[iax + 1].values[1:-1],
            label="incl",
        )
        ax.errorbar(
            hist_2d_sd.axes[1].bin_centers[1:-1],
            hist_2d_sd[iax + 1].values[1:-1],
            label="sd",
        )
        pax.errorbar(
            profile.axes[1].bin_centers[1:-1],
            profile[iax + 1].values[1:-1],
        )
        ax.set_title(f"{jpt_bins[iax]}-{jpt_bins[iax + 1]}")
        # ax.set_aspect("equal")
    ax.legend()
    # mo.mpl.interactive(fig)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
