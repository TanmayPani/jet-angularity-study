import os
import json

import pyarrow as pa
import torch

import matplotlib.pyplot as plt

from preprocessing import preprocess_data
from histograms import write_histograms_nominal_like
from systematics import SysVar, get_jet_pt_bins, get_unfolding_iter

jet_columns = (
    "ch_ang_k1_b0.5",
    "ch_ang_k1_b1",
    "ch_ang_k1_b2",
    "ch_ang_k2_b0",
)


def main(table, prefix, binning_json_file):
    jpt_bins = get_jet_pt_bins(SysVar.NONE)
    weights = torch.as_tensor(table["weight"].to_numpy(), dtype=torch.float32)
    weights_sq = weights * weights

    x_arr = torch.as_tensor(table["pt"].to_numpy(), dtype=torch.float32)

    print("Reading histogram bins from:", binning_json_file)
    with open(binning_json_file, "rb") as file:
        bins = json.load(file)

    for var_name in jet_columns:
        print(f"Setting up 2D histograms for {var_name}...")
        save_path = os.path.join("outputs/histograms", prefix, var_name)
        write_histograms_nominal_like(
            save_path,
            jpt_bins,
            torch.as_tensor(bins[var_name], dtype=torch.float32),
            x_arr,
            torch.as_tensor(table[var_name].to_numpy(), dtype=torch.float32),
            torch.as_tensor(table[f"sd_{var_name}"].to_numpy(), dtype=torch.float32),
            weights,
            weights_sq,
            False,
        )


if __name__ == "__main__":
    print("Computing histograms for pythia8 and pythia6")
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
    # main(py6_table, "pythia6", binning_json_file)

    py8_dir = "./datasets/STAR_pp200GeV_production_2012"
    py8_file = "Pythia8_pp200GeV.arrow"
    py8_path = os.path.join(py8_dir, f"preproc_{py8_file}")

    preprocess_data(py8_dir, py8_file)
    py8_buffer = pa.memory_map(py8_path, "rb")
    py8_table = pa.ipc.open_file(py8_buffer).read_all()
    main(py8_table, "pythia8", binning_json_file)

    # plt.hist(py8_table["pt"].to_numpy(), bins=25, range=(10, 60), histtype="step", density=True)
    # plt.hist(py8_table["pt"].to_numpy(), weights=py8_table["weight"].to_numpy(), bins=25, range=(10, 60), histtype="step", density=True)
    # plt.yscale("log")
    # plt.show()

    # print(py8_table[0:10])
    print("Done!")
