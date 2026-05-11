import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import os
    import json
    import copy
    from collections import defaultdict

    import numpy as np
    import pyarrow as pa
    import pandas as pd
    from matplotlib import pyplot as plt

    import torch
    from scipy.stats import chisquare
    from sklearn.gaussian_process import GaussianProcessRegressor

    from thoda import Histogram, Profile, bayesian_blocks

    from sklearn.decomposition import PCA, KernelPCA

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
        # "sd_pt",
        # "sd_m",
        # "sd_dR",
        # "sd_symmetry",
        # "sd_ch_ang_k1_b0.5",
        # "sd_ch_ang_k1_b1",
        # "sd_ch_ang_k1_b2",
        # "sd_ch_ang_k2_b0",
    ]
    return KernelPCA, jet_columns, os, pa


@app.cell
def _(jet_columns, os, pa):
    _source_dir = "./datasets/STAR_pp200GeV_production_2012/clustered_jets"
    _data_buffer = pa.memory_map(os.path.join(_source_dir, "preproc_data.arrow"), "rb")
    data_table = pa.ipc.open_file(_data_buffer).read_all().select(jet_columns)
    data_df = data_table.to_pandas()
    return (data_df,)


@app.cell
def _(data_df):
    data_df.head()
    return


@app.cell
def _(KernelPCA, data_df):
    _pca_transformer = KernelPCA()
    data_subset = data_df.sample(frac=0.1)
    transformed_data_subset = _pca_transformer.fit_transform(data_subset)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
