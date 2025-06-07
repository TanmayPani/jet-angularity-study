from typing import Optional
from datetime import datetime

import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch
import pyarrow as pa

from torchmodel import archs
from torchmodel import torchmodel
from torchmodel import datasets
from torchmodel import callbacks


def train_val_split(
    features,
    labels,
    sample_weights=None,
    val_size=0.2,
):
    if sample_weights is None:
        sample_weights = np.ones(len(features))

    indices = np.arange(len(features))

    train_indices, val_indices = train_test_split(
        indices, test_size=val_size, stratify=labels
    )

    train_features = features.take(train_indices)
    train_labels = labels[train_indices]
    train_sample_weights = sample_weights[train_indices]

    val_features = features.take(val_indices)
    val_labels = labels[val_indices]
    val_sample_weights = sample_weights[val_indices]

    train_dataset = datasets.JetDataset(
        train_features,
        train_labels,
        sample_weights=train_sample_weights,
        do_scale=True,
    )
    val_dataset = datasets.JetDataset(
        val_features,
        val_labels,
        sample_weights=val_sample_weights,
        do_scale=True,
        scale_from=train_dataset,
    )

    return train_dataset, val_dataset


def make_data_loaders(train_dataset, val_dataset, batch_size=5000, num_dl_workers=-1):
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        pin_memory=True,
        collate_fn=datasets.batch_collate,
        num_workers=num_dl_workers,
    )

    val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        pin_memory=True,
        collate_fn=datasets.batch_collate,
        num_workers=num_dl_workers,
    )

    return train_loader, val_loader


def fit(
    model,
    features,
    labels,
    sample_weights=None,
    in_keys=[("features")],
    out_key="targets",
    val_size=0.2,
    batch_size=100000,
    num_dl_workers=10,
    num_epochs=10,
    model_weights_file=None,
    checkpoint_folder: str = "outputs/checkpoints",
):
    train_dataset, val_dataset = train_val_split(
        features, labels, sample_weights=sample_weights, val_size=val_size
    )
    train_loader, val_loader = make_data_loaders(
        train_dataset, val_dataset, batch_size=batch_size, num_dl_workers=num_dl_workers
    )

    print(f"Training for {num_epochs} epochs ...")

    early_stopping = callbacks.EarlyStopping(
        patience=20,
        checkpointFolder=f"{checkpoint_folder}/{datetime.now().strftime('%Y%m%d_%H%M')}",
        checkPointFileName="model-epoch[EPOCH].pth",
    )

    if model_weights_file is not None:
        model.load(model_weights_file, weights_only=True)

    history = model.fit(
        train_loader,
        val_loader,
        epochs=num_epochs,
        in_keys=in_keys,
        out_key=out_key,
        early_stopping=early_stopping,
    )

    return history


def predict_reweight(
    model,
    features,
    in_keys=[("features")],
    batch_size=100000,
    num_dl_workers=0,
    scale_from_ds=None,
):
    predict_dataset = datasets.JetDataset(
        features, np.zeros(len(features)), scale_from=scale_from_ds
    )
    predict_sampler = torch.utils.data.SequentialSampler(predict_dataset)
    predict_loader = torch.utils.data.DataLoader(
        predict_dataset,
        batch_size=batch_size,
        sampler=predict_sampler,
        pin_memory=True,
        collate_fn=datasets.batch_collate,
        num_workers=num_dl_workers,
    )

    predictions = (
        model.predict(
            predict_loader, in_keys=in_keys, out_activation=torch.nn.Sigmoid()
        )
        .cpu()
        .detach()
        .numpy()
    )

    return predictions / (1 - predictions + 1e-20)


def get_model(dnn_layers, device=torch.device("cpu")):
    classifier = torchmodel.TensorDictModel(
        archs.DNN(dnn_layers, dropout_prob=0.2, do_batch_norm=True), device=device
    )
    optimizer = torch.optim.Adam(classifier().parameters(), lr=0.01)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    classifier.compile(
        criterion=torch.nn.BCEWithLogitsLoss(reduction="none"),
        optimizer=optimizer,
        compile_mode=None,
        scheduler=lr_scheduler,
    )

    return classifier


def process_data(column_names, data_folder="outputs", phase_space_bin_folders=None):
    if phase_space_bin_folders is None:
        ptHatBins = ["11", "15", "20", "25", "35", "45", "55", "infty"]
        phase_space_bin_folders = [
            f"ptHat{pth_low}to{pth_high}"
            for pth_low, pth_high in zip(ptHatBins[:-1], ptHatBins[1:])
        ]

    match_buffer_list = [
        pa.memory_map(f"{data_folder}/{phsp_bin_folder}/matches.arrow", "rb")
        for phsp_bin_folder in phase_space_bin_folders
    ]
    miss_buffer_list = [
        pa.memory_map(f"{data_folder}/{phsp_bin_folder}/misses.arrow", "rb")
        for phsp_bin_folder in phase_space_bin_folders
    ]
    fake_buffer_list = [
        pa.memory_map(f"{data_folder}/{phsp_bin_folder}/fakes.arrow", "rb")
        for phsp_bin_folder in phase_space_bin_folders
    ]

    match_table = pa.concat_tables(
        [pa.ipc.open_file(buffer).read_all() for buffer in match_buffer_list]
    )
    miss_table = pa.concat_tables(
        [pa.ipc.open_file(buffer).read_all() for buffer in miss_buffer_list]
    )
    fake_table = pa.concat_tables(
        [pa.ipc.open_file(buffer).read_all() for buffer in fake_buffer_list]
    )

    data_buffer = pa.memory_map(f"{data_folder}/jets.arrow")

    data_column_names = [f"data_{col}" for col in column_names]
    data_table = (
        pa.ipc.open_file(data_buffer)
        .read_all()
        .select(data_column_names)
        .rename_columns(column_names)
    )

    reco_column_names = [f"reco_{col}" for col in column_names]
    reco_table = pa.concat_tables(
        [match_table.select(reco_column_names), fake_table.select(reco_column_names)]
    ).rename_columns(column_names)
    is_reco_matched = np.concatenate(
        [np.ones(len(match_table)), np.zeros(len(fake_table))]
    )
    reco_table = reco_table.append_column("is_matched", [is_reco_matched])

    gen_column_names = [f"gen_{col}" for col in column_names]
    gen_table = pa.concat_tables(
        [match_table.select(gen_column_names), miss_table.select(gen_column_names)]
    ).rename_columns(column_names)
    is_gen_matched = np.concatenate(
        [np.ones(len(match_table)), np.zeros(len(miss_table))]
    )
    gen_table = gen_table.append_column("is_matched", [is_gen_matched])

    return data_table, reco_table, gen_table


def main(device: torch.device = torch.device("cuda")):
    jet_columns = [
        "pt",
        "nef",
        "ch_ang_k1_b0.5",
        "ch_ang_k1_b1",
        "ch_ang_k1_b2",
        "ch_ang_k2_b0",
    ]
    constit_columns = ["constit_pt", "constit_eta", "constit_phi", "constit_charge"]

    data_table, reco_table, gen_table = process_data(["weight"] + jet_columns)
    is_reco_matched = reco_table["is_matched"].to_numpy() > 0.5
    is_gen_matched = gen_table["is_matched"].to_numpy() > 0.5

    reco_matched_indices = np.arange(len(reco_table))[is_reco_matched]
    reco_fake_indices = np.arange(len(reco_table))[~is_reco_matched]
    gen_matched_indices = np.arange(len(gen_table))[is_gen_matched]
    gen_miss_indices = np.arange(len(gen_table))[~is_gen_matched]

    det_lvl_table = pa.concat_tables(
        [data_table.select(jet_columns), reco_table.select(jet_columns)]
    )
    det_lvl_label = np.concatenate(
        [np.ones(len(data_table)), np.zeros(len(reco_table))]
    )
    data_weights = data_table["weight"].to_numpy()
    reco_weights = reco_table["weight"].to_numpy()
    data_weights = data_weights / np.sum(data_weights) * len(det_lvl_table) / 2.0
    reco_weights = reco_weights / np.sum(reco_weights) * len(det_lvl_table) / 2.0

    gen_lvl_table = pa.concat_tables(
        [gen_table.select(jet_columns), gen_table.select(jet_columns)]
    )
    gen_lvl_label = np.concatenate([np.ones(len(gen_table)), np.zeros(len(gen_table))])
    gen_weights = gen_table["weight"].to_numpy()
    gen_weights = gen_weights / np.sum(gen_weights) * len(gen_lvl_table) / 2.0
    gen_lvl_weights = np.concatenate([gen_weights, gen_weights])

    dnn_layers = [len(jet_columns), 256, 256, 256, 1]

    det_lvl_model = get_model(dnn_layers, device=device)
    gen_lvl_model = get_model(dnn_layers, device=device)
    fake_rew_model = get_model(dnn_layers, device=device)
    miss_rew_model = get_model(dnn_layers, device=device)

    n_iter = 2
    prev_gen_wt = -2

    unfolded_wts = [reco_weights]

    for iter in range(0, n_iter):
        det_lvl_weights = np.concatenate([data_weights, unfolded_wts[-1]])
        det_lvl_history = fit(
            det_lvl_model, det_lvl_table, det_lvl_label, sample_weights=det_lvl_weights
        )

        pull_reweights = predict_reweight(det_lvl_model, reco_table.select(jet_columns))


if __name__ == "__main__":
    main()
