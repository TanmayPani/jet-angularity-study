import os
import shutil
import json

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid
from tensordict import TensorDict

from churten.stateless import StatelessModule
from churten.optimizer import Adam
from churten.utils.nn.archs import MLP
from churten.utils.nn.transform import Normalize
from churten.callbacks import Checkpoint, EarlyStopping
from churten.utils.data import TensorBatchSampler

from dataset import (
    TensorDictDataset,
    get_stacked_batch_loader,
    classwise_undersample_and_split,
)
from systematics import SysVar


def reweight_inference_loaders(
    data: TensorDictDataset,
    batch_size: int = 32,
    num_replicas: int = 2,
    drop_last: bool = False,
    load_only_first=None,
    has_unmatched: bool = False,
):
    all_indices = torch.arange(len(data))
    neg_indices = all_indices[data.target < 0.5]

    if not has_unmatched:
        sampler = TensorBatchSampler(
            torch.stack([neg_indices] * num_replicas),
            batch_size=batch_size,
            batch_dim=1,
            drop_last=drop_last,
        )
        return get_stacked_batch_loader(data, sampler, load_only_first=load_only_first)

    matched_mask = data.td["is_matched"] == 1
    unmatched_mask = data.td["is_matched"] == 0

    matched_indices = all_indices[matched_mask]
    unmatched_indices = all_indices[unmatched_mask]

    matched_sampler = TensorBatchSampler(
        torch.stack([matched_indices] * num_replicas),
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )
    matched_loader = get_stacked_batch_loader(
        data, matched_sampler, load_only_first=load_only_first
    )

    unmatched_sampler = TensorBatchSampler(
        torch.stack([unmatched_indices] * num_replicas),
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )
    unmatched_loader = get_stacked_batch_loader(
        data, unmatched_sampler, load_only_first=load_only_first
    )

    return (matched_loader, unmatched_loader)


def train_test_multi_loaders(
    data: TensorDictDataset,
    *,
    num_replicas: int,
    batch_size: int,
    train_size: float = 0.5,
    undersample_size: float | int = 1.0,
    stratifys: tuple[bool, bool] = (False, False),
    vmap_randomness: str = "different",
    vmap_in_dims: int | tuple = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
    pos_mask: torch.Tensor | None = None,
    neg_mask: torch.Tensor | None = None,
    load_only_first: int | None = None,
):
    pos_mask = pos_mask or data.target > 0.5
    neg_mask = neg_mask or data.target < 0.5

    all_indices = torch.arange(len(data))
    pos_indices = all_indices[pos_mask]
    neg_indices = all_indices[neg_mask]

    print(
        f"--- Undersampling and Train-Test splitting {pos_indices.shape[0]} positive samples and "
        f"{neg_indices.shape[0]} negative samples, with {num_replicas} replicas."
    )

    pos_stratify = data.td["pth_bin"][data.indices][pos_mask] if stratifys[0] else None
    neg_stratify = data.td["pth_bin"][data.indices][neg_mask] if stratifys[1] else None

    stacked_train_indices, stacked_valid_indices = classwise_undersample_and_split(
        pos_indices,
        neg_indices,
        undersample_size=undersample_size,
        split_sizes=(train_size, 1.0 - train_size),
        pos_stratify=pos_stratify,
        neg_stratify=neg_stratify,
        num_replicas=num_replicas,
        vmap_randomness=vmap_randomness,
        vmap_in_dims=vmap_in_dims,
        generator=generator,
    )

    print(
        "------ Subsampled "
        f"{str(undersample_size * 100) + '%' if isinstance(undersample_size, float) else 2 * undersample_size}, "
        f"with Train size: {stacked_train_indices.shape} and Valid size: {stacked_valid_indices.shape}"
    )

    train_sampler = TensorBatchSampler(
        stacked_train_indices,
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )
    valid_sampler = TensorBatchSampler(
        stacked_valid_indices,
        batch_size=batch_size,
        batch_dim=1,
        drop_last=drop_last,
    )

    train_loader = get_stacked_batch_loader(
        data,
        train_sampler,
        load_only_first=load_only_first,
    )
    valid_loader = get_stacked_batch_loader(
        data,
        valid_sampler,
        load_only_first=load_only_first,
    )

    return train_loader, valid_loader


if __name__ == "__main__":
    with open("runtime-files/config.json", "r") as config_file:
        cfg = json.load(config_file)

    generator = torch.Generator().manual_seed(cfg["dataseed"])

    sys_var = SysVar.UNFOLDING_PRIOR
    sys_var_dir = str(sys_var)
    src = f"./datasets/STAR_pp200GeV_production_2012/clustered_jets/embedding/{sys_var_dir}/tensordicts"
    print("Reading files from:", src)

    detlvl_src = f"{src}/det_lvl"
    detlvl_td = TensorDict.load_memmap(detlvl_src)
    print("Loaded (data, reco) TensorDict from:", detlvl_src)

    detlvl_ds = TensorDictDataset(
        detlvl_td, is_categorical=True, num_replicas=cfg["num_replicas"]
    )
    detlvl_pos_mask = detlvl_ds.target > 0.5
    detlvl_neg_mask = detlvl_ds.target < 0.5

    detlvl_matched_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 1]
    fake_scaling_indices = torch.cat([detlvl_matched_indices] * 2)
    fake_scaling_targets = torch.cat(
        [
            torch.ones_like(detlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(detlvl_matched_indices, dtype=torch.float32),
        ]
    )
    fake_scaling_weights = torch.ones_like(fake_scaling_indices, dtype=torch.float32)
    fake_scaling_ds = TensorDictDataset(
        detlvl_td,
        indices=fake_scaling_indices,
        target=fake_scaling_targets,
        sample_weight=fake_scaling_weights,
        is_categorical=True,
        num_replicas=cfg["num_replicas"],
    )
    fake_scaling_pos_mask = fake_scaling_ds.target > 0.5
    fake_scaling_neg_mask = fake_scaling_ds.target < 0.5

    detlvl_train_loader, detlvl_valid_loader = train_test_multi_loaders(
        detlvl_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        stratifys=(False, True),
    )
    fake_scaling_train_loader, fake_scaling_valid_loader = train_test_multi_loaders(
        fake_scaling_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        stratifys=(True, True),
    )
    reco_match_loader, reco_fake_loader = reweight_inference_loaders(
        detlvl_ds,
        batch_size=cfg["batch_size"] * 5,
        num_replicas=cfg["num_replicas"],
        has_unmatched=True,
    )
    print(
        "Initialized loaders to get"
        " f : p(reco) --> p(data) and"
        " g : p(reco fake) --> p(reco match)"
    )

    partlvl_src = f"{src}/part_lvl"
    partlvl_td = TensorDict.load_memmap(partlvl_src)
    print("Loaded particle level TensorDict from:", partlvl_src)

    partlvl_ds = TensorDictDataset(
        partlvl_td, is_categorical=True, num_replicas=cfg["num_replicas"]
    )
    partlvl_pos_mask = partlvl_ds.target > 0.5
    partlvl_neg_mask = partlvl_ds.target < 0.5

    partlvl_matched_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 1]
    miss_scaling_indices = torch.cat([partlvl_matched_indices] * 2)
    miss_scaling_targets = torch.cat(
        [
            torch.ones_like(partlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(partlvl_matched_indices, dtype=torch.float32),
        ]
    )
    miss_scaling_weights = torch.ones_like(miss_scaling_indices, dtype=torch.float32)
    miss_scaling_ds = TensorDictDataset(
        partlvl_td,
        indices=miss_scaling_indices,
        target=miss_scaling_targets,
        sample_weight=miss_scaling_weights,
        is_categorical=True,
        num_replicas=cfg["num_replicas"],
    )
    miss_scaling_pos_mask = miss_scaling_ds.target > 0.5
    miss_scaling_neg_mask = miss_scaling_ds.target < 0.5

    partlvl_train_loader, partlvl_valid_loader = train_test_multi_loaders(
        partlvl_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        stratifys=(True, True),
    )
    miss_scaling_train_loader, miss_scaling_valid_loader = train_test_multi_loaders(
        miss_scaling_ds,
        train_size=cfg["train_size"],
        undersample_size=cfg["num_data_subsample"],
        batch_size=cfg["batch_size"],
        num_replicas=cfg["num_replicas"],
        generator=generator,
        stratifys=(True, True),
    )
    gen_match_loader, gen_miss_loader = reweight_inference_loaders(
        partlvl_ds,
        batch_size=cfg["batch_size"] * 5,
        num_replicas=cfg["num_replicas"],
        has_unmatched=True,
    )
    print(
        "Initialized loaders to get"
        " f : p(gen) --> p(unfolded) and"
        " g : p(gen miss) --> p(gen match)"
    )

    n_data = detlvl_ds.td[detlvl_ds.td["is_data"]].size(0)
    n_reco = detlvl_ds.td[detlvl_ds.td["is_data"].logical_not()].size(0)
    n_gen = partlvl_ds.td.size(0)

    print("Number of data jets:", n_data)
    print("Number of gen jets, reco jets:", n_gen, n_reco)

    num_features = 23
    layer_sizes = [num_features, 256, 256, 256, 1]
    device = "cuda"

    optimizer_kwargs = dict(
        lr=1e-3,
        eps=1e-8,
        weight_decay=0.01,
        decoupled_weight_decay=True,
    )

    torch.manual_seed(cfg["modelseed"])

    detlvl_std, detlvl_mean = torch.std_mean(detlvl_ds.input)
    detlvl_ensemble, _, detlvl_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": Normalize(detlvl_mean.to(device), detlvl_std.to(device)),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    gen_match_std, gen_match_mean = torch.std_mean(
        partlvl_ds.input[partlvl_matched_indices]
    )
    miss_scaling_ensemble, _, miss_scaling_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": Normalize(
                gen_match_mean.to(device), gen_match_std.to(device)
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    partlvl_std, partlvl_mean = torch.std_mean(partlvl_ds.input)
    partlvl_ensemble, _, partlvl_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": Normalize(
                partlvl_mean.to(device), partlvl_std.to(device)
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    reco_match_std, reco_match_mean = torch.std_mean(
        detlvl_ds.input[detlvl_matched_indices]
    )

    fake_scaling_ensemble, _, fake_scaling_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": Normalize(
                reco_match_mean.to(device), reco_match_std.to(device)
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    w_unfolding = [
        partlvl_ds.sample_weight[..., partlvl_neg_mask],
        detlvl_ds.sample_weight[..., detlvl_neg_mask],
    ]

    w_data = (
        detlvl_td["weight"][detlvl_td["is_data"]]
        .expand(cfg["num_replicas"], -1)
        .clone()
    )
    w_matched = (
        partlvl_td["weight"][partlvl_td["is_matched"] == 1]
        .expand(cfg["num_replicas"], -1)
        .clone()
    )
    w_miss = (
        partlvl_td["weight"][partlvl_td["is_matched"] == 0]
        .expand(cfg["num_replicas"], -1)
        .clone()
    )
    w_fake = (
        detlvl_td["weight"][detlvl_td["is_matched"] == 0]
        .expand(cfg["num_replicas"], -1)
        .clone()
    )

    # dir_prefix = f"./outputs/unfolding_{sys_var_dir}_{filename_stub}"
    dir_prefix = f"./outputs/unfolding_{sys_var_dir}"
    if not os.path.exists(dir_prefix):
        print("Creating folder", dir_prefix, "...")
        os.makedirs(dir_prefix)

    print("Saving config file to folder", dir_prefix, "...")
    shutil.copy("runtime-files/config.json", f"{dir_prefix}/config.json")

    early_stopping_kwargs = dict(patience=10)

    for iteration in range(cfg["num_iterations"]):
        print(
            "###############################################################################################"
        )
        print(f"Iteration: {iteration + 1}/{cfg['num_iterations']}")
        print(
            "###############################################################################################"
        )

        detlvl_ds.sample_weight = torch.cat([w_data, w_unfolding[-1]], dim=1)
        detlvl_history = detlvl_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits,
            detlvl_state,
            detlvl_train_loader,
            valid_iterator=detlvl_valid_loader,
            num_epochs=cfg["num_epochs"],
            callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
            randomness="different",
        )

        reco_match_preds = detlvl_ensemble.predict(
            detlvl_state, reco_match_loader, non_linearity=sigmoid
        )
        reco_match_reweights = (
            reco_match_preds / (1.0 - reco_match_preds + 1e-20)
        ).squeeze_()
        fake_preds = detlvl_ensemble.predict(
            detlvl_state, reco_fake_loader, non_linearity=sigmoid
        )
        fake_reweights = (fake_preds / (1.0 - fake_preds + 1e-20)).squeeze_()

        detlvl_state.reset_status()

        w_matched *= reco_match_reweights
        w_fake *= fake_reweights

        miss_scaling_ds.sample_weight[..., miss_scaling_pos_mask] = reco_match_reweights
        miss_scaling_history = miss_scaling_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits,
            miss_scaling_state,
            miss_scaling_train_loader,
            valid_iterator=miss_scaling_valid_loader,
            num_epochs=cfg["num_epochs"],
            callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
            randomness="different",
        )

        miss_preds = miss_scaling_ensemble.predict(
            miss_scaling_state, gen_miss_loader, non_linearity=sigmoid
        )
        miss_reweights = (miss_preds / (1.0 - miss_preds + 1e-20)).squeeze_()

        miss_scaling_state.reset_status()

        w_miss *= miss_reweights

        gen_weights = torch.cat([w_matched, w_miss], dim=1)
        w_unfolding.append(
            gen_weights.mul(n_data).div_(gen_weights.sum(1).unsqueeze_(-1))
        )

        partlvl_ds.sample_weight = torch.cat([w_unfolding[-1], w_unfolding[-3]], dim=1)
        partlvl_history = partlvl_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits,
            partlvl_state,
            partlvl_train_loader,
            valid_iterator=partlvl_valid_loader,
            num_epochs=cfg["num_epochs"],
            callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
            randomness="different",
        )
        gen_match_preds = partlvl_ensemble.predict(
            partlvl_state, gen_match_loader, non_linearity=sigmoid
        )
        gen_match_reweights = (
            gen_match_preds / (1.0 - gen_match_preds + 1e-20)
        ).squeeze_()
        miss_preds = partlvl_ensemble.predict(
            partlvl_state, gen_miss_loader, non_linearity=sigmoid
        )
        miss_reweights = (miss_preds / (1.0 - miss_preds + 1e-20)).squeeze_()

        partlvl_state.reset_status()

        w_matched *= gen_match_reweights
        w_miss *= miss_reweights

        fake_scaling_ds.sample_weight[..., fake_scaling_pos_mask] = gen_match_reweights
        fake_scaling_history = fake_scaling_ensemble.fit(
            Adam,
            binary_cross_entropy_with_logits,
            fake_scaling_state,
            fake_scaling_train_loader,
            valid_iterator=fake_scaling_valid_loader,
            num_epochs=cfg["num_epochs"],
            callbacks=[("early_stopping", EarlyStopping(**early_stopping_kwargs))],
            randomness="different",
        )

        fake_preds = fake_scaling_ensemble.predict(
            fake_scaling_state, reco_fake_loader, non_linearity=sigmoid
        )
        fake_reweights = (fake_preds / (1.0 - fake_preds + 1e-20)).squeeze_()

        fake_scaling_state.reset_status()

        w_fake *= fake_reweights

        reco_weights = torch.cat([w_matched, w_fake], dim=1)
        w_unfolding.append(
            reco_weights.mul(n_data).div_(reco_weights.sum(1).unsqueeze_(-1))
        )

        with open(f"{dir_prefix}/w_unfolding.npz", "wb") as f:
            np.savez(f, *[w.detach().numpy() for w in w_unfolding])

    print("Done !")
