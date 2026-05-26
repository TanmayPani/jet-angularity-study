import os
import shutil
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid
from tensordict import TensorDict

from churten.stateless import StatelessModule
from churten.optimizer import Adam
from churten.utils.nn.archs import MLP
from churten.callbacks import Checkpoint, EarlyStopping
from churten.utils.data import TensorBatchSampler, random_split

from dataset import (
    TensorDictDataset,
    get_stacked_batch_loader,
    classwise_undersample_and_split,
    build_input_transform,
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
    neg_mask = data.target < 0.5
    neg_indices = all_indices[neg_mask]

    if not has_unmatched:
        sampler = TensorBatchSampler(
            torch.stack([neg_indices] * num_replicas),
            # Inference is just memory chunking; never drop indices. Clamp the
            # batch to the class size so a small class (e.g. B-side fakes in an
            # AB-split closure) still yields one partial batch.
            batch_size=min(batch_size, neg_indices.shape[0]),
            batch_dim=1,
            drop_last=drop_last,
        )
        return get_stacked_batch_loader(data, sampler, load_only_first=load_only_first)

    is_matched = data.td["is_matched"][data.indices][neg_mask]
    matched_indices = neg_indices[is_matched == 1]
    unmatched_indices = neg_indices[is_matched == 0]

    matched_sampler = TensorBatchSampler(
        torch.stack([matched_indices] * num_replicas),
        batch_size=min(batch_size, matched_indices.shape[0]),
        batch_dim=1,
        drop_last=drop_last,
    )
    matched_loader = get_stacked_batch_loader(
        data, matched_sampler, load_only_first=load_only_first
    )

    unmatched_sampler = TensorBatchSampler(
        torch.stack([unmatched_indices] * num_replicas),
        # B-side fakes can be < batch_size in AB-split closure; clamp so all
        # fakes are still covered in a single batch rather than erroring.
        batch_size=min(batch_size, unmatched_indices.shape[0]),
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


def read_datasets(
    detlvl_src: Path,
    partlvl_src: Path,
    *,
    num_replicas: int = 1,
    a_size=0.5,
    mode: Literal["normal", "ab_closure"] = "normal",
    generator: torch.Generator | None = None,
    out_path: Path | None = None,
):
    detlvl_td = TensorDict.load_memmap(detlvl_src)
    print("Loaded (data, reco) TensorDict from:", detlvl_src)

    partlvl_td = TensorDict.load_memmap(partlvl_src)
    print("Loaded particle level TensorDict from:", partlvl_src)

    if mode == "ab_closure":
        _detlvl_all = torch.arange(len(detlvl_td), dtype=torch.long)
        _detlvl_matched_indices = _detlvl_all[detlvl_td["is_matched"] == 1]
        _detlvl_fake_indices = _detlvl_all[detlvl_td["is_matched"] == 0]

        _partlvl_all = torch.arange(len(partlvl_td))
        _partlvl_matched_indices = _partlvl_all[partlvl_td["is_matched"] == 1]

        _partlvl_missed_indices = _partlvl_all[partlvl_td["is_matched"] == 0]

        (
            _detlvl_matched_indices_a,
            _partlvl_matched_indices_a,
            _detlvl_matched_indices_b,
            _partlvl_matched_indices_b,
        ) = random_split(
            _detlvl_matched_indices,
            _partlvl_matched_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=detlvl_td["pth_bin"][_detlvl_matched_indices],
        )

        _detlvl_fake_indices_a, _detlvl_fake_indices_b = random_split(
            _detlvl_fake_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=detlvl_td["pth_bin"][_detlvl_fake_indices],
        )

        _partlvl_missed_indices_a, _partlvl_missed_indices_b = random_split(
            _partlvl_missed_indices,
            sizes=(a_size, 1.0 - a_size),
            stratify=partlvl_td["pth_bin"][_partlvl_missed_indices],
        )

        _detlvl_a_indices = torch.cat(
            [_detlvl_matched_indices_a, _detlvl_fake_indices_a], dim=-1
        )

        _detlvl_b_indices = torch.cat(
            [_detlvl_matched_indices_b, _detlvl_fake_indices_b], dim=-1
        )

        _partlvl_a_indices = torch.cat(
            [_partlvl_matched_indices_a, _partlvl_missed_indices_a], dim=-1
        )

        _partlvl_b_indices = torch.cat(
            [_partlvl_matched_indices_b, _partlvl_missed_indices_b], dim=-1
        )

        detlvl_ds = TensorDictDataset(
            detlvl_td,
            is_categorical=True,
            indices=torch.cat([_detlvl_a_indices, _detlvl_b_indices], dim=-1),
            target=torch.cat(
                [
                    torch.ones_like(_detlvl_a_indices),
                    torch.zeros_like(_detlvl_b_indices),
                ],
                dim=-1,
            ),
            num_replicas=num_replicas,
        )

        partlvl_ds = TensorDictDataset(
            partlvl_td,
            is_categorical=True,
            indices=torch.cat([_partlvl_b_indices, _partlvl_b_indices], dim=-1),
            target=torch.cat(
                [
                    torch.ones_like(_partlvl_b_indices),
                    torch.zeros_like(_partlvl_b_indices),
                ],
                dim=-1,
            ),
            num_replicas=num_replicas,
        )

        detlvl_matched_indices = _detlvl_matched_indices_b
        detlvl_fake_indices = _detlvl_fake_indices_b

        partlvl_matched_indices = _partlvl_matched_indices_b
        partlvl_missed_indices = _partlvl_missed_indices_b

        if out_path is not None:
            with (out_path / "index_split.npz").open("wb") as outf:
                np.savez(
                    outf,
                    detlvl_matched_indices=detlvl_matched_indices.detach().numpy(),
                    partlvl_matched_indices=partlvl_matched_indices.detach().numpy(),
                    detlvl_fake_indices=detlvl_fake_indices.detach().numpy(),
                    partlvl_missed_indices=partlvl_missed_indices.detach().numpy(),
                )

    else:
        detlvl_ds = TensorDictDataset(
            detlvl_td, is_categorical=True, num_replicas=num_replicas
        )
        partlvl_ds = TensorDictDataset(
            partlvl_td, is_categorical=True, num_replicas=num_replicas
        )

        detlvl_matched_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 1]
        detlvl_fake_indices = detlvl_ds.indices[detlvl_td["is_matched"] == 0]

        partlvl_matched_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 1]
        partlvl_missed_indices = partlvl_ds.indices[partlvl_td["is_matched"] == 0]

    fake_scaling_indices = torch.cat([detlvl_matched_indices] * 2)
    fake_scaling_targets = torch.cat(
        [
            torch.ones_like(detlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(detlvl_matched_indices, dtype=torch.float32),
        ]
    )

    miss_scaling_indices = torch.cat([partlvl_matched_indices] * 2)
    miss_scaling_targets = torch.cat(
        [
            torch.ones_like(partlvl_matched_indices, dtype=torch.float32),
            torch.zeros_like(partlvl_matched_indices, dtype=torch.float32),
        ]
    )

    fake_scaling_ds = TensorDictDataset(
        detlvl_td,
        indices=fake_scaling_indices,
        target=fake_scaling_targets,
        sample_weight=torch.ones_like(fake_scaling_indices, dtype=torch.float32),
        is_categorical=True,
        num_replicas=num_replicas,
    )

    miss_scaling_ds = TensorDictDataset(
        partlvl_td,
        indices=miss_scaling_indices,
        target=miss_scaling_targets,
        sample_weight=torch.ones_like(miss_scaling_indices, dtype=torch.float32),
        is_categorical=True,
        num_replicas=num_replicas,
    )

    return (
        detlvl_ds,
        partlvl_ds,
        fake_scaling_ds,
        miss_scaling_ds,
        detlvl_matched_indices,
        partlvl_matched_indices,
        detlvl_fake_indices,
        partlvl_missed_indices,
    )


def _eps(x: torch.Tensor):
    return x[x > 0.0].min() * torch.finfo(x.dtype).eps


def _weight_stats(w: torch.Tensor, *, clamp_min: float, clamp_max: float):
    """Per-replica summary statistics for a (num_replicas, N) weight tensor.

    Processed one replica row at a time so transient memory stays bounded
    even when N ~ 1.5e7 — a single (R, N) float64 promotion of the
    renormalised gen weights would alone exceed ~2 GiB.
    """
    w = w.detach()
    if w.dim() == 1:
        w = w.unsqueeze(0)
    R = w.shape[0]
    keys = (
        "mean",
        "std",
        "min",
        "max",
        "median",
        "sum",
        "ess",
        "frac_at_clamp_min",
        "frac_at_clamp_max",
    )
    out = {k: np.empty(R, dtype=np.float64) for k in keys}
    for r in range(R):
        wr = w[r].to(torch.float64)
        s = wr.sum()
        s2 = (wr * wr).sum()
        out["mean"][r] = float(wr.mean())
        out["std"][r] = float(wr.std())
        out["min"][r] = float(wr.min())
        out["max"][r] = float(wr.max())
        out["median"][r] = float(wr.median())
        out["sum"][r] = float(s)
        out["ess"][r] = float((s * s) / s2.clamp_min(1e-30))
        out["frac_at_clamp_min"][r] = float((wr <= clamp_min * 1.001).double().mean())
        out["frac_at_clamp_max"][r] = float((wr >= clamp_max * 0.999).double().mean())
        del wr
    return out


def get_sample_reweights(
    model, model_state, data_loader, eps=None, clamp_min=1e-3, clamp_max=1e3
):
    pdata = model.predict(model_state, data_loader, non_linearity=sigmoid)
    preco = 1.0 - pdata
    eps = eps or _eps(preco)

    rewt = (pdata / preco.add_(eps)).squeeze_(-1)
    if clamp_min is not None or clamp_max is not None:
        return rewt.clamp_(min=clamp_min, max=clamp_max)

    return rewt


if __name__ == "__main__":
    with open("runtime-files/config.json", "r") as config_file:
        cfg = json.load(config_file)

    generator = torch.Generator().manual_seed(cfg["dataseed"])

    sys_var = SysVar.UNFOLDING_PRIOR_LIKE_DATA
    feature_mode = cfg.get("feature_mode", "angularities")

    # UNFOLDING_PRIOR_SAME does the AB-split closure in-memory on the
    # nominal tensordicts; no separate on-disk artefact is materialised
    # for it, so redirect the source path back to nominal.
    root_dir = Path("./datasets/STAR_pp200GeV_production_2012/features") / feature_mode
    src = (
        root_dir
        / "tensordicts"
        / str(SysVar.NONE if sys_var == SysVar.UNFOLDING_PRIOR_SAME else sys_var)
    )
    print("Reading files from:", src)

    # out_dir = Path("./outputs") / f"unfolding_{sys_var_dir}" / cfg["feature_mode"]
    out_dir = root_dir / "embedding" / str(sys_var)
    out_dir.mkdir(parents=True, exist_ok=True)

    (
        detlvl_ds,
        partlvl_ds,
        fake_scaling_ds,
        miss_scaling_ds,
        detlvl_matched_indices,
        partlvl_matched_indices,
        detlvl_fake_indices,
        partlvl_missed_indices,
    ) = read_datasets(
        src / "det_lvl",
        src / "part_lvl",
        num_replicas=cfg["num_replicas"],
        mode=("ab_closure" if sys_var == SysVar.UNFOLDING_PRIOR_SAME else "normal"),
        out_path=(out_dir if sys_var == SysVar.UNFOLDING_PRIOR_SAME else None),
    )

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

    n_data = detlvl_ds.target.gt(0.5).sum()
    n_reco = detlvl_ds.target.lt(0.5).sum()
    n_gen = partlvl_ds.target.lt(0.5).sum()

    print("Number of data jets:", n_data)
    print("Number of gen jets, reco jets:", n_gen, n_reco)

    num_features = detlvl_ds.td["input"].shape[-1]
    layer_sizes = [num_features, 256, 256, 256, 1]
    device = "cuda"

    optimizer_kwargs = dict(
        lr=1e-3,
        eps=1e-8,
        weight_decay=0.01,
        decoupled_weight_decay=True,
    )

    torch.manual_seed(cfg["modelseed"])

    detlvl_ensemble, _, detlvl_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": build_input_transform(
                cfg["input_transform"], detlvl_ds.input, device=device
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    miss_scaling_ensemble, _, miss_scaling_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": build_input_transform(
                cfg["input_transform"],
                partlvl_ds.td["input"][partlvl_matched_indices],
                device=device,
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    partlvl_ensemble, _, partlvl_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": build_input_transform(
                cfg["input_transform"], partlvl_ds.input, device=device
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    fake_scaling_ensemble, _, fake_scaling_state = StatelessModule.init(
        MLP,
        Adam,
        model_init_kwargs={
            "layer_sizes": layer_sizes,
            # "batch_norm" : torch.nn.BatchNorm1d,
            "input_transform": build_input_transform(
                cfg["input_transform"],
                detlvl_ds.td["input"][detlvl_matched_indices],
                device=device,
            ),
            "dropout_prob": 0.2,
        },
        num_replicas=cfg["num_replicas"],
        device=device,
        init_randomness="different",
        optimizer_kwargs=optimizer_kwargs,
    )

    w_unfolding = [
        partlvl_ds.sample_weight[..., partlvl_ds.target < 0.5],
        detlvl_ds.sample_weight[..., detlvl_ds.target < 0.5],
    ]

    w_data = (
        (
            detlvl_ds.td["weight"][detlvl_ds.indices][detlvl_ds.target > 0.5]
            .expand(cfg["num_replicas"], -1)
            .clone()
        )
        if cfg["num_replicas"] > 1
        else detlvl_ds.td["weight"][detlvl_ds.indices][detlvl_ds.target > 0.5].clone()
    )
    w_matched = (
        (
            partlvl_ds.td["weight"][partlvl_matched_indices]
            .expand(cfg["num_replicas"], -1)
            .clone()
        )
        if cfg["num_replicas"] > 1
        else partlvl_ds.td["weight"][partlvl_matched_indices].clone()
    )
    w_miss = (
        (
            partlvl_ds.td["weight"][partlvl_missed_indices]
            .expand(cfg["num_replicas"], -1)
            .clone()
        )
        if cfg["num_replicas"] > 1
        else partlvl_ds.td["weight"][partlvl_missed_indices].clone()
    )
    w_fake = (
        (
            detlvl_ds.td["weight"][detlvl_fake_indices]
            .expand(cfg["num_replicas"], -1)
            .clone()
        )
        if cfg["num_replicas"] > 1
        else detlvl_ds.td["weight"][detlvl_fake_indices].clone()
    )

    print("Saving config file to folder", out_dir, "...")
    shutil.copy("runtime-files/config.json", out_dir / "config.json")

    early_stopping_kwargs = dict(patience=10)

    # Phase-1 diagnostics: per-iteration weight stats and training history.
    # See plan: /home/tanmaypani/.claude/plans/the-prior-reweighing-given-piped-fiddle.md
    reweight_clamp = dict(clamp_min=1e-3, clamp_max=1e3)
    weight_stats_history: list[dict[str, np.ndarray]] = []
    fit_history_dir = out_dir / "fit_history"
    fit_history_dir.mkdir(parents=True, exist_ok=True)

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

        reco_match_reweights = get_sample_reweights(
            detlvl_ensemble, detlvl_state, reco_match_loader
        )

        fake_reweights = get_sample_reweights(
            detlvl_ensemble, detlvl_state, reco_fake_loader
        )

        detlvl_state.reset_status()

        w_matched *= reco_match_reweights
        w_fake *= fake_reweights

        miss_scaling_ds.sample_weight[..., miss_scaling_ds.target > 0.5] = (
            reco_match_reweights
        )
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

        miss_reweights = get_sample_reweights(
            miss_scaling_ensemble, miss_scaling_state, gen_miss_loader
        )

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
        gen_match_reweights = get_sample_reweights(
            partlvl_ensemble, partlvl_state, gen_match_loader
        )

        miss_reweights = get_sample_reweights(
            partlvl_ensemble, partlvl_state, gen_miss_loader
        )

        partlvl_state.reset_status()

        w_matched *= gen_match_reweights
        w_miss *= miss_reweights

        fake_scaling_ds.sample_weight[..., fake_scaling_ds.target > 0.5] = (
            gen_match_reweights
        )
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

        fake_reweights = get_sample_reweights(
            fake_scaling_ensemble, fake_scaling_state, reco_fake_loader
        )

        fake_scaling_state.reset_status()

        w_fake *= fake_reweights

        reco_weights = torch.cat([w_matched, w_fake], dim=1)
        w_unfolding.append(
            reco_weights.mul(n_data).div_(reco_weights.sum(1).unsqueeze_(-1))
        )

        with (out_dir / "w_unfolding.npz").open("wb") as outf:
            np.savez(outf, *[w.detach().numpy() for w in w_unfolding])

        # --- Phase-1 diagnostics: weight stats + training history --------
        iter_stats: dict[str, np.ndarray] = {}
        for _name, _w in (
            ("w_data", w_data),
            ("w_matched", w_matched),
            ("w_miss", w_miss),
            ("w_fake", w_fake),
            ("gen", w_unfolding[-2]),
            ("reco", w_unfolding[-1]),
        ):
            for _k, _v in _weight_stats(_w, **reweight_clamp).items():
                iter_stats[f"{_name}/{_k}"] = _v
        weight_stats_history.append(iter_stats)

        stacked = {
            k: np.stack([h[k] for h in weight_stats_history])
            for k in weight_stats_history[0]
        }
        np.savez(out_dir / "weight_stats.npz", **stacked)

        for _name, _hist in (
            ("detlvl", detlvl_history),
            ("miss_scaling", miss_scaling_history),
            ("partlvl", partlvl_history),
            ("fake_scaling", fake_scaling_history),
        ):
            _hist.to_file(fit_history_dir / f"iter{iteration:02d}_{_name}.json")

    print("Done !")
