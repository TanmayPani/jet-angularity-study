from __future__ import annotations
from functools import singledispatchmethod
from typing import Self, Callable, Optional
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from tensordict import TensorDict

from torchdata.nodes import SamplerWrapper
from torchdata.nodes import ParallelMapper
from torchdata.nodes import Prefetcher
from torchdata.nodes import PinMemory
from torchdata.nodes import Loader, Header

from churten.utils.data import (
    TensorBatchSampler,
    undersample_and_random_split,
    random_split,
)

from churten.utils.nn.transform import Normalize


class Log1p(torch.nn.Module):
    def forward(self, x):
        return torch.log1p(x)


def _chunked_std_mean(
    x: torch.Tensor,
    dim: int = 0,
    chunk_size: int = 500_000,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    eps=1e-6,
    dtype=torch.float32,
    device="cpu",
):
    dim = dim if dim >= 0 else x.dim() + dim
    num_elem = x.shape[dim]
    output_shape = x.shape[:dim] + x.shape[dim + 1 :]
    sum = torch.zeros(output_shape, dtype=dtype, device=device)
    sum_sq = torch.zeros(output_shape, dtype=dtype, device=device)
    for chunk_idx in torch.arange(len(x)).split(chunk_size):
        chunk = x[chunk_idx] if transform is None else transform(x[chunk_idx])
        sum += chunk.sum(dim=dim)
        sum_sq += chunk.square().sum(dim=dim)
    mean = sum / num_elem
    var = (sum_sq / num_elem) - mean * mean
    std = var.clamp_min(0).sqrt().clamp_min(eps)
    return std, mean


def build_input_transform(
    count_transform: str, input, dtype=torch.float32, device="cpu"
):
    if count_transform == "none":
        return None
    if count_transform == "z_norm":
        std, mean = _chunked_std_mean(input)
        return Normalize(mean.to(device), std.to(device))
    if count_transform == "log1p_z_norm":
        std, mean = _chunked_std_mean(input, transform=torch.log1p)
        return torch.nn.Sequential(
            Log1p(),
            Normalize(mean.to(device), std.to(device)),
        )
    raise ValueError(
        f"count_transform must be one of (none, z_norm, log1p_z_norm), but got {count_transform!r}"
    )


class TensorDictDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self,
        tdict: TensorDict | str,
        *,
        input_key: str = "input",
        target: str | Tensor = "target",
        sample_weight: str | Tensor = "weight",
        indices: Optional[Tensor] = None,
        is_categorical: bool = False,
        num_replicas: int = 1,
    ) -> None:

        self.td = (
            tdict if isinstance(tdict, TensorDict) else TensorDict.load_memmap(tdict)
        )
        self.input = self.td[input_key]
        self.indices = indices if indices is not None else torch.arange(len(self.td))
        self.length = self.indices.shape[0]
        self.target = (
            target if isinstance(target, Tensor) else self.td[target][self.indices]
        )
        assert self.target.shape[0] == self.length

        self.num_replicas = num_replicas
        self.is_categorical = is_categorical

        if self.is_categorical:
            self.classes, self.class_labels, self.class_sizes = torch.unique(
                self.target,
                dim=0,
                return_inverse=True,
                return_counts=True,
            )

            self.num_classes = self.classes.shape[0]
            self.one_hot_label = torch.nn.functional.one_hot(
                self.class_labels,
                num_classes=self.num_classes,
            )

        self.sample_weight = (
            sample_weight
            if isinstance(sample_weight, Tensor)
            else self.td[sample_weight][self.indices]
        )

        assert self.sample_weight.shape == (self.num_replicas, self.length)

    @property
    def sample_weight(self):
        return self.weight

    @sample_weight.setter
    def sample_weight(self, w: Tensor):
        # print(w.dim())
        if w.dim() == 1:
            assert len(w) == self.length
            self.weight = w.expand(self.num_replicas, -1).clone()
        elif w.dim() == 0:
            self.weight = torch.full((self.num_replicas, self.length), w.item())
        else:
            assert w.shape[0] == self.num_replicas
            assert w.shape[1] == self.length
            self.weight = w.clone()

        if self.is_categorical:
            self.class_weights = self.class_sizes.div(
                (self.weight.unsqueeze(-1) * self.one_hot_label).sum(dim=-2)
            )
            self.weight.mul_(self.class_weights[..., self.class_labels])

        # print("setter called!")
        # self.weight.unsqueeze_(-1)

    @singledispatchmethod
    def getitem(self, idx: int | Sequence | Tensor) -> tuple[Tensor, ...]:
        # print("int getitem")
        return (
            self.input[self.indices[idx]],
            self.target[idx],
            self.weight[idx],
        )

    @getitem.register
    def _(self, idx: Tensor) -> tuple[Tensor, ...]:
        # print("tensor getitem")
        flat_indices = self.indices[idx.ravel()]
        return (
            self.input[flat_indices].reshape((*(idx.shape), *(self.input.shape[1:]))),
            self.target[idx.ravel()]
            .reshape((*(idx.shape), *(self.target.shape[1:])))
            .unsqueeze_(-1),
            self.sample_weight.gather(-1, idx).unsqueeze_(-1),
        )

    @getitem.register
    def _(self, idx: Sequence) -> tuple[Tensor, ...]:
        # print("list, tuple getitem")
        return (
            self.input[self.indices[idx]],
            self.target[idx].unsqueeze_(-1),
            self.weight[idx].unsqueeze_(-1),
        )

    def __getitem__(self, index: int | Sequence | Tensor) -> tuple[Tensor, ...]:
        return self.getitem(index)

    __getitems__: Callable[[Self, int | Sequence | Tensor], tuple[Tensor, ...]] = (
        __getitem__
    )

    def __len__(self) -> int:
        return self.length


def classwise_split(
    pos_indices: Tensor,
    neg_indices: Tensor,
    *,
    pos_stratify: Optional[Tensor] = None,
    neg_stratify: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):

    pos_train_indices, pos_valid_indices = random_split(
        pos_indices,
        stratify=pos_stratify,
        generator=generator,
        **kwargs,
    )

    neg_train_indices, neg_valid_indices = random_split(
        neg_indices,
        stratify=neg_stratify,
        generator=generator,
        **kwargs,
    )

    train_indices = torch.atleast_2d(
        torch.cat([pos_train_indices, neg_train_indices], dim=-1)
    )
    valid_indices = torch.atleast_2d(
        torch.cat([pos_valid_indices, neg_valid_indices], dim=-1)
    )

    # The vmap branch of random_split returns (num_replicas, n) tensors;
    # the non-vmap branch (num_replicas=1) returns 1D (n,) tensors. Downstream
    # TensorBatchSampler uses batch_dim=1, so promote 1D to (1, n) to keep a uniform
    # contract regardless of num_replicas.
    # if train_indices.dim() == 1:
    #    train_indices = train_indices.unsqueeze(0)
    # if valid_indices.dim() == 1:
    #    valid_indices = valid_indices.unsqueeze(0)

    train_idx_perm = torch.randperm(train_indices.shape[-1], generator=generator)
    valid_idx_perm = torch.randperm(valid_indices.shape[-1], generator=generator)

    return (
        train_indices[..., train_idx_perm],
        valid_indices[..., valid_idx_perm],
    )


def classwise_undersample_and_split(
    pos_indices: Tensor,
    neg_indices: Tensor,
    *,
    pos_stratify: Optional[Tensor] = None,
    neg_stratify: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
    **kwargs,
):

    pos_train_indices, pos_valid_indices = undersample_and_random_split(
        pos_indices,
        stratify=pos_stratify,
        generator=generator,
        **kwargs,
    )

    neg_train_indices, neg_valid_indices = undersample_and_random_split(
        neg_indices,
        stratify=neg_stratify,
        generator=generator,
        **kwargs,
    )

    train_indices = torch.atleast_2d(
        torch.cat([pos_train_indices, neg_train_indices], dim=-1)
    )
    valid_indices = torch.atleast_2d(
        torch.cat([pos_valid_indices, neg_valid_indices], dim=-1)
    )

    # The vmap branch of undersample_and_random_split returns (num_replicas, n) tensors;
    # the non-vmap branch (num_replicas=1) returns 1D (n,) tensors. Downstream
    # TensorBatchSampler uses batch_dim=1, so promote 1D to (1, n) to keep a uniform
    # contract regardless of num_replicas.
    # if train_indices.dim() == 1:
    #    train_indices = train_indices.unsqueeze(0)
    # if valid_indices.dim() == 1:
    #    valid_indices = valid_indices.unsqueeze(0)

    train_idx_perm = torch.randperm(train_indices.shape[-1], generator=generator)
    valid_idx_perm = torch.randperm(valid_indices.shape[-1], generator=generator)

    return (
        train_indices[..., train_idx_perm],
        valid_indices[..., valid_idx_perm],
    )


def get_stacked_batch_loader(
    dataset: Dataset,
    batch_sampler: TensorBatchSampler,
    num_workers: int = 0,
    collate_fn: Optional[Callable] = None,
    pin_memory: bool = False,
    load_only_first: Optional[int] = None,
):
    map_fn = collate_fn or dataset.__getitem__

    node = SamplerWrapper(batch_sampler)
    node = ParallelMapper(
        node,
        map_fn=map_fn,
        num_workers=num_workers,
        method="process",
        in_order=True,
    )

    if pin_memory:
        node = PinMemory(node)

    if num_workers > 0:
        node = Prefetcher(node, prefetch_factor=num_workers * 2)

    if load_only_first is not None:
        node = Header(node, load_only_first)

    return Loader(node)
