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

from churten.data import TensorBatchSampler, undersample_and_random_split

class TensorDictDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(
        self, 
        tdict : TensorDict | str, 
        *,
        input_key : str = "input",
        target : str | Tensor = "target",
        sample_weight : str | Tensor = "weight",
        indices : Optional[Tensor] = None,
        is_categorical : bool = False,
        num_replicas : int = 1,
    ) -> None:

        self.td = tdict if isinstance(tdict, TensorDict) else TensorDict.load_memmap(tdict)
        self.input = self.td[input_key]
        self.indices = indices if indices is not None else torch.arange(len(self.td))
        self.length = self.indices.shape[0]
        self.target = target if isinstance(target, Tensor) else self.td[target][self.indices]
        assert self.target.shape[0] == self.length

        self.num_replicas = num_replicas
        self.is_categorical = is_categorical
        
        if self.is_categorical:
            self.classes, self.class_labels, self.class_sizes = torch.unique(
                self.target,
                dim = 0,
                return_inverse = True,
                return_counts =  True,
            )

            self.num_classes = self.classes.shape[0]
            self.one_hot_label = torch.nn.functional.one_hot(
                self.class_labels, 
                num_classes=self.num_classes,
            )

        self.sample_weight = sample_weight if isinstance(sample_weight, Tensor) \
                            else self.td[sample_weight][self.indices]   

        assert self.sample_weight.shape == (self.num_replicas, self.length)


    @property 
    def sample_weight(self):
        return self.weight

    @sample_weight.setter
    def sample_weight(self, w : Tensor):
        #print(w.dim())
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
                    (self.weight.unsqueeze(-1)*self.one_hot_label).sum(dim=-2)
                )
            self.weight.mul_(self.class_weights[..., self.class_labels])
        
        #print("setter called!")
        #self.weight.unsqueeze_(-1)

    @singledispatchmethod
    def getitem(
            self, 
            idx : int | Sequence | Tensor
    ) -> tuple[Tensor, ...]:
        #print("int getitem")
        return (
            self.input[self.indices[idx]], 
            self.target[idx], self.weight[idx],
        )

    @getitem.register
    def _(self, idx : Tensor) -> tuple[Tensor, ...]:
        #print("tensor getitem")
        flat_indices = self.indices[idx.ravel()]
        return (
            self.input[flat_indices].reshape(
                (*(idx.shape), *(self.input.shape[1:]))
            ),
            self.target[idx.ravel()].reshape(
                (*(idx.shape), *(self.target.shape[1:]))
            ).unsqueeze_(-1), 
            self.sample_weight.gather(-1, idx).unsqueeze_(-1),
        )

    @getitem.register
    def _(self, idx : Sequence) -> tuple[Tensor, ...]:
        #print("list, tuple getitem")
        return (
            self.input[self.indices[idx]], 
            self.target[idx].unsqueeze_(-1), 
            self.weight[idx].unsqueeze_(-1),
        )

    def __getitem__(self, index : int | Sequence | Tensor) -> tuple[Tensor, ...]:
        return self.getitem(index)

    __getitems__ : Callable[
        [Self, int | Sequence | Tensor], tuple[Tensor, ...]
    ] = __getitem__

    def __len__(self) -> int : 
        return self.length

def reweight_inference_loaders(
    data : TensorDictDataset,
    batch_size : int = 32,
    num_replicas : int = 2,
    drop_last : bool = False,
    load_only_first = None,
):
    matched_mask = data.td["is_matched"] == 1
    unmatched_mask = data.td["is_matched"] == 0

    all_indices = torch.arange(len(data))
    matched_indices = all_indices[matched_mask]
    unmatched_indices = all_indices[unmatched_mask]

    matched_sampler = TensorBatchSampler(
        torch.stack([matched_indices]*num_replicas),
        batch_size = batch_size,
        batch_dim = 1,
        drop_last = drop_last,
    )
    matched_loader = get_stacked_batch_loader(
        data, matched_sampler, load_only_first=load_only_first
    )

    unmatched_sampler = TensorBatchSampler(
        torch.stack([unmatched_indices]*num_replicas),
        batch_size = batch_size,
        batch_dim = 1,
        drop_last = drop_last,
    )
    unmatched_loader = get_stacked_batch_loader(
        data, unmatched_sampler, load_only_first=load_only_first
    )
    
    return (matched_loader, unmatched_loader) 

def train_test_multi_loaders(
    data : TensorDictDataset,
    *,
    num_replicas : int,
    batch_size : int,
    train_size : float = 0.5,
    undersample_size : float | int = 1.0,
    stratifys : tuple[bool, bool] = (False, False),
    vmap_randomness : str = "different",
    vmap_in_dims : int | tuple = 0,
    drop_last : bool = False,
    generator : Optional[torch.Generator] = None,
    pos_mask : Optional[Tensor] = None,
    neg_mask : Optional[Tensor] = None,
    load_only_first : Optional[int] = None,
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
    
    stratify = data.td["table_id"][data.indices]
    pos_stratify = stratify[pos_mask] if stratifys[0] else None 
    neg_stratify = stratify[neg_mask] if stratifys[1] else None

    stacked_train_indices, stacked_valid_indices = classwise_undersample_and_split(
        pos_indices, neg_indices,
        undersample_size = undersample_size,
        split_sizes = (train_size, 1. - train_size),
        pos_stratify = pos_stratify,
        neg_stratify = neg_stratify,
        num_replicas = num_replicas,
        vmap_randomness = vmap_randomness,
        vmap_in_dims = vmap_in_dims,
        generator = generator,
    )

    print(
    "------ Subsampled "
    f"{str(undersample_size*100)+"%" if isinstance(undersample_size, float) else 2*undersample_size}, "
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
        data, train_sampler,
        load_only_first=load_only_first,
    )
    valid_loader = get_stacked_batch_loader(
        data, valid_sampler,
        load_only_first=load_only_first,
    )

    return train_loader, valid_loader

def classwise_undersample_and_split(
    pos_indices : Tensor,
    neg_indices : Tensor,
    *,
    pos_stratify : Optional[Tensor] = None,
    neg_stratify : Optional[Tensor] = None,
    generator : Optional[torch.Generator] = None, 
    **kwargs,
) :

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

    train_indices = torch.cat([pos_train_indices, neg_train_indices], dim=-1)
    valid_indices = torch.cat([pos_valid_indices, neg_valid_indices], dim=-1)

    train_idx_perm = torch.randperm(train_indices.shape[-1], generator=generator)
    valid_idx_perm = torch.randperm(valid_indices.shape[-1], generator=generator)

    return ( 
        train_indices[..., train_idx_perm], 
        valid_indices[..., valid_idx_perm],
    )

def get_stacked_batch_loader(
    dataset: Dataset,
    batch_sampler : TensorBatchSampler,
    num_workers: int = 0,
    collate_fn : Optional[Callable] = None,
    pin_memory : bool = False,
    load_only_first : Optional[int] = None,
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
        node = Prefetcher(node, prefetch_factor=num_workers*2)

    if load_only_first is not None:
        node = Header(node, load_only_first)

    return Loader(node)

if __name__ == "__main__": 
    dir = "/home/tanmaypani/star-workspace/jet-angularity-study/partitioned_datasets/nominal"
    src = f"{dir}/det_lvl/all"

    td = TensorDict.load_memmap(src)
    all_indices = torch.arange(len(td))
    print(all_indices.shape)
    matched_mask = td["is_matched"] == 1 
    matched_indices = all_indices[matched_mask]
    print(matched_indices.shape)
    indices = torch.cat([matched_indices, matched_indices])
    target = torch.cat([torch.ones_like(matched_indices), torch.zeros_like(matched_indices)])
    print(indices.shape, target.shape)
    ds = TensorDictDataset(
        td, 
        indices = indices, 
        target=target, 
        num_replicas=10, 
        is_categorical=True,
    )

    print(ds.sample_weight.shape)

    indices = len(ds) - 1 - torch.arange(3)

    ex = ds[indices.expand(10,-1)]
    print(ex[0].shape, ex[1].shape, ex[2].shape)

    print(ds.sample_weight[0], ds.target[0])

    ds.sample_weight[..., ds.target > 0.5] = 0
    print(ds.sample_weight[0], ds.target[0])


