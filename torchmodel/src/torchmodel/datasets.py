from __future__ import annotations
import random
from copy import deepcopy
from typing import Any, Union, Optional, Protocol
from collections.abc import Sequence, Iterator, Sized
from numbers import Number

import numpy as np
from numpy.ma import indices
import numpy.typing as npt

from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import check_cv

import torch
from torch.utils.data import Dataset, Sampler

import pyarrow as pa

from .utils import params_for
from .utils import to_numpy
from .utils import set_global_random_seed

type Tensors = Union[torch.Tensor, tuple[torch.Tensor, ...], list[torch.Tensor]] 
type TensorsSeq = Union[Tensors, Sequence[Tensors]]
type TensorLike = Optional[torch.Tensor | Sequence[int] | Sequence[float]| npt.NDArray[np.float64]| npt.NDArray[np.int_]]
type TableLike = torch.Tensor | Sequence[Sequence[int] | Sequence[float]| npt.NDArray[np.float64]| npt.NDArray[np.int_]]
    
def seed_worker(worker_id, make_cuda_deterministic=False):
    init_seed = torch.initial_seed()
    #print(torch.utils.data.get_worker_info().id, ":", init_seed)
    worker_seed = init_seed % 2**32
    #print(worker_seed)
    set_global_random_seed(worker_seed, make_cuda_deterministic=make_cuda_deterministic, verbose=False)

def collate_fn(tensor_like : Optional[TensorsSeq], batch_size=-1):
    if tensor_like is None:
        return None

    if torch.is_tensor(tensor_like):
        if batch_size > 0:
            assert tensor_like.shape[0] == batch_size
        return tensor_like

    if isinstance(tensor_like, (tuple, list)):
        if isinstance(tensor_like[0], (tuple, list)):
            return [collate_fn(tensor) for tensor in tensor_like] 

        if isinstance(tensor_like[0], torch.Tensor):
            if batch_size > 0:
                sum_batch_dim_sizes = sum([tensor.shape[0] for tensor in tensor_like])
                assert sum_batch_dim_sizes == batch_size
            return torch.cat(tensor_like, dim=0)
    
    raise ValueError("Can't collate given type of tensor collection")

def pin_memory_fn(tensor : Optional[torch.Tensor]):
    if tensor is None:
        return None

    if tensor.is_pinned():
        return tensor

    return tensor.pin_memory()

def get_tensor_kwargs(tensor):
    return {"dtype" : tensor.dtype, 
            "device" : tensor.device, 
            "requires_grad" : tensor.requires_grad}
            #"pin_memory" : tensor.is_pinned()}


def copy_or_create(src : Optional[torch.Tensor], dest : Optional[torch.Tensor], to_dest = True):
    if dest is None or src is None:
        #print("dest is none")
        dest = src
        #print(dest)
        return dest
    
    if dest.shape == src.shape:
        #print("copying")
        return dest.copy_(src)

    if src.shape[0] < dest.shape[0] and src.shape[1:] == dest.shape[1:]:
        batch_size = src.shape[0]
        indices = torch.as_tensor(list(range(batch_size)), dtype=torch.long)
        dest.index_copy_(0, indices, src)
        return dest[:batch_size]


    kwargs = get_tensor_kwargs(dest) if to_dest else get_tensor_kwargs(src) 
    dest = torch.tensor(src, **kwargs)
    return dest

class ValidSplit:
    """Class that performs the internal train/valid split on a dataset.
    """
    def __init__(
            self,
            cv=5,
            stratified=False,
            random_state=None,
    ):
        self.stratified = stratified
        self.random_state = random_state

        print(f"In ValidSplit __init__ random_state set to {random_state}")

        if isinstance(cv, Number) and (cv <= 0):
            raise ValueError("Numbers less than 0 are not allowed for cv "
                             "but ValidSplit got {}".format(cv))

        if not self._is_float(cv) and random_state is not None:
            raise ValueError(
                "Setting a random_state has no effect since cv is not a float. "
                "You should leave random_state to its default (None), or set cv "
                "to a float value.",
            )

        self.cv = cv

    def _is_stratified(self, cv):
        return isinstance(cv, (StratifiedKFold, StratifiedShuffleSplit))

    def _is_float(self, x):
        if not isinstance(x, Number):
            return False
        return not float(x).is_integer()

    def _check_cv_float(self):
        cv_cls = StratifiedShuffleSplit if self.stratified else ShuffleSplit
        return cv_cls(test_size=self.cv, random_state=self.random_state)

    def _check_cv_non_float(self, y):
        return check_cv(
            self.cv,
            y=y,
            classifier=self.stratified,
        )

    def check_cv(self, y):
        """Resolve which cross validation strategy is used."""
        y_arr = None
        if self.stratified:
            # Try to convert y to numpy for sklearn's check_cv; if conversion
            # doesn't work, still try.
            try:
                y_arr = to_numpy(y)
            except (AttributeError, TypeError):
                y_arr = y

        if self._is_float(self.cv):
            return self._check_cv_float()
        return self._check_cv_non_float(y_arr)

    def _is_regular(self, x):
        return (x is None) or isinstance(x, np.ndarray)

    def __call__(self, dataset, y=None, groups=None):
        if hasattr(dataset, "stratification"):
            self.stratified = dataset.has_stratification
            y = dataset.stratification

        bad_y_error = ValueError(
            "Stratified CV requires explicitly passing a suitable y.")
        if (y is None) and self.stratified:
            raise bad_y_error

        cv = self.check_cv(y)
        if self.stratified and not self._is_stratified(cv):
            raise bad_y_error

        # pylint: disable=invalid-name
        len_dataset = len(dataset)
        if y is not None:
            len_y = len(y)
            if len_dataset != len_y:
                raise ValueError("Cannot perform a CV split if dataset and y "
                                 "have different lengths.")

        args = (np.arange(len_dataset),)
        if self._is_stratified(cv):
            args = args + (to_numpy(y),)

        idx_train, idx_valid = next(iter(cv.split(*args, groups=groups)))
        dataset_train = Subset(dataset, idx_train)
        dataset_valid = Subset(dataset, idx_valid)
        return dataset_train, dataset_valid

    def __repr__(self):
        # pylint: disable=useless-super-delegation
        return super(ValidSplit, self).__repr__()
class Batch:
    inputs : Optional[torch.Tensor]
    targets : Optional[torch.Tensor]
    sample_weights : Optional[torch.Tensor]
    indices : Optional[torch.Tensor]
    batch_size : int = 0
    
    def __init__(self, batch_size=-1, **kwargs):
        self.batch_size = batch_size
        self.inputs = kwargs.pop("inputs", None)
        self.targets = kwargs.pop("targets", None)
        self.sample_weights = kwargs.pop("sample_weights", None)
        self.indices = kwargs.pop("indices", None)

    def batch_collate(self, inputs, targets=None, sample_weights=None, indices=None):
        try:
            batch_size_infered = inputs.shape[0]
            batch_size = batch_size_infered if batch_size_infered < self.batch_size else self.batch_size

            self.update_attr("inputs", collate_fn(inputs, batch_size=batch_size))
            self.update_attr("targets", collate_fn(targets, batch_size=batch_size))
            self.update_attr("sample_weights", collate_fn(sample_weights, batch_size=batch_size))
            self.update_attr("indices", collate_fn(indices, batch_size=batch_size))
        except AssertionError: raise ValueError("Mismatch in batch size and received tensor size! Either pass batch_size=-1 or check Dataset subclass")

    def update_attr(self, name, value):
        #print(self.__dict__)
        if name not in self.__dict__:
            raise KeyError("Can't update value of attribute that doesn't exist!!")
        self.__setattr__(name, copy_or_create(value, self.__dict__[name]))

    def __call__(self, data):
        self.batch_collate(data[0], targets=data[1], sample_weights=data[2], indices=data[3])
        return self

    def __len__(self):
        return self.inputs.shape[0] if isinstance(self.inputs, torch.Tensor) else 0 

    def unpack_data(self):
        #print(self.inputs, self.targets, self.sample_weights, self.indices)
        return self.inputs, self.targets, self.sample_weights, self.indices

    def pin_memory(self):
        self.inputs = pin_memory_fn(self.inputs)
        self.targets = pin_memory_fn(self.targets)
        self.sample_weights = pin_memory_fn(self.sample_weights)
        self.indices = pin_memory_fn(self.indices)
        return self

class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements sequential from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        for index in self.indices:
            yield index

    def __len__(self) -> int:
        return len(self.indices)

class TorchDataset(Dataset):
    inputs : Any
    targets : Optional[torch.Tensor] = None
    sample_weights : Optional[torch.Tensor] = None
    length : int = 0
    def __init__(self, inputs : Sized, targets : TensorLike = None, sample_weights : TensorLike = None, **kwargs):
        super().__init__()
        self.is_categorical = kwargs.pop("is_categorical", False)
        self.length = kwargs.pop("length", len(inputs))
        self.dtype = kwargs.pop("dtype", torch.float32)
        self.device = kwargs.pop("device", None)
        self.as_tensor_kwargs = {"dtype" : self.dtype, "device" : self.device}

        vars(self).update(kwargs)
        
        self.inputs = inputs
        self.targets = targets
        self.sample_weights = sample_weights

    def __setattr__(self, name: str, value: Any) -> None:
        #print(name, value)
        if name in ["targets", "sample_weights"]:
            self.check_length(value)
            self.__dict__[name] = self.make_tensor(name, value)
            return

        if name == "inputs":
            self.check_length(value)

        self.__dict__[name] = value
    
    def make_tensor(self, name: str, value : TensorLike | TableLike):
        if value is None:
            return
        if f"{name}__as_tensor_kwargs" not in self.__dict__:
            as_tensor_kwargs = deepcopy(self.as_tensor_kwargs)
            as_tensor_kwargs.update(params_for(name, self.__dict__))
            self.__dict__[f"{name}__as_tensor_kwargs"] = as_tensor_kwargs

        tensor_attr = torch.as_tensor(value, **(self.__dict__[f"{name}__as_tensor_kwargs"]))
        if tensor_attr.dim() == 1:
            return tensor_attr.unsqueeze_(1)

        return tensor_attr

    def check_length(self, arr : Optional[Sized]):
        if arr is None:
            return
        #print(len(arr), self.length)
        if len(arr) == self.length:
            return
        raise ValueError("TorchDataset sized input doesn't match dataset length!")

    def getitems_targets(self, idxs):
        return self.targets[idxs] if self.targets is not None else None
    
    def getitems_sample_weights(self, idxs):
        return self.sample_weights[idxs] if self.sample_weights is not None else None

    def getitems_inputs(self, idxs : list[int]):
        raise NotImplementedError("override getitems_inputs method in your inherited class to use __getitem__ or __getitems__")

   # def getitems_targets(self, idxs : list[int]) -> Optional[torch.Tensor]:
   #     raise NotImplementedError("override getitems_targets method in your inherited class to use __getitem__ or __getitems__")

   # def getitems_sample_weights(self, idxs : list[int]) -> Optional[torch.Tensor]:
   #     raise NotImplementedError("override getitems_sample_weights method in your inherited class to use __getitem__ or __getitems__")

    def __len__(self):
        return self.length

    def __getitems__(self, idxs):
        return self.getitems_inputs(idxs), self.getitems_targets(idxs), self.getitems_sample_weights(idxs), torch.as_tensor(idxs, dtype=torch.long)
    
    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.__getitems__(idx)
        return self.__getitems__([idx])

class ArrowDataset(TorchDataset):
    def __init__(
        self,
        inputs: pa.Table,
        targets : Optional[Sequence[int]] = None,
        sample_weights : Optional[Sequence[float]] = None,
        **kwargs,
    ):
        self.column_names = kwargs.pop("column_names", None)
        self.is_jagged = kwargs.pop("is_jagged", False)

        self.has_stratification = False
        stratification_labels = [np.asarray(targets)] if targets is not None else []
        if "stratification_labels" in inputs.column_names:
            stratification_labels.append(inputs["stratification_labels"].to_numpy())

        if len(stratification_labels) > 0:
            self.has_stratification = True

        self.stratification_labels = np.stack(stratification_labels, axis=-1)

        if self.column_names:
            inputs = inputs.select(self.column_names)
        else:
            inputs = inputs.select(self)
            self.column_names = inputs.column_names

        super().__init__(inputs, targets=targets, sample_weights=sample_weights, length = len(inputs), **kwargs)
        #self.inputs = inputs
    
    @property
    def stratification(self):
        return self.stratification_labels if self.has_stratification else None
        
    def getitems_inputs(self, idxs):
        feature_list = list(zip(*self.inputs.take(idxs).to_pydict().values()))
        return self.make_tensor("inputs", feature_list)

class CategoricalDataset(TorchDataset):
    def __init__(self, inputs : Sized, targets : Optional[Sequence[int]] = None, sample_weights : Optional[Sequence[float]] = None, **kwargs):
        super().__init__(inputs, targets=targets, sample_weights=sample_weights, is_categorical = True, **kwargs)
        self.unique_labels = np.unique(targets) if targets is not None else []
        self.num_classes = kwargs.pop("num_classes", len(self.unique_labels))

class ArrowCategoricalDataset(CategoricalDataset):
    def __init__(
        self,
        inputs: pa.Table,
        targets : Optional[Sequence[int]] = None,
        sample_weights : Optional[Sequence[float]] = None,
        **kwargs,
    ):
        self.column_names = kwargs.pop("column_names", None)
        self.is_jagged = kwargs.pop("is_jagged", False)

        if self.column_names:
            inputs = inputs.select(self.column_names)
        else:
            inputs = inputs.select(self)
            self.column_names = inputs.column_names

        super().__init__(inputs, targets=targets, sample_weights=sample_weights, length = len(inputs), **kwargs)

    def getitems_inputs(self, idxs):
        feature_list = list(zip(*self.inputs.take(idxs).to_pydict().values()))
        #if self.is_ragged:
        #    _features_td = pad_sequence([TensorDict({self.dict_key : torch.as_tensor(_features, dtype=torch.float32)}) for _features in _feature_list], pad_dim=-1)
        #    _dict.update_(_features_td)
        #else:
        return torch.as_tensor(feature_list, **self.as_tensor_kwargs)

class Subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    """
    #dataset: Dataset
    dataset: TorchDataset
    indices: npt.NDArray[np.long]

    def __init__(self, dataset: TorchDataset | Subset, indices: Sequence[int] | npt.NDArray[np.long]) -> None:
        if isinstance(dataset, Subset):
            self.dataset = dataset.dataset
            self.indices = dataset.indices[indices]
        else:
            self.dataset = dataset
            self.indices = np.asarray(indices, dtype=np.long)

    def __len__(self):
        return len(self.indices)

    def get_indices(self, idx : Sequence[int] | int):
        if isinstance(idx, Sequence):
            return [self.indices[i] for i in idx]
        else:
            return self.indices[idx]

    def __getitem__(self, idx : Sequence[int] | int):
        return self.dataset[self.get_indices(idx)]

    def __getitems__(self, idxs: Sequence[int]):
        return self.dataset.__getitems__(self.get_indices(idxs))
    
   # def __getitem__(self, idx):
   #     if isinstance(idx, list):
   #         return self.dataset[[self.indices[i] for i in idx]]
   #     return self.dataset[self.indices[idx]]

   # def __getitems__(self, indices: list[int]) -> list:
   #     if callable(getattr(self.dataset, "__getitems__", None)):
   #         return self.dataset.__getitems__([self.indices[idx] for idx in indices])
   #     else:
   #         return [self.dataset[self.indices[idx]] for idx in indices]





