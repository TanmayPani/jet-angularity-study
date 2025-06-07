from __future__ import annotations

from typing import Any, Callable, Iterable, List, Tuple, Union, Optional, TypeVar, Dict
from collections.abc import Sequence
from itertools import chain

import pyarrow as pa
import pyarrow.compute as pc

import awkward as ak
import numpy as np

import torch
from tensordict import TensorDict

_Tensor_t = TypeVar("_Tensor_t", TensorDict, torch.Tensor)
type _SingleBatchInput_t = Tuple[_Tensor_t, torch.Tensor]
type _BatchInput_t = Union[_SingleBatchInput_t, List[_SingleBatchInput_t]]

class Batch:
    data : TensorDict | torch.Tensor 
    indices : torch.Tensor
    def __init__(self, batch : _BatchInput_t):
        # print(len(batch))
        if isinstance(batch, tuple):
            self.data, self.indices = batch
        elif len(batch) == 0:
            raise ValueError("Batch must not be empty!")
        elif len(batch) == 1:
            self.data, self.indices = batch[0][1]
        elif isinstance(batch, list):
            self.data = torch.cat([b[0] for b in batch], dim=0)
            self.indices = torch.cat([b[1] for b in batch], dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def pin_memory(self):
        self.data = self.data.pin_memory()
        # for key, value in self.data.items():
        #    if isinstance(value, torch.Tensor):
        #        self.data[key] = value.pin_memory()
        #    elif isinstance(value, Iterable[torch.Tensor]):
        #        self.data[key] = [tensor.pin_memory() for tensor in value]
        return self


def batch_collate(batch: _BatchInput_t) -> Batch:
    return Batch(batch)

class CustomCategoricalDataset(torch.utils.data.Dataset):
    def __init__(self, labels, sample_weights=None, **kwargs):
        super().__init__()
        self.length = len(labels)
        self.do_one_hot = kwargs.pop("do_one_hot", False)
        self.label_key = kwargs.pop("label_key", "targets")
        self.num_classes = kwargs.pop("num_classes", len(np.unique(labels)))

        if self.do_one_hot:
            self.labels = torch.as_tensor(
                torch.nn.functional.one_hot(
                    torch.as_tensor(labels, dtype=torch.long),
                    num_classes=self.num_classes,
                ),
                dtype=torch.float32,
            )
        else:
            self.labels = torch.as_tensor(labels, dtype=torch.float32).unsqueeze_(1)

        if sample_weights is None:
            sample_weights = np.ones(self.length)

        self.set_sample_weights(sample_weights)

    def set_sample_weights(self, sample_weights):
        assert sample_weights.shape[0] == self.length
        self.sample_weights = torch.as_tensor(
            sample_weights, dtype=torch.float32
        ).unsqueeze_(1)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if isinstance(idx, list):
            return self.__getitems__(idx)
        return self.__getitems__([idx])

    def __getitems__(self, idxs)->Tuple[TensorDict, torch.Tensor]:
        _dict = {}
        _dict[self.label_key] = self.labels[idxs]
        _dict["sample_weights"] = self.sample_weights[idxs]
        return TensorDict(_dict, batch_size=[len(idxs)]), torch.as_tensor(idxs, dtype=torch.long)


class JetDataset(CustomCategoricalDataset):
    def __init__(
        self,
        features: pa.Table,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.column_names = kwargs.pop("column_names", None)
        if self.column_names:
            self.features = features.select(self.column_names)
        else:
            self.features = features
            self.column_names = features.column_names

        self.do_scale = kwargs.pop("do_scale", False)
        self.mean_tensor = kwargs.pop("mean", [])
        self.stddev_tensor = kwargs.pop("stddev", [])

        _scale_from = kwargs.pop("scale_from", None)
        if _scale_from is not None:
            self.do_scale = True

        if self.do_scale:
            if len(self.mean_tensor) == 0 or len(self.stddev_tensor) == 0:
                if _scale_from is not None:
                    self.set_mean_stddev(_scale_from)
                else:
                    self.set_mean_stddev(self.features)
            else:
                assert len(self.mean_tensor) == self.column_names
                assert len(self.stddev_tensor) == self.column_names

    def __getitems__(self, idxs)->Tuple[TensorDict, torch.Tensor]:
        _dict, _indices = super().__getitems__(idxs)
        _features = ak.from_arrow(self.features.take(idxs))
        _feature_list = []
        #print(idxs)
        for _col in self.column_names:
            #print(_col)
            _feature_list.append(torch.as_tensor(_features[_col], dtype=torch.float32))

        _dict["features"] = (
            (
                torch.stack(_feature_list, dim=1)
                .sub_(self.mean_tensor)
                .div_(self.stddev_tensor)
            )
            if self.do_scale
            else torch.stack(_feature_list, dim=1)
        )
        return _dict, _indices

    def set_mean_stddev(self, data: pa.Table | JetDataset):
        if isinstance(data, JetDataset):
            assert self.column_names == data.column_names
            assert data.do_scale
            self.mean_tensor = data.mean_tensor
            self.stddev_tensor = data.stddev_tensor
        else:
            mean_list = []
            stddev_list = []
            for col in self.column_names:
                if pa.types.is_list(data[col].type):
                    mean_list.append(pc.mean(pc.list_flatten(data[col])).as_py())
                    stddev_list.append(pc.stddev(pc.list_flatten(data[col])).as_py())
                else:
                    mean_list.append(pc.mean(data[col]).as_py())
                    stddev_list.append(pc.stddev(data[col]).as_py())
            self.mean_tensor = torch.as_tensor(mean_list, dtype=torch.float32)
            self.stddev_tensor = torch.as_tensor(stddev_list, dtype=torch.float32)

class JetConstituentDataset(CustomCategoricalDataset):
    def __init__(
        self,
        data,
        labels,
        sample_weights=None,
        colnames=None,
        max_num_constits=None,
        **kwargs,
    ):
        super().__init__(labels, sample_weights=sample_weights, **kwargs)
        self.data = data.select(colnames) if colnames else data
        self.column_names = colnames if colnames else data.column_names
        self.max_nconstit = (
            max_num_constits
            if max_num_constits
            else pc.max(
                pc.list_value_length(data["jet_constit_pt_scaled"])
            ).as_py()
        )

    def _make_constit_array_regular(self, arr, pad_to=None, dtype=torch.float32):
        nconstits = ak.fill_none(ak.num(arr, axis=-1), 0)
        max_ncon = ak.max(nconstits)
        if pad_to:
            max_ncon = int(pad_to) if pad_to > max_ncon else max_ncon
        value_arr = ak.from_numpy(np.full((len(arr), max_ncon), 0, dtype=np.int32))
        nconstits_brc, _ = ak.broadcast_arrays(nconstits[:, np.newaxis], value_arr)
        mask_arr = ak.local_index(value_arr, axis=-1) < nconstits_brc
        padding = ak.drop_none(ak.mask(value_arr, ~mask_arr))
        max_padded = ak.to_regular(ak.concatenate([arr, padding], axis=-1))
        if pad_to is None:
            return torch.as_tensor(max_padded, dtype=dtype), torch.as_tensor(
                mask_arr, dtype=dtype
            )
        else:
            return torch.as_tensor(
                max_padded[:, 0:pad_to], dtype=dtype
            ), torch.as_tensor(mask_arr[:, 0:pad_to], dtype=dtype)

    def __getitems__(self, idxs)->Tuple[TensorDict, torch.Tensor]:
        _dict, _indices = super().__getitems__(idxs)
        _features = ak.from_arrow(self.data.take(idxs))
        _feature_list = []
        _mask_list = []
        for _col in self.column_names:
            _padded, _mask = self._make_constit_array_regular(
                _features[_col], pad_to=self.max_nconstit
            )
            # print(_padded.size(), _mask.size())
            _feature_list.append(_padded)
            _mask_list.append(_mask)
        _dict["constituents"] = torch.stack(_feature_list, dim=-1).transpose_(1, 2)
        _dict["constituents_mask"] = torch.stack(_mask_list, dim=-1).transpose_(1, 2)
        # print(_dict["constituents"].size())
        return TensorDict(_dict, batch_size=[len(idxs)]), _indices
