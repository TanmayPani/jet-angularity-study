from functools import singledispatchmethod

import torch

from .axis import BinnedAxes, BinnedAxis
from .storage import Storage, SumStorage, WeightedSumStorage, MultiWeightedSumStorage
from .histogram import Histogram, Profile


class Accumulator:
    def __init__(
        self,
        axes: BinnedAxes,
        sum_w_storage: Storage,
        sum_wx_storage: Storage,
    ):
        self.axes = axes
        self.sum_w_storage = sum_w_storage
        self.sum_wx_storage = sum_wx_storage

    @classmethod
    def create(
        cls,
        *bins,
        num_weights: int = 1,
        device="cpu",
        dtype=torch.float32,
        use_sum_w2=True,
    ):
        axes = BinnedAxes(
            *(
                BinnedAxis(torch.as_tensor(bin_arr, device=device, dtype=dtype))
                for bin_arr in bins
            )
        )
        if num_weights == 0:
            sum_w_storage = Storage.create(
                axes.num_bins,
                device,
                dtype,
            )
            sum_wx_storage = SumStorage.create(
                axes.num_bins,
                len(axes),
                device,
                dtype,
                sum_of_squares=use_sum_w2,
            )
        elif num_weights == 1:
            sum_w_storage = SumStorage.create(
                axes.num_bins,
                1,
                device,
                dtype,
                sum_of_squares=use_sum_w2,
            )

            sum_wx_storage = WeightedSumStorage.create(
                axes.num_bins,
                len(axes),
                device,
                dtype,
                sum_of_squares=use_sum_w2,
            )
        elif num_weights > 1:
            sum_w_storage = SumStorage.create(
                axes.num_bins,
                num_weights,
                device,
                dtype,
                sum_of_squares=use_sum_w2,
            )
            sum_wx_storage = MultiWeightedSumStorage.create(
                axes.num_bins,
                len(axes),
                num_weights,
                device,
                dtype,
                sum_of_squares=use_sum_w2,
            )
        return cls(axes, sum_w_storage, sum_wx_storage)

    @property
    def ndim(self):
        return len(self.axes)

    def histogram(self):
        sum_w = self.sum_w_storage.sum_x.unflatten(-1, self.axes.shape)
        sum_w2 = self.sum_w_storage.sum_x2
        sum_w2 = sum_w2.view(sum_w.shape) if sum_w2 is not None else None
        return Histogram(
            self.axes,
            sum_w,
            sum_w2,
            self.sum_w_storage.batch_size,
        )

    def profile(self, dim):
        sum_w = self.sum_w_storage.sum_x.unflatten(-1, self.axes.shape)
        sum_w2 = self.sum_w_storage.sum_x2
        sum_w2 = sum_w2.view(sum_w.shape) if sum_w2 is not None else None

        sum_x = self.sum_wx_storage.sum_x[..., dim, :].view(sum_w.shape)
        sum_x2 = self.sum_wx_storage.sum_x2
        sum_x2 = sum_x2[..., dim, :].view(sum_w.shape) if sum_x2 is not None else None

        batch_size = self.sum_w_storage.batch_size
        dim_to_sum = dim + len(batch_size)

        return Profile(
            self.axes.remove_dims(dim),
            sum_w.sum(dim_to_sum),
            sum_x.sum(dim_to_sum),
            sum_w2.sum(dim_to_sum) if sum_w2 is not None else None,
            sum_x2.sum(dim_to_sum) if sum_x2 is not None else None,
            batch_size,
        )

    @singledispatchmethod
    def fill(self, data: torch.Tensor | tuple | list, **kwargs):
        raise NotImplementedError(
            "Method fill of Accumulator instance can only accept torch.Tensor "
            "or list and tuple of Tensor-like columns, but got input of type "
            f"{type(data)}"
        )

    @fill.register
    def _(
        self,
        data: torch.Tensor,
        column_dim: int = 0,
        **kwargs,
    ):
        data = torch.as_tensor(data)
        if data.ndim == 1 and self.ndim == 1:
            self.fill((data,), **kwargs)
        else:
            assert data.shape[column_dim] == self.ndim
            self.fill(data.unbind(column_dim), **kwargs)

    @fill.register
    def _(
        self,
        data: tuple | list,
        weights: torch.Tensor | None = None,
        weights_batch_dim: int | None = None,
    ):
        assert len(data) == self.ndim
        assert all(len(col) == len(data[0]) for col in data)
        binned_data = self.axes.loc(*data)
        if weights is None:
            self.sum_w_storage.fill(binned_data)
            self.sum_wx_storage.fill(binned_data, *data)
            return

        if weights_batch_dim:
            weights = weights.transpose(0, weights_batch_dim)

        assert weights.shape[-1] == data[0].shape[-1]
        self.sum_w_storage.fill(binned_data, weights)
        self.sum_wx_storage.fill(binned_data, weights, *data)
