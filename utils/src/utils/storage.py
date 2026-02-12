from collections.abc import Sequence
import torch


def fill_single_impl(storage, binned_data, x):
    x = torch.as_tensor(
        x,
        dtype=storage.dtype,
        device=storage.device,
    )

    storage.add_(
        binned_data.bincount(
            weights=x,
            minlength=storage.shape[-1],
        ),
    )


def fill_multi_impl(storage, binned_data, x):
    x = torch.as_tensor(
        x,
        dtype=storage.dtype,
        device=storage.device,
    )
    storage.index_add_(
        -1,
        binned_data,
        x,
    )


def weighted_fill_single_impl(storage, binned_data, x, weights):
    weights = torch.as_tensor(
        weights,
        dtype=storage.dtype,
        device=storage.device,
    )
    x = torch.as_tensor(
        x,
        dtype=storage.dtype,
        device=storage.device,
    )
    storage.add_(
        binned_data.bincount(
            weights=weights * x,
            minlength=storage.shape[-1],
        ),
    )


def weighted_fill_multi_impl(storage, binned_data, x, weights):
    weights = torch.as_tensor(
        weights,
        dtype=storage.dtype,
        device=storage.device,
    )
    x = torch.as_tensor(
        x,
        dtype=storage.dtype,
        device=storage.device,
    )

    batch_size = storage.shape[:-1]
    storage.index_add_(
        -1,
        binned_data,
        weights.expand(*batch_size, -1) * x.expand(*batch_size, -1),
    )


class Storage:
    def __init__(
        self,
        storage: torch.Tensor,
        batch_size: Sequence[int] | None = None,
        sum_of_squares: bool = True,
    ):
        self.data = storage
        self.batch_size = batch_size or tuple()
        self.has_sum_of_squares = sum_of_squares
        if self.has_sum_of_squares:
            assert self.shape[0] == 2

        storage_batch_size = self.shape[
            int(self.has_sum_of_squares) : int(self.has_sum_of_squares)
            + len(self.batch_size)
        ]
        assert storage_batch_size == self.batch_size

    @property
    def shape(self):
        return self.data.shape

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def sum_x(self):
        return self.data[0, ...] if self.has_sum_of_squares else self.data

    @property
    def sum_x2(self):
        return self.data[1, ...] if self.has_sum_of_squares else None

    @classmethod
    def create(
        cls,
        num_bins,
        device="cpu",
        dtype=torch.float32,
    ):
        return cls(torch.zeros((2, num_bins), device=device, dtype=dtype))

    def fill(self, binned_data):
        bin_counts = binned_data.bincount(minlength=self.shape[-1])
        self.sum_x.add_(bin_counts)
        if self.has_sum_of_squares:
            self.sum_x2.add_(bin_counts)


class SumStorage(Storage):
    @classmethod
    def create(
        cls,
        num_bins: int,
        ndim: int = 1,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        sum_of_squares: bool = False,
    ):
        batch_size = (ndim,) if ndim > 1 else tuple()
        shape = (
            (2, *batch_size, num_bins) if sum_of_squares else (*batch_size, num_bins)
        )
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            batch_size=batch_size,
            sum_of_squares=sum_of_squares,
        )

    def fill(self, binned_data, *cols):
        assert len(cols) > 0
        if len(cols) == 1 and len(self.batch_size) == 0:
            fill_single_impl(self.sum_x, binned_data, cols[0])
            if self.has_sum_of_squares:
                fill_single_impl(self.sum_x2, binned_data, cols[0] ** 2)
        elif len(cols) == 1:
            fill_multi_impl(self.sum_x, binned_data, cols[0])
            if self.has_sum_of_squares:
                fill_multi_impl(self.sum_x2, binned_data, cols[0] ** 2)
        else:
            for icol, col in enumerate(cols):
                fill_single_impl(self.sum_x[icol], binned_data, col)
                if self.has_sum_of_squares:
                    fill_single_impl(self.sum_x2[icol], binned_data, col**2)


class WeightedSumStorage(Storage):
    @classmethod
    def create(
        cls,
        num_bins,
        ndim=1,
        device="cpu",
        dtype=torch.float32,
        sum_of_squares: bool = False,
    ):
        batch_size = (ndim,) if ndim > 1 else tuple()
        shape = (
            (2, *batch_size, num_bins) if sum_of_squares else (*batch_size, num_bins)
        )
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            batch_size=batch_size,
            sum_of_squares=sum_of_squares,
        )

    def fill(self, binned_data, weights, *cols):
        assert len(cols) > 0
        if len(cols) == 1 and len(self.batch_size) == 0:
            weighted_fill_single_impl(self.sum_x, binned_data, cols[0], weights)
            if self.has_sum_of_squares:
                weighted_fill_single_impl(
                    self.sum_x2, binned_data, cols[0] ** 2, weights
                )
        elif len(cols) == 1:
            weighted_fill_multi_impl(self.sum_x, binned_data, cols[0], weights)
            if self.has_sum_of_squares:
                weighted_fill_multi_impl(
                    self.sum_x2, binned_data, cols[0] ** 2, weights
                )
        else:
            for icol, col in enumerate(cols):
                weighted_fill_single_impl(self.sum_x[icol], binned_data, col, weights)
                if self.has_sum_of_squares:
                    weighted_fill_single_impl(
                        self.sum_x2[icol], binned_data, col**2, weights
                    )


class MultiWeightedSumStorage(Storage):
    @classmethod
    def create(
        cls,
        num_bins,
        ndim=1,
        num_weights=1,
        device="cpu",
        dtype=torch.float32,
        sum_of_squares: bool = False,
    ):
        assert ndim > 0 and num_weights > 0
        batch_size = (num_weights,) if num_weights > 1 else tuple()
        batch_size = batch_size + (ndim,) if ndim > 1 else tuple()

        shape = (
            (2, *batch_size, num_bins) if sum_of_squares else (*batch_size, num_bins)
        )
        return cls(
            torch.zeros(shape, device=device, dtype=dtype),
            batch_size=batch_size,
            sum_of_squares=sum_of_squares,
        )

    def fill(self, binned_data, weights, *cols):
        assert len(cols) > 0
        if len(cols) == 1 and len(self.batch_size) == 0:
            weighted_fill_single_impl(self.sum_x, binned_data, cols[0], weights)
            if self.has_sum_of_squares:
                weighted_fill_single_impl(
                    self.sum_x2, binned_data, cols[0] ** 2, weights
                )
        elif len(cols) == 1:
            weighted_fill_multi_impl(self.sum_x, binned_data, cols[0], weights)
            if self.has_sum_of_squares:
                weighted_fill_multi_impl(
                    self.sum_x2, binned_data, cols[0] ** 2, weights
                )
        else:
            for icol, col in enumerate(cols):
                weighted_fill_multi_impl(
                    self.sum_x[..., icol, :], binned_data, col, weights
                )
                if self.has_sum_of_squares:
                    weighted_fill_multi_impl(
                        self.sum_x2[..., icol, :], binned_data, col**2, weights
                    )
