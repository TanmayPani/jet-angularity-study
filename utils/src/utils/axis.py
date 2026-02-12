from itertools import accumulate
from functools import reduce
import operator

import torch


class BinnedAxis:
    def __init__(
        self,
        bins: torch.Tensor,
        /,
        min: float | int | None = None,
        max: float | int | None = None,
    ):
        self.bins = bins.contiguous()
        self.range = (min, max)

    @property
    def dtype(self):
        return self.bins.dtype

    @property
    def device(self):
        return self.bins.device

    def __len__(self):
        return len(self.bins)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            idx = slice(idx, idx + 1)
        new_bins = self.bins[idx]
        return BinnedAxis(new_bins, new_bins[0].item(), new_bins[-1].item())

    def __setitem__(self, idx, x):
        new_bins = self.bins
        new_bins[idx] = x
        self.bins = new_bins

    def insert_edge(self, x):
        index = torch.searchsorted(self.bins, x)
        val = torch.as_tensor((x,), device=self.device, dtype=self.dtype)
        if index == len(self) and not torch.isclose(val, self[index - 1]):
            self.bins = torch.cat((self.bins, val))
        elif not torch.isclose(val, self[index]):
            self.bins = torch.cat((self.bins[:index], val, self.bins[index:]))

        return self.bins

    def loc(self, x, **kwargs):
        return torch.searchsorted(self.bins, x, **kwargs) - 1

    @property
    def range(self):
        return (self[0].item(), self[1].item())

    @range.setter
    def range(self, minmax):
        min, max = minmax
        min = torch.as_tensor(
            (min if min is not None else float("-inf"),),
            device=self.device,
            dtype=self.dtype,
        )
        max = torch.as_tensor(
            (max if max is not None else float("inf"),),
            device=self.device,
            dtype=self.dtype,
        )
        if min >= max:
            raise ValueError(
                f"Axis need upper range to be greater than lower range, but got ({min}, {max})."
            )

        new_bins = self.bins
        _min_idx = torch.searchsorted(new_bins, min)
        if torch.isclose(new_bins[_min_idx], min):
            new_bins = new_bins[_min_idx:]
        else:
            new_bins = torch.cat((min, new_bins[_min_idx:]))

        _max_idx = torch.searchsorted(new_bins, max, side="right")
        if torch.isclose(new_bins[_max_idx - 1], max):
            new_bins = new_bins[:_max_idx]
        else:
            new_bins = torch.cat((new_bins[:_max_idx], max))

        self.bins = new_bins

    @property
    def bins(self):
        return self._bins

    @bins.setter
    def bins(self, bins):
        self._bins = bins
        sorted_bins, _ = self.bins.sort()
        if not torch.allclose(self.bins, sorted_bins):
            raise ValueError(
                f"Given bin edges must be strictly increasing! but got {bins}"
            )

        self.bin_sizes = self._bins[1:] - self._bins[:-1]
        self.bin_centers = (self._bins[1:] + self._bins[:-1]).mul_(0.5)


class BinnedAxes(tuple):
    def __new__(cls, *axes: BinnedAxis):
        try:
            assert all(axis.device == axes[0].device for axis in axes)
            assert all(axis.dtype == axes[0].dtype for axis in axes)
        except AssertionError:
            raise ValueError(
                "All axis inputs to BinnedAxes must be of the same device and dtype!"
            )
        return super().__new__(cls, (*axes,))

    def __init__(
        self,
        *axes: BinnedAxis,
    ):
        super().__init__()
        self.device = axes[0].device
        self.dtype = axes[0].dtype
        self.reset()

    def reset(self):
        self.shape = (*(len(axis) - 1 for axis in self),)
        self.num_bins = reduce(operator.mul, self.shape)
        self.global_bin_idx = torch.arange(self.num_bins).view(*(self.shape))

    def bin_sizes(self, flat=False):
        if len(self) == 1:
            return self[0].bin_sizes
        sizes_flat = torch.cartesian_prod(*(axis.bin_sizes for axis in self)).prod(-1)
        return sizes_flat if flat else sizes_flat.view(*(self.shape))

    def bin_centers(self, flat=False):
        if len(self) == 1:
            return self[0].bin_centers
        centers_flat = torch.cartesian_prod(*(axis.bin_centers for axis in self))
        return centers_flat if flat else centers_flat.view(*(self.shape), len(self))

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, int):
            return super().__getitem__(idx)
        return BinnedAxes(*(super().__getitem__(idx)))

    def __setitem__(self, idx: int, ax: BinnedAxis):
        try:
            assert ax.device == self.device
            assert ax.dtype == self.dtype
        except AssertionError:
            raise ValueError(
                f"All axis inputs to BinnedAxes must be with device = {self.device} and dtype = {self.dtype},"
                f" but got BinnedAxis instance with device = {ax.device} and dtype = {ax.dtype}!"
            )
        self[idx].bins = ax.bins
        self.reset()

    def remove_dims(self, *dims):
        return BinnedAxes(
            *(axis for iaxis, axis in enumerate(self) if iaxis not in (*dims,))
        )

    def strides(self):
        return (*accumulate(self.shape[:0:-1], operator.mul),)[::-1] + (1,)

    def loc(self, *args, **kwargs):
        return self.global_bin_idx[
            *(
                self[iarg].loc(
                    torch.as_tensor(
                        arg, device=self[iarg].device, dtype=self[iarg].dtype
                    ),
                    **kwargs,
                )
                for iarg, arg in enumerate(args)
            )
        ]
