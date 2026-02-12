import torch

from .axis import BinnedAxes


class Histogram:
    def __init__(
        self,
        axes,
        sum_w,
        sum_w2=None,
        batch_size=None,
    ):
        batch_size = batch_size or tuple()
        assert sum_w.shape[len(batch_size) :] == axes.shape
        if sum_w2 is not None:
            assert sum_w2.shape == sum_w.shape
        self.axes = axes
        self.batch_size = batch_size or tuple()
        self.bin_sizes = self.axes.bin_sizes()
        self.sum_w = sum_w
        self.sum_w2 = sum_w2
        self.total_sum_w = (
            self.sum_w[..., *(self.rank * (slice(1, -1),))]
            .sum((*(dim + len(self.batch_size) for dim in range(self.rank)),))
            .view(self.batch_size + (1,) * self.rank)
        )

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        idx = idx + (self.rank - len(idx)) * (slice(None),)
        axes = BinnedAxes(
            *(self.axes[i][ix] for i, ix in enumerate(idx) if not isinstance(ix, int))
        )
        sum_w = self.sum_w[..., *idx]
        sum_w2 = self.sum_w2[..., *idx] if self.sum_w2 is not None else None

        return Histogram(
            axes,
            sum_w,
            sum_w2,
            self.batch_size,
        )

    def project(self, *dims):
        dims_to_sum = (*(idx for idx in range(len(self.axes)) if idx not in (*dims,)),)
        return self.sum(*dims_to_sum)

    def sum(self, *dims, overflow=True):
        sum_w = self.sum_w
        sum_w2 = self.sum_w2

        if not overflow:
            no_overflow_slice = self.rank * (slice(None),)
            for dim in dims:
                no_overflow_slice[dim] = slice(1, -1)
            sum_w = sum_w[..., *no_overflow_slice]
            sum_w2 = sum_w2[..., *no_overflow_slice] if sum_w2 is not None else None

        dims_to_sum = (*(dim + len(self.batch_size) for dim in dims),)

        return Histogram(
            self.axes.remove_dims(*dims),
            sum_w.sum(dims_to_sum),
            sum_w2.sum(dims_to_sum) if sum_w2 is not None else None,
            self.batch_size,
        )

    @property
    def rank(self):
        return len(self.axes)

    @property
    def shape(self):
        return self.sum_w.shape

    @property
    def num_bins(self):
        return self.axes.num_bins

    @property
    def bins(self):
        return (*(axis.bins for axis in self.axes),)

    @property
    def values(self):
        return self.sum_w / (self.total_sum_w * self.bin_sizes)

    @property
    def variances(self):
        return (
            self.sum_w2 / (self.total_sum_w * self.bin_sizes) ** 2
            if self.sum_w2 is not None
            else None
        )

    @property
    def counts(self):
        return self.sum_w.square() / (self.sum_w2) if self.sum_w2 is not None else None


class Profile(Histogram):
    def __init__(
        self,
        axes,
        sum_w,
        sum_x,
        sum_w2=None,
        sum_x2=None,
        batch_size=None,
        eps=None,
    ):
        super().__init__(axes, sum_w, sum_w2, batch_size)
        assert sum_x.shape == self.shape
        if sum_x2 is not None:
            assert sum_x2.shape == self.shape
        self.sum_x = sum_x
        self.sum_x2 = sum_x2
        self.eps = (
            self.sum_w[self.sum_w > 0.0].min() * torch.finfo(self.sum_w.dtype).eps
            if eps is None
            else eps
        )

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)

        idx = idx + (self.rank - len(idx)) * (slice(None),)
        axes = BinnedAxes(
            *(self.axes[i][ix] for i, ix in enumerate(idx) if not isinstance(ix, int))
        )
        sum_w = self.sum_w[..., *idx]
        sum_w2 = self.sum_w2[..., *idx] if self.sum_w2 is not None else None
        sum_x = self.sum_x[..., *idx]
        sum_x2 = self.sum_x2[..., *idx] if self.sum_x2 is not None else None

        return Profile(
            axes,
            sum_w,
            sum_x,
            sum_w2,
            sum_x2,
            self.batch_size,
            self.eps,
        )

    def sum(self, *dims, overflow=True):
        sum_w = self.sum_w
        sum_w2 = self.sum_w2
        sum_x = self.sum_x
        sum_x2 = self.sum_x2

        if not overflow:
            no_overflow_slice = self.rank * (slice(None),)
            for dim in dims:
                no_overflow_slice[dim] = slice(1, -1)
            sum_w = sum_w[..., *no_overflow_slice]
            sum_x = sum_x[..., *no_overflow_slice]
            sum_w2 = sum_w2[..., *no_overflow_slice] if sum_w2 is not None else None
            sum_x2 = sum_x2[..., *no_overflow_slice] if sum_x2 is not None else None

        dims_to_sum = (*(dim + len(self.batch_size) for dim in dims),)
        return Profile(
            self.axes.remove_dims(*dims),
            sum_w.sum(dims_to_sum),
            sum_x.sum(dims_to_sum),
            sum_w2.sum(dims_to_sum) if sum_w2 is not None else None,
            sum_x2.sum(dims_to_sum) if sum_x2 is not None else None,
            self.batch_size,
            self.eps,
        )

    @property
    def values(self):
        return self.sum_x / (self.sum_w + self.eps)

    @property
    def variances(self):
        return (
            (self.sum_x2 / (self.sum_w + self.eps)) - self.values**2
            if self.sum_x2 is not None
            else None
        )
