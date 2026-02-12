from functools import partial

import torch

eps_factor_low = torch.as_tensor([(1 - torch.finfo(torch.float32).eps)])
eps_factor_high = torch.as_tensor([(1 + torch.finfo(torch.float32).eps)])


class TorchHist1D:
    def __init__(self, x, xbins, overflow=True):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.xbins = torch.as_tensor(xbins, dtype=torch.float32)
        if overflow:
            xmin = self.x.min()
            if xmin < self.xbins[0]:
                self.xbins = torch.cat((xmin * eps_factor_low, self.xbins))

            xmax = self.x.max()
            if xmax > self.xbins[-1]:
                self.xbins = torch.cat((self.xbins, xmax * eps_factor_high))
        else:
            self.xmask = self.x.gt(self.xbins[0]).logical_and_(
                self.x.lt(self.xbins[-1])
            )
            self.x = self.x[self.xmask]

        self.num_samples = self.x.numel()

        self.num_xbins = self.xbins.shape[0] - 1
        self.h_xbin_widths = self.xbins[1:] - self.xbins[:-1]
        self.h_xbin_scales = self.h_xbin_widths.reciprocal()
        self.h_xbin_centers = (self.xbins[:-1] + self.xbins[1:]).mul_(0.5)
        self.x_bin_indices = torch.bucketize(self.x, self.xbins) - 1
        self.one_hot_xbin_indices = (
            torch.nn.functional.one_hot(
                self.x_bin_indices,
                num_classes=self.num_xbins,
            )
            .to(torch.float32)
            .T
        )

    def histogram(self, weights=None, weights_sq=None, density=True, is_batched=False):
        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            weights_sq = (
                torch.as_tensor(weights_sq, dtype=torch.float32)
                if weights_sq is not None
                else torch.square(weights)
            )
            if hasattr(self, "xmask"):
                weights = weights[..., self.xmask]
                weights_sq = weights_sq[..., self.xmask]

            assert weights.shape[-1] == self.num_samples, (
                f"Expected weights to have {self.num_samples} elements but has {weights.shape[-1]}"
            )
            assert weights_sq.shape[-1] == self.num_samples, (
                f"Expected weights_sq to have {self.num_samples} elements but has {weights_sq.shape[-1]}"
            )
        else:
            weights = torch.ones(self.num_samples, dtype=torch.float32)
            weights_sq = weights

        if is_batched:
            return torch.vmap(self._make_histogram)(
                weights,
                weights_sq,
                density=density,
            )

        return self._make_histogram(
            weights,
            weights_sq,
            density=density,
        )

    def profileY(self, y, weights=None, density=True, is_batched=False):
        if hasattr(self, "xmask"):
            assert y.shape == self.xmask.shape
            y = y[self.xmask]
        assert y.shape == self.x.shape

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            if hasattr(self, "xmask"):
                weights = weights[..., self.xmask]
            assert weights.shape[-1] == self.num_samples, (
                f"Got {weights.shape[-1]} for {self.num_samples} samples!"
            )
        else:
            weights = torch.ones(self.num_samples, dtype=torch.float32)

        eps = weights[weights > 0].min() * torch.finfo(torch.float32).eps
        if is_batched:
            shape = (weights.shape[0], self.num_xbins)
            prof_mean, prof_err = torch.vmap(
                partial(self._make_profile, self.one_hot_xbin_indices, y)
            )(
                weights,
                density=density,
                eps=eps,
            )
        else:
            shape = (self.num_xbins,)
            prof_mean, prof_err = self._make_profile(
                self.one_hot_xbin_indices, y, weights, density=density, eps=eps
            )
        return prof_mean.view(*shape), prof_err.view(*shape)

    def _make_histogram(self, weights, weights_sq, /, density=True):
        h_bin_sumw = torch.matmul(self.one_hot_xbin_indices, weights)
        h_bin_sumw2 = torch.matmul(self.one_hot_xbin_indices, weights_sq)

        if density:
            h_bin_norm = self.h_xbin_scales.div(h_bin_sumw.sum())
            return h_bin_sumw.mul_(h_bin_norm), h_bin_sumw2.sqrt_().mul_(h_bin_norm)

        return h_bin_sumw, h_bin_sumw2.sqrt_()

    def _make_profile(self, one_hot_idx, arr, weights, /, density=True, eps=1e-22):
        arr_x_wt = arr.mul(weights)
        arr2_x_wt = arr.mul(arr_x_wt)
        h_bin_sumw = torch.mv(one_hot_idx, weights)
        h_bin_sumy = torch.mv(one_hot_idx, arr_x_wt)
        h_bin_sumy2 = torch.mv(one_hot_idx, arr2_x_wt)

        h_bin_meany = h_bin_sumy.div_(h_bin_sumw + eps)
        h_bin_vary = h_bin_sumy2.div_(h_bin_sumw + eps).sub_(h_bin_meany.square())

        return h_bin_meany, h_bin_vary.sqrt_()


class TorchHist2D:
    def __init__(self, x, y, xbins, ybins, overflow=True):
        assert x.shape == y.shape

        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.xbins = torch.as_tensor(xbins, dtype=torch.float32)
        if overflow:
            xmin = self.x.min()
            xmax = self.x.max()
            if xmin < self.xbins[0]:
                self.xbins = torch.cat((xmin * eps_factor_low, self.xbins))
            if xmax > self.xbins[-1]:
                self.xbins = torch.cat((self.xbins, xmax * eps_factor_high))

        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.ybins = torch.as_tensor(ybins, dtype=torch.float32)
        if overflow:
            ymin = self.y.min()
            ymax = self.y.max()
            if ymin < self.ybins[0]:
                self.ybins = torch.cat((ymin * eps_factor_low, self.ybins))
            if ymax > self.ybins[-1]:
                self.ybins = torch.cat((self.ybins, ymax * eps_factor_high))

        if not overflow:
            self.xmask = self.x.gt(self.xbins[0]).logical_and_(
                self.x.lt(self.xbins[-1])
            )
            self.ymask = self.y.gt(self.ybins[0]).logical_and_(
                self.y.lt(self.ybins[-1])
            )
            self.mask = self.xmask.logical_and(self.ymask)
            self.x = self.x[self.mask]
            self.y = self.y[self.mask]

        self.num_samples = self.x.numel()

        self.num_xbins = self.xbins.shape[0] - 1
        self.h_xbin_widths = self.xbins[1:] - self.xbins[:-1]
        self.h_xbin_scales = self.h_xbin_widths.reciprocal()
        self.h_xbin_centers = (self.xbins[:-1] + self.xbins[1:]).mul_(0.5)
        self.x_bin_indices = torch.bucketize(self.x, self.xbins) - 1
        self.one_hot_xbin_indices = (
            torch.nn.functional.one_hot(
                self.x_bin_indices,
                num_classes=self.num_xbins,
            )
            .to(torch.float32)
            .T
        )

        self.num_ybins = self.ybins.shape[0] - 1
        self.h_ybin_widths = self.ybins[1:] - self.ybins[:-1]
        self.h_ybin_scales = self.h_ybin_widths.reciprocal()
        self.h_ybin_centers = (self.ybins[:-1] + self.ybins[1:]).mul_(0.5)
        self.y_bin_indices = torch.bucketize(self.y, self.ybins) - 1
        self.one_hot_ybin_indices = (
            torch.nn.functional.one_hot(
                self.y_bin_indices,
                num_classes=self.num_ybins,
            )
            .to(torch.float32)
            .T
        )

        self.num_bins = self.num_xbins * self.num_ybins
        self.shape = (self.num_ybins, self.num_xbins)
        self.h_bin_sizes = self.h_ybin_widths.unsqueeze(
            1
        ) @ self.h_xbin_widths.unsqueeze(0)
        self.h_bin_scales = self.h_bin_sizes.reciprocal().flatten()

        self.bin_indices = (
            # self.x_bin_indices*self.num_ybins + self.y_bin_indices
            self.y_bin_indices * self.num_xbins + self.x_bin_indices
        )

        self.one_hot_bin_indices = (
            torch.nn.functional.one_hot(
                self.bin_indices,
                num_classes=self.num_bins,
            )
            .to(torch.float32)
            .T
        )

        self.bin_counts = (
            self.one_hot_bin_indices.sum(1).to(torch.long).view(*self.shape)
        )

    def histogram(self, weights=None, weights_sq=None, density=True, is_batched=False):
        self.density = density
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.float32)
            self.weights_sq = (
                torch.as_tensor(weights_sq, dtype=torch.float32)
                if weights_sq is not None
                else torch.square(self.weights)
            )
            if hasattr(self, "mask"):
                self.weights = self.weights[..., self.mask]
                self.weights_sq = self.weights_sq[..., self.mask]

            assert self.weights.shape[-1] == self.num_samples
            assert self.weights_sq.shape[-1] == self.num_samples
        else:
            self.weights = torch.ones(self.num_samples, dtype=torch.float32)
            self.weights_sq = self.weights

        if is_batched:
            self.hist_bin_counts, self.hist_bin_errors = torch.vmap(
                self._make_histogram
            )(
                self.weights,
                self.weights_sq,
                density=density,
            )
        else:
            self.hist_bin_counts, self.hist_bin_errors = self._make_histogram(
                self.weights,
                self.weights_sq,
                density=density,
            )
        return self.hist_bin_counts, self.hist_bin_errors

    def projX(self, yrange: tuple[float, float] | None = None, is_batched=True):
        if yrange is None:
            h_bin_counts = self.hist_bin_counts
            h_bin_errors = self.hist_bin_errors
            h_bin_widths = self.h_ybin_widths
        else:
            ymin, ymax = yrange
            bmin = torch.bucketize(
                torch.as_tensor(ymin, dtype=torch.float32), self.ybins
            ).sub_(1)
            bmax = torch.bucketize(
                torch.as_tensor(ymax, dtype=torch.float32), self.ybins
            )
            h_bin_counts = self.hist_bin_counts[..., bmin:bmax, :]
            h_bin_errors = self.hist_bin_errors[..., bmin:bmax, :]
            h_bin_widths = self.h_ybin_widths[bmin:bmax]

        if is_batched:
            return torch.vmap(self._proj)(
                h_bin_counts, h_bin_errors, h_bin_widths=h_bin_widths
            )
        return self._proj(h_bin_counts, h_bin_errors, h_bin_widths=h_bin_widths)

    def projY(self, xrange: tuple[float, float] | None = None, is_batched=True):
        if xrange is None:
            h_bin_counts = self.hist_bin_counts
            h_bin_errors = self.hist_bin_errors
            h_bin_widths = self.h_xbin_widths
        else:
            xmin, xmax = xrange
            bmin = torch.bucketize(
                torch.as_tensor(xmin, dtype=torch.float32), self.xbins
            ).sub_(1)
            bmax = torch.bucketize(
                torch.as_tensor(xmax, dtype=torch.float32), self.xbins
            )
            h_bin_counts = self.hist_bin_counts[..., :, bmin:bmax]
            h_bin_errors = self.hist_bin_errors[..., :, bmin:bmax]
            h_bin_widths = self.h_xbin_widths[bmin:bmax]

        if is_batched:
            return torch.vmap(self._proj)(
                h_bin_counts, h_bin_errors, h_bin_widths=h_bin_widths
            )
        return self._proj(h_bin_counts, h_bin_errors, h_bin_widths=h_bin_widths)

    def _proj(self, h_bin_counts, h_bin_errors, *, h_bin_widths):
        if self.density:
            h_reduced = h_bin_counts @ h_bin_widths
            h_scale = torch.dot(h_reduced, self.h_ybin_widths)
            h_bin_sumw = (h_reduced).div_(h_scale)
            h_bin_err = (
                (h_bin_errors.square() @ h_bin_widths.square()).sqrt_().div_(h_scale)
            )
        else:
            h_bin_sumw = h_bin_counts.sum(-1)
            h_bin_err = h_bin_errors.square().sum(-1).sqrt_()

        return h_bin_sumw, h_bin_err

    def profileZ(self, z, weights=None, density=True, is_batched=False):
        if hasattr(self, "mask"):
            assert z.shape == self.mask.shape
            z = z[self.mask]

        assert z.shape == self.x.shape

        if weights is not None:
            weights = torch.as_tensor(weights, dtype=torch.float32)
            if hasattr(self, "mask"):
                weights = weights[..., self.mask]
            assert weights.shape[-1] == self.num_samples, (
                f"Got {weights.shape[-1]} for {self.num_samples} samples!"
            )
        else:
            weights = torch.ones(self.num_samples, dtype=torch.float32)

        eps = weights[weights > 0].min() * torch.finfo(torch.float32).eps
        if is_batched:
            shape = (weights.shape[0],) + self.shape
            prof_mean, prof_err = torch.vmap(
                partial(self._make_profile, self.one_hot_bin_indices, z)
            )(
                weights,
                density=density,
                eps=eps,
            )
        else:
            shape = self.shape
            prof_mean, prof_err = self._make_profile(
                self.one_hot_bin_indices, z, weights, density=density, eps=eps
            )
        return prof_mean.view(*shape), prof_err.view(*shape)

    def profileX(self, weights=None, density=True, is_batched=False):
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.float32)
            if hasattr(self, "mask"):
                self.weights = self.weights[..., self.mask]
            assert self.weights.shape[-1] == self.num_samples, (
                f"Got {weights.shape[-1]} for {self.num_samples} samples!"
            )
        else:
            self.weights = torch.ones(self.num_samples, dtype=torch.float32)

        eps = self.weights[self.weights > 0].min() * torch.finfo(torch.float32).eps

        if is_batched:
            return torch.vmap(
                partial(self._make_profile, self.one_hot_xbin_indices, self.y)
            )(
                self.weights,
                density=density,
                eps=eps,
            )

        return self._make_profile(
            self.one_hot_xbin_indices, self.y, self.weights, density=density, eps=eps
        )

    def profileY(self, weights=None, density=True, is_batched=False):
        if weights is not None:
            self.weights = torch.as_tensor(weights, dtype=torch.float32)
            if hasattr(self, "mask"):
                self.weights = self.weights[..., self.mask]
            assert self.weights.shape[-1] == self.num_samples, (
                f"Got {weights.shape[-1]} for {self.num_samples} samples!"
            )
        else:
            self.weights = torch.ones(self.num_samples, dtype=torch.float32)

        eps = self.weights[self.weights > 0].min() * torch.finfo(torch.float32).eps
        if is_batched:
            return torch.vmap(
                partial(self._make_profile, self.one_hot_ybin_indices, self.x)
            )(
                self.weights,
                density=density,
                eps=eps,
            )

        return self._make_profile(
            self.one_hot_ybin_indices, self.x, self.weights, density=density, eps=eps
        )

    def _make_histogram(self, weights, weights_sq, /, density=True):
        h_bin_sumw = torch.matmul(self.one_hot_bin_indices, weights)
        h_bin_sumw2 = torch.matmul(self.one_hot_bin_indices, weights_sq)

        if density:
            h_bin_norm = self.h_bin_scales.div(torch.sum(h_bin_sumw))
            return h_bin_sumw.mul_(h_bin_norm).view(
                *self.shape
            ), h_bin_sumw2.sqrt_().mul_(h_bin_norm).view(*self.shape)

        return h_bin_sumw.view(*self.shape), h_bin_sumw2.sqrt_().view(*self.shape)

    def _make_profile(self, one_hot_idx, arr, weights, /, density=True, eps=1e-22):
        arr_x_wt = arr.mul(weights)
        arr2_x_wt = arr.mul(arr_x_wt)
        h_bin_sumw = torch.mv(one_hot_idx, weights)
        h_bin_sumy = torch.mv(one_hot_idx, arr_x_wt)
        h_bin_sumy2 = torch.mv(one_hot_idx, arr2_x_wt)

        h_bin_meany = h_bin_sumy.div_(h_bin_sumw + eps)
        h_bin_vary = h_bin_sumy2.div_(h_bin_sumw + eps).sub_(h_bin_meany.square())

        return h_bin_meany, h_bin_vary.sqrt_()
