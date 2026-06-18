"""Density-ratio estimation — the classifier likelihood-ratio primitive behind one
pluggable-estimator interface.

Stage-1 groundwork (#4). "Reweight sample A to look like sample B" is the core operation of
the whole analysis (it *is* OmniFold's per-step reweight). Today the primitive is scattered:
the training atoms live in `omnitrain` but `build_classifier` / `reweigh_samples` sit inside
`multifold`, so the reweighter imports them *from the unfolder*, and the train->odds->clamp->
calibrate sequence is duplicated. This module gives that primitive one home and one
interface, so an estimator (classifier / GP-marginal / BDT) is a swappable choice.

`Estimator.fit_predict(pos, neg, predict_table, *, spec, label) -> r_mean` returns the
per-row density ratio `r = p_pos(x)/p_neg(x)` on `predict_table` (defaults to `neg`), such
that `neg` reweighted by `r` matches `pos`.

Stage 1 keeps the proven implementation: `ClassifierEstimator` delegates to the already
validated `reweight_embedding.estimate_density_ratio` (equalize -> flatten -> to_tensordict
-> fit_ensemble -> clamped odds -> geometric-mean -> calibrate). Stage 2 inlines that body
here and has `multifold` import `build_classifier`/`reweigh_samples` from this module.
`GPMarginalEstimator` / `BDTEstimator` are interface-ready extension points.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from core import features as _features
from core import paths as _paths


@runtime_checkable
class Estimator(Protocol):
    """Estimates a per-row density ratio between two arrow tables over a `FeatureSpec`."""

    def fit_predict(
        self,
        pos_table,
        neg_table,
        predict_table=None,
        *,
        spec: "_features.FeatureSpec",
        label: str = "",
        **overrides,
    ):
        """Return `r_mean` (np.float32, shape `(len(predict_table or neg_table),)`).

        `**overrides` are optional per-call estimator knobs (e.g. `weight_flatten=False`
        for the gen pull-back, where pos/neg are the same jets); estimators ignore knobs
        they don't support.
        """
        ...


class ClassifierEstimator:
    """Binary-classifier likelihood-ratio estimator over the joint feature vector.

    The robust default: captures all feature correlations in one ND function, with the
    runaway-prevention guards (clamp `[0.1, 10]`, valid-monitored LR, geometric-mean replica
    collapse, post-hoc calibration). Stage 1 delegates the body to the validated
    `reweight_embedding.estimate_density_ratio`; the constructor args mirror it.
    """

    def __init__(
        self,
        cfg,
        *,
        num_replicas: int = 1,
        num_epochs: int | None = None,
        device=None,
        cache_dir=None,
        weight_flatten: bool = True,
        audit: bool = False,
    ):
        self.cfg = cfg
        self.num_replicas = int(num_replicas)
        self.num_epochs = int(num_epochs) if num_epochs is not None else int(cfg["num_epochs"])
        self.device = device if device is not None else cfg.device
        self.cache_dir = cache_dir
        self.weight_flatten = weight_flatten
        self.audit = audit
        self.last_history = None

    def fit_predict(
        self, pos_table, neg_table, predict_table=None, *, spec, label="",
        weight_flatten=None, audit=None,
    ):
        # Lazy import keeps `core` import-light and avoids any future import cycle if
        # reweight_embedding is later rewired onto core.
        from reweight_embedding import estimate_density_ratio

        cache_dir = self.cache_dir or str(
            _paths.reweight_embedding_cache(self.cfg, "scratch", spec.name)
        )
        r_mean, history = estimate_density_ratio(
            pos_table,
            neg_table,
            feature_mode=spec.name,
            cfg=self.cfg,
            num_replicas=self.num_replicas,
            num_epochs=self.num_epochs,
            device=self.device,
            cache_dir=cache_dir,
            label=label,
            predict_table=predict_table,
            weight_flatten=self.weight_flatten if weight_flatten is None else weight_flatten,
            audit=self.audit if audit is None else audit,
        )
        self.last_history = history
        return r_mean


class GPMarginalEstimator:
    """Legacy sequential-GP *marginal* estimator (the omnisequential method) behind the same
    interface. Interface-ready; the adapter over `omnisequential.propagate_values` is wired
    in Stage 2 (kept out of Stage 1 to avoid importing the marimo notebook's setup block)."""

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs

    def fit_predict(self, pos_table, neg_table, predict_table=None, *, spec, label="", **overrides):
        raise NotImplementedError(
            "GPMarginalEstimator: Stage 2 — adapt omnisequential.propagate_values (greedy "
            "worst-χ² 1-D marginal + GP smoothing). Use ClassifierEstimator for now."
        )


class BDTEstimator:
    """Gradient-boosted-tree reweighter (e.g. hep_ml.GBReweighter) behind the same interface.
    Interface-ready stub; not implemented (no hep_ml dependency yet)."""

    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.kwargs = kwargs

    def fit_predict(self, pos_table, neg_table, predict_table=None, *, spec, label="", **overrides):
        raise NotImplementedError(
            "BDTEstimator: not implemented (would add a hep_ml.GBReweighter dependency)."
        )


def density_ratio(
    pos_table, neg_table, *, spec, estimator: Estimator, predict_table=None, label="", **overrides
):
    """Estimate `r = p_pos/p_neg` on `predict_table` (default `neg_table`) using `estimator`.

    `neg_table` reweighted by the returned per-row `r` matches `pos_table` over the joint
    feature space defined by `spec`. `**overrides` forwards per-call estimator knobs (e.g.
    `weight_flatten=False`).
    """
    return estimator.fit_predict(
        pos_table, neg_table, predict_table, spec=spec, label=label, **overrides
    )
