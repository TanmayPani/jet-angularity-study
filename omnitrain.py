"""Manual-loop training/inference drivers for the OmniFold classifier ensembles.

torchstrap dropped ``StatelessModule.fit/predict/compile`` in the manual-loop
refactor: training is now a hand-written ``torch.func.vmap`` + ``grad_and_value``
loop calling ``Adam.apply_gradient``, and callbacks are plain callables. These
helpers package that loop (and the inference path) so ``multifold.py`` can build
one set of (optionally ``torch.compile``d) callables per ensemble **once** and
reuse them across every OmniFold iteration.

Shapes: data loaders yield ``(input, target, weight)`` 3-tuples, each with a
leading ``(num_replicas, batch, ...)`` layout; everything is vmapped over the
replica dim. The classifier emits a single logit per jet; the loss is weighted
``binary_cross_entropy_with_logits`` (``reduction="mean"``), matching the old
``fit`` contract exactly.

Dropout is governed by ``module._base_model.training`` under ``functional_call``,
so we ``.train()`` for the grad pass and ``.eval()`` for validation/prediction;
the training vmap uses ``randomness="different"`` for independent per-replica
dropout masks.
"""

from __future__ import annotations
from functools import partial
from pathlib import Path

import torch
from torch import Tensor
from torch.func import vmap, grad_and_value
from torch.nn.functional import binary_cross_entropy_with_logits, sigmoid

from torchstrap.optimizer import Adam
from torchstrap.history import History
from torchstrap.callbacks import EarlyStopping, EpochScore, PrintLog, EpochTimer
from torchstrap.callbacks import LRScheduler


def _to_device(t, state):
    return None if t is None else t.to(device=state.device, non_blocking=True)


def _input_to_device(x, state):
    """Move the input batch to device AND cast to the compute dtype.

    The bin_counts CNN route stores `input` as uint8 on disk (4x smaller memmap so
    it stays in page cache -> no disk re-reads) and ships uint8 over H2D (4x smaller
    transfer). The model's first op (Log1p/Normalize) needs float, so cast on-device
    *after* the small uint8 copy. A `.to()` to the already-matching dtype (MLP route,
    float32 input) returns the tensor unchanged, so this is a no-op there.
    """
    x = x.to(device=state.device, dtype=state.param_buffer_dtype, non_blocking=True)
    if x.ndim == 5:
        x = x.to(memory_format=torch.channels_last_3d)
    return x


def _batch_weight_mass(w, x, state):
    """Per-replica `(N,)` weight mass of one batch (``sum(w)``), kept on-device.

    Used as the per-batch weight in ``EpochScore`` so the epoch score is the exact
    per-replica weighted mean. ``w`` is ``(N, B, 1)``; with ``w is None`` the mass is
    the uniform sample count ``B`` per replica (matching ``evaluate``'s mean fallback).
    """
    n = state.batch_size[0]
    if w is None:
        return torch.full((n,), float(x.shape[1]), device=state.device)
    return w.sum(dim=tuple(range(1, w.ndim)))  # sum over (B, 1) -> (N,)


def evaluate(module, params, buffers, x, y, sample_weights):
    y_pred = module(params, buffers, x)
    # Sample-weighted MEAN: normalize by sum(w), not the element count. PyTorch's
    # `binary_cross_entropy_with_logits(..., weight=w, reduction="mean")` divides by
    # N, which only equals the weighted average when mean(w)==1. For the OmniFold
    # reweighting ensembles (non-class-balanced; mean(w)~0.1) that "/N" makes the
    # reported loss ~mean(w)x too low (=0.07 vs =0.69 at init). Adam is ~scale-
    # invariant so training is barely affected, but the metric must be normalized to
    # be interpretable / comparable across ensembles.
    if sample_weights is None:
        return binary_cross_entropy_with_logits(y_pred, y)
    per_sample = binary_cross_entropy_with_logits(y_pred, y, reduction="none")
    return (per_sample * sample_weights).sum() / sample_weights.sum()


@torch.no_grad
def grad_and_loss(module, *, compile=False, compile_kwargs=None, autocast_dtype=None):
    """Vmapped per-replica ``(grads_dict, (N,) loss)`` callable for one ensemble.

    Differentiates the weighted loss w.r.t. the per-name param views; ``module``
    is the ensemble's ``StatelessModule`` (its ``forward`` is the functional call).

    ``autocast_dtype`` (e.g. ``torch.bfloat16``) runs the conv/linear compute under
    ``torch.autocast`` for the CNN ``bin_counts`` route — ~2x throughput and ~half the
    activation memory (the fwd activations retained for backward dominate training peak,
    so this is what unblocks the large batch). ``grad_and_value`` computes the backward
    *inside* the same call, so one autocast context covers fwd **and** bwd. Weights and
    the Adam state stay fp32; ``binary_cross_entropy_with_logits`` and the weighted-mean
    reduction in ``evaluate`` auto-stay fp32 under autocast, so the returned loss and the
    fp32 grads are unchanged in dtype — standard mixed precision, no ``GradScaler``
    (bf16 has fp32 exponent range). Reweighting already accepts this precision shift.
    """

    loss_fn = partial(evaluate, module)
    grad_loss = vmap(
        grad_and_value(loss_fn, argnums=0),
        in_dims=0,
        randomness="different",
    )
    if compile:
        grad_loss = torch.compile(grad_loss, **(compile_kwargs or {}))
    if autocast_dtype is None:
        return grad_loss

    def run(params, buffers, x, y, w):
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
            return grad_loss(params, buffers, x, y, w)

    return run


@torch.inference_mode()
def eval_loss(
    module, criterion, *, compile=False, compile_kwargs=None, autocast_dtype=None
):
    """Vmapped per-replica ``(N,)`` loss callable (no grad) for validation.

    ``autocast_dtype`` mirrors :func:`grad_and_loss` so the validation pass runs at the
    same precision as training (CNN route); the BCE reduction stays fp32 under autocast.
    """

    loss_fn = partial(evaluate, module)
    valid_loss = vmap(loss_fn, in_dims=(0, 0, 0, 0, 0))
    if compile:
        valid_loss = torch.compile(valid_loss, **(compile_kwargs or {}))
    if autocast_dtype is None:
        return valid_loss

    def run(params, buffers, x, y, w):
        with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
            return valid_loss(params, buffers, x, y, w)

    return run


@torch.inference_mode()
def predict_proba(
    module,
    *,
    chunk_size=None,
    compile=False,
    compile_kwargs=None,
    autocast_dtype=None,
    pad_to=None,
):
    """Vmapped forward (with ``non_linearity``) for inference reweighting.

    ``chunk_size`` chunks the vmap over the replica dimension to bound peak memory
    (the CNN ``bin_counts`` route); ``None`` means no chunking.

    ``autocast_dtype`` (e.g. ``torch.bfloat16``) runs the conv/linear compute under
    ``torch.autocast`` — ~2x throughput and ~half the activation memory on the CNN
    route. Weights stay fp32; the final ``sigmoid`` runs in fp32 (autocast keeps it
    there) so the returned probabilities are fp32. Reweighting tolerates this (the
    run already accepts TF32's ~1e-3 shift and clamps the ratios afterwards).

    ``pad_to`` zero-pads the batch (dim 1) up to this fixed size before the forward
    and trims the result back, so a ``torch.compile``d forward always sees ONE static
    batch shape. The inference loaders keep full coverage (so every loader has a
    differently-sized trailing batch); without this the compiled graph recompiles per
    distinct size and blows ``torch._dynamo`` recompile_limit, falling back to eager.
    The CNN is per-sample independent (conv + per-sample spatial pool), so the padded
    rows never affect the real outputs. Set ``pad_to`` to the max (full) batch size.
    """

    def fwd(params, buffers, x):
        return sigmoid(module(params, buffers, x))

    forward = vmap(fwd, in_dims=(0, 0, 0), chunk_size=chunk_size)
    if compile:
        forward = torch.compile(forward, **(compile_kwargs or {}))

    if autocast_dtype is None and pad_to is None:
        return forward

    def run(params, buffers, x):
        b = x.shape[1]
        if pad_to is not None and b < pad_to:
            pad = x.new_zeros((x.shape[0], pad_to - b, *x.shape[2:]))
            x = torch.cat([x, pad], dim=1)
        if autocast_dtype is not None:
            with torch.autocast(device_type=x.device.type, dtype=autocast_dtype):
                out = forward(params, buffers, x)
        else:
            out = forward(params, buffers, x)
        return out[:, :b] if pad_to is not None else out

    return run


def warmup_compiled(
    module,
    state,
    *,
    batch_size: int,
    feature_shape,
    grad_loss=None,
    valid_loss_fn=None,
    predict_forward=None,
    target_shape=(1,),
) -> None:
    """Trigger ``torch.compile`` for each compiled callable on dummy tensors of the
    real per-batch shape, so the first real batch pays no compile cost (and compile
    errors surface up-front).

    Uses the same ``train()`` / ``eval()`` + ``inference_mode`` context as the real
    loop, since dropout makes the train graph differ from the eval one. **State is not
    mutated** — ``grad_loss``'s grads are discarded and no Adam step is taken. The
    dummy batch must match the *static* training batch size (``drop_last=True`` when
    compiling), or the real partial batch would recompile.

    ``feature_shape`` is the per-sample input shape (e.g. ``(n_features,)`` for the MLP
    route, ``(C, H, W)`` for the CNN route); take it from ``dataset.input.shape[1:]``.
    """
    n = state.batch_size[0]
    dev = state.device
    dt = state.param_buffer_dtype
    x = torch.zeros((n, batch_size, *feature_shape), device=dev, dtype=dt)
    y = torch.zeros((n, batch_size, *target_shape), device=dev, dtype=dt)
    w = torch.ones((n, batch_size, *target_shape), device=dev, dtype=dt)
    base = module._base_model
    pd, bd = state.params_dict, state.buffers_dict

    if grad_loss is not None:
        base.train()
        grad_loss(pd, bd, x, y, w)  # compiles fwd+bwd; grads discarded, state untouched
    if valid_loss_fn is not None:
        base.eval()
        with torch.inference_mode():
            valid_loss_fn(pd, bd, x, y, w)
    if predict_forward is not None:
        base.eval()
        with torch.inference_mode():
            predict_forward(pd, bd, x)
    if dev.type == "cuda":
        torch.cuda.synchronize()


def load_compile_cache(path) -> bool:
    """Load a saved ``torch.compile`` artifact bundle (Mega-Cache) so a later run with
    the **same architecture/shapes** reuses compiled kernels instead of recompiling.

    Call ONCE at startup, before building/compiling the callables. No-op (returns
    ``False``) if the file is missing. Pair with :func:`save_compile_cache`. The
    on-disk Inductor/FX cache (``~/.cache/torch/inductor``) already persists kernels
    across runs automatically; this bundle additionally skips the re-tracing/lookup so
    warmup is near-instant.
    """
    p = Path(path)
    if not p.exists():
        return False
    torch.compiler.load_cache_artifacts(p.read_bytes())
    return True


def save_compile_cache(path) -> bool:
    """Persist every ``torch.compile`` artifact produced so far to ``path`` (Mega-Cache)
    for reuse by a later run. Call AFTER :func:`warmup_compiled`. Returns ``False`` if
    nothing has been compiled yet (e.g. ``compile_forward`` was off).
    """
    out = torch.compiler.save_cache_artifacts()
    if out is None:
        return False
    artifacts, _info = out
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(artifacts)
    return True


def fit_ensemble(
    module,
    state,
    train_loader,
    valid_loader,
    *,
    num_epochs: int,
    grad_loss,
    valid_loss_fn,
    patience: int,
    valid_every: int = 2,
    lr_policy=None,
    lr_kwargs=None,
    verbose: bool = False,
    name: str = "",
):  # -> History:
    """Train one ensemble in place; return a torchstrap ``History``.

    Each epoch: a vmapped grad pass over ``train_loader`` (fused Adam step), then
    an ``inference_mode`` validation pass; per-replica ``EarlyStopping`` freezes
    plateaued replicas (the fused optimizer then skips them via ``active_mask``)
    and ``restore_best`` rolls them to their best weights at the end. Losses stay
    on-device until the once-per-epoch ``EpochScore`` aggregation.

    ``valid_every`` runs the validation pass (and the ``EarlyStopping`` check) only
    every Nth epoch — and always on the final epoch — to cut the per-epoch cost when
    the validation set is large (for the detlvl ensemble the 50/50–75/25 split makes
    validation ~25% of the epoch). On skipped epochs the last computed validation loss
    is carried forward so the ``PrintLog`` columns and ``History`` lists stay
    epoch-aligned. NOTE: with ``valid_every>1`` the ``patience`` is counted in
    *validation events*, not epochs (effective epoch patience is ``patience*valid_every``).
    """
    history = History()
    score = EpochScore()
    early = EarlyStopping(patience=patience, verbose=verbose)
    # Per-replica, freeze-aware LR scheduler (torchstrap-native; writes the (N,) lr in
    # place into state.optimizer_state["lr"], consumed directly by the fused Adam step).
    # Score-driven policies (ReduceLROnPlateau) tick on validation events (need the
    # (N,) valid_score); closed-form policies (CosineAnnealingLR, …) tick every epoch.
    scheduler = LRScheduler(lr_policy, **(lr_kwargs or {})) if lr_policy else None
    sched_needs_score = (
        scheduler is not None and scheduler._get_policy_cls().needs_score
    )
    base = module._base_model
    # Carried forward to keep log/history aligned on epochs where validation is skipped.
    last_valid_host = torch.full((int(state.batch_size[0]),), float("nan"))

    # Per-epoch table (restores what the removed `.fit` default PrintLog printed).
    log = PrintLog() if verbose else None
    timer = EpochTimer()
    if verbose and name:
        print(
            f"--- training {name}: {num_epochs} epochs, {state.batch_size[0]} replicas ---"
        )

    for _epoch in range(num_epochs):
        # history.new_epoch()
        timer.tic()

        base.train()
        batch_losses: list[Tensor] = []
        batch_wsums: list[Tensor] = []  # per-batch per-replica sum(w), stays on device
        for x, y, w in train_loader:
            x = _input_to_device(x, state)
            y, w = _to_device(y, state), _to_device(w, state)
            grads, loss = grad_loss(state.params_dict, state.buffers_dict, x, y, w)
            Adam.apply_gradient(state, grads)
            batch_losses.append(loss.detach())
            batch_wsums.append(_batch_weight_mass(w, x, state))
        # Weight each batch by its per-replica sum(w) so the epoch score is the EXACT
        # per-replica weighted mean (matches evaluate's per-batch sum(w) normalization).
        train_score = score(batch_losses, batch_wsums)  # (N,)
        train_host = train_score.detach().cpu()  # one host transfer per metric

        # Validate (and check EarlyStopping) every `valid_every` epochs, always on the
        # first and last epoch (the first gives a real baseline for the log/early-stop;
        # without it row 1 would be NaN). Skipped epochs reuse `last_valid_host` so the
        # log columns and History lists stay epoch-aligned.
        do_valid = (
            valid_every <= 1
            or _epoch == 0
            or (_epoch + 1) % valid_every == 0
            or (_epoch + 1) == num_epochs
        )
        stop = False
        if do_valid:
            base.eval()
            valid_losses: list[Tensor] = []
            valid_wsums: list[Tensor] = []
            with torch.inference_mode():
                for x, y, w in valid_loader:
                    x = _input_to_device(x, state)
                    y, w = _to_device(y, state), _to_device(w, state)
                    valid_losses.append(
                        valid_loss_fn(
                            state.params_dict, state.buffers_dict, x, y, w
                        ).detach()
                    )
                    valid_wsums.append(_batch_weight_mass(w, x, state))
            valid_score = score(valid_losses, valid_wsums)  # (N,)
            last_valid_host = valid_score.detach().cpu()
            stop = early(state, valid_score)  # True once every replica is frozen

        # Advance the LR schedule (after EarlyStopping, so a just-frozen replica holds
        # its lr). Plateau policies tick on validation events with the (N,) valid_score;
        # closed-form ones tick every epoch.
        if scheduler is not None:
            if sched_needs_score:
                if do_valid:
                    # BUGFIX: plateau policies (ReduceLROnPlateau) must monitor the
                    # *validation* score, not the training score. Feeding train_score
                    # let the LR never anneal (train loss keeps falling via overfit),
                    # which drove the OmniFold per-iteration weight runaway.
                    # scheduler(state, score=train_score)  # old (train loss => LR never dropped)
                    scheduler(state, score=valid_score)
            else:
                scheduler(state)

        history.append("train_loss", train_host)
        history.append("valid_loss", last_valid_host)
        if log is not None:
            log_metrics = dict(
                epoch=_epoch + 1,
                train_loss=train_host,
                valid_loss=last_valid_host,
                dur=timer.toc(),
            )
            if scheduler is not None:
                # Pass the (N,) lr tensor (not a pre-reduced scalar) so PrintLog renders
                # it as `mean+/-std` across replicas — same as the loss columns — which
                # surfaces per-replica LR divergence once the scheduler anneals them
                # independently. One tiny (N,)-float D2H per epoch.
                log_metrics["lr"] = state.optimizer_state["lr"].detach().cpu()
            log(**log_metrics)

        if stop:
            break

    early.restore_best(state)
    return history
