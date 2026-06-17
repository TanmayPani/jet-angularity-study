"""Lightweight persistence for torchstrap ensemble model weights.

`multifold.py` trains four classifier ensembles per OmniFold iteration but only
the per-jet `w_unfolding` weights survive a run. For the explainability study we
need the trained models back. torchstrap dropped `State.to_file()`/`from_file()`
in the churten->torchstrap rename; the supported persistence path is now
`State.state_dict()`/`load_state_dict()`.

These helpers persist only the stacked params + buffers (the `(num_replicas, ...)`
tensors the model forward needs) and skip the optimizer moments, which the
post-hoc attribution never uses. That keeps the on-disk footprint ~3x smaller
than a full `state_dict()`. The saved `buffers` include the `Normalize`
(z-norm) mean/stddev, so loading restores the in-model input transform without
recomputing it.
"""

from __future__ import annotations

from pathlib import Path

import torch


def save_model_weights(state, path: str | Path) -> None:
    """Dump a State's params + buffers (CPU) to ``path`` via ``torch.save``.

    Mirrors the ``parameters``/``buffers`` blocks of ``State.state_dict()`` but
    omits the optimizer state. Each leaf keeps its leading ``num_replicas`` dim.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        # torchstrap renamed the aliased views: param_dict -> params_dict,
        # buffer_dict -> buffers_dict (still per-name (num_replicas, ...) views).
        # "parameters": {
        #     name: param.detach().cpu() for name, param in state.param_dict.items()
        # },
        # "buffers": {
        #     name: buffer.detach().cpu() for name, buffer in state.buffer_dict.items()
        # },
        "parameters": {
            name: param.detach().cpu() for name, param in state.params_dict.items()
        },
        "buffers": {
            name: buffer.detach().cpu() for name, buffer in state.buffers_dict.items()
        },
    }
    torch.save(payload, path)


@torch.no_grad()
def load_model_weights(state, path: str | Path) -> None:
    """In-place copy saved params + buffers into a freshly-``init``'d skeleton.

    ``state`` must be a State built with the same architecture / num_replicas as
    the one that was saved (e.g. via ``StatelessModule.init(MLP, Adam, ...)``);
    the leaf tensors are copied in place so the aliasing invariant with the
    optimizer state is preserved. Values land on whatever device ``state`` lives
    on (the saved payload is CPU).
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    # param_dict -> params_dict, buffer_dict -> buffers_dict (aliased views; the
    # in-place .copy_ writes through into the consolidated flat buffer).
    # for name, param in state.param_dict.items():
    #     param.copy_(payload["parameters"][name])
    # for name, buffer in state.buffer_dict.items():
    #     buffer.copy_(payload["buffers"][name])
    for name, param in state.params_dict.items():
        param.copy_(payload["parameters"][name])
    for name, buffer in state.buffers_dict.items():
        buffer.copy_(payload["buffers"][name])


def save_ensemble_state(state, path: str | Path) -> None:
    """Dump a State's FULL trainable state (CPU) to ``path`` for resume.

    Unlike ``save_model_weights`` (params + buffers only, for post-hoc XAI), this
    persists everything needed to warm-restart the OmniFold loop bit-faithfully:
    the consolidated ``params`` / ``buffers`` flat buffers, every tensor leaf of
    ``optimizer_state`` (Adam moments ``exp_avgs`` / ``exp_avg_sqs`` / optional
    ``max_exp_avg_sqs``, plus ``state_steps`` and the per-replica ``lr`` / ``beta*``
    / ``eps`` / ``weight_decay``), and the per-replica ``active_mask``. The static
    Adam-variant flags (``amsgrad`` / ``maximize`` / ``decoupled_weight_decay``,
    stored as ``NonTensorData``) are NOT saved — ``Adam.init`` rebuilds them
    identically when the skeleton is reconstructed. Payload is all tensors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": state.params.detach().cpu(),
        "buffers": state.buffers.detach().cpu(),
        "optimizer": {
            k: v.detach().cpu()
            for k, v in state.optimizer_state.items()
            if torch.is_tensor(v)
        },
        "active_mask": state.active_mask.detach().cpu(),
    }
    torch.save(payload, path)


@torch.no_grad()
def load_ensemble_state(state, path: str | Path) -> None:
    """In-place restore of a full State (params + buffers + optimizer + mask).

    ``state`` must be a freshly-``init``'d skeleton with the same architecture /
    num_replicas / Adam config as the saved one (so ``optimizer_state`` already has
    matching keys/shapes). Every leaf is copied in place so the consolidated-buffer
    aliasing (params_dict views, optimizer moment storage) is preserved. The saved
    payload is CPU; values land on whatever device ``state`` lives on.
    """
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    state.params.copy_(payload["params"])
    state.buffers.copy_(payload["buffers"])
    for key, value in payload["optimizer"].items():
        state.optimizer_state[key].copy_(value)
    state.active_mask.copy_(payload["active_mask"])
