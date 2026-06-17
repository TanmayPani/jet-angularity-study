"""MultiFold explainability for the bin_counts 2D-CNN route — Grad-CAM.

Companion to `feature_importance.py` (which does Integrated Gradients for the
`angularities` MLP). Here the classifier is a `Conv2dNN` over a 2-channel 9x9
jet image (channel 0 = charged-constituent count, channel 1 = neutral count over
the 9 pT x 9 dR constituent grid), so feature attribution is *spatial*: we use
**Grad-CAM** to produce an importance map directly on the (pT, dR) grid, plus a
**per-channel input saliency** (input x gradient) that separates the charged vs
neutral contributions Grad-CAM merges at the feature-map level.

`Conv2dNN` keeps the grid at 9x9 through every conv (`padding=1`, no stride/pool
between convs) and only collapses it at the final `AdaptiveAvgPool2d`, so the
Grad-CAM map on the last post-ReLU conv feature map is already 1:1 with the input
grid — no upsampling. The cell axes map onto `preprocessing.con_pt_bins` /
`con_dr_bins`.

Loads the per-iteration `detlvl` / `partlvl` ensembles persisted by `multifold.py`
(requires `save_model_states: true`, then a bin_counts run).

CPU-ONLY: the GPU is reserved for other running jobs, so device is pinned to
"cpu" here regardless of `cfg.device`. The CNN is tiny and the held-out sample is
small, so per-replica forward+backward on CPU is cheap.

Open with:  uv run marimo edit feature_importance_cnn.py
Plan: /home/tanmaypani/.claude/plans/analyze-this-project-the-expressive-firefly.md
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="columns")

with app.setup:
    from pathlib import Path

    import numpy as np
    import torch
    from tensordict import TensorDict
    import matplotlib.pyplot as plt

    import marimo as mo

    plt.style.use("default")
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"

    from torchstrap.utils.nn.archs import Conv2dNN
    from torchstrap.utils.nn.transform import Normalize

    from dataset import build_input_transform
    from preprocessing import con_pt_bins, con_dr_bins, N_PT, N_DR
    from config import load_config

    # --- run configuration -------------------------------------------------
    # The target sysvar (whose models multifold.py persisted) comes from config.
    cfg = load_config()
    sys_var = cfg.sys_var

    # --- bin_counts route settings are HARDCODED here ----------------------
    # This notebook only ever targets the bin_counts CNN models, so the route-
    # specific settings are pinned locally instead of read from cfg. That keeps
    # it working even when runtime-files/config.json is switched to another
    # feature_mode (e.g. angularities) for a live run — otherwise cfg's
    # feature_mode / input_transform / cnn_channels (and cfg.features_root,
    # which derives from cfg["feature_mode"]) would all resolve for the wrong
    # route.
    feature_mode = "bin_counts"
    input_transform_mode = "log1p_per_channel_z_norm"  # per-channel (C,H,W) transform
    cnn_channels = (32, 64)                            # Conv2dNN conv widths

    # CPU-ONLY: do not touch cfg.device — the GPU is in use by other jobs.
    device = "cpu"

    root_dir = cfg.dataset_root / "features" / feature_mode
    tensordict_src = root_dir / "tensordicts" / str(sys_var)
    embed_dir = root_dir / "embedding" / str(sys_var)
    model_states_dir = embed_dir / "model_states"
    fig_dir = Path("./outputs/feature_importance") / str(sys_var) / feature_mode
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Per-run settings (ensemble size, iteration count, dropout) must match the
    # bin_counts run that produced model_states — NOT the live config, which may
    # be on a different feature_mode/run (a mismatched num_replicas would index
    # the stacked payload out of range in load_replica). multifold.py copies its
    # config verbatim into embed_dir/config.json, so read them from that snapshot
    # when present and fall back to the live config otherwise.
    _snap = embed_dir / "config.json"
    run_cfg = load_config(_snap) if _snap.exists() else cfg
    num_replicas = run_cfg["num_replicas"]
    num_iterations = run_cfg["num_iterations"]
    dropout_prob = run_cfg.dropout_prob

    # Sampling / chunking knobs.
    SEED = 0
    N_HELDOUT = 25_000      # jets attributed per (step, iteration)
    CAM_CHUNK = 2048        # jets per forward/backward chunk (memory bound)
    IN_CHANNELS = 2
    CHANNEL_NAMES = ("charged", "neutral")

    STEPS = ("detlvl", "partlvl")


@app.function
def load_sim_images(td_path: Path):
    """Return (X CPU float32 (M,2,9,9), full sim count) for a tensordict.

    Attribution runs on the MC/sim side (`is_data == False`) — the physical jets
    the unfolding reweights — over a seeded held-out subsample.
    """
    td = TensorDict.load_memmap(td_path)
    n = len(td)
    is_data = (
        td["is_data"].bool() if "is_data" in td.keys() else (td["target"] > 0.5)
    )
    sim_idx = torch.arange(n)[~is_data]

    g = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(sim_idx), generator=g)[:N_HELDOUT]
    eval_idx = sim_idx[perm]

    shape = tuple(td["input"].shape[1:])
    assert shape == (IN_CHANNELS, N_PT, N_DR), (
        f"tensordict input shape {shape} != (2, {N_PT}, {N_DR}). The on-disk "
        "tensordict is not the bin_counts image — re-run preprocessing/multifold "
        "with feature_mode=bin_counts and redo_datasets=true."
    )

    X = td["input"][eval_idx].to(torch.float32)          # (M, 2, 9, 9) CPU
    return X, int(len(sim_idx))


@app.function
def build_base(sample_img: torch.Tensor) -> Conv2dNN:
    """A concrete (non-stateless) Conv2dNN matching multifold.build_classifier.

    `module._base_model` lives on the `meta` device (no storage), so we rebuild a
    real Conv2dNN with the same constructor args instead. The placeholder
    `input_transform` only fixes the Normalize buffer shapes; the trained
    per-channel stats are overwritten by `load_replica`. Normalize is forced
    out-of-place so the input-saliency backward (which differentiates through the
    transform) does not hit an in-place-on-a-leaf autograd error.
    """
    transform = build_input_transform(input_transform_mode, sample_img, device=device)
    base = Conv2dNN(
        in_channels=IN_CHANNELS,
        conv_channels=cnn_channels,
        head_sizes=(1,),
        dropout_prob=dropout_prob,
        input_transform=transform,
        device=device,
    )
    base.eval()                                  # dropout -> deterministic no-op
    for m in base.modules():
        if isinstance(m, Normalize):
            m.inplace = False
    return base


@app.function
def load_replica(base: Conv2dNN, payload: dict, r: int) -> None:
    """Copy replica `r` of a saved (stacked) payload into the concrete `base`.

    `payload` is what `model_io.save_model_weights` dumped:
    `{"parameters": {name: (R,...)}, "buffers": {name: (R,...)}}`. The names come
    from the same Conv2dNN constructor, so the per-replica slice is exactly
    `base.state_dict()`. `strict=True` will shout if the architectures drift.
    """
    sd = {name: t[r] for name, t in payload["parameters"].items()}
    sd |= {name: t[r] for name, t in payload["buffers"].items()}
    base.load_state_dict(sd, strict=True)


@app.function
def target_layer(base: Conv2dNN) -> torch.nn.Module:
    """The last post-ReLU conv feature map: the layer before AdaptiveAvgPool2d."""
    layers = list(base.layers)
    for i, m in enumerate(layers):
        if isinstance(m, torch.nn.AdaptiveAvgPool2d):
            return layers[i - 1]
    raise RuntimeError("Conv2dNN has no AdaptiveAvgPool2d to anchor Grad-CAM on.")


@app.function
def attribute_one(base: Conv2dNN, X_cpu: torch.Tensor):
    """Grad-CAM + per-channel input saliency for one loaded replica.

    Returns (cam (9,9), signed_cam (9,9), saliency (2,9,9)), each a mean over the
    held-out jets. A single backward per chunk yields both the conv-feature-map
    gradient (Grad-CAM) and the raw-input gradient (per-channel saliency):

      Grad-CAM:  alpha_k = mean_spatial dy/dA_k ;  L = ReLU(sum_k alpha_k A_k)
      saliency:  |x . dy/dx|  averaged over jets, kept per (channel, cell)
    """
    target = target_layer(base)
    store: dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):
        out.retain_grad()
        store["A"] = out

    h = target.register_forward_hook(hook)
    cam_sum = torch.zeros(N_PT, N_DR)
    scam_sum = torch.zeros(N_PT, N_DR)
    sal_sum = torch.zeros(IN_CHANNELS, N_PT, N_DR)
    n = 0
    try:
        for xb in X_cpu.split(CAM_CHUNK):
            xb = xb.clone().requires_grad_(True)
            base.zero_grad(set_to_none=True)
            logits = base(xb)                          # (B, 1) ; hook stored A
            A = store["A"]                             # (B, C, 9, 9)
            logits.sum().backward()                    # per-sample grads

            gA = A.grad                                # (B, C, 9, 9)
            alpha = gA.mean(dim=(2, 3), keepdim=True)  # (B, C, 1, 1)
            signed = (alpha * A).sum(dim=1)            # (B, 9, 9)
            cam_sum += torch.relu(signed).sum(0).detach()
            scam_sum += signed.sum(0).detach()
            sal_sum += (xb * xb.grad).detach().abs().sum(0)
            n += xb.shape[0]
    finally:
        h.remove()
    return cam_sum / n, scam_sum / n, sal_sum / n


@app.function
def attribute_step(td_path: Path, step: str):
    """Replica-averaged Grad-CAM / signed-CAM / saliency for every iteration.

    Returns dict with keys `cam`,`signed` of shape (n_iter, 9, 9) and `saliency`
    of shape (n_iter, 2, 9, 9); plus `n_jets`. Empty dict if no state files.
    """
    X, _ = load_sim_images(td_path)
    base = build_base(X[:CAM_CHUNK])

    cams, signeds, sals = [], [], []
    for it in range(num_iterations):
        wpath = model_states_dir / f"iter{it:02d}_{step}.pt"
        if not wpath.exists():
            continue
        payload = torch.load(wpath, map_location="cpu", weights_only=True)

        cam_r, signed_r, sal_r = [], [], []
        for r in range(num_replicas):
            load_replica(base, payload, r)
            cam, signed, sal = attribute_one(base, X)
            cam_r.append(cam.numpy())
            signed_r.append(signed.numpy())
            sal_r.append(sal.numpy())
        cams.append(np.mean(cam_r, axis=0))            # (9, 9)
        signeds.append(np.mean(signed_r, axis=0))      # (9, 9)
        sals.append(np.mean(sal_r, axis=0))            # (2, 9, 9)

    if not cams:
        return {}
    return {
        "cam": np.stack(cams),
        "signed": np.stack(signeds),
        "saliency": np.stack(sals),
        "n_jets": int(X.shape[0]),
    }


@app.function
def grid_heatmap(ax, grid, title, cmap, *, center_zero=False):
    """imshow a (N_PT, N_DR) map with pT rows / dR cols labeled by the bin edges.

    `origin="lower"` puts the lowest-pT row at the bottom. Cells are bounded by
    the constituent pT / dR bin edges from preprocessing.
    """
    vmax = float(np.abs(grid).max()) or 1.0
    vlim = (-vmax, vmax) if center_zero else (None, None)
    im = ax.imshow(
        grid, aspect="auto", origin="lower", cmap=cmap, vmin=vlim[0], vmax=vlim[1]
    )
    ax.set_xticks(np.arange(N_DR + 1) - 0.5)
    ax.set_xticklabels([f"{e:g}" for e in con_dr_bins], rotation=90, fontsize=6)
    ax.set_yticks(np.arange(N_PT + 1) - 0.5)
    ax.set_yticklabels([f"{e:g}" for e in con_pt_bins], fontsize=6)
    ax.set_xlabel(r"constituent $\Delta R$", fontsize=8)
    ax.set_ylabel(r"constituent $p_T$ [GeV]", fontsize=8)
    ax.set_title(title, fontsize=9)
    return im


@app.cell
def _check_states():
    have_states = model_states_dir.is_dir() and any(
        model_states_dir.glob("iter*_*.pt")
    )
    if have_states:
        msg = mo.md(
            f"**Model states found** in `{model_states_dir}`.\n\n"
            f"sysvar=`{sys_var}`  ·  replicas={num_replicas}  ·  "
            f"iterations={num_iterations}  ·  device=`{device}` (CPU-pinned)"
        )
    else:
        msg = mo.md(
            f"⚠️ **No model states** under `{model_states_dir}`.\n\n"
            "Set `save_model_states: true` in `runtime-files/config.json`, set "
            "`feature_mode: bin_counts`, and re-run `uv run multifold.py` for "
            "this sysvar first."
        )
    msg
    return (have_states,)


@app.cell
def _compute(have_states):
    # results[step] = {cam, signed, saliency, n_jets} (replica-averaged per iter).
    results: dict[str, dict] = {}
    if have_states:
        td_paths = {
            "detlvl": tensordict_src / "det_lvl",
            "partlvl": tensordict_src / "part_lvl",
        }
        for _step in STEPS:
            _res = attribute_step(td_paths[_step], _step)
            if _res:
                results[_step] = _res

    status = mo.md(
        "**Grad-CAM computed** for: "
        + ", ".join(f"{k} ({v['n_jets']:,} jets)" for k, v in results.items())
        if results
        else "_no attribution computed yet (need model states)._"
    )
    status
    return (results,)


@app.cell
def _plot_gradcam(results: dict[str, dict]):
    # Deliverable 1: replica-mean Grad-CAM heatmap per step, final iteration.
    if results:
        _fig, _axes = plt.subplots(
            1, len(results), figsize=(5.5 * len(results), 5), squeeze=False
        )
        for _ax, (_step, _res) in zip(_axes[0], results.items()):
            _im = grid_heatmap(
                _ax, _res["cam"][-1], f"{_step}: Grad-CAM (final iter)", "magma"
            )
            _fig.colorbar(_im, ax=_ax, fraction=0.046, label="ReLU(Grad-CAM)")
        _fig.suptitle("Grad-CAM importance on the (pT, ΔR) constituent grid")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "gradcam_final.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_Grad-CAM pending model states._")
    _out
    return


@app.cell
def _plot_signed(results: dict[str, dict]):
    # Deliverable 2: signed CAM — red regions push data-like, blue push sim-like.
    if results:
        _fig, _axes = plt.subplots(
            1, len(results), figsize=(5.5 * len(results), 5), squeeze=False
        )
        for _ax, (_step, _res) in zip(_axes[0], results.items()):
            _im = grid_heatmap(
                _ax, _res["signed"][-1], f"{_step}: signed CAM (final iter)",
                "RdBu_r", center_zero=True,
            )
            _fig.colorbar(_im, ax=_ax, fraction=0.046, label="signed CAM")
        _fig.suptitle("Signed Grad-CAM: data-like (red) vs sim-like (blue) regions")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "gradcam_signed_final.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_Signed CAM pending model states._")
    _out
    return


@app.cell
def _plot_channel_saliency(results: dict[str, dict]):
    # Deliverable 3: per-channel input saliency (charged vs neutral), final iter.
    if results:
        _nrow = len(results)
        _fig, _axes = plt.subplots(
            _nrow, IN_CHANNELS, figsize=(5.5 * IN_CHANNELS, 5 * _nrow), squeeze=False
        )
        for _row, (_step, _res) in enumerate(results.items()):
            _sal = _res["saliency"][-1]                # (2, 9, 9)
            for _c in range(IN_CHANNELS):
                _im = grid_heatmap(
                    _axes[_row][_c], _sal[_c],
                    f"{_step}: {CHANNEL_NAMES[_c]} |x·∂logit/∂x|", "viridis",
                )
                _fig.colorbar(_im, ax=_axes[_row][_c], fraction=0.046)
        _fig.suptitle("Per-channel input saliency (charged vs neutral)")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "channel_saliency_final.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_Channel saliency pending model states._")
    _out
    return


@app.cell
def _plot_evolution(results: dict[str, dict]):
    # Deliverable 4: Grad-CAM map across OmniFold iterations (if num_iterations>1).
    if results and num_iterations > 1:
        _figs = []
        for _step, _res in results.items():
            _n_it = _res["cam"].shape[0]
            _fig, _axes = plt.subplots(1, _n_it, figsize=(4.2 * _n_it, 4), squeeze=False)
            for _it, _ax in enumerate(_axes[0]):
                _im = grid_heatmap(_ax, _res["cam"][_it], f"{_step} iter {_it}", "magma")
                _fig.colorbar(_im, ax=_ax, fraction=0.046)
            _fig.suptitle(f"{_step}: Grad-CAM evolution across iterations")
            _fig.tight_layout()
            _fig.savefig(fig_dir / f"gradcam_evolution_{_step}.png", dpi=150)
            _figs.append(_fig)
        _out = mo.vstack(_figs)
    else:
        _out = mo.md("_Evolution heatmap needs num_iterations > 1 and results._")
    _out
    return


@app.cell
def _plot_examples(have_states):
    # Deliverable 5 (optional): a few single-jet Grad-CAM maps over their input
    # occupancy, using the final-iteration detlvl replica 0.
    _out = mo.md("_Example single-jet CAMs pending model states._")
    # Use the latest detlvl state that actually exists. The run may have saved
    # fewer iterations than the live num_iterations (e.g. a 1-iteration run saves
    # only iter00), so indexing iter{num_iterations-1} can miss the file.
    _state_files = (
        sorted(model_states_dir.glob("iter*_detlvl.pt"))
        if model_states_dir.is_dir()
        else []
    )
    if have_states and _state_files:
        _wpath = _state_files[-1]
        _X, _ = load_sim_images(tensordict_src / "det_lvl")
        _base = build_base(_X[:CAM_CHUNK])
        _payload = torch.load(_wpath, map_location="cpu", weights_only=True)
        load_replica(_base, _payload, 0)

        # Pick the 4 jets with the most total occupancy (busiest images).
        _busy = _X.flatten(1).sum(1)
        _pick = torch.topk(_busy, 4).indices
        _target = target_layer(_base)
        _store: dict[str, torch.Tensor] = {}

        def _hook(_m, _i, o):           # must return None, else it replaces output
            o.retain_grad()
            _store["A"] = o

        _h = _target.register_forward_hook(_hook)
        try:
            _xb = _X[_pick].clone().requires_grad_(True)
            _base.zero_grad(set_to_none=True)
            _logits = _base(_xb)
            _A = _store["A"]
            _logits.sum().backward()
            _alpha = _A.grad.mean(dim=(2, 3), keepdim=True)
            _cams = torch.relu((_alpha * _A).sum(1)).detach().numpy()   # (4, 9, 9)
        finally:
            _h.remove()

        _imgs = _X[_pick].numpy()                       # (4, 2, 9, 9)
        _fig, _axes = plt.subplots(4, 3, figsize=(12, 15))
        for _j in range(4):
            grid_heatmap(_axes[_j][0], _imgs[_j, 0], f"jet {int(_pick[_j])}: charged", "Greys")
            grid_heatmap(_axes[_j][1], _imgs[_j, 1], "neutral", "Greys")
            grid_heatmap(_axes[_j][2], _cams[_j], "Grad-CAM", "magma")
        _fig.suptitle("Example jets: input occupancy (charged/neutral) + Grad-CAM")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "gradcam_examples.png", dpi=150)
        _out = _fig
    _out
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
