"""MultiFold explainability — Integrated-Gradients feature importance.

Loads the per-iteration `detlvl` / `partlvl` classifier ensembles persisted by
`multifold.py` (requires `save_model_states: true` in config, then a run) and
attributes each iteration's reweighting to the input features via Integrated
Gradients, plus a data-only partial-dependence baseline on `w_unfolding`.

Open with:  uv run marimo edit feature_importance.py
Plan: /home/tanmaypani/.claude/plans/analyze-this-project-the-expressive-firefly.md
"""

import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import json
    from pathlib import Path

    import numpy as np
    import torch
    from torch.func import vmap, grad
    from tensordict import TensorDict
    import matplotlib.pyplot as plt

    import marimo as mo

    plt.style.use("default")
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"

    from torchstrap.stateless import StatelessModule
    from torchstrap.optimizer import Adam
    from torchstrap.utils.nn.archs import MLP
    from torchstrap.utils.nn.transform import Normalize

    from dataset import build_input_transform
    from model_io import load_model_weights
    from preprocessing import jet_columns
    from config import load_config

    # --- run configuration -------------------------------------------------
    # The target sysvar (whose models multifold.py persisted) comes from config.
    cfg = load_config()
    sys_var = cfg.sys_var

    # --- angularities route is HARDCODED here ------------------------------
    # This notebook only targets the angularities MLP models, so the route is
    # pinned locally instead of read from cfg — it keeps working even when
    # runtime-files/config.json is switched to another feature_mode (e.g.
    # bin_counts) for a live run. (cfg.features_root derives from
    # cfg["feature_mode"], so it would otherwise resolve to the wrong route.)
    feature_mode = "angularities"

    device = cfg.device

    root_dir = cfg.dataset_root / "features" / feature_mode
    tensordict_src = root_dir / "tensordicts" / str(sys_var)
    embed_dir = root_dir / "embedding" / str(sys_var)
    model_states_dir = embed_dir / "model_states"
    fig_dir = Path("./outputs/feature_importance") / str(sys_var) / feature_mode
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Per-run settings (ensemble size, iterations, input transform) must match
    # the angularities run that produced model_states — NOT the live config. The
    # transform especially is load-bearing: a placeholder built with the wrong
    # mode (e.g. log1p on signed phi/eta) changes the model structure and gives
    # wrong/NaN attributions. multifold.py copies its config verbatim into
    # embed_dir/config.json, so read these from that snapshot when present.
    _snap = embed_dir / "config.json"
    run_cfg = load_config(_snap) if _snap.exists() else cfg
    num_replicas = run_cfg["num_replicas"]
    num_iterations = run_cfg["num_iterations"]
    input_transform_mode = run_cfg["input_transform"]

    # IG / sampling knobs.
    SEED = 0
    N_HELDOUT = 50_000      # jets attributed per (step, iteration)
    IG_STEPS = 50           # Riemann steps along the x0 -> x path
    IG_CHUNK = 2048         # samples per forward/backward chunk (memory bound)
    PD_BINS = 20            # bins for the data-only partial-dependence curves
    PLOT_ITER = 3           # OmniFold iteration shown in the single-iteration plots
                            # (clipped to the last available iteration)

    STEPS = ("detlvl", "partlvl")
    FEATURE_NAMES = list(jet_columns)


@app.function
def load_sim_inputs(td_path: Path):
    """Return (X_eval CPU float32, x0 device, full sim count) for a tensordict.

    Attribution runs on the MC/sim side (`is_data == False`) — the physical jets
    the unfolding reweights. `x0` (the IG baseline) is the per-feature mean of
    the held-out sim sample = nominal-MC prior mean.
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

    num_features = td["input"].shape[-1]
    assert num_features == len(FEATURE_NAMES), (
        f"tensordict input width ({num_features}) != len(jet_columns) "
        f"({len(FEATURE_NAMES)}). The on-disk tensordict is stale relative to "
        "preprocessing.jet_columns — re-run preprocessing/multifold with "
        "redo_datasets=true so models and feature labels agree."
    )

    X = td["input"][eval_idx].to(torch.float32)          # (M, F) CPU
    x0 = X.mean(0).to(device)                            # (F,) device
    return X, x0, int(len(sim_idx))


@app.function
def make_skeleton(num_features: int, sample_input: torch.Tensor):
    """Rebuild the StatelessModule/State skeleton matching multifold.py's arch.

    The placeholder `input_transform` only fixes the buffer shapes; its mean/std
    are overwritten by `load_model_weights` (the trained z-norm rides in the
    saved buffers).
    """
    module, _, state = StatelessModule.init(
        MLP,
        Adam,
        layer_sizes=cfg.layer_sizes(num_features),
        input_transform=build_input_transform(
            input_transform_mode, sample_input, device=device
        ),
        dropout_prob=cfg.dropout_prob,
        num_replicas=num_replicas,
        device=device,
        init_randomness="different",
        optimizer_kwargs=cfg.optimizer_kwargs,
    )
    # The z-norm transform applies x.sub_().div_() in place, which autograd
    # rejects when we differentiate the logit w.r.t. the input (as IG does).
    # inplace vs out-of-place is mathematically identical, so disable it.
    for m in module._base_model.modules():
        if isinstance(m, Normalize):
            m.inplace = False
    return module, state


@app.function
def integrated_gradients(module, state, X_cpu, x0):
    """Integrated Gradients on raw inputs. Returns IG of shape (R, N, F) on CPU.

    IG_j = (x_j - x0_j) * mean_alpha d(logit)/dx_j |_{x0 + alpha (x - x0)}.
    eval() makes dropout a deterministic no-op so vmap sees no randomness.
    """
    module.eval()

    def logit_one(p, b, x):                      # x: (F,) -> scalar logit
        return module.forward(p, b, x.reshape(1, -1)).reshape(())

    grad_fn = vmap(
        vmap(grad(logit_one, argnums=2), in_dims=(None, None, 0)),  # over samples
        in_dims=(0, 0, None),                                       # over replicas
    )
    alphas = torch.linspace(0.0, 1.0, IG_STEPS + 1, device=device)[1:]

    chunks = []
    for xb_cpu in X_cpu.split(IG_CHUNK):
        xb = xb_cpu.to(device)
        delta = xb - x0                          # (m, F)
        acc = torch.zeros(
            state.num_replicas, xb.shape[0], xb.shape[1], device=device
        )
        for a in alphas:
            acc += grad_fn(state.param_dict, state.buffer_dict, x0 + a * delta)
        ig = delta.unsqueeze(0) * (acc / IG_STEPS)   # (R, m, F)
        chunks.append(ig.detach().cpu())
    return torch.cat(chunks, dim=1)


@app.function
def logits_for(module, state, X_cpu):
    """Per-replica logits f(x), shape (R, N) on CPU (for IG completeness)."""
    module.eval()

    def fwd(p, b, x):
        return module.forward(p, b, x).squeeze(-1)   # x: (m, F) -> (m,)

    outs = []
    with torch.no_grad():
        for xb_cpu in X_cpu.split(IG_CHUNK):
            xb = xb_cpu.to(device)
            o = vmap(fwd, in_dims=(0, 0, None))(
                state.param_dict, state.buffer_dict, xb
            )
            outs.append(o.detach().cpu())
    return torch.cat(outs, dim=1)


@app.cell
def _check_states():
    have_states = model_states_dir.is_dir() and any(
        model_states_dir.glob("iter*_*.pt")
    )
    if have_states:
        msg = mo.md(
            f"**Model states found** in `{model_states_dir}`.\n\n"
            f"sysvar=`{sys_var}`  ·  replicas={num_replicas}  ·  "
            f"iterations={num_iterations}  ·  device=`{device}`"
        )
    else:
        msg = mo.md(
            f"⚠️ **No model states** under `{model_states_dir}`.\n\n"
            "Set `save_model_states: true` in `runtime-files/config.json` and "
            "re-run `uv run multifold.py` for this sysvar first."
        )
    msg
    return (have_states,)


@app.cell
def _compute_ig(have_states):
    # ig_importance[step]: (num_iterations, R, F) of mean |IG| over the held-out
    # sample. completeness[step]: (num_iterations,) max relative residual.
    ig_importance: dict[str, np.ndarray] = {}
    completeness: dict[str, float] = {}

    if have_states:
        td_paths = {
            "detlvl": tensordict_src / "det_lvl",
            "partlvl": tensordict_src / "part_lvl",
        }
        for _step in STEPS:
            X, x0, _ = load_sim_inputs(td_paths[_step])
            module, state = make_skeleton(X.shape[-1], X[:1024])

            per_iter = []
            worst_resid = 0.0
            for it in range(num_iterations):
                wpath = model_states_dir / f"iter{it:02d}_{_step}.pt"
                if not wpath.exists():
                    continue
                load_model_weights(state, wpath)

                ig = integrated_gradients(module, state, X, x0)   # (R, N, F)
                per_iter.append(ig.abs().mean(dim=1).numpy())      # (R, F)

                # completeness: sum_j IG_j ~= f(x) - f(x0)
                fX = logits_for(module, state, X)                  # (R, N)
                fx0 = logits_for(module, state, x0.reshape(1, -1).cpu())  # (R,1)
                lhs = ig.sum(dim=-1)                               # (R, N)
                rhs = fX - fx0
                denom = rhs.abs().mean().clamp_min(1e-6)
                worst_resid = max(
                    worst_resid, float((lhs - rhs).abs().mean() / denom)
                )

            if per_iter:
                ig_importance[_step] = np.stack(per_iter)           # (n_it, R, F)
                completeness[_step] = worst_resid

        del module, state
        if device == "cuda":
            torch.cuda.empty_cache()

    completeness_md = mo.md(
        "**IG completeness** (mean |Σⱼ IGⱼ − (f(x)−f(x₀))| / mean|f(x)−f(x₀)|): "
        + ", ".join(f"{k}={v:.2%}" for k, v in completeness.items())
        if completeness
        else "_no IG computed yet_"
    )
    completeness_md
    return (ig_importance,)


@app.cell
def _plot_ig_bars(ig_importance: dict[str, np.ndarray]):
    if ig_importance:
        _fig, _axes = plt.subplots(
            1, len(STEPS), figsize=(7 * len(STEPS), 6), squeeze=False
        )
        _x = np.arange(len(FEATURE_NAMES))
        for _ax, _step in zip(_axes[0], STEPS):
            _imp = ig_importance.get(_step)
            if _imp is None:
                continue
            _it = min(PLOT_ITER, len(_imp) - 1)   # chosen iteration (clipped)
            _final = _imp[_it]                    # (R, F)
            _mean = _final.mean(0)
            _std = _final.std(0)
            _ax.bar(_x, _mean, yerr=_std, capsize=2, color="steelblue")
            _ax.set_xticks(_x)
            _ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=7)
            _ax.set_title(f"{_step}: mean |IG| (iter {_it}, ±replica std)")
            _ax.set_ylabel("mean |Integrated Gradient|")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "ig_bars_final.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_IG bar chart pending model states._")
    _out
    return


@app.cell
def _plot_ig_evolution(ig_importance: dict[str, np.ndarray]):
    # Feature x iteration heatmap of replica-mean |IG|, per step.
    if ig_importance and num_iterations > 1:
        _fig, _axes = plt.subplots(
            1, len(STEPS), figsize=(6 * len(STEPS), 7), squeeze=False
        )
        for _ax, _step in zip(_axes[0], STEPS):
            _imp = ig_importance.get(_step)
            if _imp is None:
                continue
            _grid = _imp.mean(axis=1).T            # (F, n_it)
            _im = _ax.imshow(_grid, aspect="auto", cmap="viridis", origin="lower")
            _ax.set_yticks(np.arange(len(FEATURE_NAMES)))
            _ax.set_yticklabels(FEATURE_NAMES, fontsize=7)
            _ax.set_xlabel("OmniFold iteration")
            _ax.set_title(f"{_step}: |IG| evolution")
            _fig.colorbar(_im, ax=_ax, fraction=0.046)
        _fig.tight_layout()
        _fig.savefig(fig_dir / "ig_evolution.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_Evolution heatmap needs num_iterations > 1 and IG results._")
    _out
    return


@app.cell
def _data_only_pd():
    # Data-only partial dependence: bin each part-level feature, take <w_unf>
    # per bin, importance = Var_bins(mean_w) / Var(w). No model state needed.
    # Reconstructs the gen-jet order multifold uses: [is_matched==1, ==0].
    pd_importance = None
    pd_curves = None

    _wpath = embed_dir / f"w_unfolding_niter{num_iterations}.npz"
    part_path = tensordict_src / "part_lvl"
    if _wpath.exists() and part_path.exists():
        wz = np.load(_wpath)
        # w_unfolding layout: [part_prior, det_prior, (gen, reco) * n_iter].
        # Final gen (part-level unfolded) weights = second-to-last entry.
        keys = wz.files
        w_gen = torch.as_tensor(wz[keys[-2]], dtype=torch.float32)   # (R, n_gen)
        w_gen = w_gen.mean(0)                                        # (n_gen,)

        td = TensorDict.load_memmap(part_path)
        all_idx = torch.arange(len(td))
        matched = all_idx[td["is_matched"] == 1]
        missed = all_idx[td["is_matched"] == 0]
        gen_idx = torch.cat([matched, missed])

        if gen_idx.numel() == w_gen.numel():
            _X = td["input"][gen_idx].to(torch.float32)              # (n_gen, F)
            var_w = w_gen.var().clamp_min(1e-12)
            imp = np.zeros(len(FEATURE_NAMES))
            curves = []
            for j in range(len(FEATURE_NAMES)):
                col = _X[:, j]
                edges = torch.quantile(
                    col, torch.linspace(0, 1, PD_BINS + 1)
                )
                b = torch.bucketize(col, edges[1:-1])
                mu = torch.zeros(PD_BINS)
                cnt = torch.zeros(PD_BINS)
                mu.index_add_(0, b, w_gen)
                cnt.index_add_(0, b, torch.ones_like(w_gen))
                mu = mu / cnt.clamp_min(1)
                centers = 0.5 * (edges[:-1] + edges[1:])
                # count-weighted variance of bin means / total variance.
                p = cnt / cnt.sum()
                mbar = (p * mu).sum()
                imp[j] = float(((p * (mu - mbar) ** 2).sum()) / var_w)
                curves.append((centers.numpy(), mu.numpy()))
            pd_importance = imp
            pd_curves = curves
            status = mo.md(
                f"**Data-only PD** computed over {gen_idx.numel():,} gen jets "
                f"from `{_wpath.name}`."
            )
        else:
            status = mo.md(
                f"⚠️ gen-jet count ({gen_idx.numel():,}) != w_unfolding length "
                f"({w_gen.numel():,}); skipping PD (stale artefacts?)."
            )
    else:
        status = mo.md(f"_No `{_wpath.name}` / part_lvl tensordict; PD skipped._")
    status
    return pd_curves, pd_importance


@app.cell
def _plot_pd_curves(pd_curves, pd_importance):
    # Deliverable 3: partial-dependence curves <w_unf>(x_j) for the top features.
    if pd_curves is not None:
        _top = np.argsort(pd_importance)[::-1][:9]
        _fig, _axes = plt.subplots(3, 3, figsize=(13, 11))
        for _ax, _j in zip(_axes.ravel(), _top):
            _centers, _mu = pd_curves[_j]
            _ax.plot(_centers, _mu, marker="o", ms=3, color="darkgreen")
            _ax.axhline(1.0, color="grey", ls="--", lw=0.8)
            _ax.set_title(f"{FEATURE_NAMES[_j]}  (PD={pd_importance[_j]:.3f})", fontsize=9)
            _ax.set_xlabel(FEATURE_NAMES[_j], fontsize=8)
            _ax.set_ylabel("⟨w_unf⟩", fontsize=8)
        for _ax in _axes.ravel()[len(_top):]:
            _ax.set_visible(False)
        _fig.suptitle("Data-only partial-dependence of unfolding weight")
        _fig.tight_layout()
        _fig.savefig(fig_dir / "pd_curves_top.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_PD curves pending._")
    _out
    return


@app.cell
def _plot_pd_compare(ig_importance: dict[str, np.ndarray], pd_importance):
    if pd_importance is not None:
        _order = np.argsort(pd_importance)[::-1]
        _names = [FEATURE_NAMES[i] for i in _order]

        _fig, _axes = plt.subplots(1, 2, figsize=(15, 6))
        _axes[0].bar(np.arange(len(_names)), pd_importance[_order], color="indianred")
        _axes[0].set_xticks(np.arange(len(_names)))
        _axes[0].set_xticklabels(_names, rotation=90, fontsize=7)
        _axes[0].set_ylabel("PD importance  Var(⟨w|x⟩)/Var(w)")
        _axes[0].set_title("Data-only partial-dependence importance (final)")

        # IG (partlvl final, replica mean) vs PD scatter + Spearman.
        _partlvl = ig_importance.get("partlvl") if ig_importance else None
        if _partlvl is not None:
            _it = min(PLOT_ITER, len(_partlvl) - 1)
            _ig_final = _partlvl[_it].mean(0)     # (F,)
            _axes[1].scatter(pd_importance, _ig_final, color="navy")
            for _i, _nm in enumerate(FEATURE_NAMES):
                _axes[1].annotate(_nm, (pd_importance[_i], _ig_final[_i]), fontsize=6)
            _rho = spearman(pd_importance, _ig_final)
            _axes[1].set_xlabel("data-only PD importance")
            _axes[1].set_ylabel(f"IG importance (partlvl, iter {_it})")
            _axes[1].set_title(f"IG vs PD  (Spearman ρ = {_rho:.2f})")
        else:
            _axes[1].set_visible(False)
        _fig.tight_layout()
        _fig.savefig(fig_dir / "pd_and_compare.png", dpi=150)
        _out = _fig
    else:
        _out = mo.md("_PD comparison pending._")
    _out
    return


@app.function
def spearman(a, b):
    """Spearman rank correlation without a scipy dependency."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
