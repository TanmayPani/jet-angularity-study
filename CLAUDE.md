# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Python **>= 3.14**, managed by **uv** (`pyproject.toml`, `uv.lock`); `.venv/` is the canonical interpreter (`extra-files/pyrightconfig.json` points at it).
- Two sibling packages are editable installs from `../packages/`: **`torchstrap`** (the in-house training framework — `StatelessModule`, `MLP`, `Adam`, `EarlyStopping`, `TensorBatchSampler`, `Normalize`, `random_split`, `undersample_and_random_split`) and **`thoda`** (the in-house histogramming library — `Histogram`, `Profile`, `Snapshot`, `bayesian_blocks`). They are not on PyPI; broken imports usually mean one of those packages drifted.
- Run scripts with `uv run <script>.py`. Notebooks are **marimo** apps (`omnisequential.py`, `reverse_omnisequential.py`, `histograms.py`, `plot_physics.py`, `plot_closure.py`, `test_preprocessing.py`) — open with `uv run marimo edit <file>.py`; their `if __name__ == "__main__": app.run()` is the marimo runner, not a CLI entry point.

## On-disk layout (post-refactor)

Everything is keyed off `feature_mode` as a subdirectory. The convention is now `datasets/STAR_pp200GeV_production_2012/<jets|features>/[<feature_mode>/]...`:

```
datasets/STAR_pp200GeV_production_2012/
├── jets/                                                    # stage-1 (clustering) output
│   ├── data.arrow
│   ├── embedding/<sysvar>/ptHat<lo>to<hi>/{gen-matches,reco-matches,misses,fakes}.arrow
│   └── alt_gen/{herwig7,pythia8}.arrow                      # consumed by reverse_omnisequential
└── features/<feature_mode>/                                 # stage-2+ outputs, scoped by mode
    ├── data.arrow                                            # preprocessed data (no `preproc_` prefix anymore)
    ├── embedding/<sysvar>/{gen-matches,reco-matches,misses,fakes}.arrow   # concat across pT-hat bins
    ├── embedding/<sysvar>/w_unfolding.npz                    # stage-5 multifold output
    ├── embedding/<sysvar>/config.json                        # config snapshot at training time
    ├── embedding/<sysvar>/index_split.npz                    # only for UNFOLDING_PRIOR_SAME AB-closure
    └── tensordicts/<sysvar>/{det_lvl,part_lvl}/              # stage-3 TensorDict memmaps
```

**The old `outputs/unfolding_<sysvar>/` destination is no longer used by `multifold.py`** — the unfolding weights now land next to the embedding arrows under `features/<feature_mode>/embedding/<sysvar>/`. Downstream notebooks (`histograms.py`, `plot_physics.py`, `plot_closure.py`) and `systematics.py` still reference the old path in places and have not all been updated; check the path each notebook reads before trusting its output.

`outputs/` is still used by the reweighting notebooks for their own intermediate npz dumps: `outputs/omnisequential/<feature_mode>/omniseq-{wts,diag}-iter<N>.npz` and `outputs/reverse_omnisequential/<generator>/<feature_mode>/omniseq-{wts,diag}-iter<N>.npz`.

## Pipeline (end-to-end)

1. **Clustering** (`cluster_data.py`, `cluster_embedding.py`, `cluster_pythia8.py`) — read ROOT files via uproot, apply fastjet anti-kT R=0.4, and write per-pT-hat-bin **arrow** record-batches into `jets/embedding/<sysvar>/ptHat<lo>to<hi>/`. `cluster_embedding.py` matches gen↔reco jets within ΔR < R and emits the four arrow files `{gen-matches, reco-matches, misses, fakes}.arrow` per `SysVar`. Data paths in these scripts are **hard-coded absolute paths** to the user's drives.
2. **Preprocessing** (`preprocessing.py`) — `process_table(table, feature_mode, ...)` recomputes per-jet observables (softdrop via fastjet; the branch taken depends on `feature_mode`, see below). `preprocess_embedding` concatenates the per-pT-hat-bin arrows under `jets/embedding/<sysvar>/ptHat*/` into a single `{gen,reco}-{matches,fakes,misses}.arrow` per sysvar under `features/<feature_mode>/embedding/<sysvar>/`. `preprocess_data` writes the data side to `features/<feature_mode>/data.arrow`. All file paths are passed as `pathlib.Path`.
3. **Tensorization** (`preprocessing.to_tensordict` + `make_datasets_for_unfolding`) — concatenates the data-like and sim-like arrow tables and memory-maps them into a single `TensorDict` (`det_lvl/`, `part_lvl/`) under `features/<feature_mode>/tensordicts/<sysvar>/`. Fields: `{input, target, weight, is_data, is_matched, pth_bin}` — the only object the training loop reads. For prior-systematic variants (`UNFOLDING_PRIOR_LIKE_DATA`, `UNFOLDING_PRIOR_HERWIG7`, `UNFOLDING_PRIOR_PYTHIA8`), `make_datasets_for_unfolding` treats `embedding/<sysvar>/{reco-matches,fakes}.arrow` as pseudo-data (`is_data=True`, `is_matched=-1`) and keeps the nominal embedding as the sim side.
4. **Reference reweighting** (`reweight_embedding.py`) — the production reweighter that bakes the prior weights into the four embedding arrows. It trains a **binary classifier likelihood-ratio** reweighter over the **full joint feature vector** (`f(x)` → per-jet odds `r = f/(1-f)`, clamped to `[0.1, 10]` = `ODDS_CLIP_LO/HI`, geometric-mean replica collapse, post-hoc calibration to neg-weighted mean = 1), reusing `multifold.build_classifier` (MLP for the scalar/angularities vector, `Conv2dNN` for the `bin_counts` (2,9,9) image) and `omnitrain.fit_ensemble`. Two modes: `--mode gen_prior --generator <herwig7|pythia8>` (pos = alt-gen, neg = p6 gen; `r` on gen-matches/misses, same `r` reused on reco-matches, fakes unchanged) and `--mode data_reco` (the LIKE_DATA prior; a 1-iteration OmniFold: reco step pos = `data.arrow`, neg = p6 reco-matches⊕fakes → `r_reco`; gen pull-back trains a second classifier pos = gen weighted by the pushed matched `r_reco`, neg = nominal gen → smooth `r_gen` applied to gen-matches **and** misses). Writes the four arrows to `features/<feature_mode>/embedding/<sysvar>/`; paths are cfg-driven and `--feature-mode` defaults to `cfg.feature_mode`. The classifier's headline closure metric is `valid_loss` vs `ln2=0.6931` (below ⇒ real joint signal the old marginal method missed). **Superseded:** the legacy sequential-GP *marginal* reweighters `omnisequential.py` / `reverse_omnisequential.py` (greedy worst-χ² 1-D observable + GP-smoothed ratio + 1-D `Profile` gen pull-back) are structurally blind to joint/correlation differences and are kept only as runnable 1-D cross-checks (their `outputs/{omnisequential,reverse_omnisequential/<gen>}/...` npz dumps remain valid for comparison). `make_alt_embedding.py` is retained as a helper *library* (its CLI has stale `clustered_jets/...` paths and the old 3-tuple `StatelessModule.init`; `reweight_embedding` imports its path-independent helpers + `prepare_herwig7`/`prepare_pythia8`).
5. **MultiFold unfolding** (`multifold.py`) — ensemble of `num_replicas` MLP classifiers per step, trained via `torchstrap.StatelessModule.init(MLP, Adam, ...)`. Three iterated reweighting steps per OmniFold iteration (detector-level reco→data, miss-scaling gen-only correction, particle-level gen→unfolded). For `UNFOLDING_PRIOR_SAME` the run is an in-memory AB-split closure on the nominal tensordicts (source path is redirected to `tensordicts/nominal/`), and `read_datasets(out_path=out_dir)` persists `index_split.npz` so closure plots can recover the B-side. Writes `w_unfolding.npz` + `config.json` to `features/<feature_mode>/embedding/<sysvar>/`.

Downstream of (5): `histograms.py`, `plot_physics.py`, and `plot_closure.py` project the unfolded weights into final 1D/2D distributions; `systematics.py` combines variation outputs into a total uncertainty. All prior/closure weights (LIKE_DATA, HERWIG7, PYTHIA8) are now produced by the classifier-based `reweight_embedding.py` (step 4); the GP scripts `omnisequential.py`/`reverse_omnisequential.py` and the legacy-CLI `make_alt_embedding.py` are retained only as cross-check/helper code (see step 4).

## Runtime configuration

All run settings live in **`runtime-files/config.json`** and are loaded through the
**`config.py`** helper: `from config import load_config; cfg = load_config()`. `cfg`
is a `dict` subclass (so `cfg["num_replicas"]` / `cfg.get(...)` still work) with
resolved accessors layered on top — `cfg.dataset_root` (Path), `cfg.features_root`
(`dataset_root/features/<feature_mode>`), `cfg.jets_root`, `cfg.device` (resolves
`"auto"`), `cfg.sys_var` (SysVar), and the training accessors `cfg.optimizer_kwargs`,
`cfg.layer_sizes(n)`, `cfg.dropout_prob`, `cfg.early_stopping_patience`,
`cfg.reweight_clamp`, `cfg.cnn_channels`, `cfg.predict_replica_chunk`. Every accessor
has a default matching the old hardcoded value, so a stale `config.json` copied into
an output dir still resolves. **Do not re-hardcode these settings** in new code — read
them off `cfg`.

Keys:

- `num_replicas` (ensemble size for `StatelessModule` vmap), `batch_size`, `num_data_subsample`, `train_size`, `num_iterations`, `num_epochs`
- `dataseed`, `modelseed`
- `dataset_root` — base data path; all scripts derive `features/`, `jets/`, etc. from it via `cfg.dataset_root`/`cfg.features_root`/`cfg.jets_root` (no more per-file `./datasets/STAR_pp200GeV_production_2012` literals)
- `device` ∈ `{"auto", "cuda", "cpu"}` — `"auto"` resolves to cuda-if-available (`cfg.device`)
- `sys_var` — the run's target `SysVar` (string value). Read by the unfolding/preprocessing/explainability entrypoints (`multifold.py`, `preprocessing.py`, `feature_importance.py`) via `cfg.sys_var`. The closure plotters keep their semantically-required variant hardcoded (`plot_ab_closure`=SAME, `plot_closure`=LIKE_DATA, `histograms`=NONE).
- `feature_mode` ∈ `{"angularities", "bin_counts", "combined", "kinematics"}` — selects the `process_table` and `to_tensordict` code paths and the subdirectory under `features/`
- `input_transform` ∈ `{"none", "z_norm", "log1p_z_norm", "log1p_per_channel_z_norm"}` — built by `dataset.build_input_transform`; uses chunked Welford-style sum/sum-of-squares (`dataset._chunked_std_mean`) to avoid materializing the full memmap in memory. `log1p_per_channel_z_norm` is for the `(C,H,W)` image input of the `bin_counts` CNN route — it standardizes per channel via `dataset._chunked_channel_std_mean` (reduces over sample + spatial dims, keeps channel) so sparse zero-heavy cells don't each get a tiny std.
- `training` (sub-block, read via the `cfg.*` accessors above) — `hidden_layers` (MLP hidden widths; `cfg.layer_sizes(n)` returns `[n, *hidden, 1]`), `dropout_prob`, `early_stopping_patience`, `reweight_clamp` `{clamp_min, clamp_max}`, `optimizer` `{lr, eps, weight_decay, decoupled_weight_decay}`, and the CNN-only `cnn_channels` (conv widths for `Conv2dNN`) and `predict_replica_chunk` (vmap replica chunk for `predict()`, bounds CNN inference memory). These were previously duplicated identically across `multifold.py`, `make_alt_embedding.py`, and `feature_importance.py`.
- `redo_preprocessing`, `redo_datasets`, `redo_closure_datasets` — toggles for `preprocessing.main`
- `save_model_states` (default `false`) — when true, `multifold.py` persists the converged `detlvl` and `partlvl` ensemble weights (params + buffers only, via `model_io.save_model_weights`) per iteration to `features/<feature_mode>/embedding/<sysvar>/model_states/iter<NN>_{detlvl,partlvl}.pt`, for the explainability notebooks. Off keeps standard runs lean. Two XAI marimo apps consume these: **`feature_importance.py`** (Integrated Gradients over the named scalar features, `angularities` mode only) and **`feature_importance_cnn.py`** (Grad-CAM importance map on the (pT, ΔR) grid + per-channel charged/neutral input saliency, `bin_counts` CNN mode only). Both rebuild the classifier skeleton and load states via `model_io.load_model_weights`; `feature_importance_cnn.py` is **CPU-pinned** (rebuilds a concrete single-replica `Conv2dNN` since `module._base_model` is on the `meta` device, and hooks the last post-ReLU conv map — which stays 9×9, so no upsampling) and writes figures to `outputs/feature_importance/<sysvar>/bin_counts/`.

- `resume` (default `false`) + `resume_from_iter` (default `null`) + `keep_last_checkpoint_only` (default `true`) — resume an interrupted unfolding run (e.g. after a molab 12 h timeout). Each completed OmniFold iteration writes a **full checkpoint** to `features/<feature_mode>/embedding/<sysvar>/checkpoints/iter<N>/`: `weights.npz` (the cumulative component weights `w_matched`/`w_miss`/`w_fake` + rolling priors `prev_gen`/`prev_reco`), `stats.npz` (stacked weight-stat history), and `{detlvl,miss_scaling,partlvl,fake_scaling}.pt` — each the **full** ensemble state (params + buffers + Adam moments + `active_mask`, via `model_io.save_ensemble_state`, distinct from the lean params-only `save_model_weights` used for XAI). A `complete.marker` is written **last** so a half-written checkpoint is ignored. When `resume` is true, `multifold.run()` auto-detects the highest complete `iter<N>/` and continues at iteration `N+1`, restoring all four ensembles in place (`load_ensemble_state`); `resume_from_iter` overrides the detected start explicitly. Because the ensembles warm-start across iterations, the full-state restore makes a resumed run bit-faithful to a continuous one. `keep_last_checkpoint_only` prunes older `iter<N>/` dirs after each new one commits (bounds storage to one iteration's worth of full state ×4 ensembles ×`num_replicas`). The per-jet `w_data` is re-derived from the dataset (deterministic), never checkpointed; `w_unfolding_niter0.npz` is not re-written on resume.

- `compile_forward` (default `false`) + `compile_kwargs` (default `{}`) — when true, `multifold.py` calls `StatelessModule.compile(**compile_kwargs)` on all four classifier ensembles after construction, `torch.compile`-ing the vmapped **(forward+grad)** train path and **(forward)** valid path (`_compiled_eval_wgrad` / `_compiled_eval`, auto-dispatched by `evaluate_and_gradient`/`evaluate`). The reweighting `predict()` path is **not** compiled (it uses its own `vmap(self.__call__)`). Mainly for the CNN (`bin_counts`) route: it fuses the pointwise ops (ReLU/Dropout2d/log1p/Normalize) and cuts kernel-launch overhead, but the per-replica **grouped conv** (vmap over `num_replicas`) is unchanged, so expect a moderate (not MLP-level) speedup. When on, `multifold.py` sets `drop_last=True` on the four training loaders (`train_drop_last`) so the compiled paths see static batch shapes and don't recompile on the trailing partial batch. `compile_kwargs` forwards to `torch.compile` (e.g. `{"mode": "max-autotune"}`, `{"dynamic": true}`). The train path is the fragile combo (`torch.compile` + `vmap` + `grad_and_value` + `randomness="different"` for dropout) — verify it compiled cleanly with `TORCH_LOGS="recompiles,graph_breaks"` rather than silently graph-breaking.

The same `config.json` is copied verbatim into each `features/<feature_mode>/embedding/<sysvar>/` so a run is reproducible from its output directory.

## Systematic variations (`SysVar`)

`systematics.SysVar` is a `StrEnum`; its **string value** is used directly as a directory name throughout the pipeline (`jets/embedding/<sysvar>/`, `features/<feature_mode>/embedding/<sysvar>/`, `features/<feature_mode>/tensordicts/<sysvar>/`). Adding a new variation means: add the enum, add a branch in `cluster_embedding.py` if it changes clustering, and the rest of the pipeline picks it up via the for-loop in each script's `__main__`.

Four `UNFOLDING_PRIOR*` variants are closure / prior-systematic constructs that reuse the nominal `embedding/<NONE>/` arrows but rewrite the `weight` column or relabel the data/sim sides:

- **`UNFOLDING_PRIOR_SAME`** — AB-split closure on the nominal tensordict; no separate on-disk arrow tree. `multifold.read_datasets(mode="ab_closure", out_path=...)` writes `index_split.npz` so closure plots can recover the B-side indices (note: indices are in **tensordict space** of length 2N because `to_tensordict` concatenates data-like + sim-like — subtract N to map back to arrow rows for the sim-side).
- **`UNFOLDING_PRIOR_LIKE_DATA`** — reweighted closure whose four embedding arrows are baked by `reweight_embedding.py --mode data_reco` (classifier reco→data + gen pull-back). `make_datasets_for_unfolding` then treats `embedding/<sysvar>/{reco-matches,fakes}.arrow` as pseudo-data against the nominal sim (see the LIKE_DATA pseudo-data branch in step 3).
- **`UNFOLDING_PRIOR_HERWIG7` / `UNFOLDING_PRIOR_PYTHIA8`** — prior-systematic closure where `reweight_embedding.py --mode gen_prior --generator <gen>` has baked the reweighted weights into `features/<feature_mode>/embedding/<sysvar>/{gen-matches,misses,reco-matches,fakes}.arrow` (legacy: `reverse_omnisequential.py` / `make_alt_embedding.py`).

## Ensemble / `num_replicas` invariants

The training stack is vmap-ed across replicas; this shape contract is load-bearing and easy to break:

- `TensorDictDataset.sample_weight` is always stored as `(num_replicas, length)` even when `num_replicas == 1` (`sample_weight` setter broadcasts/clones to this shape).
- `TensorBatchSampler(indices, batch_dim=1)` is fed `(num_replicas, n)` index tensors. The 1-D fallback branch of `random_split`/`undersample_and_random_split` is promoted via `torch.atleast_2d` in `dataset.classwise_*` so downstream code sees a uniform shape regardless of replicas.
- `multifold.read_datasets` builds per-class indices and targets, then in `__main__` expands the per-jet `weight` column to `(num_replicas, -1)` before training. The OmniFold iteration mutates `w_data, w_matched, w_miss, w_fake` in place and stitches them via `torch.cat(..., dim=1)`; `w_unfolding` is a list keeping every iteration's prior so the prior dataset's "previous iteration" can be referenced as `w_unfolding[-3]`.

When changing tensor shapes anywhere in `preprocessing.to_tensordict`, `dataset.TensorDictDataset`, or `multifold` model construction, address the memory consequences in the plan (on-disk size delta, whether any reduction needs chunking like `_chunked_std_mean`, per-batch GPU memory at `num_replicas × batch_size × n_features`). The dataset is O(10^7) jets memmapped; full-tensor reductions can OOM the host.

## Feature modes

`preprocessing.FEATURE_MODES = ("angularities", "bin_counts", "combined", "kinematics")` selects two things independently: which **columns `process_table` writes** to the arrow file, and which **input tensor `to_tensordict` assembles** from those columns.

`process_table` switches on `feature_mode`:

- `"angularities"` — calls `calculate_angularities`, producing the scalar columns `nef`, `ch_ang_k{1,2}_b{0,0.5,1,2}` and their `sd_*` softdrop variants (these names appear in `preprocessing.jet_columns`, which is the canonical input-column list for `make_datasets_for_unfolding`).
- `"bin_counts"` — calls `get_con_pt_dr_bins_sparse`, which writes list-typed columns per jet: `bin_index`/`bin_count` (occupied (pT, dR) cells flattened as `pt_bin * N_DR + dr_bin`, with the per-cell **charged**-constituent count), `bin_index_neutral`/`bin_count_neutral` (same for **neutral** constituents, `charge == 0`), and `bin_sum_wts` (5 channels per charged occupied cell — `Σ pT`, `Σ pT·√dR`, `Σ pT·dR`, `Σ pT·dR²`, `Σ pT²`; recover angularity κ,β as `Σ_bins(channel) / (pT_jet^κ · jet_r^β)`). Cell grid is `N_PT × N_DR = 9 × 9 = 81` (`con_pt_bins`, `con_dr_bins`). **`bin_counts` is now a 2-channel `(2, 9, 9)` image route** (channel 0 = charged count, channel 1 = neutral count) fed to a **2D-CNN** (`torchstrap.utils.nn.archs.Conv2dNN`), selected in `multifold.build_classifier`; the old flat-81 single-channel MLP path was retired for this mode (`combined` still uses the flat-81 charged block).
- `"combined"` — emits both the bin-count columns and the scalar columns; the angularity-sum reconstruction from `bin_sum_wts` is still a TODO so the scalar `ch_ang_*` columns are currently absent in this mode.
- `"kinematics"` — handled only by `to_tensordict` (not a `process_table` branch); behaves like `"angularities"` for input assembly but skips angularity-named columns. Used by `make_alt_embedding`'s classifier path.

`to_tensordict` then assembles the `input` tensor:

- `"angularities"` / `"kinematics"` — scalar columns only (driven by `columns=` arg, defaults to `jet_columns`). Fast `to_numpy + np.stack` path per chunk.
- `"bin_counts"` — a dense `(2, N_PT, N_DR) = (2, 9, 9)` image per jet (charged + neutral channels), densified per chunk by `_densify_bin_image`; the `input` field is 3D-per-jet. (`_densify_bin_counts` still produces the flat-81 charged block for `combined`.)
- `"combined"` — concatenation of both; the scalar block iterates via `to_numpy`/`np.stack` and the bin block via the sparse-densify path, then they're `torch.cat`-ed per chunk on `dim=-1`. Per-jet feature width is `len(jet_input_columns) + N_BINS`.

The bin-count columns (`bin_index`, `bin_count`, `bin_index_neutral`, `bin_count_neutral`, `bin_sum_wts`) are list-typed in arrow and must be iterated via the bin-block branch in `to_tensordict`, not the scalar `to_numpy + np.stack` path.

`leading_constit_*` and `subleading_constit_*` columns are currently commented out of `jet_columns` so they are written to arrow but not fed to the model.

## Conventions

- The `embedding/<sysvar>/` directory is treated as optional input; preprocessing/dataset-building functions print a "skipping" message and `return` when the directory is missing rather than raising, so the top-level `for sys_var in SysVar:` loops can be run before every variation has been clustered.
- Arrow files are memory-mapped (`pa.memory_map`) and kept alive via a `buffers` list — do not let those buffers go out of scope while the resulting tables are still in use.
- `outputs/`, `logs/`, `datasets/`, `slides/`, `_*`, and `extra-*` are gitignored; `extra-files/` is a scratch/archive area for old script versions and is not part of the live pipeline.
- The marimo notebooks load `runtime-files/config.json` in their `with app.setup:` block to bind `feature_mode` at module scope; `@app.function`-decorated functions can therefore be imported into plain Python modules (as long as the importer triggers the setup block).

## Planned: run `multifold.py` on marimo **molab** (cloud, Blackwell GPU)

**Goal.** Run the real `bin_counts` unfolding (`multifold.py`) on **molab** (marimo's
cloud notebook runner) instead of the local 4070 box. This is a *planning stub* — pick
it up in a fresh session and write the actual plan/notebook from here.

**Why it's now viable.** molab specs (verified 2026-06): **4 CPU / 32 GB RAM** default,
optional **NVIDIA RTX Pro 6000 Blackwell, 96 GB VRAM (~125 TFLOPS)** toggled via the
notebook-specs button, **12 h** max session / **90 min** idle shutdown, **torch
preinstalled**, free for reasonable use, auto-installs PyPI imports, sidebar file upload,
"limited persistent storage" (size unspecified). The local run **peaks ~20 GB host anon
at R=40 and is stable** (measured — see memory `omnitrain-host-ram-oom-roots`; the past
39.8 GB OOM was box oversubscription / an old `num_workers>0` run, not this path), so
32 GB RAM clears it and 96 GB VRAM dwarfs the 4070's 12 GB.

**The three real blockers + the intended path:**

1. **Data transfer (~5.8 GB).** The uint8 memmaps are `tensordicts/nominal/part_lvl`
   (5.1 GB) + `det_lvl` (0.68 GB). Don't browser-upload 6 GB into "limited" storage —
   **host them externally** (HF Hub dataset / S3 / GDrive) and `download` in a setup cell
   (~minutes inside the 12 h window). Assume you may **re-download each session**
   (persistence unconfirmed). *Open decision: where to host.*
2. **Helion on Blackwell (sm_120) — the one genuine technical risk.** The fused Adam
   (`torchstrap.optimizer.adam`) JIT-compiles a Triton/Helion CUDA kernel with a **pinned
   `config=` tuned for Ada (the 4070)**: `block_sizes=[1,2048], num_warps=4`. On Blackwell
   it should compile + run but the config will be **suboptimal** — re-sweep with
   `packages/torchstrap/test/optimizer/bench_inplace_adam.py --sweep` on the molab card.
   If Triton can't JIT for sm_120 with molab's torch, the CUDA path errors (the
   vectorized-PyTorch CPU fallback exists but is hopeless at 29M jets). **Smoke-test the
   op on a tiny tensor first**, before the full run.
3. **Packaging (neither pkg is on PyPI).** Both are their own GitHub repos and
   pip-installable from git: **torchstrap** = `git+https://github.com/TanmayPani/torchstrap.git`
   (hard deps: `helion`, `tensordict`, `beartype`, `plum-dispatch`), and this study =
   `github.com/TanmayPani/jet-angularity-study.git` (sibling modules the run imports:
   `omnitrain`, `dataset`, `config`, `model_io`, `systematics`, `preprocessing`). Clone via
   **HTTPS** (molab has no SSH key); if either repo is private, supply a PAT. `runtime-files/config.json`
   must come along (it drives everything via `config.load_config()`).

**Notebook structure.** A marimo `.py` is **not** your `if __name__ == "__main__"` script.
Refactor `multifold.py`'s `__main__` body into a callable `def run(cfg): ...` (leave the
script runnable as-is) and drive it from a thin `multifold_nb.py`: cells for
deps/clone → data download → **Helion smoke test** → `run(cfg)` → plots (the
per-iteration weight stats + the `_log_mem` curve). The loop is one long blocking
`run()` — keep it in a single "run" cell, not spread across reactive cells.

**Reuse what's already in-repo.** `multifold.py` already has `_log_mem(tag)` (per-phase
RssAnon/CUDA logging) and `_release_heap()` (gc + `malloc_trim`, called each iteration end);
the uint8 conversion is done (`convert_bin_counts_uint8.py`). Keep `num_workers=0`
(`cfg.get("num_workers",0)`) — the torchdata threaded loader leaks worker threads.
Default config: `num_replicas=40, num_iterations=3, num_epochs=30, batch_size=8000,
num_data_subsample=100000, feature_mode=bin_counts, input_transform=log1p_per_channel_z_norm`.

<!-- headroom:learn:start -->
## Headroom Learned Patterns
*Auto-generated by `headroom learn` on 2026-06-02 — do not edit manually*

### File Paths: Large Source Files
*~600 tokens/session saved*
- `preprocessing.py`, `multifold.py`, `omnisequential.py`, `make_alt_embedding.py` are all >700 lines. Use targeted `Read` with `offset`/`limit`, or `grep -n` to locate sections, instead of reading the whole file multiple times.

### Workflow: Syntax & Import Checks
*~500 tokens/session saved*
- Syntax check: `uv run python -c "import ast; ast.parse(open('file.py').read()); print('OK')"`
- Import smoke test: `uv run --no-sync python -c "import module; print('OK')"`
- Marimo notebook check: `uv run marimo check <notebook>.py` — run this after every edit to a marimo .py file before declaring it done.

### File Paths: Local Packages
*~400 tokens/session saved*
- Local packages live under `/home/tanmaypani/star-workspace/packages/`: `churten` (deep-learning utilities), `thoda` (histogram library), `torchstrap` (stateless training framework).
- Source roots: `packages/<pkg>/src/<pkg>/` — grep here for class/function definitions before searching the web.

### Environment
*~350 tokens/session saved*
- Always use `uv run python` or `uv run --no-sync python` to run Python code — bare `python` or `python3` fail with module-not-found errors because the venv is not activated.
- Use `uv run --no-sync` when you do not want uv to sync the lockfile (faster for quick checks).

### Workflow: Marimo MCP Tool Quirk
*~300 tokens/session saved*
- `mcp__marimo__get_notebook_errors` returns `status: success` even when there are live cell errors; the actual errors are buried in `next_steps`. Always follow up with `mcp__marimo__get_cell_runtime_data` on flagged cells, or use `mcp__marimo__lint_notebook` for a cleaner summary.

### Workflow: Plan Files
*~250 tokens/session saved*
- Plan files at `/home/tanmaypani/.claude/plans/*.md` must always be Read before being Written — the Write tool will error otherwise. Re-read if unsure whether the file was already read in the current context.

<!-- headroom:learn:end -->
