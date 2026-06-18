"""Typed on-disk layout registry — the single source of truth for where data lives.

Stage-1 groundwork (#1). Every data path in the pipeline is currently built by string
concat at ~109 sites across ~20 files; when the layout changed (`outputs/unfolding_<sysvar>/`
-> `features/<mode>/embedding/`) much of that drifted (`make_alt_embedding` still points at a
non-existent `clustered_jets/...`). This module centralizes the *current* layout into pure
functions so a future layout change is a one-file edit. It encodes today's tree EXACTLY — it
moves no data.

Layout (relative to `cfg.dataset_root`):

    <dataset_root>/
    ├── jets/                                              # stage-1 clustering output
    │   ├── data.arrow
    │   ├── embedding/<sysvar>/ptHat<lo>to<hi>/{gen-matches,reco-matches,misses,fakes}.arrow
    │   └── alt_gen/<generator>.arrow
    └── features/<feature_mode>/                           # stage-2+ outputs
        ├── data.arrow
        ├── embedding/<sysvar>/{gen-matches,reco-matches,misses,fakes}.arrow
        │   ├── w_unfolding.npz | config.json | index_split.npz
        │   ├── checkpoints/iter<N>/ | model_states/ | fit_history/
        └── tensordicts/<sysvar>/{det_lvl,part_lvl}/

`outputs/` holds reweighter diagnostic dumps (gitignored).

All functions take a `cfg` (anything with a `.dataset_root: Path` accessor, i.e.
`config.Config`) and return a `pathlib.Path`. None of them touch the filesystem.

`mode` / `sysvar` accept either the raw string or the enum/object whose `str()` is the
directory name (`SysVar` values stringify to their on-disk dir name, e.g. `SysVar.NONE` ->
``"nominal"``), matching how the rest of the pipeline already keys directories.
"""

from __future__ import annotations

from pathlib import Path

# Canonical arrow file stems for the four-way embedding split, in pipeline order.
EMBEDDING_ARROWS = ("gen-matches", "reco-matches", "misses", "fakes")


def _mode(cfg, mode) -> str:
    """Resolve a feature-mode argument, defaulting to the cfg's configured mode."""
    return str(mode if mode is not None else cfg["feature_mode"])


# --- roots -------------------------------------------------------------------------------
def dataset_root(cfg) -> Path:
    return cfg.dataset_root


def features_root(cfg, mode=None) -> Path:
    """`<dataset_root>/features/<feature_mode>` (the stage-2+ output tree for a mode)."""
    return cfg.dataset_root / "features" / _mode(cfg, mode)


def jets_root(cfg) -> Path:
    """`<dataset_root>/jets` (the stage-1 clustering output tree)."""
    return cfg.dataset_root / "jets"


# --- features/<mode> tree ----------------------------------------------------------------
def data_arrow(cfg, mode=None) -> Path:
    """Preprocessed real-data arrow for a feature mode."""
    return features_root(cfg, mode) / "data.arrow"


def embedding_dir(cfg, sysvar, mode=None) -> Path:
    """`features/<mode>/embedding/<sysvar>` — the four reweighted/unfolded arrows + run
    artefacts for one systematic variation."""
    return features_root(cfg, mode) / "embedding" / str(sysvar)


def embedding_arrow(cfg, sysvar, name, mode=None) -> Path:
    """One of `EMBEDDING_ARROWS` under a sysvar's embedding dir."""
    if name not in EMBEDDING_ARROWS:
        raise ValueError(f"unknown embedding arrow {name!r}; expected one of {EMBEDDING_ARROWS}")
    return embedding_dir(cfg, sysvar, mode) / f"{name}.arrow"


def embedding_arrows(cfg, sysvar, mode=None) -> dict[str, Path]:
    """All four embedding arrows as `{name: Path}`."""
    return {n: embedding_arrow(cfg, sysvar, n, mode) for n in EMBEDDING_ARROWS}


def unfolding_npz(cfg, sysvar, mode=None, *, niter=None) -> Path:
    """MultiFold per-jet weights. `niter=None` -> `w_unfolding.npz`; else
    `w_unfolding_niter<N>.npz`."""
    stem = "w_unfolding" if niter is None else f"w_unfolding_niter{niter}"
    return embedding_dir(cfg, sysvar, mode) / f"{stem}.npz"


def config_snapshot(cfg, sysvar, mode=None) -> Path:
    """The `config.json` copied into a run's output dir for reproducibility."""
    return embedding_dir(cfg, sysvar, mode) / "config.json"


def index_split_npz(cfg, sysvar, mode=None) -> Path:
    """AB-split closure index file (UNFOLDING_PRIOR_SAME)."""
    return embedding_dir(cfg, sysvar, mode) / "index_split.npz"


def checkpoints_dir(cfg, sysvar, mode=None) -> Path:
    """Resume checkpoints root; per-iteration full ensemble state in `iter<N>/`."""
    return embedding_dir(cfg, sysvar, mode) / "checkpoints"


def checkpoint_iter_dir(cfg, sysvar, iteration, mode=None) -> Path:
    return checkpoints_dir(cfg, sysvar, mode) / f"iter{iteration}"


def model_states_dir(cfg, sysvar, mode=None) -> Path:
    """Lean params-only ensemble weights for the XAI notebooks
    (`iter<NN>_{detlvl,partlvl}.pt`)."""
    return embedding_dir(cfg, sysvar, mode) / "model_states"


def fit_history_dir(cfg, sysvar, mode=None) -> Path:
    """Per-step training loss histories."""
    return embedding_dir(cfg, sysvar, mode) / "fit_history"


# --- features/<mode>/tensordicts tree ----------------------------------------------------
def tensordicts_dir(cfg, sysvar, mode=None) -> Path:
    """`features/<mode>/tensordicts/<sysvar>` (the memmapped TensorDict root)."""
    return features_root(cfg, mode) / "tensordicts" / str(sysvar)


def det_lvl_dir(cfg, sysvar, mode=None) -> Path:
    return tensordicts_dir(cfg, sysvar, mode) / "det_lvl"


def part_lvl_dir(cfg, sysvar, mode=None) -> Path:
    return tensordicts_dir(cfg, sysvar, mode) / "part_lvl"


# --- jets/ (stage-1) tree ----------------------------------------------------------------
def jets_data_arrow(cfg) -> Path:
    return jets_root(cfg) / "data.arrow"


def jets_embedding_dir(cfg, sysvar) -> Path:
    """`jets/embedding/<sysvar>` (per-pT-hat-bin clustering output root)."""
    return jets_root(cfg) / "embedding" / str(sysvar)


def jets_pthat_dir(cfg, sysvar, lo, hi) -> Path:
    """`jets/embedding/<sysvar>/ptHat<lo>to<hi>` (a single pT-hat bin's clustered arrows)."""
    return jets_embedding_dir(cfg, sysvar) / f"ptHat{lo}to{hi}"


def alt_gen_arrow(cfg, generator) -> Path:
    """`jets/alt_gen/<generator>.arrow` (alternate-generator gen jets for reweighting)."""
    return jets_root(cfg) / "alt_gen" / f"{generator}.arrow"


# --- outputs/ (gitignored reweighter / diagnostic dumps) ---------------------------------
def outputs_root(cfg=None) -> Path:
    return Path("outputs")


def omnisequential_dump(cfg, mode=None) -> Path:
    """Legacy GP data-reco reweighter npz dumps: `outputs/omnisequential/<mode>/`."""
    return outputs_root() / "omnisequential" / _mode(cfg, mode)


def reverse_omnisequential_dump(cfg, generator, mode=None) -> Path:
    """Legacy GP gen-prior reweighter npz dumps:
    `outputs/reverse_omnisequential/<generator>/<mode>/`."""
    return outputs_root() / "reverse_omnisequential" / str(generator) / _mode(cfg, mode)


def reweight_embedding_cache(cfg, kind, mode=None) -> Path:
    """Classifier reweighter scratch/TD cache: `outputs/reweight_embedding/<kind>/<mode>/`
    (`kind` is e.g. `"data_reco"` or a generator name)."""
    return outputs_root() / "reweight_embedding" / str(kind) / _mode(cfg, mode)
