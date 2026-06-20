"""Single source of truth for run settings.

`runtime-files/config.json` already drives the training loop; this module
unifies the other settings that used to be hardcoded redundantly across the
analysis (the dataset-root path, the compute device, the target `sys_var`, and
the model/optimizer hyperparameters) so there is one place to change them.

Usage:

    from config import load_config
    cfg = load_config()
    cfg["num_replicas"]          # raw json values still work (dict subclass)
    cfg.dataset_root             # Path
    cfg.features_root            # dataset_root / "features" / feature_mode
    cfg.device                   # "auto" resolved to cuda/cpu
    cfg.sys_var                  # SysVar enum (the run's target variation)
    cfg.optimizer_kwargs, cfg.layer_sizes(n), cfg.dropout_prob, ...

Every accessor has a default matching the previous hardcoded value, so a stale
`config.json` copied into an output directory (which predates these keys) still
resolves sensibly.
"""

from __future__ import annotations

import json
from pathlib import Path

CONFIG_PATH = Path("runtime-files/config.json")
DEFAULT_DATASET_ROOT = "./datasets/STAR_pp200GeV_production_2012"

# Valid options for the constrained settings (config.json is plain JSON, so the
# enumerations live here):
#   feature_mode    : "angularities" | "bin_counts" | "combined" | "kinematics"
#   input_transform : "none" | "z_norm" | "log1p_z_norm" | "log1p_per_channel_z_norm"
#   device          : "auto" | "cuda" | "cpu"
#   sys_var (SysVar string values): "nominal" | "tower_et_corr_sys" |
#       "track_pt_sys" | "jet_pt_res_sys_0" | "jet_pt_res_sys_1" |
#       "unf_prior_same" | "unf_prior_like_data" | "unf_prior_herwig7" |
#       "unf_prior_pythia8" | "unf_iter_sys_0" | "unf_iter_sys_1"


class Config(dict):
    """The parsed config.json (still a plain dict) plus resolved accessors."""

    # --- paths ---------------------------------------------------------------
    @property
    def dataset_root(self) -> Path:
        return Path(self.get("dataset_root", DEFAULT_DATASET_ROOT))

    @property
    def features_root(self) -> Path:
        """`dataset_root/features/<feature_mode>` — the stage-2+ output tree."""
        return self.dataset_root / "features" / self["feature_mode"]

    @property
    def jets_root(self) -> Path:
        """`dataset_root/jets` — the stage-1 clustering output tree."""
        return self.dataset_root / "jets"

    # --- compute -------------------------------------------------------------
    @property
    def device(self) -> str:
        d = self.get("device", "auto")
        if d == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return d

    # --- target variation ----------------------------------------------------
    @property
    def sys_var(self):
        """The run's target `SysVar` (default `nominal`).

        Only the unfolding/preprocessing/explainability entrypoints read this;
        the closure plotters keep their semantically-required variant.
        """
        from systematics import SysVar

        return SysVar(self.get("sys_var", "nominal"))

    # --- training hyperparameters (the `training` block) ---------------------
    @property
    def training(self) -> dict:
        return self.get("training", {})

    @property
    def optimizer_kwargs(self) -> dict:
        return dict(
            self.training.get(
                "optimizer",
                dict(lr=1e-3, eps=1e-8, weight_decay=0.01, decoupled_weight_decay=True),
            )
        )

    @property
    def hidden_layers(self) -> list[int]:
        return list(self.training.get("hidden_layers", [256, 256, 256]))

    def layer_sizes(self, num_features: int, num_outputs: int = 1) -> list[int]:
        """MLP layer sizes `[num_features, *hidden_layers, num_outputs]`."""
        return [num_features, *self.hidden_layers, num_outputs]

    @property
    def dropout_prob(self) -> float:
        return self.training.get("dropout_prob", 0.2)

    @property
    def early_stopping_patience(self) -> int:
        return self.training.get("early_stopping_patience", 10)

    @property
    def reweight_clamp(self) -> dict:
        rc = self.training.get("reweight_clamp", {})
        return dict(
            clamp_min=rc.get("clamp_min", 1e-3),
            clamp_max=rc.get("clamp_max", 1e3),
        )

    @property
    def cnn_channels(self) -> tuple:
        return tuple(self.training.get("cnn_channels", (32, 64)))

    @property
    def predict_replica_chunk(self):
        return self.training.get("predict_replica_chunk", 2)

    @property
    def lr_schedule(self) -> dict:
        """Optional per-replica LR scheduler for `fit_ensemble`. A dict with a `policy`
        key (a torchstrap LRScheduler policy name, e.g. "ReduceLROnPlateau",
        "CosineAnnealingLR") plus that policy's kwargs (e.g. factor/patience, T_max).
        Empty/absent → constant LR (no scheduler)."""
        return dict(self.training.get("lr_schedule", {}))

    @property
    def cnn_collapse(self) -> bool:
        """bin_counts only: if true, `Conv2dNN` collapses the spatial grid to 1×1 via
        valid convs (per-layer kernels derived from the image size) instead of
        size-preserving convs + a global average pool. Needs `cnn_channels` deep enough
        to reach 1×1 (e.g. (16,32,48,64) for a 9×9 grid → 9→7→5→3→1). Off by default."""
        return bool(self.training.get("cnn_collapse", False))

    # --- torch.compile (optional) -------------------------------------------
    @property
    def compile_forward(self) -> bool:
        """If true, `multifold.py` `torch.compile`s the vmapped (forward+grad) and
        (forward) eval paths of each classifier ensemble via
        `StatelessModule.compile()`. Off by default. Mainly helps the CNN
        (`bin_counts`) route by fusing pointwise ops / cutting launch overhead;
        the per-replica grouped conv itself is unchanged."""
        return bool(self.get("compile_forward", False))

    @property
    def compile_kwargs(self) -> dict:
        """kwargs forwarded to `StatelessModule.compile()` / `torch.compile`
        (e.g. `{"mode": "max-autotune"}` or `{"dynamic": true}`)."""
        return dict(self.get("compile_kwargs", {}))


def load_config(path: str | Path = CONFIG_PATH) -> Config:
    with open(path) as f:
        return Config(json.load(f))


def config_path_from_argv(default: str | Path = CONFIG_PATH) -> Path:
    """Resolve a config path from the command line so an entry point can be pointed at a
    private config copy without env vars, e.g.::

        uv run multifold.py runtime-files/config.noptd.json

    Returns the first ``sys.argv`` token that is an existing ``*.json`` file, else
    ``default``. The ``.json``+exists gate means marimo's own invocation (``marimo edit
    histograms.py`` — argv carries a subcommand / the notebook name, not a json file)
    safely falls through to the default ``runtime-files/config.json``.
    """
    import sys

    for arg in sys.argv[1:]:
        if arg.endswith(".json") and Path(arg).is_file():
            return Path(arg)
    return default
