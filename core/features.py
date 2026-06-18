"""`FeatureSpec` registry — one declarative source of truth for each feature mode.

Stage-1 groundwork (#3). Today a single `feature_mode` string secretly drives three
orthogonal decisions at 27 branch sites across 6 files:

  (a) which columns `process_table` writes,
  (b) which input tensor `to_tensordict` assembles (shape + dtype + bin-block handling),
  (c) which model `build_classifier` picks (MLP vs Conv2dNN).

…and `make_alt_embedding` even keeps a *divergent* column copy (`KINEMATIC_COLUMNS`,
`ALT_FEATURE_MODES`). This module captures all of that as data: a `FeatureSpec` per mode,
so a caller asks the spec instead of branching on a string. It imports the canonical
constants from `preprocessing` (it duplicates no values).

Stage 2 makes `to_tensordict` / `process_table` / `build_classifier` consume these specs;
for now this is a read-only registry that nothing depends on yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from preprocessing import (
    FEATURE_MODES,
    N_BINS,
    N_DR,
    N_PT,
    jet_columns,
)

# Angularity-observable column name fragments (vs pure kinematic/softdrop scalars). Used to
# derive the `kinematics` column subset from `jet_columns`. See the reconciliation note on
# the `kinematics` spec below.
_ANGULARITY_FRAGMENTS = ("nef", "ch_ang", "symmetry")


def _is_angularity(col: str) -> bool:
    return any(frag in col for frag in _ANGULARITY_FRAGMENTS)


@dataclass(frozen=True)
class FeatureSpec:
    """How a feature mode maps jet records -> a model input tensor + which model consumes it.

    - `columns`        scalar arrow columns selected per jet (empty for the pure image route).
    - `per_jet_shape`  shape of one jet's `input` tensor.
    - `input_dtype`    `"uint8"` for the count image, else `"float32"`.
    - `default_transform`  the `dataset.build_input_transform` key appropriate to this input
                       (scalar angularity columns carry -1 softdrop sentinels -> `z_norm`,
                       never `log1p`; the non-negative count image uses the per-channel path).
    - `architecture`   `"cnn"` for the (2,9,9) `bin_counts` image, else `"mlp"`.
    - `bin_block`      `"image"` (2-channel 9×9 CNN input), `"flat"` (flattened 81-cell charged
                       block concatenated to scalars, the `combined` mode), or `"none"`.
    """

    name: str
    columns: tuple[str, ...]
    per_jet_shape: tuple[int, ...]
    input_dtype: str
    default_transform: str
    architecture: str
    bin_block: str
    notes: str = field(default="", compare=False)

    @property
    def is_image(self) -> bool:
        return self.bin_block == "image"

    @property
    def num_features(self) -> int:
        """Per-jet feature count: flattened width for scalar/flat modes, channel*H*W for the
        image. (Equals `per_jet_shape[-1]` for the scalar modes that `cfg.layer_sizes` keys
        the MLP on.)"""
        n = 1
        for d in self.per_jet_shape:
            n *= d
        return n


_kinematic_columns = tuple(c for c in jet_columns if not _is_angularity(c))

FEATURE_SPECS: dict[str, FeatureSpec] = {
    "angularities": FeatureSpec(
        name="angularities",
        columns=tuple(jet_columns),
        per_jet_shape=(len(jet_columns),),
        input_dtype="float32",
        default_transform="z_norm",
        architecture="mlp",
        bin_block="none",
    ),
    "bin_counts": FeatureSpec(
        name="bin_counts",
        columns=(),
        per_jet_shape=(2, N_PT, N_DR),  # (charged, neutral) count image
        input_dtype="uint8",
        default_transform="log1p_per_channel_z_norm",
        architecture="cnn",
        bin_block="image",
    ),
    "combined": FeatureSpec(
        name="combined",
        columns=tuple(jet_columns),
        per_jet_shape=(len(jet_columns) + N_BINS,),  # scalars ⊕ flat-81 charged block
        input_dtype="float32",
        default_transform="z_norm",
        architecture="mlp",
        bin_block="flat",
    ),
    "kinematics": FeatureSpec(
        name="kinematics",
        columns=_kinematic_columns,
        per_jet_shape=(len(_kinematic_columns),),
        input_dtype="float32",
        default_transform="z_norm",
        architecture="mlp",
        bin_block="none",
        notes=(
            "Legacy make_alt_embedding classifier mode. Its KINEMATIC_COLUMNS list "
            "additionally carried ncharged/nconstituents/sd_pz/sd_ncharged/sd_nconstituents "
            "(not in jet_columns); reconcile that divergence into this `columns` tuple in "
            "Stage 2 when make_alt_embedding is fully retired."
        ),
    ),
}

# Cheap invariant: the registry covers exactly preprocessing.FEATURE_MODES.
assert set(FEATURE_SPECS) == set(FEATURE_MODES), (
    f"FEATURE_SPECS {sorted(FEATURE_SPECS)} != preprocessing.FEATURE_MODES {sorted(FEATURE_MODES)}"
)


def spec_for(feature_mode: str) -> FeatureSpec:
    """Return the `FeatureSpec` for a mode, raising a clear error for an unknown one."""
    try:
        return FEATURE_SPECS[feature_mode]
    except KeyError:
        raise ValueError(
            f"unknown feature_mode {feature_mode!r}; known: {sorted(FEATURE_SPECS)}"
        ) from None
