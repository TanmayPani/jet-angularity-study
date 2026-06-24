#!/usr/bin/env python
"""Per-SysVar driver for the unfolding pipeline, for any ``feature_mode``.

Runs the two compute stages for the given SysVar(s), in order, serially:

    preprocess  ->  uv run preprocessing.py <copy>   (CPU; tensordicts)
    unfold      ->  uv run multifold.py     <copy>   (GPU; embedding/<sysvar>/w_unfolding.npz)

Histogramming is intentionally NOT here — use the separate histogram script after.

Feature mode
------------
The mode to run is chosen with ``--feature-mode`` (default: the ``feature_mode`` in
``runtime-files/config.json``). It is baked into each per-SysVar config copy alongside
``sys_var``, so ``--feature-mode`` can drive a run without hand-editing the shared config.
NOTE: other mode-specific keys (``redo_preprocessing``, ``input_transform``, ...) are NOT
overridden — they come from ``config.json`` as-is. The ``angularities_*`` subset modes all
use ``redo_preprocessing=false`` + ``input_transform=z_norm``, so switching among them via
``--feature-mode`` Just Works; switching to e.g. ``bin_counts`` still needs those keys set in
``config.json`` first.

Subset modes (``angularities_noptd``, ``angularities_minimal``) reuse the ``angularities``
arrows: this script auto-creates the symlink tree
``features/<mode>/{data.arrow, embedding/<sysvar>/{gen-matches,misses,reco-matches,fakes}.arrow}``
-> ``features/angularities/...`` before the stages (idempotent; ``--relink`` to refresh).
If a sysvar's ``angularities`` source arrows don't exist yet, this script first auto-runs
the ``angularities``-mode preprocessing (``redo_preprocessing=true``) to produce them from
the clustered jets (``--no-autoprep`` to skip); if the clustered jets are also missing it
fails fast telling you to run ``cluster_embedding.py`` for that sysvar first.

Concurrency model: ONE invocation handles its --sysvars. To run several at once, open
another terminal and run the script again with a different --sysvars. Each SysVar gets its
own config copy ``runtime-files/config.<name>.<sysvar>.json`` (with its sys_var + feature_mode
baked in), so separate terminals never fight over the shared ``runtime-files/config.json``
(which is only ever READ here) or over each other. Output dirs are disjoint per SysVar.

Each stage streams to the terminal and to ``logs/<feature_mode>/<sysvar>/<stage>.log``.

Examples
--------
    # terminal 1
    uv run run_pipeline.py --feature-mode angularities_minimal --sysvars nominal
    # terminal 2 (simultaneously, different SysVar)
    uv run run_pipeline.py --feature-mode angularities_minimal --sysvars track_pt_sys

    uv run run_pipeline.py --feature-mode angularities_noptd          # whole band, serially
    uv run run_pipeline.py --stages unfold --sysvars unf_prior_herwig7
    uv run run_pipeline.py --feature-mode angularities_minimal --sysvars nominal --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONFIG_SRC = (
    REPO / "runtime-files" / "config.json"
)  # shared config; only ever READ here
LOG_ROOT = REPO / "logs"

# Modes that are pure input-subset variants of `angularities`: they have no
# `process_table` branch of their own and reuse the `angularities` arrows via
# symlinks (only their model-input column subset differs). Auto-linked below.
ANGULARITY_SUBSET_MODES = ("angularities_noptd", "angularities_minimal")
LINK_SRC_MODE = "angularities"
EMBEDDING_ARROWS = ("gen-matches", "misses", "reco-matches", "fakes")

# stage name -> script run as `uv run <script> <config-copy>`
STAGES: dict[str, str] = {
    "preprocess": "preprocessing.py",
    "unfold": "multifold.py",
}
STAGE_ORDER = ("preprocess", "unfold")

DEFAULT_SYSVARS = (
    "nominal",
    "tower_et_corr_sys",
    "tower_gain_sys",
    "track_pt_sys",
    "unf_prior_herwig7",
    "unf_prior_like_data",
)


def _feature_mode_of(path: Path) -> str | None:
    m = re.search(r'"feature_mode"\s*:\s*"([^"]*)"', path.read_text())
    return m.group(1) if m else None


def _dataset_root() -> Path:
    """Resolve the dataset root from the shared config (for the symlink targets)."""
    from config import load_config

    return load_config(CONFIG_SRC).dataset_root


def copy_path_for(config_name: str, sysvar: str) -> Path:
    return REPO / "runtime-files" / f"config.{config_name}.{sysvar}.json"


def make_copy(
    copy_path: Path, sysvar: str, feature_mode: str, *, refresh: bool
) -> None:
    """Snapshot config.json into a per-SysVar copy; bake in its sys_var + feature_mode."""
    if not copy_path.exists() or refresh:
        if not CONFIG_SRC.is_file():
            raise SystemExit(f"ERROR: {CONFIG_SRC} not found to copy from.")
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_SRC, copy_path)
    _set_json_string(copy_path, "sys_var", sysvar)
    _set_json_string(copy_path, "feature_mode", feature_mode)


def _set_json_string(copy_path: Path, key: str, value: str) -> None:
    text = copy_path.read_text()
    new_text, n = re.subn(rf'("{key}"\s*:\s*")[^"]*(")', rf"\g<1>{value}\g<2>", text)
    if n != 1:
        raise SystemExit(
            f"ERROR: expected one {key!r} substitution in {copy_path}, got {n}"
        )
    copy_path.write_text(new_text)


def _angularities_arrows_exist(sysvar: str) -> bool:
    """True iff all four `angularities`-mode embedding arrows for `sysvar` are on disk
    (the source a subset mode symlinks to)."""
    base = _dataset_root() / "features" / LINK_SRC_MODE / "embedding" / sysvar
    return all((base / f"{name}.arrow").exists() for name in EMBEDDING_ARROWS)


def _clustered_jets_exist(sysvar: str) -> bool:
    """True iff stage-1 clustering output exists for `sysvar` (at least one
    `jets/embedding/<sysvar>/ptHat*/gen-matches.arrow`). The angularities-mode
    preprocessing needs this as its input; preprocessing itself validates that
    every pT-hat bin is present."""
    base = _dataset_root() / "jets" / "embedding" / sysvar
    return base.is_dir() and any(base.glob("ptHat*/gen-matches.arrow"))


def _write_angularities_prep_config(sysvar: str) -> Path:
    """Write a one-off config copy driving the `angularities`-mode preprocessing that
    materializes the source arrows a subset mode symlinks to: feature_mode=angularities,
    redo_preprocessing=true (run process_table), redo_datasets=false (we only need the
    arrows, not angularities-mode tensordicts). JSON round-trip (not the regex setters)
    so the bool overrides are robust."""
    cfg = json.loads(CONFIG_SRC.read_text())
    cfg["feature_mode"] = LINK_SRC_MODE
    cfg["sys_var"] = sysvar
    cfg["redo_preprocessing"] = True
    cfg["redo_datasets"] = False
    copy_path = REPO / "runtime-files" / f"config.{LINK_SRC_MODE}.{sysvar}.autoprep.json"
    copy_path.write_text(json.dumps(cfg, indent=2))
    return copy_path


def ensure_angularities_arrows(sysvars: list[str], *, dry_run: bool) -> None:
    """Make sure each subset-mode `sysvar` has its `angularities` source arrows, by
    auto-running the angularities-mode preprocessing when they are missing. Fails fast
    with a clear message when the upstream clustered jets are also absent (so the user
    learns to run cluster_embedding.py first, instead of dying later in a downstream
    stage). The data side (`data.arrow`) is sysvar-independent and is (re)produced as a
    side effect of that same preprocessing run."""
    print(
        f"\n{'-' * 78}\nensuring {LINK_SRC_MODE} source arrows for subset mode"
        f"\n{'-' * 78}",
        flush=True,
    )
    for sysvar in sysvars:
        if _angularities_arrows_exist(sysvar):
            print(f"  [{sysvar}] {LINK_SRC_MODE} arrows present", flush=True)
            continue
        if not _clustered_jets_exist(sysvar):
            raise SystemExit(
                f"ERROR: [{sysvar}] no {LINK_SRC_MODE} source arrows, and no clustered "
                f"jets at {_dataset_root() / 'jets' / 'embedding' / sysvar}.\n"
                f"       Run cluster_embedding.py for {sysvar} first "
                f"(set sys_var_type = SysVar(...) for it in __main__), then re-run."
            )
        if dry_run:
            print(
                f"  [{sysvar}] {LINK_SRC_MODE} arrows missing -> (dry-run) would auto-run "
                f"angularities preprocessing from clustered jets",
                flush=True,
            )
            continue
        print(
            f"  [{sysvar}] {LINK_SRC_MODE} arrows missing -> auto-running angularities "
            f"preprocessing (process_table over clustered jets)",
            flush=True,
        )
        copy_path = _write_angularities_prep_config(sysvar)
        rc = run_stage("preprocess", sysvar, LINK_SRC_MODE, copy_path, dry_run=False)
        if rc != 0:
            raise SystemExit(
                f"ERROR: [{sysvar}] angularities-mode preprocessing failed (exit {rc}); "
                f"see the log above. Clustering may be incomplete (a missing pT-hat bin)."
            )
        if not _angularities_arrows_exist(sysvar):
            raise SystemExit(
                f"ERROR: [{sysvar}] angularities preprocessing ran but the source arrows "
                f"are still absent; aborting before the subset run produces a partial band."
            )


def link_subset_arrows(
    feature_mode: str, sysvars: list[str], *, relink: bool, dry_run: bool
) -> None:
    """Create the `features/<mode>/...` symlink tree pointing at the `angularities`
    arrows (subset modes only). Idempotent: existing links are kept unless `relink`.
    Missing source arrows only warn, so a partially-clustered band still runs."""
    # Resolve to an absolute path so the display `relative_to(...)` calls below
    # work: `src` is `.resolve()`d (absolute), so `features`/`features.parent`
    # must be absolute too (dataset_root is a relative `./datasets/...`).
    features = (_dataset_root() / "features").resolve()
    src_root = features / LINK_SRC_MODE
    dst_root = features / feature_mode

    # (relative-link-target, link-path) pairs: data.arrow + per-sysvar embedding arrows.
    pairs: list[tuple[str, Path]] = [
        (f"../{LINK_SRC_MODE}/data.arrow", dst_root / "data.arrow")
    ]
    for sysvar in sysvars:
        for name in EMBEDDING_ARROWS:
            pairs.append(
                (
                    f"../../../{LINK_SRC_MODE}/embedding/{sysvar}/{name}.arrow",
                    dst_root / "embedding" / sysvar / f"{name}.arrow",
                )
            )

    print(
        f"\n{'-' * 78}\nlinking {feature_mode} arrows -> {LINK_SRC_MODE} "
        f"(under {features})\n{'-' * 78}",
        flush=True,
    )
    n_made = n_kept = n_missing = 0
    for rel_target, link_path in pairs:
        src = (link_path.parent / rel_target).resolve()
        if not src.exists():
            print(
                f"  WARN missing source {src.relative_to(features.parent)} "
                f"(skip {link_path.name})",
                flush=True,
            )
            n_missing += 1
            continue
        if link_path.is_symlink() or link_path.exists():
            if relink:
                if not dry_run:
                    link_path.unlink()
            else:
                n_kept += 1
                continue
        if dry_run:
            print(
                f"  (dry-run) {link_path.relative_to(features)} -> {rel_target}",
                flush=True,
            )
            continue
        link_path.parent.mkdir(parents=True, exist_ok=True)
        link_path.symlink_to(rel_target)
        n_made += 1
    print(
        f"  links: {n_made} made, {n_kept} kept, {n_missing} missing-source", flush=True
    )


def run_stage(
    stage: str, sysvar: str, feature_mode: str, copy_path: Path, *, dry_run: bool
) -> int:
    script = STAGES[stage]
    log_dir = LOG_ROOT / feature_mode / sysvar
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    cmd = ["uv", "run", script, str(copy_path)]
    banner = f"[{feature_mode}/{sysvar}] {stage}: {' '.join(cmd)}"

    print(f"\n{'=' * 78}\n{banner}\n  log -> {log_path}\n{'=' * 78}", flush=True)
    if dry_run:
        print("  (dry-run: not executing)", flush=True)
        return 0

    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(
            f"# {banner}\n# started {datetime.now().isoformat(timespec='seconds')}\n\n"
        )
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        rc = proc.wait()
        elapsed = time.monotonic() - start
        log.write(f"\n# exit {rc} after {elapsed:.1f}s\n")
    print(f"  -> exit {rc} ({elapsed:.1f}s)", flush=True)
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--feature-mode",
        default=None,
        help="Feature mode to run (default: feature_mode in config.json). "
        "Baked into each per-SysVar config copy.",
    )
    ap.add_argument(
        "--sysvars",
        nargs="+",
        default=list(DEFAULT_SYSVARS),
        help=f"SysVars to process (default: {' '.join(DEFAULT_SYSVARS)})",
    )
    ap.add_argument(
        "--stages",
        nargs="+",
        default=list(STAGE_ORDER),
        choices=list(STAGE_ORDER),
        help="Stages to run, in canonical order (default: preprocess unfold)",
    )
    ap.add_argument(
        "--config-name",
        default=None,
        help="Per-SysVar copies are runtime-files/config.<name>.<sysvar>.json "
        "(default: the feature mode).",
    )
    ap.add_argument(
        "--refresh-copy",
        action="store_true",
        help="Re-snapshot config.json into the copies even if they already exist.",
    )
    ap.add_argument(
        "--relink",
        action="store_true",
        help="Re-create the subset-mode arrow symlinks even if they exist.",
    )
    ap.add_argument(
        "--no-link",
        action="store_true",
        help="Skip the subset-mode arrow-symlink setup step.",
    )
    ap.add_argument(
        "--no-autoprep",
        action="store_true",
        help="Skip auto-running the angularities-mode preprocessing for subset modes "
        "when the source arrows are missing (then a missing source only warns).",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="Print what would run; change nothing."
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going to the next SysVar if a stage fails (default: stop).",
    )
    args = ap.parse_args()

    feature_mode = args.feature_mode or _feature_mode_of(CONFIG_SRC)
    if not feature_mode:
        raise SystemExit(
            f"ERROR: no feature_mode given and none found in {CONFIG_SRC.name}."
        )

    from preprocessing import FEATURE_MODES

    if feature_mode not in FEATURE_MODES:
        raise SystemExit(
            f"ERROR: feature_mode {feature_mode!r} not in FEATURE_MODES {FEATURE_MODES}."
        )

    config_name = args.config_name or feature_mode
    stages = [s for s in STAGE_ORDER if s in set(args.stages)]
    sysvars = list(dict.fromkeys(args.sysvars))  # de-dup, preserve order

    cfg_fmode = _feature_mode_of(CONFIG_SRC)
    print(
        f"feature_mode  : {feature_mode!r} (config.json has {cfg_fmode!r}; baked into copies)"
    )
    print(f"SysVars       : {', '.join(sysvars)}")
    print(f"Stages        : {', '.join(stages)}")
    print(f"Per-SysVar cfg: runtime-files/config.{config_name}.<sysvar>.json")

    if feature_mode in ANGULARITY_SUBSET_MODES and not args.no_link:
        if not args.no_autoprep:
            ensure_angularities_arrows(sysvars, dry_run=args.dry_run)
        link_subset_arrows(
            feature_mode, sysvars, relink=args.relink, dry_run=args.dry_run
        )

    results: list[tuple[str, str, int]] = []
    for sysvar in sysvars:
        copy_path = copy_path_for(config_name, sysvar)
        if not args.dry_run:
            make_copy(copy_path, sysvar, feature_mode, refresh=args.refresh_copy)
        for stage in stages:
            rc = run_stage(stage, sysvar, feature_mode, copy_path, dry_run=args.dry_run)
            results.append((sysvar, stage, rc))
            if rc != 0:
                print(
                    f"!! [{sysvar}] {stage} FAILED (exit {rc}); skipping rest of this SysVar.",
                    flush=True,
                )
                if not args.continue_on_error:
                    print(
                        "Stopping (use --continue-on-error to keep going).", flush=True
                    )
                    _summary(results, config_name, feature_mode)
                    return rc
                break

    _summary(results, config_name, feature_mode)
    return 1 if any(rc != 0 for _, _, rc in results) else 0


def _summary(results, config_name, feature_mode) -> None:
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for sysvar, stage, rc in results:
        print(f"  {'OK ' if rc == 0 else 'FAIL'}  {sysvar:<20} {stage} (exit {rc})")
    print(
        f"\nPer-SysVar config copies: runtime-files/config.{config_name}.<sysvar>.json "
        f"(shared config.json untouched). Histogram separately."
    )
    print(f"Logs under: {LOG_ROOT / feature_mode}")


if __name__ == "__main__":
    raise SystemExit(main())
