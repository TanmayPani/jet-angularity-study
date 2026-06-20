#!/usr/bin/env python
"""Per-SysVar driver for the no-p_T^D ("angularities_noptd") reproduction.

Runs the two compute stages for the given SysVar(s), in order, serially:

    preprocess  ->  uv run preprocessing.py <copy>   (CPU; tensordicts, drops p_T^D)
    unfold      ->  uv run multifold.py     <copy>   (GPU; embedding/<sysvar>/w_unfolding.npz)

Histogramming is intentionally NOT here — use the separate histogram script after.

Concurrency model: ONE invocation handles its --sysvars. To run several at once, open
another terminal and run the script again with a different --sysvars. Each SysVar gets its
own config copy ``runtime-files/config.<name>.<sysvar>.json`` (with its sys_var baked in),
so separate terminals never fight over the shared ``runtime-files/config.json`` (which is
only ever READ here) or over each other. Output dirs are disjoint per SysVar.

Each stage streams to the terminal and to ``logs/noptd/<sysvar>/<stage>.log``.

Examples
--------
    # terminal 1
    uv run run_noptd_pipeline.py --sysvars nominal
    # terminal 2 (simultaneously, different SysVar)
    uv run run_noptd_pipeline.py --sysvars track_pt_sys

    uv run run_noptd_pipeline.py                       # whole band, serially, one terminal
    uv run run_noptd_pipeline.py --stages unfold --sysvars unf_prior_herwig7
    uv run run_noptd_pipeline.py --dry-run --sysvars nominal
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONFIG_SRC = REPO / "runtime-files" / "config.json"  # shared config; only ever READ here
LOG_ROOT = REPO / "logs" / "noptd"

EXPECTED_FEATURE_MODE = "angularities_noptd"

# stage name -> script run as `uv run <script> <config-copy>`
STAGES: dict[str, str] = {
    "preprocess": "preprocessing.py",
    "unfold": "multifold.py",
}
STAGE_ORDER = ("preprocess", "unfold")

DEFAULT_SYSVARS = (
    "nominal",
    "tower_et_corr_sys",
    "track_pt_sys",
    "unf_prior_herwig7",
)


def _feature_mode_of(path: Path) -> str | None:
    m = re.search(r'"feature_mode"\s*:\s*"([^"]*)"', path.read_text())
    return m.group(1) if m else None


def copy_path_for(config_name: str, sysvar: str) -> Path:
    return REPO / "runtime-files" / f"config.{config_name}.{sysvar}.json"


def make_copy(copy_path: Path, sysvar: str, *, refresh: bool) -> None:
    """Snapshot config.json into a per-SysVar copy and bake in its sys_var."""
    if not copy_path.exists() or refresh:
        if not CONFIG_SRC.is_file():
            raise SystemExit(f"ERROR: {CONFIG_SRC} not found to copy from.")
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_SRC, copy_path)
    _set_sys_var(copy_path, sysvar)


def _set_sys_var(copy_path: Path, sysvar: str) -> None:
    text = copy_path.read_text()
    new_text, n = re.subn(r'("sys_var"\s*:\s*")[^"]*(")', rf"\g<1>{sysvar}\g<2>", text)
    if n != 1:
        raise SystemExit(f"ERROR: expected one sys_var substitution in {copy_path}, got {n}")
    copy_path.write_text(new_text)


def run_stage(stage: str, sysvar: str, copy_path: Path, *, dry_run: bool) -> int:
    script = STAGES[stage]
    log_dir = LOG_ROOT / sysvar
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage}.log"
    cmd = ["uv", "run", script, str(copy_path)]
    banner = f"[{sysvar}] {stage}: {' '.join(cmd)}"

    print(f"\n{'=' * 78}\n{banner}\n  log -> {log_path}\n{'=' * 78}", flush=True)
    if dry_run:
        print("  (dry-run: not executing)", flush=True)
        return 0

    start = time.monotonic()
    with log_path.open("w") as log:
        log.write(f"# {banner}\n# started {datetime.now().isoformat(timespec='seconds')}\n\n")
        log.flush()
        proc = subprocess.Popen(
            cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
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
    ap.add_argument("--sysvars", nargs="+", default=list(DEFAULT_SYSVARS),
                    help=f"SysVars to process (default: {' '.join(DEFAULT_SYSVARS)})")
    ap.add_argument("--stages", nargs="+", default=list(STAGE_ORDER), choices=list(STAGE_ORDER),
                    help="Stages to run, in canonical order (default: preprocess unfold)")
    ap.add_argument("--config-name", default="noptd",
                    help="Per-SysVar copies are runtime-files/config.<name>.<sysvar>.json")
    ap.add_argument("--refresh-copy", action="store_true",
                    help="Re-snapshot config.json into the copies even if they already exist.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would run; change nothing.")
    ap.add_argument("--continue-on-error", action="store_true",
                    help="Keep going to the next SysVar if a stage fails (default: stop).")
    ap.add_argument("--force", action="store_true",
                    help=f"Run even if config feature_mode != {EXPECTED_FEATURE_MODE!r}.")
    args = ap.parse_args()

    stages = [s for s in STAGE_ORDER if s in set(args.stages)]
    sysvars = list(dict.fromkeys(args.sysvars))  # de-dup, preserve order

    fmode = _feature_mode_of(CONFIG_SRC)
    if fmode != EXPECTED_FEATURE_MODE and not args.force:
        raise SystemExit(
            f"REFUSING: {CONFIG_SRC.name} feature_mode is {fmode!r}, expected "
            f"{EXPECTED_FEATURE_MODE!r}. Fix the config (or --force)."
        )

    print(f"feature_mode  : {fmode!r} (from {CONFIG_SRC.name}, never modified)")
    print(f"SysVars       : {', '.join(sysvars)}")
    print(f"Stages        : {', '.join(stages)}")
    print(f"Per-SysVar cfg: runtime-files/config.{args.config_name}.<sysvar>.json")

    results: list[tuple[str, str, int]] = []
    for sysvar in sysvars:
        copy_path = copy_path_for(args.config_name, sysvar)
        if not args.dry_run:
            make_copy(copy_path, sysvar, refresh=args.refresh_copy)
        for stage in stages:
            rc = run_stage(stage, sysvar, copy_path, dry_run=args.dry_run)
            results.append((sysvar, stage, rc))
            if rc != 0:
                print(f"!! [{sysvar}] {stage} FAILED (exit {rc}); skipping rest of this SysVar.",
                      flush=True)
                if not args.continue_on_error:
                    print("Stopping (use --continue-on-error to keep going).", flush=True)
                    _summary(results, args.config_name)
                    return rc
                break

    _summary(results, args.config_name)
    return 1 if any(rc != 0 for _, _, rc in results) else 0


def _summary(results, config_name) -> None:
    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for sysvar, stage, rc in results:
        print(f"  {'OK ' if rc == 0 else 'FAIL'}  {sysvar:<20} {stage} (exit {rc})")
    print(f"\nPer-SysVar config copies: runtime-files/config.{config_name}.<sysvar>.json "
          f"(shared config.json untouched). Histogram separately.")
    print(f"Logs under: {LOG_ROOT}")


if __name__ == "__main__":
    raise SystemExit(main())
