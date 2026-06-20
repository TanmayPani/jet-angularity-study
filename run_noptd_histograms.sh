#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Drive the no-p_T^D histogramming: run histograms.py once per systematic
# variation, each with its own runtime-files/config.noptd.<sysvar>.json (passed
# via config.config_path_from_argv, so the global runtime-files/config.json is
# never touched).
#
#   * nominal MUST run first -- it produces the shared runtime-files/bins_perpt.json
#     edges and the MC reference hists (pythia6/8, herwig7) the others/ratios reuse.
#   * unf_iter_sys_{0,1} reuse nominal's w_unfolding.npz, read at iter1 / iter3
#     (the {1,3} iteration systematic around the noptd central iter2).
#
# Sequential + fail-fast: these snapshots feed systematics.py, so a broken sysvar
# halts the chain rather than producing a silently-partial band.
#
# Usage:
#   ./run_noptd_histograms.sh                  # all 7, in order
#   ./run_noptd_histograms.sh nominal track_pt_sys   # just these (still order-checked)
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

# Order matters: nominal first.
SYSVARS=(
  nominal
  tower_et_corr_sys
  track_pt_sys
  unf_prior_herwig7
  unf_prior_like_data
  unf_iter_sys_0
  unf_iter_sys_1
  jet_pt_res_sys_0
  jet_pt_res_sys_1
)
# Optional CLI override of the run list.
if [[ $# -gt 0 ]]; then
  SYSVARS=("$@")
fi

LOG_DIR="logs/noptd_histograms"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== no-p_T^D histogramming: ${#SYSVARS[@]} run(s) -> $LOG_DIR ==="
overall_start=$SECONDS

for sv in "${SYSVARS[@]}"; do
  cfg="runtime-files/config.noptd.${sv}.json"
  log="$LOG_DIR/${sv}_${STAMP}.log"

  if [[ ! -f "$cfg" ]]; then
    echo "!! missing config: $cfg" >&2
    exit 1
  fi

  echo ""
  echo ">>> [$sv] $(date '+%H:%M:%S')  uv run histograms.py $cfg"
  echo "    log: $log   (tail -f to watch)"
  start=$SECONDS
  if uv run histograms.py "$cfg" >"$log" 2>&1; then
    echo "    OK  ($((SECONDS - start))s)"
  else
    rc=$?
    echo "!! [$sv] FAILED (rc=$rc) after $((SECONDS - start))s -- last 20 lines:" >&2
    tail -n 20 "$log" >&2
    exit "$rc"
  fi
done

echo ""
echo "=== all ${#SYSVARS[@]} run(s) complete in $((SECONDS - overall_start))s ==="
echo "Next (these read the GLOBAL config.json, so set feature_mode=angularities_noptd there):"
echo "  uv run systematics.py && uv run plot_hp2026_prelims.py"
