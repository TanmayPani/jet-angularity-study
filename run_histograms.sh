#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Drive the histogramming for ANY feature mode: run histograms.py once per
# systematic variation, each with its own per-SysVar config copy
# runtime-files/config.<config-name>.<sysvar>.json (passed via
# config.config_path_from_argv, so the global runtime-files/config.json is
# never touched). These copies are the same ones run_pipeline.py bakes.
#
#   * nominal MUST run first -- it produces the shared runtime-files/bins_perpt.json
#     edges and the MC reference hists (pythia6/8, herwig7) the others/ratios reuse.
#   * unf_iter_sys_{0,1} reuse nominal's w_unfolding.npz, read at the central
#     iteration +/- 1 (the iteration systematic).
#
# Sequential + fail-fast on a *run* error: these snapshots feed systematics.py,
# so a broken sysvar halts the chain rather than producing a silently-partial
# band. A *missing* config (a sysvar not present for this mode) is skipped with
# a warning -- different feature modes carry different sysvar sets -- except for
# nominal, whose absence is fatal.
#
# Feature mode + config-name tag (mirrors run_pipeline.py):
#   --feature-mode MODE   feature mode (default: feature_mode in config.json);
#                         used for the log dir + summary text.
#   --config-name  NAME   tag in config.<NAME>.<sysvar>.json (default: MODE).
#                         Use --config-name noptd for the legacy noptd copies.
#
# Usage:
#   ./run_histograms.sh                                   # mode from config.json, full band
#   ./run_histograms.sh --feature-mode angularities_minimal
#   ./run_histograms.sh --feature-mode angularities_noptd --config-name noptd
#   ./run_histograms.sh --feature-mode angularities_minimal nominal track_pt_sys
# ---------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")"

FEATURE_MODE=""
CONFIG_NAME=""
# Order matters: nominal first. The full canonical band; modes that lack some
# of these (no matching config copy) skip them with a warning.
SYSVARS=(
  nominal
  tower_et_corr_sys
  tower_gain_sys
  track_pt_sys
  unf_prior_herwig7
  unf_prior_like_data
  unf_iter_sys_0
  unf_iter_sys_1
  jet_pt_res_sys_0
  jet_pt_res_sys_1
)
POSITIONAL=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --feature-mode) FEATURE_MODE="${2:?--feature-mode needs a value}"; shift 2 ;;
    --config-name)  CONFIG_NAME="${2:?--config-name needs a value}";  shift 2 ;;
    -h|--help) sed -n '2,33p' "$0"; exit 0 ;;
    --) shift; POSITIONAL+=("$@"); break ;;
    -*) echo "!! unknown option: $1" >&2; exit 2 ;;
    *)  POSITIONAL+=("$1"); shift ;;
  esac
done

# feature_mode defaults to the one in the shared config.json.
if [[ -z "$FEATURE_MODE" ]]; then
  FEATURE_MODE="$(grep -oP '"feature_mode"\s*:\s*"\K[^"]+' runtime-files/config.json)" \
    || { echo "!! could not read feature_mode from runtime-files/config.json" >&2; exit 1; }
fi
# config-name tag defaults to feature_mode (matches run_pipeline.py --config-name default).
CONFIG_NAME="${CONFIG_NAME:-$FEATURE_MODE}"

# Positional args (after flags) override the sysvar run list.
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  SYSVARS=("${POSITIONAL[@]}")
fi

LOG_DIR="logs/${FEATURE_MODE}_histograms"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"

echo "=== ${FEATURE_MODE} histogramming: up to ${#SYSVARS[@]} run(s) -> $LOG_DIR ==="
echo "    config copies: runtime-files/config.${CONFIG_NAME}.<sysvar>.json"
overall_start=$SECONDS
ran=0

for sv in "${SYSVARS[@]}"; do
  cfg="runtime-files/config.${CONFIG_NAME}.${sv}.json"
  log="$LOG_DIR/${sv}_${STAMP}.log"

  if [[ ! -f "$cfg" ]]; then
    if [[ "$sv" == "nominal" ]]; then
      echo "!! missing nominal config: $cfg" >&2
      echo "   (nominal must run first -- it produces the shared bins + MC refs)" >&2
      exit 1
    fi
    echo ""
    echo ">>> [$sv] SKIP -- no config $cfg"
    continue
  fi

  echo ""
  echo ">>> [$sv] $(date '+%H:%M:%S')  uv run histograms.py $cfg"
  echo "    log: $log   (tail -f to watch)"
  start=$SECONDS
  if uv run histograms.py "$cfg" >"$log" 2>&1; then
    echo "    OK  ($((SECONDS - start))s)"
    ran=$((ran + 1))
  else
    rc=$?
    echo "!! [$sv] FAILED (rc=$rc) after $((SECONDS - start))s -- last 20 lines:" >&2
    tail -n 20 "$log" >&2
    exit "$rc"
  fi
done

echo ""
echo "=== ${ran} run(s) complete in $((SECONDS - overall_start))s ==="
echo "Next (these read the GLOBAL config.json, so set feature_mode=${FEATURE_MODE} there):"
echo "  uv run systematics.py && uv run plot_hp2026_prelims.py"
