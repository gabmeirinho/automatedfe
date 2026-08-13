#!/bin/bash
# Run a single feature-search task (one seed) through the unified CLI.
#
# All settings can be overridden via environment variables; edit the defaults
# below or use one of the strategy presets in this directory. Intended to be
# driven by run_seeds.sh, which sets TASK_ID for each seed.

export PYTHONUNBUFFERED=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ---- Run configuration (override via environment as needed) ----
STRATEGY="${STRATEGY:-genetic}"                 # genetic | enumerative | random | enumerative_without_archive
BASE_SEED="${BASE_SEED:-42}"                    # seed = BASE_SEED + TASK_ID
TASK_ID="${TASK_ID:-0}"
SEED_COUNT="${SEED_COUNT:-1}"                   # only used for logging
TIME_BUDGET="${TIME_BUDGET:-600}"               # seconds, evaluated strategies only (10 minutes)
CANDIDATE_COUNT="${CANDIDATE_COUNT:-100}"       # enumerative_without_archive only
POPULATION_SIZE="${POPULATION_SIZE:-50}"        # genetic only
MAX_DEPTH="${MAX_DEPTH:-4}"
SCORE_METRIC="${SCORE_METRIC:-brier_improvement}"
FITNESS_RANDOM_STATE="${FITNESS_RANDOM_STATE:-42}"
DATASET="${DATASET:-data/loan/transformed/dataset.parquet}"
MAPPING="${MAPPING:-data/loan/transformed/label_mapping.json}"
MMAP_DIR="${MMAP_DIR:-data/loan/transformed/mmap}"
FEATURE_CACHE_DIR="${FEATURE_CACHE_DIR:-}"      # disabled by default

# ---- Active-set promotion (genetic only; requires SCORE_METRIC=brier_improvement) ----
USE_ACTIVE_SET="${USE_ACTIVE_SET:-false}"
PROMOTION_INTERVAL="${PROMOTION_INTERVAL:-5}"
FIRST_PROMOTION_TOP_K="${FIRST_PROMOTION_TOP_K:-2}"
PROMOTION_ADD_K="${PROMOTION_ADD_K:-1}"
PROMOTION_REFRESH_TOP_N="${PROMOTION_REFRESH_TOP_N:-50}"
ARCHIVE_QUALITY_THRESHOLD="${ARCHIVE_QUALITY_THRESHOLD:-0.001}"
ARCHIVE_CORRELATION_THRESHOLD="${ARCHIVE_CORRELATION_THRESHOLD:-0.85}"
ACTIVE_CORRELATION_THRESHOLD="${ACTIVE_CORRELATION_THRESHOLD:-0.90}"
PROMOTION_MIN_GAIN="${PROMOTION_MIN_GAIN:-0.0}"
PROMOTION_MEAN_GAIN="${PROMOTION_MEAN_GAIN:-0.0005}"

# ---- Outputs ----
RUN_GROUP="${RUN_GROUP:-${STRATEGY}_search}"
OUT_DIR="${OUT_DIR:-runs/${RUN_GROUP}}"
FORCE="${FORCE:-true}"                          # overwrite existing seed outputs

SEED=$((BASE_SEED + TASK_ID))
RUN_DIR="${OUT_DIR}/seed_${SEED}"

echo "Running ${STRATEGY} search locally"
echo "task_id=${TASK_ID} seed=${SEED} seed_count=${SEED_COUNT}"
echo "run_group=${RUN_GROUP} time_budget=${TIME_BUDGET} population_size=${POPULATION_SIZE} max_depth=${MAX_DEPTH}"
echo "score_metric=${SCORE_METRIC} use_active_set=${USE_ACTIVE_SET}"
echo "out_dir=${OUT_DIR}"

ARGS=(
  --strategy "${STRATEGY}"
  --seed "${SEED}"
  --dataset "${DATASET}"
  --mapping "${MAPPING}"
  --mmap-dir "${MMAP_DIR}"
  --score-metric "${SCORE_METRIC}"
  --fitness-random-state "${FITNESS_RANDOM_STATE}"
  --max-depth "${MAX_DEPTH}"
)

case "${STRATEGY}" in
  enumerative_without_archive)
    ARGS+=(--candidate-count "${CANDIDATE_COUNT}")
    ;;
  *)
    ARGS+=(--time-budget "${TIME_BUDGET}")
    ARGS+=(--csv "${RUN_DIR}/diagnostics.csv")
    ARGS+=(--archive "${RUN_DIR}/archive.json")
    ;;
esac

if [[ "${STRATEGY}" == "genetic" ]]; then
  ARGS+=(--population-size "${POPULATION_SIZE}")
  if [[ "${USE_ACTIVE_SET}" == "true" ]]; then
    ARGS+=(
      --use-active-set
      --promotion-interval "${PROMOTION_INTERVAL}"
      --first-promotion-top-k "${FIRST_PROMOTION_TOP_K}"
      --promotion-add-k "${PROMOTION_ADD_K}"
      --promotion-refresh-top-n "${PROMOTION_REFRESH_TOP_N}"
      --archive-quality-threshold "${ARCHIVE_QUALITY_THRESHOLD}"
      --archive-correlation-threshold "${ARCHIVE_CORRELATION_THRESHOLD}"
      --active-correlation-threshold "${ACTIVE_CORRELATION_THRESHOLD}"
      --promotion-min-gain "${PROMOTION_MIN_GAIN}"
      --promotion-mean-gain "${PROMOTION_MEAN_GAIN}"
      --history "${RUN_DIR}/history.json"
      --active-archive "${RUN_DIR}/active_archive.json"
    )
  fi
fi

if [[ -n "${FEATURE_CACHE_DIR}" ]]; then
  ARGS+=(--feature-cache-dir "${FEATURE_CACHE_DIR}")
fi

if [[ "${FORCE}" == "true" ]]; then
  ARGS+=(--force)
fi

ARGS+=(--summary "${RUN_DIR}/summary.json")

uv run automatedfe search "${ARGS[@]}"

