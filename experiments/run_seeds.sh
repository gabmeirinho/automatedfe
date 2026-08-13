#!/bin/bash
# Run a batch of seeds for one or more search strategies locally.
#
# For each run type in RUN_TYPES, launches SEED_COUNT tasks (seed =
# BASE_SEED + task id) with up to MAX_CONCURRENT tasks in flight. A run type
# is either a preset script in this directory (e.g. gp_active.sh) or the name
# of a strategy passed directly to run_search.sh (e.g. "random").

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEED_COUNT="${SEED_COUNT:-30}"
BASE_SEED="${BASE_SEED:-42}"
MAX_CONCURRENT="${MAX_CONCURRENT:-1}"
RUN_TYPES="${RUN_TYPES:-gp_active gp_without_active random enumerative}"

if (( SEED_COUNT < 1 )); then
  echo "SEED_COUNT must be >= 1" >&2
  exit 1
fi

if (( MAX_CONCURRENT < 1 )); then
  echo "MAX_CONCURRENT must be >= 1" >&2
  exit 1
fi

if [[ -z "${RUN_TYPES}" ]]; then
  echo "RUN_TYPES must name at least one preset or strategy" >&2
  exit 1
fi

run_one() {
  local run_type="$1"
  local task_id="$2"
  local script_path="${SCRIPT_DIR}/${run_type}.sh"

  if [[ -x "${script_path}" ]]; then
    TASK_ID="${task_id}" SEED_COUNT="${SEED_COUNT}" BASE_SEED="${BASE_SEED}" "${script_path}"
  else
    TASK_ID="${task_id}" SEED_COUNT="${SEED_COUNT}" BASE_SEED="${BASE_SEED}" \
      STRATEGY="${run_type}" "${SCRIPT_DIR}/run_search.sh"
  fi
}

failures=0

for run_type in ${RUN_TYPES}; do
  echo "==> ${run_type}: ${SEED_COUNT} seeds, MAX_CONCURRENT=${MAX_CONCURRENT}"

  running=0
  for ((task_id = 0; task_id < SEED_COUNT; task_id++)); do
    run_one "${run_type}" "${task_id}" &
    running=$((running + 1))

    if (( running >= MAX_CONCURRENT )); then
      wait -n || failures=$((failures + 1))
      running=$((running - 1))
    fi
  done

  for ((i = 0; i < running; i++)); do
    wait -n || failures=$((failures + 1))
  done
done

if (( failures > 0 )); then
  echo "${failures} seed run(s) failed" >&2
  exit 1
fi

echo "All seed runs finished successfully"

