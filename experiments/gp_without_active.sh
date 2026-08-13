#!/bin/bash
# Plain genetic search preset, active set disabled (one seed).

export STRATEGY="${STRATEGY:-genetic}"
export USE_ACTIVE_SET="${USE_ACTIVE_SET:-false}"
export RUN_GROUP="${RUN_GROUP:-gp_without_active}"
export TIME_BUDGET="${TIME_BUDGET:-600}"
export POPULATION_SIZE="${POPULATION_SIZE:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_search.sh"

