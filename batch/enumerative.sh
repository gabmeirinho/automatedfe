#!/bin/bash
# Enumerative search preset (one seed; see run_seeds.sh for batching).

export STRATEGY="${STRATEGY:-enumerative}"
export USE_ACTIVE_SET="${USE_ACTIVE_SET:-false}"
export RUN_GROUP="${RUN_GROUP:-enumerative_search}"
export TIME_BUDGET="${TIME_BUDGET:-600}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_search.sh"
