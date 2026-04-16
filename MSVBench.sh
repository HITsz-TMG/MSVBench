#!/usr/bin/env bash

# Evaluate (story_id, method) pairs one by one.
# This script drives single-case execution by setting environment variables
# and invoking MSVBench.py directly.

# Re-exec with bash if needed
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="python3"

# API key should be provided from env for open-source safety:
# - GEMINI_API_KEYS: comma-separated string or JSON array string
# - GEMINI_API_KEYS_FILE: path to file containing keys
if [[ -z "${GEMINI_API_KEYS:-}" && -z "${GEMINI_API_KEYS_FILE:-}" ]]; then
  echo "[WARN] GEMINI_API_KEYS/GEMINI_API_KEYS_FILE is not set; Gemini-dependent metrics may fail."
fi

# Story IDs to evaluate
STORY_IDS=(
  "01" "02" "08" "09" "14" "17" "19" "22" "24" "27"
  "28" "29" "32" "41" "55" "57" "60" "64" "68" "79"
)

# Method list.
METHODS=(
  "LongLive"
)

# Skip existing results when set to 1
SKIP_IF_EXISTS=${SKIP_IF_EXISTS:-0}

RESULTS_DIR="$SCRIPT_DIR/Evaluation/results"

echo "========== MSVBench Batch Evaluation Script =========="
echo "Working directory: $SCRIPT_DIR"
echo "Python: $PYTHON_BIN"
if [[ -n "${GEMINI_API_KEYS_FILE:-}" ]]; then
  echo "Gemini keys source: file $GEMINI_API_KEYS_FILE"
elif [[ -n "${GEMINI_API_KEYS:-}" ]]; then
  IFS=',' read -r -a __keys <<< "$GEMINI_API_KEYS"
  echo "Gemini key count: ${#__keys[@]}"
else
  echo "Gemini keys not set"
fi
echo "Skip existing results: $SKIP_IF_EXISTS (1=skip, 0=run)"
echo

mkdir -p "$RESULTS_DIR"

MODULES_ENV=${MODULES:-}
SUBMETRICS_ENV=${SUBMETRICS:-}

# Iterate all (method, story_id) pairs
for method in "${METHODS[@]}"; do
  for sid in "${STORY_IDS[@]}"; do
    printf "\n----- Start evaluation: method=%s, story_id=%s -----\n" "$method" "$sid"

    RESULT_JSON="$RESULTS_DIR/$method/$sid.json"
    if [[ "${SKIP_IF_EXISTS:-0}" -eq 1 && -f "$RESULT_JSON" ]]; then
      echo "Result exists: $RESULT_JSON, skipped (set SKIP_IF_EXISTS=0 to re-run)"
      continue
    fi

    if [[ -n "$MODULES_ENV" ]]; then
      echo "Modules: $MODULES_ENV"
    else
      echo "Modules: (default all)"
    fi
    if [[ -n "$SUBMETRICS_ENV" ]]; then
      echo "Submetrics: $SUBMETRICS_ENV"
    else
      echo "Submetrics: (not specified, module defaults)"
    fi

    METHOD="$method" STORY_ID="$sid" MODULES="$MODULES_ENV" SUBMETRICS="$SUBMETRICS_ENV" \
      "$PYTHON_BIN" "$SCRIPT_DIR/MSVBench.py"
  done
done
