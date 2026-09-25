#!/usr/bin/env bash
# Compatibility entry point; this run is direct and needs no scheduler.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/run_original_grpo_libero_10_a100.sh" "$@"
