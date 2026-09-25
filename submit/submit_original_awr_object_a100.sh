#!/usr/bin/env bash
# Direct-run entry point for A100 servers. There is no scheduler submission.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "DiT4DiT A100 launcher: NVIDIA-EGL physical-GPU mode (2026-08-29)" >&2
exec bash "${SCRIPT_DIR}/run_original_awr_object_a100.sh" "$@"
