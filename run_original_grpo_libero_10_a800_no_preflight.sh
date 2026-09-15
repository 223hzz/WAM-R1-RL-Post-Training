#!/usr/bin/env bash
# LIBERO-10 A800 GRPO entry point without the SFT preflight gate.
# Set PREFLIGHT_EVAL_EPISODES or REQUIRE_PREFLIGHT_SUCCESS explicitly to override.
set -Eeuo pipefail

export PREFLIGHT_EVAL_EPISODES="${PREFLIGHT_EVAL_EPISODES:-0}"
export REQUIRE_PREFLIGHT_SUCCESS="${REQUIRE_PREFLIGHT_SUCCESS:-0}"
exec bash "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/run_original_grpo_libero_10_a800.sh" "$@"
