#!/usr/bin/env bash
# One GPU worker, offline inference, resumable evidence; default compute cap: 8h.
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${SEMPTR_PYTHON:-$PWD/warden/bin/python}"
[[ -x "$python_bin" ]] || { echo "Missing project Python: $python_bin" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1
exec "$python_bin" reviewer_suite.py "$@"
