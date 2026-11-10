#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1
exec "$PWD/.venv-local-suite/bin/python" -m evaluation.bounded_full "${@:-status}"
