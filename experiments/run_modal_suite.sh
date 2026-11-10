#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
project_python="$PWD/warden/bin/python"
modal_python="$PWD/.venv-modal/bin/python"
run_root="${SEMPTR_MODAL_RUN_DIR:-$PWD/modal_runs/ten_qwen35_v1}"
if [[ -z "${SEMPTR_MODAL_RUN_DIR:-}" && -f "$PWD/modal_runs/selected_inference_bundle.json" ]]; then
    run_root="$("$project_python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$PWD/modal_runs/selected_inference_bundle.json")"
fi
bundle="$run_root/bundle"
stage="${1:-plan}"
if (($#)); then shift; fi
if [[ "$stage" == plan ]]; then
    echo 'Ten families, maximum 300 labeled eligible questions per family, six systems (full context where it fits).'
    echo 'Qwen3.5-4B; frozen inference configuration, independent routing development, cost pilot, paired CWE cap audit.'
    echo "Selected run directory: $run_root"
    echo 'One-time prerequisite: Modal workspace usage cap <= $25 and net spend cap $0.'
    echo 'Run: bash run_modal_suite.sh start --workspace-budget-confirmed'
    echo 'Preparation and upload finish locally before the detached cloud job begins.'
    exit 0
fi
if [[ "$stage" == start || "$stage" == resume ]]; then
    confirmed=false
    for arg in "$@"; do [[ "$arg" != --workspace-budget-confirmed ]] || confirmed=true; done
    if [[ "$confirmed" != true ]]; then
        echo 'Launch blocked: set Modal workspace usage limit <= $25 and net spend limit $0, then add --workspace-budget-confirmed.' >&2
        echo 'This is required to enforce the strict ceiling; the script cannot verify or set those account limits.' >&2
        exit 2
    fi
fi
if [[ "$stage" == prepare || "$stage" == start ]]; then
    preparation_args=()
    [[ -z "${SEMPTR_PREP_WORKERS:-}" ]] || preparation_args+=(--workers "$SEMPTR_PREP_WORKERS")
    [[ -z "${SEMPTR_TOKENIZER_THREADS:-}" ]] || preparation_args+=(--tokenizer-threads "$SEMPTR_TOKENIZER_THREADS")
    "$project_python" prepare_modal.py --output "$bundle" "${preparation_args[@]}"
    [[ "$stage" != prepare ]] || exit 0
fi
case "$stage" in start|resume|status|download|cancel) ;; *) echo 'Usage: bash run_modal_suite.sh [plan|prepare|start|resume|status|download|cancel] [--workspace-budget-confirmed]' >&2; exit 2;; esac
[[ -x "$modal_python" ]] || { echo 'Install the isolated client with: uv venv --python warden/bin/python .venv-modal && uv pip install --python .venv-modal/bin/python modal==1.6.3' >&2; exit 1; }
exec "$modal_python" modal_client.py "$stage" --bundle "$bundle" "$@"
