#!/usr/bin/env bash
# One entry point; preserves previous runs and never trains model weights.
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${SEMPTR_PYTHON:-$PWD/warden/bin/python}"
output_root="${SEMPTR_OUTPUT_ROOT:-$PWD/results/downloaded_suite}"
default_model="Qwen/Qwen2.5-3B-Instruct"
if [[ -f ../models/Qwen2.5-3B-Instruct/config.json ]]; then
    default_model="$PWD/../models/Qwen2.5-3B-Instruct"
fi
model_id="${SEMPTR_MODEL:-$default_model}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1
stage="${1:-check}"
case "$stage" in
    model)
        # Explicit setup stage only. Evaluation stages remain offline.
        HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$python_bin" -c \
            'from pathlib import Path; from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen2.5-3B-Instruct", local_dir=Path.cwd().parent/"models/Qwen2.5-3B-Instruct", allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.md"], token=False, max_workers=2)'
        ;;
    check)
        "$python_bin" run_all_datasets.py plan --output "$output_root/preflight"
        "$python_bin" run_all_datasets.py validate --output "$output_root/preflight"
        "$python_bin" run_all_datasets.py model-check --model "$model_id" --output "$output_root/preflight"
        ;;
    smoke)
        "$python_bin" run_all_datasets.py run --model "$model_id" \
            --families ruler --tasks niah_single_1 --context-lengths 4096 \
            --ruler-samples 1 --partition all --limit 1 --output "$output_root/smoke"
        ;;
    full)
        status=0
        "$python_bin" run_all_datasets.py run --model "$model_id" --limit 0 \
            --partition test --output "$output_root/qa" || status=$?
        "$python_bin" run_all_datasets.py beir --model "$model_id" --limit 0 \
            --output "$output_root/beir" || status=$?
        exit "$status"
        ;;
    report)
        for run in qa beir; do
            if [[ -f "$output_root/$run/config.json" ]]; then
                "$python_bin" run_all_datasets.py summarize --output "$output_root/$run"
            fi
        done
        ;;
    *) echo 'Usage: bash run_downloaded_suite.sh [model|check|smoke|full|report]' >&2; exit 2 ;;
esac
