#!/usr/bin/env bash
# New model/new evidence directory; historical Qwen2.5 launchers remain available.
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${SEMPTR_PYTHON:-$PWD/warden/bin/python}"
precision="${SEMPTR_QWEN35_DTYPE:-nf4}"
[[ "$precision" == nf4 || "$precision" == bf16 ]] || { echo 'Precision must be nf4 or bf16' >&2; exit 2; }
model_path="$PWD/../models/Qwen3.5-4B"
output_root="${SEMPTR_QWEN35_OUTPUT:-$PWD/results/reviewer_qwen35_4b_${precision}_qa128_cwe512}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DEACTIVATE_ASYNC_LOAD=1 HF_HUB_DISABLE_PROGRESS_BARS=1
stage="${1:-dry-run}"
if (($#)); then shift; fi
common=(--model "$model_path" --engine qwen35 --dtype "$precision" --qa-output-tokens 128 --cwe-output-tokens 512)
case "$stage" in
    model)
        HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 "$python_bin" -c \
          'from pathlib import Path; from huggingface_hub import snapshot_download; snapshot_download("Qwen/Qwen3.5-4B", local_dir=Path.cwd().parent/"models/Qwen3.5-4B", allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "*.md"], token=False, max_workers=2)'
        ;;
    check)
        exec "$python_bin" run_all_datasets.py model-check "${common[@]}" --output "$output_root/preflight" "$@"
        ;;
    smoke)
        exec "$python_bin" run_all_datasets.py run "${common[@]}" --families ruler \
          --tasks niah_single_1,cwe --context-lengths 4096 --ruler-samples 1 --partition all \
          --limit 1 --seed 20261109 --systems sempointer,pointer_off,bm25_2,bm25_4,bm25_matched_budget,full_context \
          --output "$output_root/standalone_smoke" "$@"
        ;;
    dry-run|run|report)
        extra=()
        [[ "$stage" != dry-run ]] || extra+=(--dry-run)
        [[ "$stage" != report ]] || extra+=(--report-only)
        exec "$python_bin" reviewer_suite.py "${common[@]}" --output "$output_root" "${extra[@]}" "$@"
        ;;
    *) echo 'Usage: bash run_qwen35_suite.sh [model|check|smoke|dry-run|run|report] [options]' >&2; exit 2 ;;
esac
