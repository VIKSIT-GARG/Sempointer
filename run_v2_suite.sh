#!/usr/bin/env bash
# V2 runs use new result names; the V1 paper and paper_* results are untouched.
set -euo pipefail
cd "$(dirname "$0")"
PY="$PWD/warden/bin/python"
[[ -x "$PY" ]] || { echo "Missing $PY" >&2; exit 1; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)"
[[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "Cannot read free GPU memory" >&2; exit 1; }
if ((free_mib < 5500)); then
    echo "Only ${free_mib} MiB GPU memory free; need at least 5500 MiB before V2 inference." >&2
    echo "Check nvidia-smi and stop the idle Ollama model or other GPU jobs first." >&2
    exit 1
fi

stage="${1:-smoke}"
common=(--model Qwen/Qwen2.5-3B-Instruct --dtype nf4 --device cuda
        --seed 42 --block-size 512 --k 16 --m 2 --resume)
case "$stage" in
    smoke)
        args=(--benchmark ruler --ruler-tasks niah_single_1,niah_multikey_1
              --context-lengths 4096 --limit 3 --max-new-tokens 32
              --run-name v2_smoke --systems 'sempointer_v2,bm25:2,bm25:4') ;;
    ruler)
        args=(--benchmark ruler
              --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe
              --context-lengths 4096,8192,16384 --limit 10 --max-new-tokens 32
              --run-name v2_ruler --systems 'sempointer_v2,bm25:2,bm25:4') ;;
    ruler_holdout)
        args=(--benchmark ruler --seed 143
              --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe
              --context-lengths 4096,8192,16384 --limit 10 --max-new-tokens 32
              --run-name v2_ruler_holdout
              --systems 'sempointer,sempointer_v2,bm25:2,bm25:4,full') ;;
    longbench)
        args=(--benchmark longbench
              --longbench-tasks hotpotqa,2wikimqa,musique,multifieldqa_en,narrativeqa,qasper
              --limit 100 --max-new-tokens 64
              --run-name v2_longbench --systems 'sempointer_v2,bm25:4') ;;
    locomo)
        args=(--benchmark locomo --locomo-conversations 0,1,2
              --max-new-tokens 64 --run-name v2_locomo
              --systems 'sempointer_v2,bm25:4') ;;
    rescue)
        args=(--benchmark ruler --ruler-tasks niah_single_1,niah_multikey_1
              --context-lengths 4096,8192 --limit 10 --max-new-tokens 32
              --run-name v2_rescue
              --systems 'sempointer_v2,sempointer_v2[semantic_rescue=true],bm25:4') ;;
    *) echo 'Usage: ./run_v2_suite.sh [smoke|ruler|ruler_holdout|longbench|locomo|rescue]' >&2; exit 2 ;;
esac
"$PY" run_benchmark.py "${common[@]}" "${args[@]}"
