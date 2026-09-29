#!/usr/bin/env bash
# Resumable, offline paper runs on the 8 GB RTX 5060.
# Usage: ./run_paper_suite.sh [--dry-run] [--max-hours 10] [paper_ruler ...]
# The wall-clock cap interrupts the active stage; completed prediction rows
# remain on disk and the same command resumes them on the next invocation.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/warden/bin/python"
MODEL="Qwen/Qwen2.5-3B-Instruct"
EMBEDDER="sentence-transformers/all-MiniLM-L6-v2"
DTYPE=nf4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

usage() {
    cat <<'EOF'
Usage: ./run_paper_suite.sh [--dry-run] [--max-hours HOURS] [STAGE ...]

Stages (default order): paper_ruler paper_ablation paper_kv paper_longbench paper_locomo
--dry-run validates cached models, datasets, and every system spec on CPU,
          then prints the commands. It does not require CUDA or run inference.
--max-hours sets a wall-clock cap for this invocation (default: 10).
Re-run the same command to resume scored examples in fixed results/paper_* dirs.
EOF
}

DRY_RUN=0
MAX_HOURS=10
STAGES=()
while (($#)); do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --max-hours)
            (($# >= 2)) || { echo 'ERROR: --max-hours needs a value' >&2; exit 2; }
            MAX_HOURS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        -*) echo "ERROR: unknown option $1" >&2; usage; exit 2 ;;
        *) STAGES+=("$1"); shift ;;
    esac
done
[[ "$MAX_HOURS" =~ ^[1-9][0-9]*$ ]] || { echo 'ERROR: --max-hours must be a positive integer' >&2; exit 2; }
((${#STAGES[@]})) || STAGES=(paper_ruler paper_ablation paper_kv paper_longbench paper_locomo)
[[ -x "$PY" ]] || { echo "ERROR: missing venv Python: $PY" >&2; exit 1; }

# All stage settings live here so validation and execution use identical argv.
stage_args() {
    local stage="$1"
    ARGS=(--model "$MODEL" --dtype "$DTYPE" --device cuda
          --seed 42 --block-size 512 --k 16 --m 2 --run-name "$stage" --resume)
    case "$stage" in
        paper_ruler)
            ARGS+=(--benchmark ruler
                --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe
                --context-lengths 4096,8192,16384 --limit 10
                --systems 'sempointer,sempointer[selector=hybrid],full,rag:1,rag:4,bm25:4,hybrid:4'
                --max-new-tokens 32) ;;
        paper_ablation)
            ARGS+=(--benchmark ruler
                --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe
                --context-lengths 4096,8192 --limit 10
                --systems 'sempointer,sempointer[ptype=random_id],sempointer[ptype=none],sempointer[ptype=lead],sempointer[k=8],sempointer[k=32],sempointer[m=1],sempointer[m=adaptive],sempointer[hops=2],sempointer[selector=dense],sempointer[selector=bm25],sempointer[selector=attention]'
                --max-new-tokens 32) ;;
        paper_kv)
            ARGS+=(--benchmark kv_compare
                --ruler-tasks niah_single_1,niah_multikey_1
                --context-lengths 4096 --limit 25
                --systems 'sempointer,sempointer[substrate=kv],full,kvpress:snapkv:0.3,kvpress:tova:0.3,kvpress:knorm:0.3'
                --max-new-tokens 32) ;;
        paper_longbench)
            ARGS+=(--benchmark longbench
                --longbench-tasks hotpotqa,2wikimqa,musique,multifieldqa_en,narrativeqa,qasper
                --limit 100 --systems 'sempointer,full,rag:4,hybrid:4'
                --max-new-tokens 64) ;;
        paper_locomo)
            ARGS+=(--benchmark locomo --locomo-conversations 0,1,2
                --systems 'sempointer,full,rag:4,hybrid:4'
                --max-new-tokens 64) ;;
        *) echo "ERROR: unknown stage $stage" >&2; usage; return 2 ;;
    esac
}

# This is intentionally CPU-only: run_benchmark.py's own --dry-run currently
# requires CUDA even when asked to use --device cpu.
validate_stage() {
    "$PY" - "${ARGS[@]}" <<'PYCODE'
import json
import os
import sys

sys.argv = ["run_benchmark.py", *sys.argv[1:]]
from run_benchmark import build_examples, parse_args, parse_specs, system_class
from system_specs import validate_kwargs

args = parse_args()
specs = parse_specs(args)
if args.dtype == "nf4":
    import bitsandbytes  # noqa: F401 - fail before a long run if NF4 support is absent
if any(spec.kind == "kvpress" for spec in specs):
    import kvpress  # noqa: F401
for spec in specs:
    validate_kwargs(system_class(spec.kind), spec.kwargs, spec.raw)
args.limit = 1
args.context_lengths = str(min(int(x) for x in args.context_lengths.split(",")))
examples = build_examples(args)
if not examples:
    raise RuntimeError("no examples loaded")

path = os.path.join("results", args.run_name, "config.json")
if os.path.isfile(path):
    with open(path, encoding="utf-8") as f:
        previous = json.load(f)
    keys = ("benchmark", "systems", "model", "dtype", "device", "seed", "limit",
            "max_new_tokens", "k", "m", "rag_m", "block_size", "context_lengths",
            "ruler_tasks", "longbench_tasks", "locomo_conversations", "locomo_categories")
    # Restore the actual stage values after the one-example dataset probe.
    sys.argv = ["run_benchmark.py", *sys.argv[1:]]
    current = parse_args()
    changed = [k for k in keys if str(previous.get(k)) != str(getattr(current, k))]
    if changed:
        raise RuntimeError(f"{path} has incompatible settings: {', '.join(changed)}")
print(f"  {args.run_name}: {len(specs)} system specs valid; {len(examples)} sample examples accessible")
PYCODE
}

echo "Paper suite: ${STAGES[*]} (model=$MODEL, dtype=$DTYPE, cap=${MAX_HOURS}h)"
"$PY" - "$MODEL" "$EMBEDDER" <<'PYCODE' || exit 1
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
for model in sys.argv[1:]:
    snapshot_download(model, local_files_only=True)
    print(f"  cached: {model}")
AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True)
print("  tokenizer: OK")
PYCODE
for stage in "${STAGES[@]}"; do
    stage_args "$stage" || exit $?
    validate_stage || exit 1
    if ((DRY_RUN)); then
        printf '  command:'
        printf ' %q' "$PY" run_benchmark.py "${ARGS[@]}"
        printf '\n'
    fi
done
if ((DRY_RUN)); then
    echo 'Dry run passed. No GPU work started.'
    exit 0
fi

command -v timeout >/dev/null || { echo 'ERROR: coreutils timeout is required for the wall-clock cap' >&2; exit 1; }
"$PY" - <<'PYCODE' || exit 1
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")
props = torch.cuda.get_device_properties(0)
if props.total_memory < 7 * 1024**3:
    raise SystemExit(f"ERROR: GPU has only {props.total_memory / 1024**3:.1f} GiB VRAM")
print(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f} GiB)")
PYCODE

mkdir -p logs results
LOG="logs/paper_suite_$(date +%Y%m%d_%H%M%S).log"
START=$(date +%s)
DEADLINE=$((START + MAX_HOURS * 3600))
FAILED=()
TIMED_OUT=0
for stage in "${STAGES[@]}"; do
    stage_args "$stage"
    remaining=$((DEADLINE - $(date +%s)))
    if ((remaining <= 0)); then TIMED_OUT=1; break; fi
    echo "[$(date +%T)] START $stage (remaining ${remaining}s)" | tee -a "$LOG"
    timeout --signal=INT --kill-after=30s "${remaining}s" "$PY" run_benchmark.py "${ARGS[@]}" 2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}
    if ((rc == 124 || rc == 137 || rc == 130 || rc == 143)); then
        echo "[$(date +%T)] INTERRUPTED $stage (exit $rc); re-run to resume." | tee -a "$LOG"
        TIMED_OUT=1
        break
    fi
    if ((rc != 0)); then
        echo "[$(date +%T)] FAILED $stage (exit $rc); continuing." | tee -a "$LOG"
        FAILED+=("$stage")
        continue
    fi
    "$PY" - "$stage" <<'PYCODE' | tee -a "$LOG"
import json
import sys
stage = sys.argv[1]
with open(f"results/{stage}/metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)
incomplete = {name: (value.get("n_scored", 0), value.get("n_errors", 0))
              for name, value in metrics["systems"].items()
              if value.get("n_scored", 0) != metrics["n_examples"] or value.get("n_errors", 0)}
if incomplete:
    print(f"INCOMPLETE {stage}: {incomplete}; re-run to retry errors")
    sys.exit(1)
print(f"DONE {stage}: {metrics['n_examples']} examples per system")
PYCODE
    check_rc=${PIPESTATUS[0]}
    ((check_rc == 0)) || FAILED+=("$stage")
done

echo "Results: $SCRIPT_DIR/results/paper_*/metrics.json" | tee -a "$LOG"
echo "Log: $SCRIPT_DIR/$LOG" | tee -a "$LOG"
if ((TIMED_OUT)); then exit 124; fi
if ((${#FAILED[@]})); then
    echo "Incomplete stages: ${FAILED[*]} (re-run those names to resume)" | tee -a "$LOG"
    exit 1
fi
echo 'All requested stages complete.' | tee -a "$LOG"
