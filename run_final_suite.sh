#!/usr/bin/env bash
# =============================================================================
# SemPointer — full final benchmark suite (P1 → P4), sequential, resumable.
#
# Usage:
#   ./run_final_suite.sh                  # run everything: P1 P4 P3 P2
#   ./run_final_suite.sh ruler kvcompare  # run only these, in this order
#
# Every run is idempotent: each stage calls run_benchmark.py with --resume and
# a fixed --run-name, so re-running this script (after a crash, reboot, OOM)
# skips every (system, example) pair already recorded in predictions.jsonl.
#
# Results: results/final_{ruler,kvcompare,locomo,longbench}/
# Logs:    logs/final_suite_<timestamp>.log  (+ per-stage console output)
# =============================================================================
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="$SCRIPT_DIR/warden/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found — create the venv first (RUNNING_EXPERIMENTS.md §1)"; exit 1; }

MODEL="Qwen/Qwen2.5-3B-Instruct"
DTYPE="nf4"
mkdir -p logs results

STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(ruler kvcompare locomo longbench)

TS="$(date +%Y%m%d_%H%M%S)"
SUITE_LOG="logs/final_suite_${TS}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$SUITE_LOG"; }

# ---------------------------------------------------------------- preflight
log "=== SemPointer final suite — preflight ==="
if ! "$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'" 2>/dev/null; then
    log "FATAL: CUDA is not available. Fix the GPU/driver before running the suite."
    exit 1
fi
log "GPU: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader)"
log "Model: $MODEL ($DTYPE) — cached check:"
"$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" \
    >>"$SUITE_LOG" 2>&1 || { log "FATAL: model download failed (see $SUITE_LOG)"; exit 1; }
log "Preflight OK."

# ---------------------------------------------------------------- stages
run_stage() {
    local name="$1"; shift
    log ""
    log "================================================================"
    log ">>> STAGE: $name  (args: $*)"
    log "================================================================"
    "$PY" run_benchmark.py "$@" 2>&1 | tee -a "$SUITE_LOG"
    local rc=${PIPESTATUS[0]}
    if [ "$rc" -eq 0 ]; then
        log "[DONE] $name"
    else
        log "[FAILED] $name exited with code $rc — continuing with next stage."
        log "         Resume later with: ./run_final_suite.sh $name"
        FAILED_STAGES+=("$name")
    fi
    return 0   # always continue the suite
}

FAILED_STAGES=()

case "${STAGES[0]}" in *) ;; esac
for stage in "${STAGES[@]}"; do
    case "$stage" in
        ruler)
            run_stage ruler --benchmark ruler \
                --ruler-tasks niah_single_1,niah_single_2,niah_single_3,niah_multikey_1,vt,cwe \
                --context-lengths 4096,8192,16384 \
                --limit 10 --model "$MODEL" --dtype "$DTYPE" \
                --systems sempointer,full,rag,bm25 --max-new-tokens 32 \
                --run-name final_ruler --resume
            ;;
        kvcompare)
            run_stage kvcompare --benchmark kv_compare \
                --ruler-tasks niah_single_1,niah_multikey_1 \
                --context-lengths 4096 \
                --limit 5 --model "$MODEL" --dtype "$DTYPE" \
                --systems sempointer,full,kvpress:snapkv:0.3,kvpress:tova:0.3,kvpress:knorm:0.3 \
                --max-new-tokens 32 \
                --run-name final_kvcompare --resume
            ;;
        locomo)
            run_stage locomo --benchmark locomo \
                --locomo-conversations 0 \
                --model "$MODEL" --dtype "$DTYPE" \
                --systems sempointer,full,rag,bm25 --max-new-tokens 64 \
                --run-name final_locomo --resume
            ;;
        longbench)
            run_stage longbench --benchmark longbench \
                --longbench-tasks hotpotqa,2wikimqa,musique,multifieldqa_en,narrativeqa,qasper \
                --model "$MODEL" --dtype "$DTYPE" \
                --systems sempointer,full,rag,bm25 --max-new-tokens 64 \
                --run-name final_longbench --resume
            ;;
        *)
            log "Unknown stage: $stage (valid: ruler kvcompare locomo longbench)"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------- summary
log ""
log "================================================================"
log "SUITE COMPLETE"
if [ ${#FAILED_STAGES[@]} -eq 0 ]; then
    log "All requested stages finished with exit code 0."
else
    log "Stages with failures: ${FAILED_STAGES[*]}"
    log "Re-run: ./run_final_suite.sh ${FAILED_STAGES[*]}   (resumes where they stopped)"
fi
log "Results:"
for d in results/final_ruler results/final_kvcompare results/final_locomo results/final_longbench; do
    [ -d "$d" ] && log "  $d/metrics.json"
done
log "================================================================"

# quick score digest from whatever finished
"$PY" - <<'PYEOF' 2>/dev/null | tee -a "$SUITE_LOG"
import json, os
for run in ["final_ruler", "final_kvcompare", "final_locomo", "final_longbench"]:
    p = os.path.join("results", run, "metrics.json")
    if not os.path.exists(p):
        continue
    m = json.load(open(p))
    print(f"\n=== {run} ({m.get('n_examples')} examples) ===")
    for sysname, s in m.get("systems", {}).items():
        if s.get("n_scored"):
            print(f"  {sysname:24s} score={s['score_mean']:.3f} "
                  f"ci={s['score_ci95'][0]:.2f}-{s['score_ci95'][1]:.2f} "
                  f"tok={s['active_tokens_mean']:.0f} "
                  f"lat={s['latency_ms_mean']:.0f}ms "
                  f"vram={s['peak_vram_mb_mean']:.0f}MB "
                  f"errs={s.get('n_errors', 0)}")
        else:
            print(f"  {sysname:24s} NO SCORED EXAMPLES  errors={s.get('n_errors', '?')}")
PYEOF
