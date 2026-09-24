#!/usr/bin/env bash
# ==============================================================================
# SemPointer Empirical Validation Suite — Master Runner
#
# Runs real empirical experiments validating, falsifying, or qualifying
# the theoretical claims in "SemPointer: Toward Random-Access Semantic Memory for LLMs"
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Defaults
TIER="1"
DRY_RUN=""
DEVICE="cpu"
SINGLE_EXP=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --tier) TIER="$2"; shift ;;
        --dry-run) DRY_RUN="--dry-run" ;;
        --device) DEVICE="$2"; shift ;;
        --exp|--experiment) SINGLE_EXP="$2"; shift ;;
        -h|--help)
            echo "Usage: bash run_all.sh [OPTIONS]"
            echo "Options:"
            echo "  --tier {1, 2, 3, all}   Experiment priority tier (default: 1)"
            echo "  --dry-run               Run minimal verification passes for quick testing"
            echo "  --device {cpu, cuda}    Compute device (default: cpu)"
            echo "  --exp {e01, e02, ...}   Run a single specific experiment"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

# Detect Python environment
if [[ -f "/home/viksit/miniconda3/envs/warden/bin/python" ]]; then
    PYTHON="/home/viksit/miniconda3/envs/warden/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON="$(command -v python3)"
else
    PYTHON="python"
fi

# Pre-flight device check
if [[ "$DEVICE" == "cuda" ]]; then
    if ! "$PYTHON" -c "import torch; torch.randn(1, device='cuda')" &>/dev/null; then
        echo "[WARNING] PyTorch CUDA allocation failed (NVML driver/kernel mismatch or no active GPU)."
        echo "[INFO] Automatically failing over to --device cpu so experiments run cleanly."
        echo "[INFO] (To restore native CUDA later, perform a 'sudo reboot' to sync the kernel driver)."
        DEVICE="cpu"
    fi
fi

echo "================================================================================"
echo "SemPointer Master Experiment Orchestrator"
echo "Python binary: $PYTHON"
echo "Device:        $DEVICE"
echo "Tier:          $TIER"
echo "Dry Run:       ${DRY_RUN:-Full Production Run}"
echo "================================================================================"

mkdir -p logs results figures tables

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG="logs/run_all_${TIMESTAMP}.log"

# Define experiments by tier
TIER_1_EXPS=(
    "e03_compression"
    "e01_retrieval"
    "e05_kstar"
    "e02_qa"
    "e04_flops"
    "e06_scaling"
    "e07_pointer_length"
    "e14_resolution_fidelity"
)

TIER_2_EXPS=(
    "e08_payload_size"
    "e09_selected_pointers"
    "e10_query_length"
    "e11_compositionality"
    "e12_collision"
    "e13_drift"
    "e15_kv_cache"
    "e17_cross_model"
)

TIER_3_EXPS=(
    "e16_paged"
)

EXPS_TO_RUN=()
if [[ -n "$SINGLE_EXP" ]]; then
    EXPS_TO_RUN=("$SINGLE_EXP")
elif [[ "$TIER" == "1" ]]; then
    EXPS_TO_RUN=("${TIER_1_EXPS[@]}")
elif [[ "$TIER" == "2" ]]; then
    EXPS_TO_RUN=("${TIER_1_EXPS[@]}" "${TIER_2_EXPS[@]}")
elif [[ "$TIER" == "3" || "$TIER" == "all" ]]; then
    EXPS_TO_RUN=("${TIER_1_EXPS[@]}" "${TIER_2_EXPS[@]}" "${TIER_3_EXPS[@]}")
else
    echo "Unknown tier: $TIER"; exit 1
fi

TOTAL=${#EXPS_TO_RUN[@]}
PASSED=0
FAILED=0

echo "Starting execution of $TOTAL experiments..." | tee -a "$MASTER_LOG"

for exp in "${EXPS_TO_RUN[@]}"; do
    echo "" | tee -a "$MASTER_LOG"
    echo "--------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"
    echo ">>> Running $exp" | tee -a "$MASTER_LOG"
    echo "--------------------------------------------------------------------------------" | tee -a "$MASTER_LOG"

    CONFIG="config/${exp}.yaml"
    EXTRA_FLAGS=()
    if [[ -f "$CONFIG" ]]; then
        EXTRA_FLAGS+=(--config "$CONFIG")
    fi
    if [[ -n "$DRY_RUN" ]]; then
        EXTRA_FLAGS+=($DRY_RUN)
    fi
    if [[ -n "$DEVICE" ]]; then
        EXTRA_FLAGS+=(--device "$DEVICE")
    fi

    set +e
    "$PYTHON" -m "runners.${exp}" "${EXTRA_FLAGS[@]}" 2>&1 | tee -a "$MASTER_LOG"
    EXIT_CODE=$?
    set -e

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[SUCCESS] $exp completed cleanly." | tee -a "$MASTER_LOG"
        PASSED=$((PASSED + 1))
    else
        echo "[ERROR] $exp exited with code $EXIT_CODE." | tee -a "$MASTER_LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo "" | tee -a "$MASTER_LOG"
echo "================================================================================" | tee -a "$MASTER_LOG"
echo "Experiment Suite Completed" | tee -a "$MASTER_LOG"
echo "Passed: $PASSED / $TOTAL" | tee -a "$MASTER_LOG"
echo "Failed: $FAILED / $TOTAL" | tee -a "$MASTER_LOG"
echo "Log file: $MASTER_LOG" | tee -a "$MASTER_LOG"
echo "================================================================================" | tee -a "$MASTER_LOG"

# Generate publication figures and tables from results
echo "Generating publication artifacts..." | tee -a "$MASTER_LOG"
set +e
"$PYTHON" analysis/generate_tables.py 2>&1 | tee -a "$MASTER_LOG"
"$PYTHON" analysis/generate_figures.py 2>&1 | tee -a "$MASTER_LOG"
set -e
echo "All done." | tee -a "$MASTER_LOG"
