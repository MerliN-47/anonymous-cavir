#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG / CAVIR - Table 1 Main Zero-Shot Forecasting Sweeps
# Double-Blind Compliant Modular Architecture
# Sweeps across Foundation Backbones (Chronos-Bolt, Chronos-2, Moirai-2.0, TimesFM-2.5)
# and Fusion Modes (Vanilla, Token In-Context, Latent ARM, Output Vincentization)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/table1_main"
mkdir -p "$OUT_DIR"

DATASETS=("ETTh1" "ETTh2" "ETTm1" "ETTm2" "weather" "electricity")
BACKBONES=("chronos-bolt" "chronos-2" "moirai-2.0" "timesfm-2.5")
FUSIONS=("none" "latent" "token" "output")

echo "=== Starting Table 1 Main Zero-Shot Forecasting Sweeps ==="
echo "Python Executable: $PYTHON"
echo "Results Directory: $OUT_DIR"
echo "=========================================================="

for DATASET in "${DATASETS[@]}"; do
    for BACKBONE in "${BACKBONES[@]}"; do
        for FUSION in "${FUSIONS[@]}"; do
            SAVE_NAME="table1_${DATASET}_${BACKBONE}_${FUSION}.json"
            echo ">>> Running Benchmark: Dataset=$DATASET | Backbone=$BACKBONE | Fusion=$FUSION"
            
            $PYTHON zeroshot.py \
                --data "$DATASET" \
                --backbone "$BACKBONE" \
                --injection_point "$FUSION" \
                --seq_len 512 \
                --pred_len 64 \
                --top_k 10 \
                --eval_metrics distributional \
                --save_file_name "$SAVE_NAME" \
                "$@" || echo "Warning: Run failed for $DATASET-$BACKBONE-$FUSION"
        done
    done
done

echo "=== Table 1 Main Zero-Shot Sweeps Completed Successfully ==="
