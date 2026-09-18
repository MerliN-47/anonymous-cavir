#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG / CAVIR - Table 2 Macro Shock & Conformal Stress Tests
# Double-Blind Compliant Modular Architecture
# Evaluates multi-backbone NonExchangeableConformalGuard under variance shocks
# Verifies finite-sample coverage guarantees and interval width dynamics
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/table2_shocks"
mkdir -p "$OUT_DIR"

BACKBONES=("chronos-bolt" "chronos-2" "moirai-2.0" "timesfm-2.5")
DATASETS=("ETTh1" "weather" "electricity")
MODES=("nominal" "shock_uncalibrated" "shock_conformal_guard")

echo "=== Starting Table 2 Macro Shock & Conformal Stress Tests ==="
echo "Python Executable: $PYTHON"
echo "Results Directory: $OUT_DIR"
echo "=============================================================="

for BACKBONE in "${BACKBONES[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        for MODE in "${MODES[@]}"; do
            SAVE_NAME="table2_${DATASET}_${BACKBONE}_${MODE}.json"
            echo ">>> Running Evaluation: Backbone=$BACKBONE | Dataset=$DATASET | Mode=$MODE"

            if [ "$MODE" == "nominal" ]; then
                $PYTHON zeroshot.py \
                    --data "$DATASET" \
                    --backbone "$BACKBONE" \
                    --injection_point "latent" \
                    --conformal "none" \
                    --cf_shift "none" \
                    --eval_metrics distributional \
                    --save_file_name "$SAVE_NAME" \
                    "$@"
            elif [ "$MODE" == "shock_uncalibrated" ]; then
                $PYTHON zeroshot.py \
                    --data "$DATASET" \
                    --backbone "$BACKBONE" \
                    --injection_point "latent" \
                    --conformal "none" \
                    --cf_shift "active" \
                    --eval_metrics distributional \
                    --save_file_name "$SAVE_NAME" \
                    "$@"
            elif [ "$MODE" == "shock_conformal_guard" ]; then
                $PYTHON zeroshot.py \
                    --data "$DATASET" \
                    --backbone "$BACKBONE" \
                    --injection_point "latent" \
                    --conformal "active" \
                    --cf_shift "active" \
                    --eval_metrics distributional \
                    --save_file_name "$SAVE_NAME" \
                    "$@"
            fi
        done
    done
done

echo "=== Table 2 Conformal Shock Tests Completed Successfully ==="
