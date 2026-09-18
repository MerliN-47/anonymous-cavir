#!/bin/bash
# ==============================================================================
# [Anonymous] TS-RAG / CAVIR - Downstream Covariate RAG (fev-bench)
# Double-Blind Compliant Modular Architecture
# Evaluates CovariateQueryEmbedder and ChannelBlockEncoder across
# the 30 known-covariate planning tasks in fev-bench
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
OUT_DIR="./results/fevbench"
mkdir -p "$OUT_DIR"

BACKBONES=("chronos-bolt" "chronos-2" "moirai-2.0")
COVARIATE_MODES=("none" "future_known")
CHANNEL_MODES=("channel_independent" "channel_block")

echo "=== Starting fev-bench Downstream Covariate Evaluations ==="
echo "Python Executable: $PYTHON"
echo "Results Directory: $OUT_DIR"
echo "==========================================================="

for BACKBONE in "${BACKBONES[@]}"; do
    for COV_MODE in "${COVARIATE_MODES[@]}"; do
        for CH_MODE in "${CHANNEL_MODES[@]}"; do
            SAVE_NAME="fevbench_${BACKBONE}_cov_${COV_MODE}_ch_${CH_MODE}.json"
            echo ">>> Running fev-bench: Backbone=$BACKBONE | Covariate=$COV_MODE | Channel=$CH_MODE"

            $PYTHON zeroshot.py \
                --benchmark "fev_bench" \
                --backbone "$BACKBONE" \
                --injection_point "latent" \
                --covariate_mode "$COV_MODE" \
                --channel_mode "$CH_MODE" \
                --seq_len 512 \
                --pred_len 64 \
                --top_k 10 \
                --eval_metrics distributional \
                --save_file_name "$SAVE_NAME" \
                "$@"
        done
    done
done

echo "=== fev-bench Evaluations Completed Successfully ==="
