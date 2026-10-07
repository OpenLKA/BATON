#!/usr/bin/env bash
# Multi-label Task-1 (6-class multi-hot, BCEWithLogits) — GRU baselines.
# Modalities: T1-RuleFree-Struct and VJEPA-only-FrontVideo; seeds 42/123/7;
# cross_driver split; 15 epochs (same as run_task1_ablation.sh).
# Labels: benchmark_v2/task1_action_samples_multilabel.csv
# WAITS for the RTX 5090 sequential job (run_review_backfill.sh) to finish
# before touching the GPU.
set -u
cd "$(dirname "$0")"
source /home/henry/miniconda3/etc/profile.d/conda.sh
conda activate hci

echo "[$(date)] queue started; waiting for run_review_backfill to release the GPU..."
while pgrep -f run_review_backfill >/dev/null; do sleep 60; done
echo "[$(date)] GPU free — starting multi-label Task-1 runs"

RD=results_task1_ml
mkdir -p "$RD"
SEEDS=(42 123 7)

for seed in "${SEEDS[@]}"; do
  echo ">>> [task1-ml] T1-RuleFree-Struct seed=$seed [$(date)]"
  python3 train_task1_multilabel.py --modality T1-RuleFree-Struct \
    --seed "$seed" --split cross_driver --epochs 15 --results-dir "$RD" || true
done

for seed in "${SEEDS[@]}"; do
  echo ">>> [task1-ml] VJEPA-only-FrontVideo seed=$seed [$(date)]"
  python3 train_task1_multilabel.py --modality VJEPA-only-FrontVideo --use-vjepa \
    --seed "$seed" --split cross_driver --epochs 15 --results-dir "$RD" || true
done

echo "[$(date)] ALL TASK1-ML RUNS DONE"
