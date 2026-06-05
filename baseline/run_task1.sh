#!/usr/bin/env bash
# T1 — original baselines (XGBoost, GRU) on the leak-safe benchmark_v2.
# Headline: leaky Full-Struct vs leak-safe Safe-Full-Struct, T2+T3, 3 seeds.
set -u
cd "$(dirname "$0")"
RD=results_task1
mkdir -p "$RD"
SEEDS=(42 123 7)
MODS=(Full-Struct Safe-Full-Struct)

for task in task2 task3; do
  for mod in "${MODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      echo ">>> XGB [$task] $mod seed=$seed"
      python3 train_classical.py --task "$task" --model xgb --modality "$mod" \
        --split cross_driver --horizon 3 --seed "$seed" --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
      echo ">>> GRU [$task] $mod seed=$seed"
      python3 train_nn.py --task "$task" --model gru --modality "$mod" \
        --split cross_driver --horizon 3 --seed "$seed" --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
    done
  done
done
echo "ALL T1 RUNS DONE"
