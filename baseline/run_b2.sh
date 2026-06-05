#!/usr/bin/env bash
# B2 leakage-safe + hierarchical CAN ablation (cross-driver, h=3, GRU).
# Headline configs get 3 seeds; ablation ladder gets 1 seed (42).
set -u
cd "$(dirname "$0")"
RD=results_b2
mkdir -p "$RD"

# Ablation ladder + leaky references (seed 42)
LADDER=(ADASctrl-only Full-Struct Safe-Full-Struct \
        Safe-Ego Safe-Ego+Drv_in Safe-Ego+Drv_in+Lead \
        Safe-Ego+Drv_in+Lead+Road Safe-Ego+Drv_in+Lead+Road+DMS)

for task in task2 task3; do
  for mod in "${LADDER[@]}"; do
    echo ">>> [$task] $mod seed=42"
    python3 train_nn.py --task "$task" --modality "$mod" --model gru \
      --split cross_driver --horizon 3 --seed 42 --results-dir "$RD" 2>&1 \
      | grep -E "TEST RESULTS|auc_roc:|auprc:|event_auprc:|event_detection_recall:|event_median_lead|n_events" || true
  done
done

# 3-seed headline: leaky Full-Struct vs leak-safe Safe-Full-Struct
for task in task2 task3; do
  for mod in Full-Struct Safe-Full-Struct; do
    for seed in 123 7; do
      echo ">>> [$task] $mod seed=$seed"
      python3 train_nn.py --task "$task" --modality "$mod" --model gru \
        --split cross_driver --horizon 3 --seed "$seed" --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|auprc:|event_auprc:" || true
    done
  done
done

echo "ALL B2 RUNS DONE"
