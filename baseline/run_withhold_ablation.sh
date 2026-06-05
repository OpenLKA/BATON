#!/usr/bin/env bash
# Incremental-withhold ablation: which fields leak, and the difference between choices.
# Arg1 = model (xgb|gru). XGBoost is fast; run it first. GRU later (heavier, GPU).
set -u
cd "$(dirname "$0")"
MODEL=${1:-xgb}
RD=results_withhold; mkdir -p "$RD"
CONFIGS=(WH-Full WH-noFlags WH-noFlags-noCS WH-noActuator WH-Safe WH-Safe-noPlanner)

for task in task2 task3; do
  for mod in "${CONFIGS[@]}"; do
    rn="${task}_${mod}_${MODEL}_cross_driver_h3_s42"
    [ -f "$RD/$rn/results.json" ] && { echo "skip $rn"; continue; }
    echo ">>> [$task] $MODEL $mod"
    if [ "$MODEL" = "xgb" ]; then
      nice -n 10 python3 train_classical.py --task "$task" --model xgb --modality "$mod" \
        --split cross_driver --horizon 3 --seed 42 --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
    else
      nice -n 15 python3 train_nn.py --task "$task" --model gru --modality "$mod" \
        --split cross_driver --horizon 3 --seed 42 --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
    fi
  done
done
echo "WITHHOLD $MODEL DONE"
