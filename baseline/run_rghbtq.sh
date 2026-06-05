#!/usr/bin/env bash
# RG-HBT-Q-lite headline: hierarchical leak-safe multimodal fusion (front video +
# cabin video + 6-category CAN) via reliability-gated bottleneck + transition queries.
# Three settings x 3 seeds, sequential (each run already saturates the GPU):
#   task2          handover (leak-safe)               — standard samples
#   task3          takeover DETECTION                 — standard samples
#   task3 _antsafe takeover ANTICIPATION (window-buffer) — anticipation-safe samples
set -u
cd "$(dirname "$0")"
RD=results_rghbtq; mkdir -p "$RD"
COMMON="--model rghbtq --modality Hier-Safe-MM --use-vjepa --split cross_driver \
        --horizon 3 --batch-size 2048 --num-workers 2 --results-dir $RD"

run(){  # task suffix seed
  local task=$1 suf=$2 seed=$3
  local rn="${task}_Hier-Safe-MM_rghbtq_cross_driver_h3${suf}_s${seed}"
  if [ -f "$RD/$rn/results.json" ]; then echo "=== skip (done): $rn"; return; fi
  echo ">>> $task ${suf:-std} seed=$seed"
  python3 train_nn.py --task "$task" --seed "$seed" --t3-suffix "$suf" $COMMON \
    2>&1 | grep -E "params=|TEST RESULTS|  auprc:|  event_auprc:|Event-level" || true
}

for seed in 42 123 7; do
  run task2 ""        "$seed"
  run task3 ""        "$seed"
  run task3 _antsafe  "$seed"
done
echo "RG-HBT-Q RUNS DONE"
