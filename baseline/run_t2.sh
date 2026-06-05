#!/usr/bin/env bash
set -u; cd "$(dirname "$0")"; RD=results_t2; mkdir -p "$RD"
for task in task2 task3; do
  for mod in Full-Struct Safe-Full-Struct; do
    for seed in 42 123 7; do
      echo ">>> [$task] transformer $mod seed=$seed"
      python3 train_nn.py --task "$task" --model transformer --modality "$mod" \
        --split cross_driver --horizon 3 --seed "$seed" --results-dir "$RD" 2>&1 \
        | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
    done
  done
done
echo "ALL T2 RUNS DONE"
