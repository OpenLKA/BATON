#!/usr/bin/env bash
# Soak the idle GPU during cabin extraction: run only the V-JEPA2 ablation configs that
# DO NOT need cabin features (front-video + struct-only). Writes into results_vjepa2/ so
# the post-cabin full ablation skips them. CPU-gentle (sequential, 2 dataloader workers,
# niced) so it doesn't slow the CPU-decode-bound cabin extraction.
set -u
cd "$(dirname "$0")"
RD=results_vjepa2; mkdir -p "$RD"
COMMON="--model transformer --use-vjepa --split cross_driver --horizon 3 --batch-size 1024 --num-workers 2 --results-dir $RD"

run(){  # task modality seeds...
  local task="$1" mod="$2"; shift 2
  for s in "$@"; do
    if [ -f "$RD/${task}_${mod}_transformer_cross_driver_h3_s${s}/results.json" ]; then
      echo "=== skip (done): $task $mod s$s"; continue; fi
    echo ">>> [$task] $mod seed=$s"
    nice -n 15 python3 train_nn.py --task "$task" --modality "$mod" --seed "$s" $COMMON 2>&1 \
      | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
  done
}

for task in task2 task3; do
  run "$task" VJEPA-no-CabinVideo 42 123 7      # front+struct, no cabin (3 seeds)
  for mod in VJEPA-only-FrontVideo VJEPA-only-Ego VJEPA-only-DrvInput \
             VJEPA-only-Lead VJEPA-only-DMS VJEPA-only-RoadGeom; do
    run "$task" "$mod" 42
  done
done
echo "NOCABIN ABLATION SUBSET DONE"
