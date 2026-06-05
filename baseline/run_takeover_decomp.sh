#!/usr/bin/env bash
# Complete takeover (T3) leakage decomposition, 3 seeds, XGBoost + GRU.
#   Full-Struct      : leaky (ADAS flags present)            — standard samples
#   WH-Safe          : canonical (no flags) = DETECTION       — standard samples
#   WH-Safe-noOverride: canonical − override fields (field anticipation) — standard
#   WH-Safe + antsafe: canonical on window-buffered samples = ANTICIPATION
set -u
cd "$(dirname "$0")"
RD=results_antsafe; mkdir -p "$RD"
run(){  # task model modality suffix seed
  local task=$1 model=$2 mod=$3 suf=$4 seed=$5
  local rn="${task}_${mod}_${model}_cross_driver_h3${suf}_s${seed}"
  [ -f "$RD/$rn/results.json" ] && { echo "skip $rn"; return; }
  echo ">>> $task $model $mod ${suf:-std} s$seed"
  if [ "$model" = xgb ]; then
    nice -n 10 python3 train_classical.py --task "$task" --model xgb --modality "$mod" \
      --split cross_driver --horizon 3 --seed "$seed" --t3-suffix "$suf" --results-dir "$RD" 2>&1 \
      | grep -E "  auprc:|  event_auprc:|n_pos:|n_neg:" || true
  else
    nice -n 12 python3 train_nn.py --task "$task" --model gru --modality "$mod" \
      --split cross_driver --horizon 3 --seed "$seed" --t3-suffix "$suf" --results-dir "$RD" 2>&1 \
      | grep -E "  auprc:|  event_auprc:" || true
  fi
}
for seed in 42 123 7; do
  for model in xgb gru; do
    run task3 $model WH-Full            ""        $seed   # leaky
    run task3 $model WH-Safe            ""        $seed   # detection
    run task3 $model WH-Safe-noOverride ""        $seed   # field anticipation
    run task3 $model WH-Safe            _antsafe  $seed   # window anticipation
  done
done
echo "TAKEOVER DECOMP DONE"
