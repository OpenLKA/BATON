#!/usr/bin/env bash
# A — Task 1 rule-free ablation: quantify how much of 0.910 is rule reconstruction.
# Full-Struct (repro) vs rule-free struct vs single non-rule modalities vs video-only
# (rules use no video). GRU, cross-driver. Niced so it yields to cabin extraction.
set -u
cd "$(dirname "$0")"
RD=results_task1_abl
mkdir -p "$RD"
EP=${1:-15}
COMMON="--task task1 --model gru --split cross_driver --epochs $EP --results-dir $RD"

# struct configs (no video needed)
for mod in Full-Struct T1-RuleFree-Struct T1-Drv-only T1-IMU-only; do
  echo ">>> [task1] $mod"
  nice -n 19 python3 train_nn.py --modality "$mod" --seed 42 $COMMON 2>&1 \
    | grep -E "TEST RESULTS|  accuracy:|  macro_f1:|f1_LaneChange" || true
done

# video-only via V-JEPA2 (front features ready; rules use no video → independent test)
echo ">>> [task1] VJEPA-only-FrontVideo (--use-vjepa)"
nice -n 19 python3 train_nn.py --modality VJEPA-only-FrontVideo --use-vjepa --seed 42 $COMMON 2>&1 \
  | grep -E "TEST RESULTS|  accuracy:|  macro_f1:|f1_LaneChange" || true

echo "ALL TASK1 ABLATION RUNS DONE"
