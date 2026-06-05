#!/usr/bin/env bash
# T3 — V-JEPA2 cross-modal Transformer + comprehensive modality ablation.
# Requires V-JEPA2 front+cabin features (data/vjepa2_*_video_features/).
# Pool of P parallel jobs (P=2, auto-drops to 1 if VRAM/RAM tight). Each job = one
# modality GROUP and runs all its seeds sequentially → same-modality runs never
# overlap → no norm-stats write race.
set -u
cd "$(dirname "$0")"
RD=results_vjepa2; mkdir -p "$RD"
export RD
COMMON_ARGS="--model transformer --use-vjepa --split cross_driver --horizon 3 --batch-size 1024 --results-dir $RD"
export COMMON_ARGS

# ── per-group runner: $1=task $2=modality $3.. = seeds ──
run_group(){
  local task="$1" mod="$2"; shift 2
  for s in "$@"; do
    if [ -f "$RD/${task}_${mod}_transformer_cross_driver_h3_s${s}/results.json" ]; then
      echo "=== skip (done): $task $mod s$s"; continue
    fi
    echo ">>> [$task] $mod seed=$s"
    python3 train_nn.py --task "$task" --modality "$mod" --seed "$s" $COMMON_ARGS \
      2>&1 | grep -E "TEST RESULTS|  auprc:|  event_auprc:|event_detection" || true
  done
}
export -f run_group

LOO="VJEPA-Full VJEPA-no-FrontVideo VJEPA-no-CabinVideo VJEPA-no-Ego VJEPA-no-DrvInput VJEPA-no-Lead VJEPA-no-DMS VJEPA-no-RoadGeom"
SINGLE="VJEPA-only-FrontVideo VJEPA-only-CabinVideo VJEPA-only-Ego VJEPA-only-DrvInput VJEPA-only-Lead VJEPA-only-DMS VJEPA-only-RoadGeom"

# ── decide pool size from free VRAM + available RAM ──
free_vram=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
avail_ram=$(free -g | awk '/Mem:/{print $7}')
P=2
if [ "${free_vram:-0}" -lt 4000 ] || [ "${avail_ram:-0}" -lt 34 ]; then P=1; fi
echo "free_vram=${free_vram}MiB avail_ram=${avail_ram}GB -> pool P=$P"

# emit one line per modality group ("task mod seed...") and run with a pool of P
{
  for task in task2 task3; do
    for mod in $LOO;    do echo "$task $mod 42 123 7"; done
    for mod in $SINGLE; do echo "$task $mod 42"; done
  done
} | xargs -P "$P" -L1 bash -c 'run_group "$@"' _

echo "ALL VJEPA2 ABLATION RUNS DONE"
