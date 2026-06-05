#!/usr/bin/env bash
# V-JEPA2 ablation, TAKEOVER (task3) only, on the anticipation-safe (window-buffered)
# samples. Writes into results_vjepa2 (run_name carries _antsafe → no collision with the
# detection-protocol runs). Pool of P (auto 1/2 from VRAM+RAM).
set -u
cd "$(dirname "$0")"
RD=results_vjepa2; mkdir -p "$RD"
export RD
COMMON_ARGS="--model transformer --use-vjepa --split cross_driver --horizon 3 --batch-size 2048 --num-workers 2 --t3-suffix _antsafe --results-dir $RD"
export COMMON_ARGS

run_group(){
  local mod="$1"; shift
  for s in "$@"; do
    if [ -f "$RD/task3_${mod}_transformer_cross_driver_h3_antsafe_s${s}/results.json" ]; then
      echo "=== skip (done): task3 $mod s$s"; continue; fi
    echo ">>> [task3-antsafe] $mod seed=$s"
    python3 train_nn.py --task task3 --modality "$mod" --seed "$s" $COMMON_ARGS \
      2>&1 | grep -E "TEST RESULTS|  auprc:|  event_auprc:" || true
  done
}
export -f run_group

LOO="VJEPA-Full VJEPA-no-FrontVideo VJEPA-no-CabinVideo VJEPA-no-Ego VJEPA-no-DrvInput VJEPA-no-Lead VJEPA-no-DMS VJEPA-no-RoadGeom"
SINGLE="VJEPA-only-FrontVideo VJEPA-only-CabinVideo VJEPA-only-Ego VJEPA-only-DrvInput VJEPA-only-Lead VJEPA-only-DMS VJEPA-only-RoadGeom"

# Pool size: arg1 overrides; else auto from RAM (~15GB/proc observed). Cap at 2 for safety.
avail_ram=$(free -g | awk '/Mem:/{print $7}')
P=${1:-$(( avail_ram / 18 ))}
[ "$P" -lt 1 ] && P=1; [ "$P" -gt 2 ] && P=2
echo "avail_ram=${avail_ram}GB -> pool P=$P"

{ for mod in $LOO;    do echo "$mod 42 123 7"; done
  for mod in $SINGLE; do echo "$mod 42"; done
} | xargs -P "$P" -L1 bash -c 'run_group "$@"' _
echo "VJEPA2 T3-ANTSAFE ABLATION DONE"
