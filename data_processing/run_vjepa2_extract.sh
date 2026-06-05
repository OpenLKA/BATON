#!/usr/bin/env bash
# Parallel sharded V-JEPA2 extraction: N workers/camera (each own model+batch).
set -u
cd "$(dirname "$0")"
N=4; BATCH=64; THREADS=5
for cam in front cabin; do
  echo "=== $cam V-JEPA2 ($N workers, batch=$BATCH) ==="; date
  pids=()
  for s in $(seq 0 $((N-1))); do
    python3 extract_vjepa2_features.py --camera "$cam" \
      --shard "$s" --num-shards "$N" --batch "$BATCH" --threads "$THREADS" \
      > "/tmp/vjepa2_${cam}_shard${s}.log" 2>&1 &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p"; done
  echo "=== $cam DONE ==="; date
done
echo "=== ALL V-JEPA2 EXTRACTION DONE ==="; date
