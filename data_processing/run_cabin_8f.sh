#!/usr/bin/env bash
# Wait for the running front extraction to finish, then extract CABIN at 8-frame clips
# (4 parallel workers). Emits the "ALL V-JEPA2 EXTRACTION DONE" marker so the check-in
# cron auto-launches the ablation.
set -u
cd "$(dirname "$0")"
echo "=== waiting for front workers to finish ==="; date
while pgrep -f 'extract_vjepa2_features.py --camera front' >/dev/null; do sleep 30; done
echo "=== front DONE — launching CABIN (8-frame, 4 workers) ==="; date
N=4
pids=()
for s in $(seq 0 $((N-1))); do
  python3 extract_vjepa2_features.py --camera cabin --shard "$s" --num-shards "$N" \
    --batch 64 --threads 5 --clip-frames 8 > "/tmp/vjepa2_cabin_shard${s}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== cabin DONE ==="; date
echo "=== ALL V-JEPA2 EXTRACTION DONE ==="; date
