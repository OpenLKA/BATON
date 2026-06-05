#!/usr/bin/env bash
# Finish cabin extraction with a larger pool to parallelize the remaining tail.
# Resumable: each worker skips routes whose npz already exists, so the ~50 undone
# cabin routes get split across N workers (no overlap — single pool, disjoint shards).
# Emits the DONE marker so the check-in cron auto-launches the ablation.
set -u
cd "$(dirname "$0")"
N=8; BATCH=64; THREADS=3
echo "=== CABIN finish: $N workers (8-frame) ==="; date
pids=()
for s in $(seq 0 $((N-1))); do
  python3 extract_vjepa2_features.py --camera cabin --shard "$s" --num-shards "$N" \
    --batch "$BATCH" --threads "$THREADS" --clip-frames 8 \
    > "/tmp/vjepa2_cabin_fin_shard${s}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
echo "=== cabin DONE ==="; date
echo "=== ALL V-JEPA2 EXTRACTION DONE ==="; date
