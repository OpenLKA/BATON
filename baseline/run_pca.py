#!/usr/bin/env python3
"""
run_pca.py — Run PCA-128 video feature experiments (v2: residual fusion).

Uses in-process training with shared RouteCache — features loaded once.
Results saved to results_pca128_v2/.
"""
import gc
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BASELINE_DIR = Path(__file__).parent
RESULTS_DIR = BASELINE_DIR / "results_pca128_v2"

SEEDS = [42, 123, 7]
TASKS = ["task1", "task2", "task3"]
VIDEO_MODALITIES = ["FV", "CV", "FV+CV", "Full-Multimodal", "Full-All"]


def is_done(task, modality, seed):
    run_name = f"{task}_{modality}_gru_cross_driver_h3_s{seed}"
    return (RESULTS_DIR / run_name / "results.json").exists()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    total = len(VIDEO_MODALITIES) * len(TASKS) * len(SEEDS)
    done = sum(1 for m in VIDEO_MODALITIES for t in TASKS for s in SEEDS
               if is_done(t, m, s))
    print(f"\nPCA-128 v2 (Residual Fusion): {done}/{total} done")
    print(f"Results dir: {RESULTS_DIR}\n")

    if done == total:
        print("All done!")
        return

    # Build shared cache once — covers all modalities
    from config import BENCHMARK_DIR, MODALITY_CONFIGS
    from dataset import RouteCache, PassingCtrlDataset

    cache = RouteCache(use_pca=True)

    # Collect all routes
    split_file = BENCHMARK_DIR / "split_cross_driver.json"
    all_routes = set()
    for task in TASKS:
        for split_name in ["train", "val", "test"]:
            ds = PassingCtrlDataset(
                task, split_name, split_file,
                MODALITY_CONFIGS["Full-All"],
                horizon=3, route_cache=cache,
            )
            all_routes.update(ds.route_ids)

    print(f"Pre-loading {len(all_routes)} routes (struct + GPS + PCA video)...")
    t0 = time.time()
    cache.preload(list(all_routes), load_gps=True,
                  load_front_video=True, load_cabin_video=True)
    print(f"Cache loaded in {time.time()-t0:.0f}s\n")

    # Training loop
    from train_nn import train_run

    t0 = time.time()

    for seed in SEEDS:
        for modality in VIDEO_MODALITIES:
            for task in TASKS:
                run_name = f"{task}_{modality}_gru_cross_driver_h3_s{seed}"
                if is_done(task, modality, seed):
                    print(f"  [SKIP] {run_name}")
                    continue

                print(f"\n  [RUN] {run_name}")
                train_run(
                    task=task, modality=modality, model_type="gru",
                    split="cross_driver", horizon=3, seed=seed,
                    use_pca=True, video_dropout=0.5,
                    results_dir=str(RESULTS_DIR),
                    route_cache=cache,
                )
                done += 1
                elapsed = time.time() - t0
                remaining = total - done
                rate = (done - (total - len(VIDEO_MODALITIES)*len(TASKS)*len(SEEDS)))
                print(f"  Progress: {done}/{total}, elapsed: {elapsed/60:.0f}min")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PCA v2 DONE: {elapsed/3600:.1f}h")
    print(f"{'='*60}")

    del cache
    gc.collect()


if __name__ == "__main__":
    main()
