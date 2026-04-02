#!/usr/bin/env python3
"""
run_experiments.py — Master orchestration for all experiments.

Runs all training in-process with shared RouteCache to avoid redundant I/O.

Priority order:
  Phase 1: XGB/LR single-frame ablation        (CPU, ~5min)
  Phase 2: GRU single-frame ablation            (GPU, ~30min)
  Phase 3: PCA v2 — residual fusion + video LN  (GPU, ~4h)
  Phase 4: CLIP feature extraction + training    (GPU, ~1h)

Usage:
  python3 run_experiments.py              # run all phases
  python3 run_experiments.py --phase 3    # run specific phase
"""
import gc
import json
import logging
import subprocess
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import BENCHMARK_DIR, MODALITY_CONFIGS, CACHE_DIR
from dataset import RouteCache, PassingCtrlDataset, compute_norm_stats

BASELINE_DIR = Path(__file__).parent
DATA_DIR = BASELINE_DIR.parent / "data"
PYTHON = sys.executable

SEEDS = [42, 123, 7]
TASKS = ["task1", "task2", "task3"]

RESULTS_ABLATION = BASELINE_DIR / "results_ablation"
RESULTS_PCA_V2 = BASELINE_DIR / "results_pca128_v2"
RESULTS_CLIP = BASELINE_DIR / "results_clip"

VIDEO_MODALITIES = ["FV", "CV", "FV+CV", "Full-Multimodal", "Full-All"]

logger = logging.getLogger("experiments")


def banner(msg, char="═"):
    w = 70
    print(f"\n{char * w}")
    print(f"  {msg}")
    print(f"{char * w}\n")


def is_done(results_dir, run_name):
    return (results_dir / run_name / "results.json").exists()


def build_shared_cache(use_pca=False, use_clip=False, load_video=False, load_gps=True):
    """Build a RouteCache pre-loaded with all routes, shared across runs."""
    cache = RouteCache(use_pca=use_pca, use_clip=use_clip)
    split_file = BENCHMARK_DIR / "split_cross_driver.json"

    # Collect all route_ids from all tasks
    all_routes = set()
    for task in TASKS:
        for horizon in [3]:
            ds = PassingCtrlDataset(
                task, "train", split_file,
                MODALITY_CONFIGS["Full-Struct+GPS"],
                horizon=horizon, route_cache=cache,
            )
            all_routes.update(ds.route_ids)
            ds = PassingCtrlDataset(
                task, "val", split_file,
                MODALITY_CONFIGS["Full-Struct+GPS"],
                horizon=horizon, route_cache=cache,
            )
            all_routes.update(ds.route_ids)
            ds = PassingCtrlDataset(
                task, "test", split_file,
                MODALITY_CONFIGS["Full-Struct+GPS"],
                horizon=horizon, route_cache=cache,
            )
            all_routes.update(ds.route_ids)

    print(f"  Pre-loading {len(all_routes)} routes (video={load_video}, gps={load_gps})...")
    t0 = time.time()
    cache.preload(
        list(all_routes),
        load_gps=load_gps,
        load_front_video=load_video,
        load_cabin_video=load_video,
    )
    print(f"  Cache loaded in {time.time()-t0:.0f}s")
    return cache


# ═══════════════════════════════════════════════════════════
# PHASE 1: XGB/LR single-frame ablation (CPU, ~5min)
# ═══════════════════════════════════════════════════════════
def phase1_classical_ablation():
    banner("PHASE 1: Classical Single-Frame Ablation (XGB/LR)", "▓")
    RESULTS_ABLATION.mkdir(parents=True, exist_ok=True)

    from train_classical import extract_statistical_features, train_xgb, train_lr
    from metrics import evaluate_task1, evaluate_binary, find_optimal_f1_threshold
    from config import LABEL2IDX

    FIXED_MODALITY = "Full-Struct+GPS"
    mod_cfg = MODALITY_CONFIGS[FIXED_MODALITY]
    split_file = BENCHMARK_DIR / "split_cross_driver.json"

    # Build shared cache (struct + GPS only, no video)
    cache = build_shared_cache(load_video=False, load_gps=True)

    total = len(TASKS) * len(SEEDS) * 2
    done = 0
    t0 = time.time()

    for task in TASKS:
        # Load norm stats
        norm_path = CACHE_DIR / f"norm_{FIXED_MODALITY}_{task}_cross_driver_h3.npz"
        if norm_path.exists():
            data = np.load(norm_path)
            norm_stats = {"mean": data["mean"], "std": data["std"]}
        else:
            norm_stats = compute_norm_stats(task, split_file, mod_cfg, 3)
            if norm_stats is not None:
                np.savez(norm_path, mean=norm_stats["mean"], std=norm_stats["std"])

        # Build datasets once per task (shared across seeds — seeds only affect model init)
        train_ds = PassingCtrlDataset(task, "train", split_file, mod_cfg,
                                      horizon=3, norm_stats=norm_stats, route_cache=cache)
        val_ds = PassingCtrlDataset(task, "val", split_file, mod_cfg,
                                    horizon=3, norm_stats=norm_stats, route_cache=cache)
        test_ds = PassingCtrlDataset(task, "test", split_file, mod_cfg,
                                     horizon=3, norm_stats=norm_stats, route_cache=cache)

        # Extract features once (last-only)
        print(f"  Extracting last-frame features for {task}...")
        X_train, y_train = extract_statistical_features(train_ds, last_only=True)
        X_val, y_val = extract_statistical_features(val_ds, last_only=True)
        X_test, y_test = extract_statistical_features(test_ds, last_only=True)
        for arr in [X_train, X_val, X_test]:
            np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"    Features: {X_train.shape}")

        for model_name in ["xgb", "lr"]:
            for seed in SEEDS:
                run_name = f"{task}_{FIXED_MODALITY}_{model_name}_lastonly_cross_driver_h3_s{seed}"
                if is_done(RESULTS_ABLATION, run_name):
                    print(f"  [SKIP] {run_name}")
                    done += 1
                    continue

                np.random.seed(seed)
                run_dir = RESULTS_ABLATION / run_name
                run_dir.mkdir(parents=True, exist_ok=True)

                if model_name == "xgb":
                    model = train_xgb(X_train, y_train, X_val, y_val, task)
                else:
                    model = train_lr(X_train, y_train, X_val, y_val, task)

                if task == "task1":
                    test_proba = model.predict_proba(X_test)
                    test_metrics = evaluate_task1(y_test.astype(np.int64), test_proba)
                else:
                    val_proba = model.predict_proba(X_val)[:, 1]
                    test_proba = model.predict_proba(X_test)[:, 1]
                    opt_thresh, _ = find_optimal_f1_threshold(y_val, val_proba)
                    test_metrics = evaluate_binary(y_test, test_proba, threshold=opt_thresh)

                result = {
                    "run_name": run_name, "task": task,
                    "modality": FIXED_MODALITY, "model": model_name,
                    "split": "cross_driver", "horizon": 3, "seed": seed,
                    "test_metrics": {k: v for k, v in test_metrics.items()
                                     if k != "confusion_matrix"},
                }
                with open(run_dir / "results.json", "w") as f:
                    json.dump(result, f, indent=2)

                done += 1
                print(f"  [OK] {run_name}  ({done}/{total})")

    print(f"\n  Phase 1 done: {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════
# PHASE 2: GRU single-frame ablation (GPU, ~30min)
# ═══════════════════════════════════════════════════════════
def phase2_gru_singleframe():
    banner("PHASE 2: GRU Single-Frame Ablation", "▓")
    RESULTS_ABLATION.mkdir(parents=True, exist_ok=True)

    from train_nn import train_run

    # Shared cache: struct + GPS, no video
    cache = build_shared_cache(load_video=False, load_gps=True)

    total = len(TASKS) * len(SEEDS)
    done = 0
    t0 = time.time()

    for task in TASKS:
        for seed in SEEDS:
            run_name = f"{task}_Full-Struct+GPS_gru_sf_cross_driver_h3_s{seed}"
            if is_done(RESULTS_ABLATION, run_name):
                print(f"  [SKIP] {run_name}")
                done += 1
                continue

            print(f"  [RUN] {run_name}")
            train_run(
                task=task, modality="Full-Struct+GPS", model_type="gru",
                split="cross_driver", horizon=3, seed=seed,
                single_frame=True, results_dir=str(RESULTS_ABLATION),
                route_cache=cache,
            )
            done += 1
            print(f"  Progress: {done}/{total}")

    print(f"\n  Phase 2 done: {time.time()-t0:.0f}s")
    del cache
    gc.collect()


# ═══════════════════════════════════════════════════════════
# PHASE 3: PCA v2 — improved multimodal fusion (GPU, ~4h)
# ═══════════════════════════════════════════════════════════
def phase3_pca_v2():
    banner("PHASE 3: PCA v2 — Residual Fusion + Video LayerNorm", "▓")
    RESULTS_PCA_V2.mkdir(parents=True, exist_ok=True)

    from train_nn import train_run
    from concurrent.futures import ThreadPoolExecutor

    # Shared cache: struct + GPS + PCA video (all modalities covered)
    cache = build_shared_cache(use_pca=True, load_video=True, load_gps=True)

    total = len(VIDEO_MODALITIES) * len(TASKS) * len(SEEDS)
    done = sum(1 for m in VIDEO_MODALITIES for t in TASKS for s in SEEDS
               if is_done(RESULTS_PCA_V2, f"{t}_{m}_gru_cross_driver_h3_s{s}"))
    print(f"  {done}/{total} already done\n")

    t0 = time.time()
    newly_done = 0

    for seed in SEEDS:
        for modality in VIDEO_MODALITIES:
            group = f"{modality}_s{seed}"
            print(f"\n  -- Group: {group} --")

            # Task1 alone (big dataset, needs full GPU)
            rn1 = f"task1_{modality}_gru_cross_driver_h3_s{seed}"
            if is_done(RESULTS_PCA_V2, rn1):
                print(f"  [SKIP] {rn1}")
            else:
                print(f"  [RUN] {rn1}")
                train_run(
                    task="task1", modality=modality, model_type="gru",
                    split="cross_driver", horizon=3, seed=seed,
                    use_pca=True, video_dropout=0.5,
                    results_dir=str(RESULTS_PCA_V2),
                    route_cache=cache,
                )
                newly_done += 1

            # Task2 + Task3 sequentially (num_workers=0 to avoid fork issues)
            for task in ["task2", "task3"]:
                rn = f"{task}_{modality}_gru_cross_driver_h3_s{seed}"
                if is_done(RESULTS_PCA_V2, rn):
                    print(f"  [SKIP] {rn}")
                else:
                    print(f"  [RUN] {rn}")
                    train_run(
                        task=task, modality=modality, model_type="gru",
                        split="cross_driver", horizon=3, seed=seed,
                        use_pca=True, video_dropout=0.5,
                        results_dir=str(RESULTS_PCA_V2),
                        route_cache=cache, num_workers=0,
                    )
                newly_done += 1

            done_now = sum(1 for m in VIDEO_MODALITIES for t in TASKS for s in SEEDS
                          if is_done(RESULTS_PCA_V2, f"{t}_{m}_gru_cross_driver_h3_s{s}"))
            elapsed = time.time() - t0
            rate = newly_done / elapsed if newly_done > 0 else 0
            remaining = total - done_now
            eta_min = remaining / rate / 60 if rate > 0 else 0
            print(f"  Progress: {done_now}/{total}, elapsed: {elapsed/60:.0f}min, "
                  f"ETA: {eta_min:.0f}min")

    elapsed = time.time() - t0
    print(f"\n  Phase 3 done: {elapsed/3600:.1f}h")
    del cache
    gc.collect()


# ═══════════════════════════════════════════════════════════
# PHASE 4: CLIP feature extraction + training (GPU, ~1h)
# ═══════════════════════════════════════════════════════════
def phase4_clip():
    banner("PHASE 4: CLIP Baseline", "▓")

    # Step 4a: Install open-clip if needed
    try:
        import open_clip
        print("  open-clip already installed")
    except ImportError:
        print("  Installing open-clip-torch...")
        subprocess.run([PYTHON, "-m", "pip", "install", "open-clip-torch", "-q"])

    # Step 4b: Extract CLIP features
    clip_front_dir = DATA_DIR / "clip_front_video_features"
    clip_cabin_dir = DATA_DIR / "clip_cabin_video_features"

    for camera, out_dir in [("front", clip_front_dir), ("cabin", clip_cabin_dir)]:
        existing = len(list(out_dir.glob("*.npz"))) if out_dir.exists() else 0
        if existing >= 370:
            print(f"  [SKIP] CLIP {camera}: {existing} routes already extracted")
            continue
        print(f"\n  Extracting CLIP features ({camera})...")
        subprocess.run(
            [PYTHON, str(DATA_DIR / "extract_clip_features.py"), "--camera", camera],
            cwd=str(DATA_DIR),
        )

    # Step 4c: Train with CLIP features (in-process with shared cache)
    RESULTS_CLIP.mkdir(parents=True, exist_ok=True)

    from train_nn import train_run

    cache = build_shared_cache(use_clip=True, load_video=True, load_gps=True)

    total = len(VIDEO_MODALITIES) * len(TASKS) * len(SEEDS)
    done = sum(1 for m in VIDEO_MODALITIES for t in TASKS for s in SEEDS
               if is_done(RESULTS_CLIP, f"{t}_{m}_gru_cross_driver_h3_s{s}"))
    print(f"\n  CLIP training: {done}/{total} already done")

    from concurrent.futures import ThreadPoolExecutor

    t0 = time.time()
    for seed in SEEDS:
        for modality in VIDEO_MODALITIES:
            print(f"\n  -- CLIP {modality}_s{seed} --")

            # Task1 alone
            rn1 = f"task1_{modality}_gru_cross_driver_h3_s{seed}"
            if not is_done(RESULTS_CLIP, rn1):
                print(f"  [RUN] {rn1}")
                train_run(
                    task="task1", modality=modality, model_type="gru",
                    split="cross_driver", horizon=3, seed=seed,
                    use_clip=True, video_dropout=0.5,
                    results_dir=str(RESULTS_CLIP),
                    route_cache=cache,
                )
            else:
                print(f"  [SKIP] {rn1}")

            # Task2 + Task3 sequentially (avoid fork issues with ThreadPool)
            for task in ["task2", "task3"]:
                rn = f"{task}_{modality}_gru_cross_driver_h3_s{seed}"
                if is_done(RESULTS_CLIP, rn):
                    print(f"  [SKIP] {rn}")
                else:
                    print(f"  [RUN] {rn}")
                    train_run(
                        task=task, modality=modality, model_type="gru",
                        split="cross_driver", horizon=3, seed=seed,
                        use_clip=True, video_dropout=0.5,
                        results_dir=str(RESULTS_CLIP),
                        route_cache=cache, num_workers=0,
                    )

            done_now = sum(1 for m in VIDEO_MODALITIES for t in TASKS for s in SEEDS
                          if is_done(RESULTS_CLIP, f"{t}_{m}_gru_cross_driver_h3_s{s}"))
            print(f"  CLIP progress: {done_now}/{total}")

    elapsed = time.time() - t0
    print(f"\n  Phase 4 done: {elapsed/60:.0f}min")
    del cache
    gc.collect()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, default=None,
                        help="Run only this phase (1-4)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    banner("PassingCtrl Experiment Suite", "█")
    print("  Phase 1: Classical single-frame ablation  (CPU, ~5min)")
    print("  Phase 2: GRU single-frame ablation        (GPU, ~30min)")
    print("  Phase 3: PCA v2 residual fusion            (GPU, ~4h)")
    print("  Phase 4: CLIP baseline                     (GPU, ~1h)")
    print(f"\n  Results dirs:")
    print(f"    Ablation: {RESULTS_ABLATION}")
    print(f"    PCA v2:   {RESULTS_PCA_V2}")
    print(f"    CLIP:     {RESULTS_CLIP}")

    t_global = time.time()

    phases = {
        1: phase1_classical_ablation,
        2: phase2_gru_singleframe,
        3: phase3_pca_v2,
        4: phase4_clip,
    }

    if args.phase:
        phases[args.phase]()
    else:
        for num, func in phases.items():
            func()

    total_elapsed = time.time() - t_global
    banner(f"ALL DONE — Total: {total_elapsed/3600:.1f}h", "█")


if __name__ == "__main__":
    main()
