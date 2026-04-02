#!/usr/bin/env python3
"""
train_classical.py — Train Logistic Regression or XGBoost baseline on PassingCtrl.

Uses flattened/statistical features from structured signals.

Usage:
  python3 train_classical.py --task task1 --model lr --seed 42
  python3 train_classical.py --task task2 --model xgb --seed 42
"""
import argparse
import json
import logging
import sys
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BENCHMARK_DIR, MODALITY_CONFIGS, RESULTS_DIR, CACHE_DIR,
    LABEL2IDX, STRUCT_SEQ_LEN,
)
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats
from metrics import evaluate_task1, evaluate_binary, find_optimal_f1_threshold

logger = logging.getLogger("baseline")

# Always use Full-Struct+GPS for model comparison (all structured + GPS)
FIXED_MODALITY = "Full-Struct+GPS"


def extract_statistical_features(dataset, max_samples=None, last_only=False):
    """Extract flattened statistical features from structured time series.

    For each channel, compute: mean, std, min, max, last value.
    If last_only=True, use only the last timestep (single-frame baseline).
    Concatenates struct + GPS (if available).
    Returns: X [N, D*5] or [N, D], y [N]
    """
    n = len(dataset)
    if max_samples and n > max_samples:
        n = max_samples

    d = dataset.struct_dim + dataset.gps_dim
    n_feats = d if last_only else d * 5
    X = np.zeros((n, n_feats), dtype=np.float32)
    y = np.zeros(n, dtype=dataset.labels.dtype)

    for i in tqdm(range(n), desc="  features", leave=False, unit="sample"):
        item = dataset[i]
        parts = []
        if "struct" in item:
            parts.append(item["struct"].numpy())
        if "gps" in item:
            parts.append(item["gps"].numpy())
        seq = np.concatenate(parts, axis=1) if parts else item["struct"].numpy()

        if last_only:
            X[i, :d] = seq[-1]  # last timestep only
        else:
            # 5 statistics per channel
            X[i, 0*d:1*d] = seq.mean(axis=0)
            X[i, 1*d:2*d] = seq.std(axis=0)
            X[i, 2*d:3*d] = seq.min(axis=0)
            X[i, 3*d:4*d] = seq.max(axis=0)
            X[i, 4*d:5*d] = seq[-1]  # last timestep

        y[i] = item["label"] if isinstance(item["label"], (int, float, np.integer, np.floating)) else item["label"].item()

    return X, y


def train_lr(X_train, y_train, X_val, y_val, task):
    """Train Logistic Regression."""
    from sklearn.linear_model import LogisticRegression

    if task == "task1":
        model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced", multi_class="multinomial",
            n_jobs=-1,
        )
    else:
        model = LogisticRegression(
            C=1.0, max_iter=1000, solver="lbfgs",
            class_weight="balanced",
            n_jobs=-1,
        )

    model.fit(X_train, y_train)
    return model


def train_xgb(X_train, y_train, X_val, y_val, task):
    """Train XGBoost."""
    from xgboost import XGBClassifier

    if task == "task1":
        n_classes = len(LABEL2IDX)
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss",
            early_stopping_rounds=20,
            n_jobs=-1, tree_method="hist",
            num_class=n_classes,
        )
    else:
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        spw = n_neg / max(n_pos, 1)
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=min(spw, 10.0),
            eval_metric="logloss",
            early_stopping_rounds=20,
            n_jobs=-1, tree_method="hist",
        )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def main():
    parser = argparse.ArgumentParser(description="Train classical baseline")
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--model", required=True, choices=["lr", "xgb"])
    parser.add_argument("--split", default="cross_driver",
                        choices=["cross_driver", "cross_vehicle", "random"])
    parser.add_argument("--horizon", type=int, default=3, choices=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--last-only", action="store_true",
                        help="Use only last timestep (single-frame ablation)")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Override results directory")
    args = parser.parse_args()

    np.random.seed(args.seed)
    split_file = BENCHMARK_DIR / f"split_{args.split}.json"
    mod_cfg = MODALITY_CONFIGS[FIXED_MODALITY]

    suffix = "_lastonly" if args.last_only else ""
    run_name = f"{args.task}_{FIXED_MODALITY}_{args.model}{suffix}_{args.split}_h{args.horizon}_s{args.seed}"
    results_base = Path(args.results_dir) if args.results_dir else RESULTS_DIR
    run_dir = results_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(run_dir / "train.log"),
        ],
    )
    logger.info(f"Run: {run_name}")

    # Norm stats
    norm_path = CACHE_DIR / f"norm_{FIXED_MODALITY}_{args.task}_{args.split}_h{args.horizon}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if norm_path.exists():
        data = np.load(norm_path)
        norm_stats = {"mean": data["mean"], "std": data["std"]}
    else:
        norm_stats = compute_norm_stats(args.task, split_file, mod_cfg, args.horizon)
        if norm_stats is not None:
            np.savez(norm_path, mean=norm_stats["mean"], std=norm_stats["std"])

    # Datasets
    cache = RouteCache()
    train_ds = PassingCtrlDataset(
        args.task, "train", split_file, mod_cfg,
        horizon=args.horizon, norm_stats=norm_stats, route_cache=cache,
    )
    val_ds = PassingCtrlDataset(
        args.task, "val", split_file, mod_cfg,
        horizon=args.horizon, norm_stats=norm_stats, route_cache=cache,
    )
    test_ds = PassingCtrlDataset(
        args.task, "test", split_file, mod_cfg,
        horizon=args.horizon, norm_stats=norm_stats, route_cache=cache,
    )

    # Eagerly pre-load all route data
    all_routes = list(set(
        list(train_ds.route_ids) + list(val_ds.route_ids) + list(test_ds.route_ids)
    ))
    cache.preload(all_routes, load_gps=mod_cfg["gps"])

    # Extract features
    logger.info("Extracting train features...")
    t0 = time.time()
    X_train, y_train = extract_statistical_features(train_ds, last_only=args.last_only)
    logger.info(f"  Train: {X_train.shape}, {time.time()-t0:.1f}s")

    logger.info("Extracting val features...")
    X_val, y_val = extract_statistical_features(val_ds, last_only=args.last_only)
    logger.info(f"  Val: {X_val.shape}")

    logger.info("Extracting test features...")
    X_test, y_test = extract_statistical_features(test_ds, last_only=args.last_only)
    logger.info(f"  Test: {X_test.shape}")

    # Replace NaN/Inf
    for arr in [X_train, X_val, X_test]:
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Train
    logger.info(f"Training {args.model}...")
    t0 = time.time()
    if args.model == "lr":
        model = train_lr(X_train, y_train, X_val, y_val, args.task)
    else:
        model = train_xgb(X_train, y_train, X_val, y_val, args.task)
    logger.info(f"Training done in {time.time()-t0:.1f}s")

    # Evaluate
    if args.task == "task1":
        # Get class probabilities as logits proxy
        val_proba = model.predict_proba(X_val)
        test_proba = model.predict_proba(X_test)
        val_metrics = evaluate_task1(y_val.astype(np.int64), val_proba)
        test_metrics = evaluate_task1(y_test.astype(np.int64), test_proba)
    else:
        val_proba = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        # Find optimal threshold on val
        opt_threshold, _ = find_optimal_f1_threshold(y_val, val_proba)
        logger.info(f"Optimal threshold from val: {opt_threshold:.4f}")

        val_metrics = evaluate_binary(y_val, val_proba, threshold=opt_threshold)
        test_metrics = evaluate_binary(y_test, test_proba, threshold=opt_threshold)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEST RESULTS: {run_name}")
    logger.info(f"{'='*60}")
    for k, v in test_metrics.items():
        if k == "confusion_matrix":
            continue
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save
    result = {
        "run_name": run_name,
        "task": args.task,
        "modality": FIXED_MODALITY,
        "model": args.model,
        "split": args.split,
        "horizon": args.horizon,
        "seed": args.seed,
        "val_metrics": {k: v for k, v in val_metrics.items()
                        if k != "confusion_matrix"},
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if k != "confusion_matrix"},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"Results saved to {run_dir}/results.json")
    return result


if __name__ == "__main__":
    main()
