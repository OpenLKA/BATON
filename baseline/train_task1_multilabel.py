#!/usr/bin/env python3
"""
train_task1_multilabel.py — Multi-label variant of Task-1 action recognition.

Reuses PassingCtrlDataset (task1) for the INPUTS (same windows, same feature
caches) but swaps the single 7-class label for the 6-dim multi-hot label from
benchmark_v2/task1_action_samples_multilabel.csv (made by
benchmark/make_task1_multilabel.py). Cruising = all-zero vector.

Backbone: the standard GRUBackbone with its head replaced by a 6-logit linear
layer; loss = BCEWithLogitsLoss with per-class pos_weight (clipped like the
binary tasks). Metrics: per-class AP, macro-AP, micro/macro-F1 @ 0.5.

Usage:
  python3 train_task1_multilabel.py --modality T1-RuleFree-Struct --seed 42
  python3 train_task1_multilabel.py --modality VJEPA-only-FrontVideo --use-vjepa --seed 42
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import (  # noqa: E402
    BENCHMARK_DIR, MODALITY_CONFIGS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, CACHE_DIR, MAX_POS_WEIGHT, WARMUP_EPOCHS, FUSION_DIM,
    BASELINE_DIR,
)
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats  # noqa: E402
from models import GRUBackbone  # noqa: E402
from train_nn import set_seed, collate_fn  # noqa: E402

logger = logging.getLogger("baseline")

ML_CLASSES = ["Stopped", "LaneChange", "Turning", "Braking",
              "Accelerating", "CarFollowing"]
ML_CSV = BENCHMARK_DIR / "task1_action_samples_multilabel.csv"
DEFAULT_RESULTS_DIR = BASELINE_DIR / "results_task1_ml"


def attach_multilabels(ds, ml_df):
    """Replace ds.labels (single-class ints) with the [N, 6] multi-hot matrix."""
    Y = ml_df.reindex(ds.sample_ids).values
    if np.isnan(Y).any():
        missing = int(np.isnan(Y).any(axis=1).sum())
        raise RuntimeError(f"{missing} samples missing from {ML_CSV}")
    ds.labels = Y.astype(np.float32)
    return ds


def get_pos_weights(labels):
    """Per-class pos_weight = clip(neg/pos, 1, MAX_POS_WEIGHT)."""
    n = labels.shape[0]
    pos = labels.sum(axis=0).clip(min=1)
    pw = ((n - pos) / pos).clip(1.0, MAX_POS_WEIGHT)
    return torch.from_numpy(pw.astype(np.float32))


def evaluate_multilabel(y_true, y_prob, threshold=0.5):
    """Per-class AP, macro-AP, micro/macro F1 at fixed threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    result = {}
    aps = []
    for c, name in enumerate(ML_CLASSES):
        yc = y_true[:, c]
        if 0 < yc.sum() < len(yc):
            ap = float(average_precision_score(yc, y_prob[:, c]))
        else:
            ap = float("nan")
        result[f"ap_{name}"] = ap
        if np.isfinite(ap):
            aps.append(ap)
        result[f"f1_{name}"] = float(f1_score(yc, y_pred[:, c], zero_division=0))
        result[f"pos_rate_{name}"] = float(yc.mean())
    result["macro_ap"] = float(np.mean(aps)) if aps else float("nan")
    result["micro_f1"] = float(f1_score(y_true, y_pred, average="micro",
                                        zero_division=0))
    result["macro_f1"] = float(f1_score(y_true, y_pred, average="macro",
                                        zero_division=0))
    return result


def run_epoch(model, loader, criterion, optimizer, device, scaler, train=True):
    model.train() if train else model.eval()
    total_loss, n_batches = 0.0, 0
    all_y, all_p = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            kwargs = {}
            for k in ("struct", "gps", "front_video", "cabin_video"):
                if k in batch:
                    kwargs[k] = batch[k].to(device, non_blocking=True)
            labels = batch["label"].float().to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = model(**kwargs)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                all_y.append(labels.cpu().numpy())
                all_p.append(torch.sigmoid(logits.float()).cpu().numpy())

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    if train:
        return avg_loss, None, None
    return avg_loss, np.concatenate(all_y), np.concatenate(all_p)


def train_run(modality, seed=42, split="cross_driver", epochs=EPOCHS,
              batch_size=None, lr=LR, num_workers=4, device="cuda",
              use_vjepa=False, results_dir=None):
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    split_file = BENCHMARK_DIR / f"split_{split}.json"
    mod_cfg = MODALITY_CONFIGS[modality]

    run_name = f"task1ml_{modality}_gru_{split}_s{seed}"
    results_base = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
    run_dir = results_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(run_dir / "train.log"))
    logger.setLevel(logging.INFO)
    logger.info(f"Run: {run_name} (multi-label Task-1, classes={ML_CLASSES})")

    # Norm stats — same cache path pattern as train_nn (inputs are identical).
    norm_path = CACHE_DIR / f"norm_{modality}_task1_{split}_h3.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if norm_path.exists():
        d = np.load(norm_path)
        norm_stats = {"mean": d["mean"], "std": d["std"]}
        logger.info(f"Loaded norm stats from {norm_path}")
    else:
        norm_stats = compute_norm_stats("task1", split_file, mod_cfg)
        if norm_stats is not None:
            np.savez(norm_path, mean=norm_stats["mean"], std=norm_stats["std"])
            logger.info(f"Saved norm stats to {norm_path}")

    # Multi-hot labels indexed by sample_id
    ml_df = pd.read_csv(ML_CSV, usecols=["sample_id"] + [f"ml_{c}" for c in ML_CLASSES],
                        index_col="sample_id")
    ml_df = ml_df[[f"ml_{c}" for c in ML_CLASSES]]

    cache = RouteCache(use_vjepa=use_vjepa)
    datasets = {}
    for part in ("train", "val", "test"):
        ds = PassingCtrlDataset("task1", part, split_file, mod_cfg,
                                norm_stats=norm_stats, route_cache=cache)
        datasets[part] = attach_multilabels(ds, ml_df)
    train_ds, val_ds, test_ds = datasets["train"], datasets["val"], datasets["test"]

    all_routes = list(set(list(train_ds.route_ids) + list(val_ds.route_ids)
                          + list(test_ds.route_ids)))
    cache.preload(all_routes, load_gps=mod_cfg["gps"],
                  load_front_video=mod_cfg["front_video"],
                  load_cabin_video=mod_cfg["cabin_video"])

    bs = batch_size if batch_size is not None else \
        (BATCH_SIZE if len(train_ds) > 200000 else 512)
    logger.info(f"Batch size: {bs} ({len(train_ds)//bs + 1} batches/epoch)")
    logger.info("Train positive rates: " + ", ".join(
        f"{c}={train_ds.labels[:, i].mean():.4f}" for i, c in enumerate(ML_CLASSES)))

    loaders = {
        part: DataLoader(datasets[part], batch_size=bs, shuffle=(part == "train"),
                         num_workers=num_workers, collate_fn=collate_fn,
                         pin_memory=True,
                         prefetch_factor=4 if num_workers > 0 else None)
        for part in ("train", "val", "test")
    }

    model = GRUBackbone(
        struct_dim=train_ds.struct_dim,
        use_gps=mod_cfg["gps"],
        use_front_video=mod_cfg["front_video"],
        use_cabin_video=mod_cfg["cabin_video"],
        task="task1",
        video_feature_dim=cache.video_feature_dim,
    )
    model.head = nn.Linear(FUSION_DIM, len(ML_CLASSES))  # 6-logit multi-label head
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: gru + 6-logit BCE head, params={n_params:,}")

    pos_weight = get_pos_weights(train_ds.labels).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"Per-class pos_weight: {pos_weight.cpu().numpy().round(2)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  weight_decay=WEIGHT_DECAY)
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=WARMUP_EPOCHS)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - WARMUP_EPOCHS, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])
    scaler = torch.amp.GradScaler("cuda")

    best_val, best_epoch, patience_counter = -1.0, 0, 0
    metric_name = "macro_ap"
    logger.info(f"Model-selection metric: val {metric_name}")
    logger.info(f"{'Epoch':>5} | {'Loss':>8} | {'Val':>8} | {'Best':>8} | {'Time':>5} | Note")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, _, _ = run_epoch(model, loaders["train"], criterion,
                                     optimizer, device, scaler, train=True)
        scheduler.step()
        _, vy, vp = run_epoch(model, loaders["val"], criterion, optimizer,
                              device, scaler, train=False)
        val_metrics = evaluate_multilabel(vy, vp)
        val_metric = val_metrics[metric_name]
        elapsed = time.time() - t0

        note = ""
        if val_metric > best_val:
            best_val, best_epoch, patience_counter = val_metric, epoch, 0
            torch.save(model.state_dict(), run_dir / "best_model.pt")
            note = "* new best"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                note = "x early stop"
        logger.info(f"{epoch:>5} | {train_loss:>8.4f} | {val_metric:>8.4f} | "
                    f"{best_val:>8.4f} | {elapsed:>4.0f}s | {note}")
        if patience_counter >= PATIENCE:
            break

    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
    logger.info(f"Best epoch: {best_epoch}, val_{metric_name}={best_val:.4f}")

    _, vy, vp = run_epoch(model, loaders["val"], criterion, optimizer, device,
                          scaler, train=False)
    val_metrics = evaluate_multilabel(vy, vp)
    _, ty, tp = run_epoch(model, loaders["test"], criterion, optimizer, device,
                          scaler, train=False)
    test_metrics = evaluate_multilabel(ty, tp)

    logger.info("=" * 60)
    logger.info(f"TEST RESULTS: {run_name}")
    logger.info("=" * 60)
    for k, v in test_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    result = {
        "run_name": run_name,
        "task": "task1_multilabel",
        "classes": ML_CLASSES,
        "modality": modality,
        "model": "gru",
        "split": split,
        "seed": seed,
        "use_vjepa": use_vjepa,
        "label_csv": str(ML_CSV),
        "best_epoch": best_epoch,
        "best_val_metric": float(best_val),
        "n_params": n_params,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Results saved to {run_dir}/results.json")

    for loader in loaders.values():
        if hasattr(loader, "_iterator") and loader._iterator is not None:
            loader._iterator._shutdown_workers()
    del model, optimizer, scaler, loaders
    torch.cuda.empty_cache()
    import gc; gc.collect()
    return result


def main():
    parser = argparse.ArgumentParser(description="Train multi-label Task-1 GRU")
    parser.add_argument("--modality", required=True,
                        choices=list(MODALITY_CONFIGS.keys()))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="cross_driver",
                        choices=["cross_driver", "cross_vehicle", "random"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-vjepa", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    args = parser.parse_args()

    train_run(modality=args.modality, seed=args.seed, split=args.split,
              epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
              num_workers=args.num_workers, device=args.device,
              use_vjepa=args.use_vjepa, results_dir=args.results_dir)


if __name__ == "__main__":
    main()
