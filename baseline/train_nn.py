#!/usr/bin/env python3
"""
train_nn.py — Train GRU or TCN baseline on PassingCtrl benchmark.

Usage:
  python3 train_nn.py --task task1 --modality Full-Struct --model gru --seed 42
  python3 train_nn.py --task task2 --modality Full-Multimodal --model tcn --seed 42
"""
import argparse
import json
import logging
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BENCHMARK_DIR, MODALITY_CONFIGS, BATCH_SIZE, LR, WEIGHT_DECAY,
    EPOCHS, PATIENCE, NUM_WORKERS, RESULTS_DIR, CACHE_DIR,
    LABEL2IDX, MAX_CLASS_WEIGHT, MAX_POS_WEIGHT, GPS_COLS,
    LABEL_SMOOTHING, WARMUP_EPOCHS,
)
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats
from models import GRUBackbone, TCNBackbone
from metrics import evaluate_task1, evaluate_binary, find_optimal_f1_threshold

logger = logging.getLogger("baseline")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(dataset):
    """Compute inverse-frequency class weights for Task 1."""
    labels = dataset.labels
    counts = np.bincount(labels, minlength=len(LABEL2IDX))
    counts = counts.clip(min=1)
    weights = 1.0 / counts.astype(np.float32)
    weights = weights / weights.min()
    weights = np.clip(weights, 1.0, MAX_CLASS_WEIGHT)
    return torch.from_numpy(weights)


def get_pos_weight(dataset):
    """Compute positive class weight for binary tasks."""
    labels = dataset.labels
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0:
        return torch.tensor([1.0])
    pw = min(n_neg / n_pos, MAX_POS_WEIGHT)
    return torch.tensor([pw])


def collate_fn(batch):
    """Custom collate that handles variable keys."""
    out = {}
    out["label"] = torch.stack([
        torch.tensor(b["label"]) if not isinstance(b["label"], torch.Tensor)
        else b["label"] for b in batch
    ])
    if "struct" in batch[0]:
        out["struct"] = torch.stack([b["struct"] for b in batch])
    if "gps" in batch[0]:
        out["gps"] = torch.stack([b["gps"] for b in batch])
    if "front_video" in batch[0]:
        out["front_video"] = torch.stack([b["front_video"] for b in batch])
    if "cabin_video" in batch[0]:
        out["cabin_video"] = torch.stack([b["cabin_video"] for b in batch])
    return out


def train_one_epoch(model, loader, criterion, optimizer, device, task, scaler):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        kwargs = {}
        if "struct" in batch:
            kwargs["struct"] = batch["struct"].to(device, non_blocking=True)
        if "gps" in batch:
            kwargs["gps"] = batch["gps"].to(device, non_blocking=True)
        if "front_video" in batch:
            kwargs["front_video"] = batch["front_video"].to(device, non_blocking=True)
        if "cabin_video" in batch:
            kwargs["cabin_video"] = batch["cabin_video"].to(device, non_blocking=True)

        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            logits = model(**kwargs)
            if task == "task1":
                loss = criterion(logits, labels)
            else:
                loss = criterion(logits.squeeze(1), labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, task):
    model.eval()
    all_labels = []
    all_outputs = []

    for batch in loader:
        kwargs = {}
        if "struct" in batch:
            kwargs["struct"] = batch["struct"].to(device, non_blocking=True)
        if "gps" in batch:
            kwargs["gps"] = batch["gps"].to(device, non_blocking=True)
        if "front_video" in batch:
            kwargs["front_video"] = batch["front_video"].to(device, non_blocking=True)
        if "cabin_video" in batch:
            kwargs["cabin_video"] = batch["cabin_video"].to(device, non_blocking=True)

        labels = batch["label"]
        with torch.amp.autocast("cuda"):
            logits = model(**kwargs)

        all_labels.append(labels.numpy())
        if task == "task1":
            all_outputs.append(logits.float().cpu().numpy())
        else:
            probs = torch.sigmoid(logits.float().squeeze(1)).cpu().numpy()
            all_outputs.append(probs)

    y_true = np.concatenate(all_labels)
    y_out = np.concatenate(all_outputs)

    if task == "task1":
        return evaluate_task1(y_true, y_out), y_true, y_out
    else:
        return evaluate_binary(y_true, y_out), y_true, y_out


def train_run(task, modality, model_type="gru", split="cross_driver",
              horizon=3, seed=42, device="cuda", epochs=EPOCHS,
              batch_size=None, lr=LR, num_workers=4,
              use_pca=False, use_clip=False, results_dir=None,
              video_dropout=None, single_frame=False,
              route_cache=None):
    """Core training function. Can be called in-process with a shared RouteCache.

    Args:
        route_cache: Optional pre-loaded RouteCache to avoid reloading features.
                     If None, creates and loads a new one.

    Returns:
        dict with run results, or None on failure.
    """
    set_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    split_file = BENCHMARK_DIR / f"split_{split}.json"
    mod_cfg = MODALITY_CONFIGS[modality]

    # Run name
    sf_tag = "_sf" if single_frame else ""
    run_name = f"{task}_{modality}_{model_type}{sf_tag}_{split}_h{horizon}_s{seed}"
    results_base = Path(results_dir) if results_dir else RESULTS_DIR
    run_dir = results_base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Reset logging handlers for this run (close old file handles first)
    for h in logger.handlers[:]:
        h.close()
        logger.removeHandler(h)
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(run_dir / "train.log"))
    logger.setLevel(logging.INFO)

    logger.info(f"Run: {run_name}")
    logger.info(f"Config: task={task}, modality={modality}, model={model_type}, "
                f"split={split}, horizon={horizon}, seed={seed}")

    # Norm stats
    norm_path = CACHE_DIR / f"norm_{modality}_{task}_{split}_h{horizon}.npz"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if norm_path.exists():
        data = np.load(norm_path)
        norm_stats = {"mean": data["mean"], "std": data["std"]}
        logger.info(f"Loaded norm stats from {norm_path}")
    else:
        norm_stats = compute_norm_stats(task, split_file, mod_cfg, horizon)
        if norm_stats is not None:
            np.savez(norm_path, mean=norm_stats["mean"], std=norm_stats["std"])
            logger.info(f"Saved norm stats to {norm_path}")

    # Datasets — use shared RouteCache if provided
    if route_cache is None:
        cache = RouteCache(use_pca=use_pca, use_clip=use_clip)
    else:
        cache = route_cache
    vf_dim = cache.video_feature_dim

    train_ds = PassingCtrlDataset(
        task, "train", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame,
    )
    val_ds = PassingCtrlDataset(
        task, "val", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame,
    )
    test_ds = PassingCtrlDataset(
        task, "test", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame,
    )

    # Preload only if we created a fresh cache
    if route_cache is None:
        all_routes = list(set(
            list(train_ds.route_ids) + list(val_ds.route_ids) + list(test_ds.route_ids)
        ))
        cache.preload(
            all_routes,
            load_gps=mod_cfg["gps"],
            load_front_video=mod_cfg["front_video"],
            load_cabin_video=mod_cfg["cabin_video"],
        )

    # Adaptive batch size
    if batch_size is not None:
        bs = batch_size
    else:
        n_train = len(train_ds)
        if n_train > 200000:
            bs = BATCH_SIZE
        else:
            bs = 512
    logger.info(f"Batch size: {bs} ({len(train_ds)//bs + 1} batches/epoch)")

    nw = num_workers
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=nw, collate_fn=collate_fn,
                              pin_memory=True, drop_last=False,
                              persistent_workers=False, prefetch_factor=4 if nw > 0 else None)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False,
                            num_workers=nw, collate_fn=collate_fn,
                            pin_memory=True,
                            persistent_workers=False, prefetch_factor=4 if nw > 0 else None)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False,
                             num_workers=nw, collate_fn=collate_fn,
                             pin_memory=True,
                             persistent_workers=False, prefetch_factor=4 if nw > 0 else None)

    # Model
    ModelClass = GRUBackbone if model_type == "gru" else TCNBackbone
    model = ModelClass(
        struct_dim=train_ds.struct_dim,
        use_gps=mod_cfg["gps"],
        use_front_video=mod_cfg["front_video"],
        use_cabin_video=mod_cfg["cabin_video"],
        task=task,
        video_feature_dim=vf_dim,
        video_dropout=video_dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_type}, params={n_params:,}")

    # Loss
    if task == "task1":
        weights = get_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights,
                                         label_smoothing=LABEL_SMOOTHING)
        logger.info(f"Class weights: {weights.cpu().numpy().round(2)}")
    else:
        pos_weight = get_pos_weight(train_ds).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Pos weight: {pos_weight.item():.2f}")

    # Optimizer with differential LR: video/GPS branches get lower LR
    video_gps_params = []
    other_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in ["fv_", "cv_", "gps_"]):
            video_gps_params.append(param)
        else:
            other_params.append(param)

    if video_gps_params:
        aux_lr = lr * 0.3
        param_groups = [
            {"params": other_params, "lr": lr},
            {"params": video_gps_params, "lr": aux_lr},
        ]
        logger.info(f"Differential LR: main={lr}, aux={aux_lr} "
                    f"({len(video_gps_params)} aux params)")
    else:
        param_groups = [{"params": other_params, "lr": lr}]

    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0,
        total_iters=WARMUP_EPOCHS,
    )
    cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - WARMUP_EPOCHS,
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched],
        milestones=[WARMUP_EPOCHS],
    )

    scaler = torch.amp.GradScaler("cuda")

    # Training loop
    best_val_metric = -1.0
    best_epoch = 0
    patience_counter = 0
    metric_name = "macro_f1" if task == "task1" else "auc_roc"

    logger.info(f"{'Epoch':>5} | {'Loss':>8} | {'Val':>8} | {'Best':>8} | {'Time':>5} | Note")
    logger.info(f"{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+------")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, task, scaler)
        scheduler.step()

        val_metrics, _, _ = evaluate(model, val_loader, device, task)
        elapsed = time.time() - t0

        val_metric = val_metrics[metric_name]
        improved = val_metric > best_val_metric
        note = ""

        if improved:
            best_val_metric = val_metric
            best_epoch = epoch
            patience_counter = 0
            ckpt_path = run_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            if not ckpt_path.exists():
                logger.error(f"FAILED to save checkpoint to {ckpt_path}")
            note = "★ new best"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                note = "✗ early stop"

        logger.info(f"{epoch:>5} | {train_loss:>8.4f} | {val_metric:>8.4f} | "
                    f"{best_val_metric:>8.4f} | {elapsed:>4.0f}s | {note}")

        if patience_counter >= PATIENCE:
            break

    # Load best model and evaluate on test
    model.load_state_dict(torch.load(run_dir / "best_model.pt", weights_only=True))
    logger.info(f"\nBest epoch: {best_epoch}, val_{metric_name}={best_val_metric:.4f}")

    if task != "task1":
        val_metrics_final, val_true, val_scores = evaluate(model, val_loader, device, task)
        opt_threshold, _ = find_optimal_f1_threshold(val_true, val_scores)
        logger.info(f"Optimal threshold from val: {opt_threshold:.4f}")

        test_metrics, test_true, test_scores = evaluate(model, test_loader, device, task)
        test_metrics = evaluate_binary(test_true, test_scores, threshold=opt_threshold)
    else:
        val_metrics_final, _, _ = evaluate(model, val_loader, device, task)
        test_metrics, _, _ = evaluate(model, test_loader, device, task)

    logger.info(f"\n{'='*60}")
    logger.info(f"TEST RESULTS: {run_name}")
    logger.info(f"{'='*60}")
    for k, v in test_metrics.items():
        if k == "confusion_matrix":
            continue
        logger.info(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    result = {
        "run_name": run_name,
        "task": task,
        "modality": modality,
        "model": model_type,
        "split": split,
        "horizon": horizon,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_metric": float(best_val_metric),
        "n_params": n_params,
        "val_metrics": {k: v for k, v in val_metrics_final.items()
                        if k != "confusion_matrix"},
        "test_metrics": {k: v for k, v in test_metrics.items()
                         if k != "confusion_matrix"},
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nResults saved to {run_dir}/results.json")

    # Cleanup: shut down DataLoader workers to release file descriptors
    for loader in [train_loader, val_loader, test_loader]:
        if hasattr(loader, '_iterator') and loader._iterator is not None:
            loader._iterator._shutdown_workers()
    del model, optimizer, scaler, train_loader, val_loader, test_loader
    torch.cuda.empty_cache()
    import gc; gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser(description="Train NN baseline")
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--modality", required=True, choices=list(MODALITY_CONFIGS.keys()))
    parser.add_argument("--model", default="gru", choices=["gru", "tcn"])
    parser.add_argument("--split", default="cross_driver",
                        choices=["cross_driver", "cross_vehicle", "random"])
    parser.add_argument("--horizon", type=int, default=3, choices=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--video-dropout", type=float, default=None)
    parser.add_argument("--single-frame", action="store_true")
    args = parser.parse_args()

    train_run(
        task=args.task, modality=args.modality, model_type=args.model,
        split=args.split, horizon=args.horizon, seed=args.seed,
        device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers,
        use_pca=args.use_pca, use_clip=args.use_clip,
        results_dir=args.results_dir, video_dropout=args.video_dropout,
        single_frame=args.single_frame,
    )


if __name__ == "__main__":
    main()
