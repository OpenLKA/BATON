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
import torch.nn.functional as F
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
from models import GRUBackbone, TCNBackbone, CrossModalTransformer, RGHBTQ, DIRGHBTQ
from metrics import (evaluate_task1, evaluate_binary, find_optimal_f1_threshold,
                     evaluate_event_level)

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


class FocalLossBin(nn.Module):
    """Binary focal loss on logits (for imbalanced takeover/handover)."""

    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        return (alpha_t * (1 - p_t).pow(self.gamma) * ce).mean()


def _as_2class(logits):
    """Map a [B,1] binary logit to [B,2] so softmax-KL matches the multiclass path."""
    return torch.cat([torch.zeros_like(logits), logits], dim=-1)


def _distill_loss(logits, logits_teacher, z, z_teacher, task):
    """KL(student||teacher) on detached teacher + feature alignment. Binary-safe."""
    s = logits if task == "task1" else _as_2class(logits)
    t = logits_teacher if task == "task1" else _as_2class(logits_teacher)
    kl = F.kl_div(F.log_softmax(s, dim=-1),
                  F.softmax(t.detach(), dim=-1), reduction="batchmean")
    align = (F.normalize(z, dim=-1) - F.normalize(z_teacher.detach(), dim=-1)).pow(2).sum(-1).mean()
    return 0.1 * kl + 0.05 * align


def train_one_epoch(model, loader, criterion, optimizer, device, task, scaler,
                    mm_distill=False, aux_loss_w=0.0):
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
            aux_logits = []
            if mm_distill:
                # Teacher: clean pass (no modality dropout). Student: dropped pass.
                logits_t, z_t = model(**kwargs, apply_modality_dropout=False, return_repr=True)
                logits, z = model(**kwargs, apply_modality_dropout=True, return_repr=True)
            elif aux_loss_w > 0:
                logits, aux_logits = model(**kwargs, return_aux=True)
            else:
                logits = model(**kwargs)
            if task == "task1":
                loss = criterion(logits, labels)
                for al in aux_logits:
                    loss = loss + aux_loss_w * criterion(al, labels)
            else:
                loss = criterion(logits.squeeze(1), labels)
                for al in aux_logits:
                    loss = loss + aux_loss_w * criterion(al.squeeze(1), labels)
            if mm_distill:
                loss = loss + _distill_loss(logits, logits_t, z, z_t, task)

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
              use_pca=False, use_clip=False, use_vjepa=False, results_dir=None,
              video_dropout=None, single_frame=False, t3_suffix="",
              route_cache=None, mm_distill=False, aux_loss_w=0.0, select_metric=None,
              loss_type="bce", weight_decay=None, reweight_leadtime=0.0):
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
    run_name = f"{task}_{modality}_{model_type}{sf_tag}_{split}_h{horizon}{t3_suffix}_s{seed}"
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
        cache = RouteCache(use_pca=use_pca, use_clip=use_clip, use_vjepa=use_vjepa)
    else:
        cache = route_cache
    vf_dim = cache.video_feature_dim

    train_ds = PassingCtrlDataset(
        task, "train", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame, t3_suffix=t3_suffix,
    )
    val_ds = PassingCtrlDataset(
        task, "val", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame, t3_suffix=t3_suffix,
    )
    test_ds = PassingCtrlDataset(
        task, "test", split_file, mod_cfg,
        horizon=horizon, norm_stats=norm_stats, route_cache=cache,
        single_frame=single_frame, t3_suffix=t3_suffix,
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
    # Lead-time reweighting: upweight EARLY positives so the model does not only
    # exploit the last-second override spike (where trees already dominate).
    train_sampler, train_shuffle = None, True
    if reweight_leadtime and reweight_leadtime > 0:
        from torch.utils.data import WeightedRandomSampler
        lab = np.asarray(train_ds.labels)
        lt = np.asarray(train_ds.event_times, dtype=np.float64) - np.asarray(train_ds.ends, dtype=np.float64)
        w = np.ones(len(train_ds), dtype=np.float64)
        pos = (lab == 1) & np.isfinite(lt)
        w[pos] = 1.0 + reweight_leadtime * np.clip(lt[pos], 0, horizon)
        train_sampler = WeightedRandomSampler(torch.as_tensor(w), num_samples=len(w), replacement=True)
        train_shuffle = False
        logger.info(f"Lead-time reweight alpha={reweight_leadtime}: "
                    f"pos weight [{w[pos].min():.2f}, {w[pos].max():.2f}]")
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=train_shuffle,
                              sampler=train_sampler,
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
    ModelClass = {"gru": GRUBackbone, "tcn": TCNBackbone,
                  "transformer": CrossModalTransformer, "rghbtq": RGHBTQ,
                  "dirghbtq": DIRGHBTQ}[model_type]
    model_kwargs = dict(
        struct_dim=train_ds.struct_dim,
        use_gps=mod_cfg["gps"],
        use_front_video=mod_cfg["front_video"],
        use_cabin_video=mod_cfg["cabin_video"],
        task=task,
        video_feature_dim=vf_dim,
        video_dropout=video_dropout,
    )
    if model_type in ("rghbtq", "dirghbtq"):
        # These models need the per-category split of the concatenated struct tensor.
        model_kwargs["struct_group_dims"] = [len(cols) for _, cols in train_ds._struct_sources]
        model_kwargs["struct_group_names"] = list(mod_cfg["struct"])
    model = ModelClass(**model_kwargs).to(device)
    # DI-RG-HBT-Q: video auxiliary loss is on by default; select on AUPRC for imbalance.
    if model_type == "dirghbtq":
        if aux_loss_w == 0.0:
            aux_loss_w = 0.1
        if select_metric is None and task != "task1":
            select_metric = "auprc"

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {model_type}, params={n_params:,}")

    # Loss
    if task == "task1":
        weights = get_class_weights(train_ds).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights,
                                         label_smoothing=LABEL_SMOOTHING)
        logger.info(f"Class weights: {weights.cpu().numpy().round(2)}")
    elif loss_type == "focal":
        criterion = FocalLossBin(gamma=2.0, alpha=0.75)
        logger.info("Loss: binary focal (gamma=2, alpha=0.75)")
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

    wd = weight_decay if weight_decay is not None else WEIGHT_DECAY
    optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
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
    metric_name = select_metric or ("macro_f1" if task == "task1" else "auc_roc")
    logger.info(f"Model-selection metric: {metric_name}; aux_loss_w={aux_loss_w}")

    logger.info(f"{'Epoch':>5} | {'Loss':>8} | {'Val':>8} | {'Best':>8} | {'Time':>5} | Note")
    logger.info(f"{'-'*5}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*5}-+------")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     device, task, scaler, mm_distill=mm_distill,
                                     aux_loss_w=aux_loss_w)
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

        # Event-level metrics (B3): aggregate overlapping windows → one unit/event.
        # test_loader is shuffle=False, so test_scores align with test_ds order.
        event_metrics = evaluate_event_level(
            route_ids=test_ds.route_ids, end_times=test_ds.ends,
            event_times=test_ds.event_times, y_true=test_true, y_scores=test_scores,
            horizon=horizon, threshold=opt_threshold,
        )
        test_metrics.update(event_metrics)
        logger.info(f"Event-level: AUPRC={event_metrics['event_auprc']:.4f} "
                    f"det_recall={event_metrics['event_detection_recall']:.4f} "
                    f"median_lead={event_metrics['event_median_lead_time_s']:.2f}s "
                    f"({event_metrics['n_events_pos']}+/{event_metrics['n_events_neg']}-)")
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
    parser.add_argument("--model", default="gru",
                        choices=["gru", "tcn", "transformer", "rghbtq", "dirghbtq"])
    parser.add_argument("--select-metric", default=None,
                        choices=["auc_roc", "auprc", "macro_f1"],
                        help="validation metric for model selection / early stopping")
    parser.add_argument("--aux-loss-w", type=float, default=0.0,
                        help="(dirghbtq) weight of the front/cabin auxiliary BCE loss")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal"],
                        help="binary loss type (focal helps imbalanced takeover)")
    parser.add_argument("--weight-decay", type=float, default=None,
                        help="override AdamW weight decay (default 1e-4)")
    parser.add_argument("--reweight-leadtime", type=float, default=0.0,
                        help="upweight early positives: w = 1 + alpha*lead_time")
    parser.add_argument("--split", default="cross_driver",
                        choices=["cross_driver", "cross_vehicle", "random",
                                 "within_device_temporal"])
    parser.add_argument("--horizon", type=int, default=3, choices=[1, 3, 5])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--use-pca", action="store_true")
    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--use-vjepa", action="store_true")
    parser.add_argument("--t3-suffix", type=str, default="",
                        help="e.g. _antsafe → use anticipation-safe T3 samples")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--video-dropout", type=float, default=None)
    parser.add_argument("--single-frame", action="store_true")
    parser.add_argument("--mm-distill", action="store_true",
                        help="(rghbtq only) complete<->missing distillation: clean "
                             "teacher + dropped student, KL + feature alignment")
    args = parser.parse_args()
    if args.mm_distill and args.model != "rghbtq":
        parser.error("--mm-distill is only supported with --model rghbtq")

    train_run(
        task=args.task, modality=args.modality, model_type=args.model,
        split=args.split, horizon=args.horizon, seed=args.seed,
        device=args.device, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, num_workers=args.num_workers,
        use_pca=args.use_pca, use_clip=args.use_clip, use_vjepa=args.use_vjepa,
        t3_suffix=args.t3_suffix,
        results_dir=args.results_dir, video_dropout=args.video_dropout,
        single_frame=args.single_frame, mm_distill=args.mm_distill,
        aux_loss_w=args.aux_loss_w, select_metric=args.select_metric,
        loss_type=args.loss, weight_decay=args.weight_decay,
        reweight_leadtime=args.reweight_leadtime,
    )


if __name__ == "__main__":
    main()
