#!/usr/bin/env python3
"""
eval_rghbtq_robustness.py — missing-modality robustness at test time.

Loads a trained checkpoint and re-evaluates the test set under four input
conditions: full, drop-front-video, drop-cabin-video, drop-CAN (each "drop" =
zero that input tensor). Reports sample AUPRC per condition and, for RG-HBT-Q,
the learned reliability-gate means. Works for both --model rghbtq and the plain
--model transformer (the latter has no gate / no missing-modality training), so
the two can be contrasted on identical inputs.

Usage:
  python3 eval_rghbtq_robustness.py --model rghbtq --modality Hier-Safe-MM \
      --task task2 --seed 42 --results-dir results_rghbtq
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, MODALITY_CONFIGS, CACHE_DIR
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats
from models import GRUBackbone, TCNBackbone, CrossModalTransformer, RGHBTQ
from metrics import evaluate_binary
from torch.utils.data import DataLoader
from train_nn import collate_fn

CONDITIONS = ["full", "drop_front", "drop_cabin", "drop_can"]


def build_model(model_type, ds, mod_cfg, task, vf_dim, device):
    cls = {"gru": GRUBackbone, "tcn": TCNBackbone,
           "transformer": CrossModalTransformer, "rghbtq": RGHBTQ}[model_type]
    kw = dict(struct_dim=ds.struct_dim, use_gps=mod_cfg["gps"],
              use_front_video=mod_cfg["front_video"],
              use_cabin_video=mod_cfg["cabin_video"], task=task,
              video_feature_dim=vf_dim)
    if model_type == "rghbtq":
        kw["struct_group_dims"] = [len(c) for _, c in ds._struct_sources]
        kw["struct_group_names"] = list(mod_cfg["struct"])
    return cls(**kw).to(device)


@torch.no_grad()
def eval_condition(model, loader, device, condition):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        kw = {}
        for k in ["struct", "front_video", "cabin_video", "gps"]:
            if k in batch:
                kw[k] = batch[k].to(device, non_blocking=True)
        if condition == "drop_front" and "front_video" in kw:
            kw["front_video"] = torch.zeros_like(kw["front_video"])
        if condition == "drop_cabin" and "cabin_video" in kw:
            kw["cabin_video"] = torch.zeros_like(kw["cabin_video"])
        if condition == "drop_can" and "struct" in kw:
            kw["struct"] = torch.zeros_like(kw["struct"])
        with torch.amp.autocast("cuda"):
            logits = model(**kw)
        ps.append(torch.sigmoid(logits.float().squeeze(1)).cpu().numpy())
        ys.append(batch["label"].numpy())
    return np.concatenate(ys), np.concatenate(ps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["gru", "tcn", "transformer", "rghbtq"])
    ap.add_argument("--modality", required=True)
    ap.add_argument("--task", default="task2")
    ap.add_argument("--split", default="cross_driver")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t3-suffix", default="")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    mod_cfg = MODALITY_CONFIGS[args.modality]
    split_file = BENCHMARK_DIR / f"split_{args.split}.json"

    run_name = (f"{args.task}_{args.modality}_{args.model}_{args.split}"
                f"_h{args.horizon}{args.t3_suffix}_s{args.seed}")
    ckpt = Path(args.results_dir) / run_name / "best_model.pt"
    assert ckpt.exists(), f"checkpoint not found: {ckpt}"

    norm_path = CACHE_DIR / f"norm_{args.modality}_{args.task}_{args.split}_h{args.horizon}.npz"
    if norm_path.exists():
        d = np.load(norm_path); norm_stats = {"mean": d["mean"], "std": d["std"]}
    else:
        norm_stats = compute_norm_stats(args.task, split_file, mod_cfg, args.horizon)

    cache = RouteCache(use_vjepa=True)
    test_ds = PassingCtrlDataset(args.task, "test", split_file, mod_cfg,
                                 horizon=args.horizon, norm_stats=norm_stats,
                                 route_cache=cache, t3_suffix=args.t3_suffix)
    cache.preload(list(set(test_ds.route_ids)), load_gps=mod_cfg["gps"],
                  load_front_video=mod_cfg["front_video"],
                  load_cabin_video=mod_cfg["cabin_video"])
    loader = DataLoader(test_ds, batch_size=2048, shuffle=False, num_workers=2,
                        collate_fn=collate_fn, pin_memory=True)

    vf_dim = cache.video_feature_dim
    model = build_model(args.model, test_ds, mod_cfg, args.task, vf_dim, device)
    model.load_state_dict(torch.load(ckpt, weights_only=True))

    print(f"\n=== Missing-modality robustness: {run_name} ===")
    print(f"{'condition':<14}{'sample AUPRC':>14}{'rel. drop':>12}")
    print("-" * 40)
    full_auprc = None
    for cond in CONDITIONS:
        y, p = eval_condition(model, loader, device, cond)
        auprc = evaluate_binary(y, p)["auprc"]
        if cond == "full":
            full_auprc = auprc
            rel = ""
            gates = getattr(model, "last_gate_mean", {})
        else:
            rel = f"{100*(auprc-full_auprc)/max(full_auprc,1e-9):+.0f}%"
        print(f"{cond:<14}{auprc:>14.3f}{rel:>12}")
    if getattr(model, "last_gate_mean", None):
        gm = model.last_gate_mean; gs = model.last_gate_std
        print("\nreliability-gate means (full input):",
              {k: f"{gm[k]:.3f}±{gs[k]:.3f}" for k in gm})
    print()


if __name__ == "__main__":
    main()
