#!/usr/bin/env python3
"""
stack_rghbtq_xgb.py — hybrid that fuses the multimodal RG-HBT-Q (temporal + video)
with XGBoost (tabular controller-signal strength) at the probability level.

Both models are evaluated on the identical leak-safe samples (override KEPT, only the
ADAS control flags removed). The combiner is fit on the validation split and applied
to the test split. Reports test AUPRC for XGBoost, RG-HBT-Q, simple average, the
best validation-tuned weight, and logistic stacking.

Usage:
  python3 stack_rghbtq_xgb.py --task task3 --nn-results-dir results_rghbtq2 --seed 42
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, MODALITY_CONFIGS, CACHE_DIR
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats
from train_classical import extract_statistical_features, train_xgb
from train_nn import collate_fn
from models import RGHBTQ, DIRGHBTQ
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

XGB_MODALITY = "Safe-Full-Struct"     # leak-safe structured, override kept
NN_CLASS = {"rghbtq": RGHBTQ, "dirghbtq": DIRGHBTQ}


def load_norm(modality, task, split, horizon):
    mod_cfg = MODALITY_CONFIGS[modality]
    p = CACHE_DIR / f"norm_{modality}_{task}_{split}_h{horizon}.npz"
    if p.exists():
        d = np.load(p); return {"mean": d["mean"], "std": d["std"]}
    ns = compute_norm_stats(task, BENCHMARK_DIR / f"split_{split}.json", mod_cfg, horizon)
    if ns is not None:
        np.savez(p, mean=ns["mean"], std=ns["std"])
    return ns


def xgb_probs(task, split, horizon):
    mod_cfg = MODALITY_CONFIGS[XGB_MODALITY]
    ns = load_norm(XGB_MODALITY, task, split, horizon)
    cache = RouteCache()
    sf = BENCHMARK_DIR / f"split_{split}.json"
    ds = {s: PassingCtrlDataset(task, s, sf, mod_cfg, horizon=horizon,
                                norm_stats=ns, route_cache=cache)
          for s in ["train", "val", "test"]}
    cache.preload(sorted({r for s in ds for r in ds[s].route_ids}), load_gps=mod_cfg["gps"])
    X, y = {}, {}
    for s in ds:
        X[s], y[s] = extract_statistical_features(ds[s])
        np.nan_to_num(X[s], copy=False)
    model = train_xgb(X["train"], y["train"], X["val"], y["val"], task)
    return (model.predict_proba(X["val"])[:, 1], model.predict_proba(X["test"])[:, 1],
            y["val"], y["test"])


def nn_probs(task, split, horizon, ckpt, nn_model, nn_modality):
    mod_cfg = MODALITY_CONFIGS[nn_modality]
    ns = load_norm(nn_modality, task, split, horizon)
    cache = RouteCache(use_vjepa=True)
    sf = BENCHMARK_DIR / f"split_{split}.json"
    ds = {s: PassingCtrlDataset(task, s, sf, mod_cfg, horizon=horizon,
                                norm_stats=ns, route_cache=cache)
          for s in ["val", "test"]}
    cache.preload(sorted({r for s in ds for r in ds[s].route_ids}),
                  load_front_video=True, load_cabin_video=True)
    dev = "cuda"
    m = NN_CLASS[nn_model](
        struct_dim=ds["val"].struct_dim, use_front_video=True,
        use_cabin_video=True, task=task, video_feature_dim=cache.video_feature_dim,
        struct_group_dims=[len(c) for _, c in ds["val"]._struct_sources],
        struct_group_names=list(mod_cfg["struct"])).to(dev)
    m.load_state_dict(torch.load(ckpt, weights_only=True)); m.eval()
    out = {}
    for s in ["val", "test"]:
        loader = DataLoader(ds[s], batch_size=2048, shuffle=False, num_workers=2,
                            collate_fn=collate_fn, pin_memory=True)
        ps, ys = [], []
        for b in loader:
            kw = {k: b[k].to(dev) for k in ["struct", "front_video", "cabin_video"] if k in b}
            with torch.no_grad(), torch.amp.autocast("cuda"):
                lo = m(**kw)
            ps.append(torch.sigmoid(lo.float().squeeze(1)).cpu().numpy())
            ys.append(b["label"].numpy())
        out[s] = (np.concatenate(ps), np.concatenate(ys))
    return out["val"][0], out["test"][0], out["val"][1], out["test"][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task3")
    ap.add_argument("--split", default="cross_driver")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nn-results-dir", default="results_rghbtq2")
    ap.add_argument("--nn-model", default="rghbtq", choices=["rghbtq", "dirghbtq"])
    ap.add_argument("--nn-modality", default="Hier-Safe-MM")
    args = ap.parse_args()

    ckpt = (Path(args.nn_results_dir) /
            f"{args.task}_{args.nn_modality}_{args.nn_model}_{args.split}_h{args.horizon}_s{args.seed}" /
            "best_model.pt")
    assert ckpt.exists(), f"NN checkpoint not found: {ckpt}"

    xv, xt, yv, yt = xgb_probs(args.task, args.split, args.horizon)
    nv, nt, yv2, yt2 = nn_probs(args.task, args.split, args.horizon, ckpt,
                                args.nn_model, args.nn_modality)
    assert np.array_equal(yt, yt2) and np.array_equal(yv, yv2), \
        "sample order mismatch between XGBoost and NN pipelines"

    ap_x = average_precision_score(yt, xt)
    ap_n = average_precision_score(yt, nt)
    ap_avg = average_precision_score(yt, (xt + nt) / 2)

    # validation-tuned weight  w*xgb + (1-w)*nn
    ws = np.linspace(0, 1, 51)
    w = ws[np.argmax([average_precision_score(yv, w * xv + (1 - w) * nv) for w in ws])]
    ap_w = average_precision_score(yt, w * xt + (1 - w) * nt)

    # logistic stacking on val
    lr = LogisticRegression(max_iter=1000).fit(np.c_[xv, nv], yv)
    ap_s = average_precision_score(yt, lr.predict_proba(np.c_[xt, nt])[:, 1])

    print(f"\n=== Hybrid stacking: {args.task} (seed {args.seed}) ===  base rate {yt.mean():.3f}")
    print(f"  XGBoost                 AUPRC = {ap_x:.3f}")
    print(f"  RG-HBT-Q                AUPRC = {ap_n:.3f}")
    print(f"  average                 AUPRC = {ap_avg:.3f}")
    print(f"  weighted (w_xgb={w:.2f})      AUPRC = {ap_w:.3f}")
    print(f"  logistic stack          AUPRC = {ap_s:.3f}")
    print(f"  --> best hybrid beats XGBoost by {max(ap_avg, ap_w, ap_s) - ap_x:+.3f}\n")


if __name__ == "__main__":
    main()
