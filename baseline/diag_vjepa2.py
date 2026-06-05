#!/usr/bin/env python3
"""
diag_vjepa2.py — V-JEPA2-only signal diagnosis (Run 1).

Does the frozen V-JEPA2 video carry ANY independent signal for the transition
tasks? Builds temporal-stat features (mean/last/max/delta-late) from the
pre-extracted front/cabin V-JEPA2 embeddings and trains XGBoost on them, for
front-only / cabin-only / front+cabin. Compares to the base rate.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, MODALITY_CONFIGS
from dataset import PassingCtrlDataset, RouteCache
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from tqdm import tqdm


def vid_stats(v):                       # v: [10, 1024] -> [4096]
    return np.concatenate([v.mean(0), v[-1], v.max(0), v[-1] - v[-3]]).astype(np.float32)


def extract(ds, max_n=None):
    n = len(ds) if max_n is None else min(len(ds), max_n)
    F = np.zeros((n, 4096), np.float32)
    C = np.zeros((n, 4096), np.float32)
    y = np.zeros(n, np.int64)
    for i in tqdm(range(n), desc="extract", leave=False):
        it = ds[i]
        F[i] = vid_stats(it["front_video"].numpy())
        C[i] = vid_stats(it["cabin_video"].numpy())
        y[i] = int(it["label"]) if not hasattr(it["label"], "item") else int(it["label"].item())
    return F, C, y


def xgb_ap(Xtr, ytr, Xva, yva, Xte, yte):
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(spw, 10),
                      eval_metric="logloss", early_stopping_rounds=20,
                      tree_method="hist", device="cuda", n_jobs=-1)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return average_precision_score(yte, m.predict_proba(Xte)[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task3")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=60000)
    args = ap.parse_args()

    mod = MODALITY_CONFIGS["FV+CV"]
    sf = BENCHMARK_DIR / "split_cross_driver.json"
    cache = RouteCache(use_vjepa=True)
    ds = {s: PassingCtrlDataset(args.task, s, sf, mod, horizon=args.horizon, route_cache=cache)
          for s in ["train", "val", "test"]}
    cache.preload(sorted({r for s in ds for r in ds[s].route_ids}),
                  load_front_video=True, load_cabin_video=True)

    Ftr, Ctr, ytr = extract(ds["train"], args.max_train)
    Fva, Cva, yva = extract(ds["val"])
    Fte, Cte, yte = extract(ds["test"])
    base = yte.mean()
    print(f"\n=== V-JEPA2-only diagnosis: {args.task} (h={args.horizon}) ===")
    print(f"test base rate = {base:.3f}   (n_train={len(ytr)}, n_test={len(yte)})")
    print(f"{'input':<16}{'test AUPRC':>12}{'lift x chance':>16}")
    print("-" * 44)
    for name, tr, va, te in [
        ("front only", Ftr, Fva, Fte),
        ("cabin only", Ctr, Cva, Cte),
        ("front+cabin", np.hstack([Ftr, Ctr]), np.hstack([Fva, Cva]), np.hstack([Fte, Cte])),
    ]:
        apv = xgb_ap(tr, ytr, va, yva, te, yte)
        print(f"{name:<16}{apv:>12.3f}{apv/base:>15.2f}x")
    print()


if __name__ == "__main__":
    main()
