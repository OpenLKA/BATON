#!/usr/bin/env python3
"""
diag_vjepa2_fusion.py — does CAN + V-JEPA2 (tabular fusion) beat CAN-only?

Trains XGBoost on (a) CAN window-stats only, (b) V-JEPA2 video stats only,
(c) CAN + video stats concatenated. If (c) > (a), the video is complementary
and a multimodal model can beat the CAN-only tree baseline.
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, MODALITY_CONFIGS
from dataset import PassingCtrlDataset, RouteCache
from train_classical import extract_statistical_features
from metrics import evaluate_event_level
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
from tqdm import tqdm

CAN_MOD = "DI-Safe-Struct"      # leak-safe CAN, no DMS, no video


def vid_stats(v):
    return np.concatenate([v.mean(0), v[-1], v.max(0), v[-1] - v[-3]]).astype(np.float32)


def extract_video(ds, max_n=None):
    n = len(ds) if max_n is None else min(len(ds), max_n)
    V = np.zeros((n, 8192), np.float32)
    for i in tqdm(range(n), desc="vid", leave=False):
        it = ds[i]
        V[i] = np.concatenate([vid_stats(it["front_video"].numpy()),
                               vid_stats(it["cabin_video"].numpy())])
    return V


def xgb_probs(Xtr, ytr, Xva, yva, Xte, seed=0):
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(spw, 10),
                      eval_metric="logloss", early_stopping_rounds=20,
                      tree_method="hist", device="cuda", n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]


def xgb_ap(Xtr, ytr, Xva, yva, Xte, yte):
    return average_precision_score(yte, xgb_probs(Xtr, ytr, Xva, yva, Xte)[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task3")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--can-mod", default=CAN_MOD)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()
    sf = BENCHMARK_DIR / "split_cross_driver.json"

    # CAN stats (struct cache, no video)
    can_cache = RouteCache()
    dca = {s: PassingCtrlDataset(args.task, s, sf, MODALITY_CONFIGS[args.can_mod],
                                 horizon=args.horizon, route_cache=can_cache)
           for s in ["train", "val", "test"]}
    can_cache.preload(sorted({r for s in dca for r in dca[s].route_ids}))
    Xc, yc = {}, {}
    for s in dca:
        Xc[s], yc[s] = extract_statistical_features(dca[s],
                        max_samples=args.max_train if s == "train" else None)
        np.nan_to_num(Xc[s], copy=False)

    # video stats (vjepa cache)
    vid_cache = RouteCache(use_vjepa=True)
    dv = {s: PassingCtrlDataset(args.task, s, sf, MODALITY_CONFIGS["FV+CV"],
                                horizon=args.horizon, route_cache=vid_cache)
          for s in ["train", "val", "test"]}
    vid_cache.preload(sorted({r for s in dv for r in dv[s].route_ids}),
                      load_front_video=True, load_cabin_video=True)
    Xv = {s: extract_video(dv[s], args.max_train if s == "train" else None) for s in dv}

    # align lengths (max_train cap applies to both via same order)
    for s in ["train", "val", "test"]:
        n = min(len(Xc[s]), len(Xv[s]))
        Xc[s], Xv[s], yc[s] = Xc[s][:n], Xv[s][:n], yc[s][:n]

    yte = yc["test"]; yva = yc["val"]; base = yte.mean()
    te_ds = dca["test"]
    rid, ends, evt = te_ds.route_ids, te_ds.ends, te_ds.event_times

    def ev_ap(scores):
        m = evaluate_event_level(route_ids=rid, end_times=ends, event_times=evt,
                                 y_true=yte, y_scores=scores, horizon=args.horizon,
                                 threshold=0.5)
        return m["event_auprc"]

    from sklearn.linear_model import LogisticRegression
    Xtr = np.hstack([Xc["train"], Xv["train"]]); Xva = np.hstack([Xc["val"], Xv["val"]])
    Xte_b = np.hstack([Xc["test"], Xv["test"]])
    seeds = [int(s) for s in args.seeds.split(",")]
    acc = {k: {"s": [], "e": []} for k in
           ["CAN only", "V-JEPA2 only", "CAN+video (concat)", "CAN+video (avg)", "CAN+video (stack)"]}
    for sd in seeds:
        cv_v, cv_t = xgb_probs(Xc["train"], yc["train"], Xc["val"], yva, Xc["test"], sd)
        vv_v, vv_t = xgb_probs(Xv["train"], yc["train"], Xv["val"], yva, Xv["test"], sd)
        _, both_t = xgb_probs(Xtr, yc["train"], Xva, yva, Xte_b, sd)
        lr = LogisticRegression(max_iter=1000).fit(np.c_[cv_v, vv_v], yva)
        stack_t = lr.predict_proba(np.c_[cv_t, vv_t])[:, 1]
        avg_t = (cv_t + vv_t) / 2
        for name, sc in [("CAN only", cv_t), ("V-JEPA2 only", vv_t),
                         ("CAN+video (concat)", both_t), ("CAN+video (avg)", avg_t),
                         ("CAN+video (stack)", stack_t)]:
            acc[name]["s"].append(average_precision_score(yte, sc))
            acc[name]["e"].append(ev_ap(sc))

    print(f"\n=== CAN + V-JEPA2 fusion: {args.task} (h={args.horizon}), {len(seeds)} seeds ===")
    print(f"test base rate = {base:.3f}   (honest mean +/- std, no seed selection)")
    print(f"{'input':<22}{'sample AUPRC':>18}{'event AUPRC':>18}")
    print("-" * 58)
    for name in acc:
        s = np.array(acc[name]["s"]); e = np.array(acc[name]["e"])
        print(f"{name:<22}{f'{s.mean():.3f}+/-{s.std():.3f}':>18}{f'{e.mean():.3f}+/-{e.std():.3f}':>18}")
    print()


if __name__ == "__main__":
    main()
