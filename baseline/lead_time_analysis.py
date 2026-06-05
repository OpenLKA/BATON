#!/usr/bin/env python3
"""
lead_time_analysis.py — average prediction lead time of the multimodal FUSION model.

Trains the late-fusion model (CAN-XGB + V-JEPA2-XGB, logistic meta) in the
no-override setting, then for each test event scores sliding windows ending at
ev - Delta for Delta in [0, 5] s. Reports the detection-recall@Delta curve and
the mean/median sustained anticipation time (how many seconds before the event
the alarm reliably comes on).
"""
import argparse
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (BENCHMARK_DIR, MODALITY_CONFIGS, STRUCT_GROUPS, CACHE_DIR,
                    STRUCT_SEQ_LEN, INPUT_WINDOW_SEC)
from dataset import PassingCtrlDataset, RouteCache, compute_norm_stats
from train_classical import extract_statistical_features
from metrics import find_optimal_f1_threshold
from diag_vjepa2_fusion import vid_stats, extract_video
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

CAN_MOD = "DI-Safe-Struct-noOver"
VID_MOD = "FV+CV"


def fit_xgb(Xtr, ytr, Xva, yva, seed=0):
    spw = (len(ytr) - ytr.sum()) / max(ytr.sum(), 1)
    m = XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=min(spw, 10),
                      eval_metric="logloss", early_stopping_rounds=20,
                      tree_method="hist", device="cuda", n_jobs=-1, random_state=seed)
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task3")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--max-lead", type=float, default=5.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--max-train", type=int, default=60000)
    ap.add_argument("--target-recall", type=float, default=0.0,
                    help="if >0, operate at the val threshold giving this recall "
                         "(else F1-optimal)")
    args = ap.parse_args()
    sf = BENCHMARK_DIR / "split_cross_driver.json"

    can_cfg = MODALITY_CONFIGS[CAN_MOD]
    sources = [STRUCT_GROUPS[g] for g in can_cfg["struct"]]      # [(csv, cols), ...]
    D = sum(len(c) for _, c in sources)
    npz = CACHE_DIR / f"norm_{CAN_MOD}_{args.task}_cross_driver_h{args.horizon}.npz"
    if npz.exists():
        nd = np.load(npz); nmean, nstd = nd["mean"][:D], nd["std"][:D]
    else:
        ns = compute_norm_stats(args.task, sf, can_cfg, args.horizon)
        nmean, nstd = ns["mean"][:D], ns["std"][:D]

    # ---- train fusion (CAN-XGB + video-XGB + logistic meta), threshold on val ----
    can_cache = RouteCache()
    dca = {s: PassingCtrlDataset(args.task, s, sf, can_cfg, horizon=args.horizon,
                                 route_cache=can_cache) for s in ["train", "val", "test"]}
    can_cache.preload(sorted({r for s in dca for r in dca[s].route_ids}))
    Xc, yc = {}, {}
    for s in dca:
        Xc[s], yc[s] = extract_statistical_features(dca[s],
                        max_samples=args.max_train if s == "train" else None)
        np.nan_to_num(Xc[s], copy=False)

    vid_cache = RouteCache(use_vjepa=True)
    dv = {s: PassingCtrlDataset(args.task, s, sf, MODALITY_CONFIGS[VID_MOD],
                                horizon=args.horizon, route_cache=vid_cache)
          for s in ["train", "val"]}
    vid_cache.preload(sorted({r for s in dv for r in dv[s].route_ids}),
                      load_front_video=True, load_cabin_video=True)
    Xv = {s: extract_video(dv[s], args.max_train if s == "train" else None) for s in dv}
    n = {s: min(len(Xc[s]), len(Xv[s])) for s in ["train", "val"]}
    for s in ["train", "val"]:
        Xc[s], Xv[s], yc[s] = Xc[s][:n[s]], Xv[s][:n[s]], yc[s][:n[s]]

    can_m = fit_xgb(Xc["train"], yc["train"], Xc["val"], yc["val"])
    vid_m = fit_xgb(Xv["train"], yc["train"], Xv["val"], yc["val"])
    pc_v = can_m.predict_proba(Xc["val"])[:, 1]; pv_v = vid_m.predict_proba(Xv["val"])[:, 1]
    meta = LogisticRegression(max_iter=1000).fit(np.c_[pc_v, pv_v], yc["val"])
    fused_val = meta.predict_proba(np.c_[pc_v, pv_v])[:, 1]
    yv = np.asarray(yc["val"])
    if args.target_recall > 0:
        thr = float(np.quantile(fused_val[yv == 1], 1 - args.target_recall))
        print(f"\n[{args.task} h{args.horizon}] fusion trained. "
              f"val threshold @ recall~{args.target_recall} = {thr:.3f}")
    else:
        thr, _ = find_optimal_f1_threshold(yv, fused_val)
        print(f"\n[{args.task} h{args.horizon}] fusion trained. val F1-opt threshold = {thr:.3f}")

    # ---- per-event sliding-window scoring on the test set ----
    test_ds = dca["test"]
    # route time bounds (from cached vehicle_dynamics)
    def route_bounds(rid):
        rd = can_cache._load_struct_npz(rid)
        t0, step, data, _ = rd["vehicle_dynamics.csv"]
        return t0, t0 + (data.shape[0] - 1) * step

    # unique positive events
    evset = {}
    for rid, et in zip(test_ds.route_ids, test_ds.event_times):
        if np.isfinite(et):
            evset[(rid, round(float(et), 2))] = (rid, float(et))
    events = list(evset.values())

    def can_feat(rid, t):
        parts = [can_cache.load_struct_signals(rid, csv, cols, t - INPUT_WINDOW_SEC, t)
                 for csv, cols in sources]
        x = np.concatenate(parts, axis=1)
        x = (x - nmean) / (nstd + 1e-8)
        return np.concatenate([x.mean(0), x.std(0), x.min(0), x.max(0), x[-1]])

    def vid_feat(rid, t):
        fv = vid_cache.load_video_features("front", rid, t - INPUT_WINDOW_SEC, t)
        cv = vid_cache.load_video_features("cabin", rid, t - INPUT_WINDOW_SEC, t)
        return np.concatenate([vid_stats(fv), vid_stats(cv)])

    vid_cache.preload(sorted({r for r, _ in events}),
                      load_front_video=True, load_cabin_video=True)
    deltas = np.arange(0, args.max_lead + 1e-9, args.step)
    # score[event_idx, delta_idx] = fused prob (nan if window invalid)
    scores = np.full((len(events), len(deltas)), np.nan)
    for i, (rid, et) in enumerate(events):
        t0, t1 = route_bounds(rid)
        cf, vf, valid = [], [], []
        for d in deltas:
            t = et - d
            ok = (t - INPUT_WINDOW_SEC >= t0) and (t <= t1)
            valid.append(ok)
            cf.append(can_feat(rid, t) if ok else np.zeros(5 * D))
            vf.append(vid_feat(rid, t) if ok else np.zeros(8192))
        cf = np.array(cf); vf = np.array(vf)
        pc = can_m.predict_proba(cf)[:, 1]; pv = vid_m.predict_proba(vf)[:, 1]
        fp = meta.predict_proba(np.c_[pc, pv])[:, 1]
        for j, ok in enumerate(valid):
            if ok:
                scores[i, j] = fp[j]

    fired = scores > thr
    # detection recall at each lead Delta (over events with a valid window there)
    print(f"\n=== Prediction lead time: {args.task} (fusion, no-override) ===")
    print(f"events = {len(events)}   max-lead = {args.max_lead}s   step = {args.step}s")
    print(f"{'lead Δ (s)':>10}{'recall@Δ':>12}{'n valid':>10}")
    for j, d in enumerate(deltas):
        col = scores[:, j]; vmask = np.isfinite(col)
        rec = (col[vmask] > thr).mean() if vmask.any() else float("nan")
        print(f"{d:>10.1f}{rec:>12.3f}{int(vmask.sum()):>10}")

    # sustained anticipation time: largest Δ with continuous firing from event back to Δ
    leads = []
    for i in range(len(events)):
        if not (np.isfinite(scores[i, 0]) and fired[i, 0]):
            continue                                   # not detected at/near event
        lead = 0.0
        for j in range(1, len(deltas)):
            if np.isfinite(scores[i, j]) and fired[i, j]:
                lead = deltas[j]
            else:
                break
        leads.append(lead)
    leads = np.array(leads)
    det_rate = len(leads) / len(events)
    print(f"\ndetected events (fire within window of event): {len(leads)}/{len(events)} "
          f"({det_rate:.1%})")
    if len(leads):
        print(f"mean sustained anticipation time   = {leads.mean():.2f} s")
        print(f"median sustained anticipation time = {np.median(leads):.2f} s")
        print(f"(capped at max-lead {args.max_lead}s)\n")


if __name__ == "__main__":
    main()
