#!/usr/bin/env python3
"""
precursor_analysis.py — Temporal-precursor evidence for the handover (T2) leak.

For each candidate leaky field, on a sample of T2 (handover) positive vs negative 5 s
windows, reports:
  (a) precursor prevalence  — % of POSITIVE windows where the field is non-constant
      over the 5 s before the event (the field "moves" ahead of engagement);
  (b) temporal AUPRC        — single-field, using its within-window {mean,std,min,max,
      last,slope} via logistic regression → can the field's *trajectory* predict the
      imminent handover? (binary flags are within-window constant → ≈ base rate;
      actuators_accel should be well above);
  (c) event-aligned mean trajectory (positives vs negatives), saved for a figure.

Usage:  python3 precursor_analysis.py [--max-pos 4000 --neg-ratio 5]
"""
import argparse, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, STRUCT_SEQ_LEN
from dataset import RouteCache
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

OUT = BENCHMARK_DIR.parent / "baseline" / "results_withhold"
# field -> source csv, per task
FIELDS_T2 = {  # handover: ADAS flags + DA-internal + planner
    "cc_latActive": "vehicle_dynamics.csv",
    "cruiseState_enabled": "vehicle_dynamics.csv",
    "cs_longControlState": "vehicle_dynamics.csv",
    "actuators_accel": "vehicle_dynamics.csv",
    "model_desiredAcceleration": "planning.csv",
}
FIELDS_T3 = {  # takeover: driver-override (disengagement trigger) + ADAS flags (reference)
    "brakePressed": "vehicle_dynamics.csv",
    "steeringPressed": "vehicle_dynamics.csv",
    "gasPressed": "vehicle_dynamics.csv",
    "steeringTorque": "vehicle_dynamics.csv",
    "brake": "vehicle_dynamics.csv",
    "gas": "vehicle_dynamics.csv",
    "cc_latActive": "vehicle_dynamics.csv",
    "cruiseState_enabled": "vehicle_dynamics.csv",
}


def stats(seq):  # seq: [T] -> 6 temporal stats
    t = np.arange(len(seq))
    slope = np.polyfit(t, seq, 1)[0] if seq.std() > 1e-9 else 0.0
    return [seq.mean(), seq.std(), seq.min(), seq.max(), seq[-1], slope]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task2", choices=["task2", "task3"])
    ap.add_argument("--max-pos", type=int, default=4000)
    ap.add_argument("--neg-ratio", type=int, default=5)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    global FIELDS
    FIELDS = FIELDS_T2 if args.task == "task2" else FIELDS_T3
    csv_name = ("task2_activation_samples_h3.csv" if args.task == "task2"
                else "task3_takeover_samples_h3.csv")
    print(f"=== precursor analysis: {args.task} ({csv_name}) ===")
    rows = list(csv.DictReader(open(BENCHMARK_DIR / csv_name)))
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]
    rng = np.random.RandomState(0)
    rng.shuffle(pos); rng.shuffle(neg)
    pos = pos[:args.max_pos]
    neg = neg[:args.max_pos * args.neg_ratio]
    samp = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    print(f"sample: {len(pos)} pos / {len(neg)} neg (base rate {y.mean():.3f})")

    cache = RouteCache()
    cache.preload(sorted({r["route_id"] for r in samp}))

    by_src = defaultdict(list)
    for f, src in FIELDS.items():
        by_src[src].append(f)

    # collect per-field: full sequences [N, T]
    seqs = {f: np.zeros((len(samp), STRUCT_SEQ_LEN), np.float32) for f in FIELDS}
    for i, r in enumerate(samp):
        rid, s, e = r["route_id"], float(r["start_time_sec"]), float(r["end_time_sec"])
        for src, flds in by_src.items():
            arr = cache.load_struct_signals(rid, src, flds, s, e)  # [T, len(flds)]
            for j, f in enumerate(flds):
                seqs[f][i] = arr[:, j]

    print(f"\n{'field':<26}{'precursor prevalence':>22}{'temporal AUPRC':>16}{'within-win const%':>18}")
    print("-" * 84)
    results = {}
    for f in FIELDS:
        S = seqs[f]
        nonconst = S.std(axis=1) > 1e-6
        prevalence = float(nonconst[y == 1].mean())          # among positives
        const_rate = float((~nonconst).mean())
        X = np.array([stats(S[i]) for i in range(len(S))])
        X = np.nan_to_num(X)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
        auprc = float(average_precision_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]))
        results[f] = (prevalence, auprc, const_rate)
        print(f"{f:<26}{100*prevalence:>20.1f}%{auprc:>16.3f}{100*const_rate:>16.1f}%")
    print(f"\n(base rate AUPRC = {y.mean():.3f}; temporal AUPRC ≫ base ⇒ the field's "
          "within-window trajectory is a precursor of the imminent handover.)")

    # event-aligned mean trajectory (downsample to 50 pts), positives vs negatives
    ds = np.linspace(0, STRUCT_SEQ_LEN - 1, 50).astype(int)
    with open(OUT / f"precursor_trajectory_{args.task}.csv", "w", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["t_norm"] + [f"{f}_{c}" for f in FIELDS for c in ("pos", "neg")])
        posmean = {f: seqs[f][y == 1].mean(0)[ds] for f in FIELDS}
        negmean = {f: seqs[f][y == 0].mean(0)[ds] for f in FIELDS}
        for k, t in enumerate(ds):
            row = [round(t / (STRUCT_SEQ_LEN - 1), 3)]
            for f in FIELDS:
                row += [round(float(posmean[f][k]), 5), round(float(negmean[f][k]), 5)]
            w.writerow(row)
    print(f"\nWrote event-aligned trajectory → {OUT/'precursor_trajectory.csv'}")


if __name__ == "__main__":
    main()
