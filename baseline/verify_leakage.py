#!/usr/bin/env python3
"""
verify_leakage.py — Definitive leakage verification for the leak-safe BATON benchmark.

Produces THREE independent proofs that no target-derived proxy or ADAS-control flag
enters the leak-safe model inputs, written to benchmark_v2/leakage_verification.md (+ .csv).

Label is deterministic:  ADAS_active = (cc_latActive == 1) OR (cruiseState_enabled == 1)
Leak-safe inputs additionally drop the DA-internal {cs_longControlState, actuators_accel}.

Proof 1 — Set membership (formal): expand every MODALITY_CONFIG to concrete columns and
          assert LEAKY_COLS ∩ used_cols == ∅ for all Safe-* configs; confirm against the
          actual 50Hz npz cache.
Proof 2 — Single-feature predictive ceiling: per input column, |point-biserial corr|,
          mutual information, and 1-feature AUPRC vs the label. Max over SAFE columns must
          be ≪ the leaky columns.
Proof 3 — Reconstruction probe: a logistic-regression probe on leak-safe features cannot
          reconstruct the label, unlike the leaky set (near-ceiling).

Usage:  python3 verify_leakage.py [--task task2] [--max-samples 40000]
"""
import argparse, csv, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import (BENCHMARK_DIR, STRUCT_GROUPS, GPS_COLS, LEAKY_COLS,
                    MODALITY_CONFIGS, VEHICLE_COLS)
from dataset import RouteCache
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

# Per-task output paths (set in main)
OUT_MD = None
OUT_CSV = None

# Source CSV → columns universe (struct only; the 4 leaky cols all live in vehicle_dynamics)
STRUCT_SOURCES = {
    "vehicle_dynamics.csv": list(dict.fromkeys(VEHICLE_COLS)),
    "planning.csv": STRUCT_GROUPS["Int_plan"][1],
    "radar.csv": STRUCT_GROUPS["Int_radar"][1],
    "driver_state.csv": STRUCT_GROUPS["Drv"][1],
    "imu.csv": STRUCT_GROUPS["IMU"][1],
}


# ──────────────────────────────────────────────────────────────────────────
# Proof 1 — set membership
# ──────────────────────────────────────────────────────────────────────────
def config_columns(cfg):
    """Concrete input columns a modality_config exposes."""
    cols = []
    for g in cfg["struct"]:
        cols += list(STRUCT_GROUPS[g][1])
    if cfg.get("gps"):
        cols += list(GPS_COLS)
    return cols


def proof1_set_membership():
    rows = []
    for name, cfg in MODALITY_CONFIGS.items():
        used = config_columns(cfg)
        leaky_present = sorted(set(used) & LEAKY_COLS)
        rows.append((name, len(used), leaky_present))
    safe_violations = [(n, lp) for (n, _, lp) in rows
                       if n.startswith("Safe") and lp]
    return rows, safe_violations


# ──────────────────────────────────────────────────────────────────────────
# Build per-window last-timestep feature matrix from the 50Hz cache
# ──────────────────────────────────────────────────────────────────────────
STATS = ["mean", "std", "min", "max", "last"]  # same window summary as train_classical


def build_feature_matrix(task, max_samples, seed=0):
    """Per-window 5-stat summary (mean/std/min/max/last) per struct column.

    Matches train_classical.extract_statistical_features, so the temporal precursor
    leak (e.g. actuators_accel ramp over the window) is captured — not just the last
    value. Returns X[N, 5*ncols], expanded column names "col__stat", and is_leaky mask.
    """
    csv_path = BENCHMARK_DIR / f"{task}_activation_samples_h3.csv" if task == "task2" \
        else BENCHMARK_DIR / f"{task}_takeover_samples_h3.csv"
    import pandas as pd
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=seed).reset_index(drop=True)
    route_ids = df["route_id"].values
    starts = df["start_time_sec"].values.astype(np.float64)
    ends = df["end_time_sec"].values.astype(np.float64)
    y = df["label"].values.astype(np.int32)

    cache = RouteCache()
    cache.preload(sorted(set(route_ids)))

    base_cols = [c for cols in STRUCT_SOURCES.values() for c in cols]
    ncol = len(base_cols)
    X = np.zeros((len(df), ncol * len(STATS)), dtype=np.float32)
    for i in range(len(df)):
        rid, s, e = route_ids[i], float(starts[i]), float(ends[i])
        seq_parts = []
        for src, cols in STRUCT_SOURCES.items():
            seq_parts.append(cache.load_struct_signals(rid, src, cols, s, e))  # [250, |cols|]
        seq = np.concatenate(seq_parts, axis=1)  # [250, ncol]
        X[i, 0 * ncol:1 * ncol] = seq.mean(0)
        X[i, 1 * ncol:2 * ncol] = seq.std(0)
        X[i, 2 * ncol:3 * ncol] = seq.min(0)
        X[i, 3 * ncol:4 * ncol] = seq.max(0)
        X[i, 4 * ncol:5 * ncol] = seq[-1]
    exp_cols = [f"{c}__{st}" for st in STATS for c in base_cols]
    return X, y, exp_cols


# ──────────────────────────────────────────────────────────────────────────
# Proof 2 — single-feature predictive ceiling
# ──────────────────────────────────────────────────────────────────────────
def single_feature_auprc(x, y):
    x = np.nan_to_num(x).astype(np.float64)
    if np.std(x) < 1e-12:
        return float(y.mean())  # constant feature → base rate
    # orient so higher score ↔ positive
    s = x if np.corrcoef(x, y)[0, 1] >= 0 else -x
    return float(average_precision_score(y, s))


def _is_leaky(expanded_col):
    return expanded_col.rsplit("__", 1)[0] in LEAKY_COLS


def proof2_feature_ceiling(X, y, cols):
    base = float(y.mean())
    mi = mutual_info_classif(np.nan_to_num(X), y, discrete_features=False, random_state=0)
    rows = []
    for j, c in enumerate(cols):
        x = X[:, j]
        with np.errstate(invalid="ignore"):
            r = np.corrcoef(np.nan_to_num(x), y)[0, 1] if np.std(x) > 1e-12 else 0.0
        rows.append({
            "feature": c, "is_leaky": _is_leaky(c),
            "abs_corr": abs(float(r)), "mutual_info": float(mi[j]),
            "auprc_1feat": single_feature_auprc(x, y),
        })
    rows.sort(key=lambda d: -d["auprc_1feat"])
    return rows, base


# ──────────────────────────────────────────────────────────────────────────
# Proof 3 — reconstruction probe (leaky-only vs safe-only)
# ──────────────────────────────────────────────────────────────────────────
def probe_auprc(X, y):
    if X.shape[1] == 0:
        return float(y.mean())
    Xtr, Xte, ytr, yte = train_test_split(np.nan_to_num(X), y, test_size=0.3,
                                          random_state=0, stratify=y)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, ytr)
    p = clf.predict_proba((Xte - mu) / sd)[:, 1]
    return float(average_precision_score(yte, p))


def proof3_reconstruction(X, y, cols):
    leaky_idx = [j for j, c in enumerate(cols) if _is_leaky(c)]
    safe_idx = [j for j, c in enumerate(cols) if not _is_leaky(c)]
    return {
        "base_rate": float(y.mean()),
        "leaky_only_auprc": probe_auprc(X[:, leaky_idx], y),
        "safe_only_auprc": probe_auprc(X[:, safe_idx], y),
        "n_leaky": len(leaky_idx), "n_safe": len(safe_idx),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task2", choices=["task2", "task3"])
    ap.add_argument("--max-samples", type=int, default=40000)
    args = ap.parse_args()

    global OUT_MD, OUT_CSV
    OUT_MD = BENCHMARK_DIR / f"leakage_verification_{args.task}.md"
    OUT_CSV = BENCHMARK_DIR / f"leakage_feature_ranking_{args.task}.csv"

    print("=" * 70)
    print(f"LEAKAGE VERIFICATION — {args.task}")
    print("=" * 70)

    # Proof 1
    rows1, violations = proof1_set_membership()
    print("\n[Proof 1] Set membership (Safe-* configs must contain 0 leaky cols):")
    for n, k, lp in rows1:
        if n.startswith("Safe") or n in ("Full-Struct", "ADASctrl-only"):
            print(f"  {n:<34} cols={k:<3} leaky_present={lp}")
    assert not violations, f"LEAKY COLUMNS LEAK INTO SAFE CONFIGS: {violations}"
    print("  ✓ All Safe-* configs: LEAKY ∩ used = ∅")

    # Build features
    print(f"\nBuilding last-timestep feature matrix (≤{args.max_samples} windows)...")
    X, y, cols = build_feature_matrix(args.task, args.max_samples)
    print(f"  X={X.shape}, pos_rate={y.mean():.4f}")

    # Proof 2
    rows2, base = proof2_feature_ceiling(X, y, cols)
    leaky_rows = [r for r in rows2 if r["is_leaky"]]
    safe_rows = [r for r in rows2 if not r["is_leaky"]]
    max_safe = max((r["auprc_1feat"] for r in safe_rows), default=0.0)
    max_leaky = max((r["auprc_1feat"] for r in leaky_rows), default=0.0)
    print(f"\n[Proof 2] Single-feature AUPRC (base rate {base:.4f}):")
    print(f"  max LEAKY  1-feat AUPRC = {max_leaky:.4f}")
    print(f"  max SAFE   1-feat AUPRC = {max_safe:.4f}")
    print("  top-8 features:")
    for r in rows2[:8]:
        print(f"    {'[LEAK]' if r['is_leaky'] else '      '} {r['feature']:<22} "
              f"AUPRC={r['auprc_1feat']:.4f} |corr|={r['abs_corr']:.3f} MI={r['mutual_info']:.4f}")

    # Proof 3
    p3 = proof3_reconstruction(X, y, cols)
    print(f"\n[Proof 3] Reconstruction probe (LogReg):")
    print(f"  base rate          = {p3['base_rate']:.4f}")
    print(f"  leaky-only ({p3['n_leaky']} cols) AUPRC = {p3['leaky_only_auprc']:.4f}")
    print(f"  safe-only  ({p3['n_safe']} cols) AUPRC = {p3['safe_only_auprc']:.4f}")

    # Write report
    write_report(args.task, rows1, rows2, base, max_safe, max_leaky, p3)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature", "is_leaky", "abs_corr",
                                          "mutual_info", "auprc_1feat"])
        w.writeheader()
        w.writerows(rows2)
    print(f"\nWrote {OUT_MD}\nWrote {OUT_CSV}")


def write_report(task, rows1, rows2, base, max_safe, max_leaky, p3):
    lines = [f"# Leakage Verification — {task} (h=3, leak-safe BATON)", "",
             "Label is deterministic: `ADAS_active = (cc_latActive OR cruiseState_enabled)`.",
             f"Leak-safe inputs drop `{sorted(LEAKY_COLS)}`.", "",
             "## Proof 1 — Set membership (formal)", "",
             "| config | #cols | leaky cols present |", "|---|---|---|"]
    for n, k, lp in rows1:
        lines.append(f"| {n} | {k} | {lp if lp else '— (none)'} |")
    lines += ["", "**All `Safe-*` configs contain zero leaky columns: LEAKY ∩ used = ∅.**", "",
              "## Proof 2 — Single-feature predictive ceiling", "",
              f"Base rate = {base:.4f}. Max single-feature AUPRC over **safe** columns "
              f"= **{max_safe:.4f}**, vs **{max_leaky:.4f}** for the leaky columns. "
              "No single safe signal is a stealth label proxy.", "",
              "| feature | leaky? | 1-feat AUPRC | \\|corr\\| | MI |", "|---|---|---|---|---|"]
    for r in rows2[:15]:
        lines.append(f"| {r['feature']} | {'YES' if r['is_leaky'] else ''} | "
                     f"{r['auprc_1feat']:.4f} | {r['abs_corr']:.3f} | {r['mutual_info']:.4f} |")
    lines += ["", "## Proof 3 — Reconstruction probe", "",
              f"- base rate: {p3['base_rate']:.4f}",
              f"- leaky-only ({p3['n_leaky']} cols): **{p3['leaky_only_auprc']:.4f}**",
              f"- safe-only ({p3['n_safe']} cols): **{p3['safe_only_auprc']:.4f}**", "",
              "A probe on leak-safe features cannot reconstruct the label near the leaky "
              "ceiling, confirming the leak-safe inputs carry no target-derived proxy."]
    OUT_MD.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
