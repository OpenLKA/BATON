#!/usr/bin/env python3
"""
task1_threshold_sensitivity.py (B) — Robustness of Task-1 labels to rule thresholds.

Re-derives the 1 Hz action labels on a route sample at ±20% / ±30% on each rule
threshold (reusing the official assign_action_1hz / load_route_data / resample_to_1hz),
and reports, per perturbation: the % of 1 Hz labels that FLIP vs the default thresholds
and the label-distribution shift. Shows the benchmark is not an artifact of arbitrary
cutoffs. Label-recompute only (no training).

Usage:  python3 task1_threshold_sensitivity.py [--n-routes 40]
"""
import argparse, sys
from pathlib import Path
from collections import Counter
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_benchmark as gb
from paths import discover_segments

THRESHOLDS = ["TURN_THRESH", "ACCEL_THRESH", "BRAKE_THRESH", "STOPPED_SPEED", "LEAD_DIST_THRESH"]


def label_route(seg):
    """Return the default-threshold 1 Hz label list for a route, plus the resampled inputs."""
    data = gb.load_route_data(seg)
    if data is None or len(data["times"]) < 10:
        return None
    rs = gb.resample_to_1hz(
        data["times"], {"vego": data["vego"], "aego": data["aego"],
                        "steer": data["steer"], "brake_pressed": data["brake_pressed"],
                        "blinker_l": data["blinker_l"], "blinker_r": data["blinker_r"]},
        data["lcs_times"], data["lcs_vals"],
        data["lead_times"], data["lead_status"], data["lead_drel"],
        data["blinker_l"], data["blinker_r"])
    if not rs[0]:
        return None
    return rs


def labels_at(rs, **overrides):
    """Compute 1 Hz labels with temporarily overridden thresholds."""
    saved = {k: getattr(gb, k) for k in overrides}
    for k, v in overrides.items():
        setattr(gb, k, v)
    try:
        labels, _ = gb.assign_action_1hz(*rs)
    finally:
        for k, v in saved.items():
            setattr(gb, k, v)
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-routes", type=int, default=40)
    args = ap.parse_args()

    rng = np.random.RandomState(0)
    segs = discover_segments()
    rng.shuffle(segs)
    samples = []
    for seg in segs:
        rs = label_route(seg)
        if rs is not None:
            samples.append(rs)
        if len(samples) >= args.n_routes:
            break
    print(f"Sampled {len(samples)} routes for threshold sensitivity.\n")

    base = [labels_at(rs) for rs in samples]
    base_flat = [l for seq in base for l in seq]
    base_dist = Counter(base_flat)
    n = len(base_flat)
    print(f"Default-threshold label distribution ({n} 1Hz windows):")
    for k, v in base_dist.most_common():
        print(f"  {k:<13} {100*v/n:5.1f}%")
    print()

    print(f"{'threshold':<18}{'perturb':>9}{'value':>10}{'label-flip%':>13}")
    print("-" * 50)
    for thr in THRESHOLDS:
        default = getattr(gb, thr)
        for frac in (-0.30, -0.20, 0.20, 0.30):
            val = default * (1 + frac)
            flips = 0
            for rs, b in zip(samples, base):
                new = labels_at(rs, **{thr: val})
                flips += sum(1 for x, y in zip(b, new) if x != y)
            print(f"{thr:<18}{frac*100:>+8.0f}%{val:>10.3f}{100*flips/n:>12.2f}%")
    print("\nLower flip% = labels are robust to that threshold's exact value.")


if __name__ == "__main__":
    main()
