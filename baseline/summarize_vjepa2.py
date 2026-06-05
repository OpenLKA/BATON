#!/usr/bin/env python3
"""Summarize the V-JEPA2 modality ablation (results_vjepa2/).

Reports, per task: Full + leave-one-out (with per-type contribution ΔAUPRC =
AUPRC[Full] - AUPRC[no-Type], the marginal value of each modality) and the
single-modality table. Sample- and event-level AUPRC.
"""
import json, glob, os
from collections import defaultdict
import numpy as np

RD = os.path.join(os.path.dirname(__file__), "results_vjepa2")
g = defaultdict(list)
for f in glob.glob(os.path.join(RD, "*", "results.json")):
    r = json.load(open(f))
    g[(r["task"], r["modality"])].append(r)


def agg(task, mod, key):
    vals = [r["test_metrics"].get(key) for r in g.get((task, mod), [])
            if r["test_metrics"].get(key) is not None]
    return (float(np.mean(vals)), float(np.std(vals)), len(vals)) if vals else None


TYPES = ["FrontVideo", "CabinVideo", "Ego", "DrvInput", "Lead", "DMS", "RoadGeom"]

for task in ["task2", "task3"]:
    print(f"\n{'='*84}\n{task}  V-JEPA2 cross-modal Transformer (cross-driver, h=3)\n{'='*84}")
    full = agg(task, "VJEPA-Full", "auprc")
    full_e = agg(task, "VJEPA-Full", "event_auprc")
    if full:
        print(f"VJEPA-Full: sampleAUPRC={full[0]:.3f}±{full[1]:.3f}  "
              f"eventAUPRC={full_e[0]:.3f}±{full_e[1]:.3f}  (n{full[2]})")
    print(f"\nLeave-one-out — contribution ΔAUPRC = Full − (no-Type):")
    print(f"{'removed type':<14}{'sampleAUPRC':>14}{'Δsample':>10}{'eventAUPRC':>14}{'Δevent':>10}")
    print("-" * 62)
    for t in TYPES:
        a = agg(task, f"VJEPA-no-{t}", "auprc")
        e = agg(task, f"VJEPA-no-{t}", "event_auprc")
        if not a:
            continue
        ds = (full[0] - a[0]) if full else 0.0
        de = (full_e[0] - e[0]) if full_e else 0.0
        print(f"{t:<14}{a[0]:>9.3f}±{a[1]:.2f}{ds:>+10.3f}"
              f"{e[0]:>9.3f}±{e[1]:.2f}{de:>+10.3f}")
    print(f"\nSingle-modality (1 seed):")
    print(f"{'type':<14}{'sampleAUPRC':>14}{'eventAUPRC':>14}")
    print("-" * 42)
    for t in TYPES:
        a = agg(task, f"VJEPA-only-{t}", "auprc")
        e = agg(task, f"VJEPA-only-{t}", "event_auprc")
        if a:
            print(f"{t:<14}{a[0]:>14.3f}{e[0]:>14.3f}")
