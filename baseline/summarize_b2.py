#!/usr/bin/env python3
"""Summarize B2 leakage-safe + ablation results from results_b2/."""
import json, glob, os
from collections import defaultdict
import numpy as np

RD = os.path.join(os.path.dirname(__file__), "results_b2")

rows = []
for f in glob.glob(os.path.join(RD, "*", "results.json")):
    rows.append(json.load(open(f)))

# group by (task, modality) → list of seed results
g = defaultdict(list)
for r in rows:
    g[(r["task"], r["modality"])].append(r)


def agg(task, mod, key):
    vals = [r["test_metrics"].get(key) for r in g.get((task, mod), [])
            if r["test_metrics"].get(key) is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


LADDER = ["ADASctrl-only", "Full-Struct", "Safe-Full-Struct",
          "Safe-Ego", "Safe-Ego+Drv_in", "Safe-Ego+Drv_in+Lead",
          "Safe-Ego+Drv_in+Lead+Road", "Safe-Ego+Drv_in+Lead+Road+DMS"]

for task in ["task2", "task3"]:
    print(f"\n{'='*92}\n{task}  (cross-driver, h=3, GRU)\n{'='*92}")
    print(f"{'modality':<34} {'sampleAUPRC':>14} {'eventAUPRC':>14} {'evtDetRec':>10} {'medLead(s)':>11}")
    print("-" * 92)
    for mod in LADDER:
        sa = agg(task, mod, "auprc")
        ea = agg(task, mod, "event_auprc")
        dr = agg(task, mod, "event_detection_recall")
        ml = agg(task, mod, "event_median_lead_time_s")
        if sa is None:
            continue
        def fmt(x):
            return f"{x[0]:.3f}±{x[1]:.3f}(n{x[2]})" if x else "    -    "
        tag = " [LEAKY]" if mod in ("ADASctrl-only", "Full-Struct") else ""
        print(f"{mod:<34} {fmt(sa):>14} {fmt(ea):>14} "
              f"{dr[0]:>10.3f} {ml[0]:>11.2f}{tag}")

# Headline leak delta
print(f"\n{'='*60}\nHEADLINE: leaky Full-Struct vs leak-safe Safe-Full-Struct\n{'='*60}")
for task in ["task2", "task3"]:
    fs = agg(task, "Full-Struct", "auprc")
    ss = agg(task, "Safe-Full-Struct", "auprc")
    fe = agg(task, "Full-Struct", "event_auprc")
    se = agg(task, "Safe-Full-Struct", "event_auprc")
    if fs and ss:
        print(f"{task}: sampleAUPRC {fs[0]:.3f}→{ss[0]:.3f} (Δ{ss[0]-fs[0]:+.3f}) | "
              f"eventAUPRC {fe[0]:.3f}→{se[0]:.3f} (Δ{se[0]-fe[0]:+.3f})")
