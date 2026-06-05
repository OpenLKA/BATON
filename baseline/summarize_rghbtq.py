#!/usr/bin/env python3
"""
summarize_rghbtq.py — aggregate RG-HBT-Q seeds into the §3 comparison rows.

Scans results_rghbtq/*/results.json, groups by (task, protocol), and prints
mean ± std of sample-level and event-level AUPRC across seeds, alongside the
existing baselines for context.
"""
import json
import glob
import os
from collections import defaultdict
import numpy as np

RD = os.path.join(os.path.dirname(__file__), "results_rghbtq")

# (task, protocol) -> human label, and the existing-baseline reference numbers
ROWS = [
    (("task2", "std"),     "Handover"),
    (("task3", "std"),     "Takeover - detection"),
    (("task3", "antsafe"), "Takeover - anticipation"),
]
REF = {  # sample AUPRC of the prior baselines (for context)
    ("task2", "std"):     {"XGBoost": 0.222, "GRU": 0.202, "Transformer": 0.242},
    ("task3", "std"):     {"XGBoost": 0.478, "GRU": 0.280, "Transformer": 0.316},
    ("task3", "antsafe"): {"XGBoost": 0.080, "GRU": 0.077, "Transformer": 0.071},
}


def fmt(vals):
    a = np.array([v for v in vals if v is not None], dtype=float)
    return f"{a.mean():.3f} ± {a.std():.3f}" if len(a) else "n/a"


def main():
    groups = defaultdict(list)
    for f in glob.glob(os.path.join(RD, "*", "results.json")):
        d = json.load(open(f))
        rn = d["run_name"]
        proto = "antsafe" if "_antsafe" in rn else "std"
        tm = d["test_metrics"]
        groups[(d["task"], proto)].append((tm.get("auprc"), tm.get("event_auprc")))

    print(f"\n{'Setting':<26}{'n':>3}{'RG-HBT-Q sample':>20}{'RG-HBT-Q event':>18}"
          f"{'Transf.':>10}{'GRU':>8}{'XGB':>8}")
    print("-" * 93)
    for key, name in ROWS:
        g = groups.get(key, [])
        s = [x[0] for x in g]
        e = [x[1] for x in g]
        ref = REF[key]
        print(f"{name:<26}{len(g):>3}{fmt(s):>20}{fmt(e):>18}"
              f"{ref['Transformer']:>10.3f}{ref['GRU']:>8.3f}{ref['XGBoost']:>8.3f}")
    print()


if __name__ == "__main__":
    main()
