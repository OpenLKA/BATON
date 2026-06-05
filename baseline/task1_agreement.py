#!/usr/bin/env python3
"""
task1_agreement.py — Rule-label vs human-annotation agreement for Task 1 validation.

Reads the annotations.csv exported from the HTML kit (idx, sample_id, route_id,
rule_label, human_label, agree) and reports overall accuracy, Cohen's kappa, and
per-class precision/recall/F1 of the RULE labels treating HUMAN labels as ground truth
(LaneChange highlighted). Only labeled rows (non-empty human_label) are scored.

Usage:  python3 task1_agreement.py /path/to/annotations.csv
"""
import csv, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report, confusion_matrix, accuracy_score

sys.path.insert(0, str(Path(__file__).parent))
from config import TASK1_LABELS


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "annotations.csv"
    rows = [r for r in csv.DictReader(open(path)) if r.get("human_label")]
    if not rows:
        print("No labeled rows found."); return
    rule = [r["rule_label"] for r in rows]
    human = [r["human_label"] for r in rows]
    labels = TASK1_LABELS

    acc = accuracy_score(human, rule)
    kappa = cohen_kappa_score(human, rule, labels=labels)
    print(f"Labeled windows: {len(rows)}")
    print(f"Overall rule↔human agreement (accuracy): {acc:.3f}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print("\nPer-class (rule labels vs human as ground truth):")
    print(classification_report(human, rule, labels=labels, zero_division=0, digits=3))

    cm = confusion_matrix(human, rule, labels=labels)
    print("Confusion matrix (rows=human, cols=rule):")
    print("      " + " ".join(f"{l[:5]:>6}" for l in labels))
    for i, l in enumerate(labels):
        print(f"{l[:5]:>5} " + " ".join(f"{cm[i,j]:>6}" for j in range(len(labels))))

    # LaneChange highlight (reviewers cite 0.925 F1)
    if "LaneChange" in labels:
        j = labels.index("LaneChange")
        tp = cm[j, j]; fp = cm[:, j].sum() - tp; fn = cm[j, :].sum() - tp
        p = tp / (tp + fp) if tp + fp else 0; r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        print(f"\nLaneChange: precision={p:.3f} recall={r:.3f} F1={f1:.3f} "
              f"(human-confirmed rule LaneChanges)")


if __name__ == "__main__":
    main()
