#!/usr/bin/env python3
"""
collect_results.py — Aggregate results from all baseline runs into summary tables.

Outputs:
  Table 1: Modality ablation (GRU, cross-driver, h=3)
  Table 2: Model comparison (Full-Struct, cross-driver, h=3)
  Table S1: Supplementary single-modality (Ctx, IMU, FV, CV, FV+CV)
  Table S2: Split comparison (cross-driver vs cross-vehicle vs random)
  Table S3: Horizon sensitivity (h=1, h=3, h=5)

Usage:
  python3 collect_results.py
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"

# Metrics to exclude from aggregation (not meaningful across seeds)
EXCLUDE_METRICS = {"n_pos", "n_neg", "threshold"}


def load_all_results():
    """Load all results.json files."""
    results = []
    for rdir in sorted(RESULTS_DIR.iterdir()):
        rfile = rdir / "results.json"
        if rfile.exists():
            with open(rfile) as f:
                results.append(json.load(f))
    return results


def aggregate_seeds(results):
    """Group by (task, modality, model, split, horizon) and compute mean +/- std across seeds."""
    df = pd.DataFrame(results)
    if df.empty:
        return df

    # Flatten test_metrics into columns
    test_cols = pd.json_normalize(df["test_metrics"])
    test_cols.columns = [f"test_{c}" for c in test_cols.columns]

    val_cols = pd.json_normalize(df["val_metrics"])
    val_cols.columns = [f"val_{c}" for c in val_cols.columns]

    df = pd.concat([df.drop(columns=["test_metrics", "val_metrics"]), test_cols, val_cols], axis=1)

    group_keys = ["task", "modality", "model", "split", "horizon"]
    metric_cols = [c for c in df.columns
                   if (c.startswith("test_") or c.startswith("val_"))
                   and not any(ex in c for ex in EXCLUDE_METRICS)]

    rows = []
    for key, group in df.groupby(group_keys):
        row = dict(zip(group_keys, key))
        row["n_seeds"] = len(group)
        for mc in metric_cols:
            vals = group[mc].dropna().values
            if len(vals) > 0 and isinstance(vals[0], (int, float, np.integer, np.floating)):
                row[f"{mc}_mean"] = float(np.mean(vals))
                row[f"{mc}_std"] = float(np.std(vals))
        rows.append(row)

    return pd.DataFrame(rows)


def fmt(mean, std, f=".4f"):
    """Format as mean +/- std."""
    if pd.isna(mean) or pd.isna(std):
        return "—"
    return f"{mean:{f}} +/- {std:{f}}"


def get_metric(agg, task, modality, model, split, horizon, metric_key):
    """Get formatted mean+/-std for a specific experiment config."""
    sub = agg[(agg["task"] == task) & (agg["modality"] == modality) &
              (agg["model"] == model) & (agg["split"] == split) &
              (agg["horizon"] == horizon)]
    if len(sub) == 0:
        return "—", 0
    mean_col = f"test_{metric_key}_mean"
    std_col = f"test_{metric_key}_std"
    if mean_col not in sub.columns:
        return "—", 0
    m = sub[mean_col].values[0]
    s = sub[std_col].values[0]
    if pd.isna(m):
        return "—", 0
    return fmt(m, s), int(sub["n_seeds"].values[0])


def print_modality_table(agg, config_order, title, model="gru", split="cross_driver", horizon=3):
    """Generic modality ablation table."""
    print(f"\n{'='*95}")
    print(title)
    print(f"{'='*95}")

    header = f"{'Config':<20} | {'n':>1} | {'T1 Macro-F1':>20} | {'T2 AUC-ROC':>20} | {'T3 AUC-ROC':>20}"
    print(header)
    print("-" * len(header))

    for cfg in config_order:
        t1, n1 = get_metric(agg, "task1", cfg, model, split, horizon, "macro_f1")
        t2, n2 = get_metric(agg, "task2", cfg, model, split, horizon, "auc_roc")
        t3, n3 = get_metric(agg, "task3", cfg, model, split, horizon, "auc_roc")
        n = max(n1, n2, n3)
        if n == 0:
            continue
        print(f"{cfg:<20} | {n:>1} | {t1:>20} | {t2:>20} | {t3:>20}")


def print_table1(agg):
    """Table 1: Main modality ablation."""
    config_order = [
        "Veh", "Drv", "Int", "Veh+Int", "Veh+Int+Drv",
        "Full-Struct", "Full-Struct+FV", "Full-Struct+CV", "Full-Multimodal",
    ]
    print_modality_table(
        agg, config_order,
        "TABLE 1: Modality Ablation (GRU, cross-driver, h=3s)",
    )


def print_table2(agg):
    """Table 2: Model comparison."""
    print(f"\n{'='*95}")
    print("TABLE 2: Model Comparison (Full-Struct, cross-driver, h=3s)")
    print(f"{'='*95}")

    model_order = [("lr", "Logistic Reg."), ("xgb", "XGBoost"),
                   ("gru", "GRU"), ("tcn", "TCN")]

    header = f"{'Model':<20} | {'n':>1} | {'T1 Macro-F1':>20} | {'T2 AUC-ROC':>20} | {'T3 AUC-ROC':>20}"
    print(header)
    print("-" * len(header))

    for m, mname in model_order:
        t1, n1 = get_metric(agg, "task1", "Full-Struct", m, "cross_driver", 3, "macro_f1")
        t2, n2 = get_metric(agg, "task2", "Full-Struct", m, "cross_driver", 3, "auc_roc")
        t3, n3 = get_metric(agg, "task3", "Full-Struct", m, "cross_driver", 3, "auc_roc")
        n = max(n1, n2, n3)
        if n == 0:
            continue
        print(f"{mname:<20} | {n:>1} | {t1:>20} | {t2:>20} | {t3:>20}")


def print_table_s1(agg):
    """Table S1: Supplementary single-modality."""
    config_order = ["Ctx", "IMU", "FV", "CV", "FV+CV"]
    print_modality_table(
        agg, config_order,
        "TABLE S1: Supplementary Single-Modality (GRU, cross-driver, h=3s)",
    )


def print_table_s2(agg):
    """Table S2: Split comparison."""
    print(f"\n{'='*95}")
    print("TABLE S2: Split Comparison (Full-Struct, GRU, h=3s)")
    print(f"{'='*95}")

    splits = ["cross_driver", "cross_vehicle", "random"]
    header = f"{'Split':<20} | {'n':>1} | {'T1 Macro-F1':>20} | {'T2 AUC-ROC':>20} | {'T3 AUC-ROC':>20}"
    print(header)
    print("-" * len(header))

    for sp in splits:
        t1, n1 = get_metric(agg, "task1", "Full-Struct", "gru", sp, 3, "macro_f1")
        t2, n2 = get_metric(agg, "task2", "Full-Struct", "gru", sp, 3, "auc_roc")
        t3, n3 = get_metric(agg, "task3", "Full-Struct", "gru", sp, 3, "auc_roc")
        n = max(n1, n2, n3)
        if n == 0:
            continue
        print(f"{sp:<20} | {n:>1} | {t1:>20} | {t2:>20} | {t3:>20}")


def print_table_s3(agg):
    """Table S3: Horizon sensitivity."""
    print(f"\n{'='*95}")
    print("TABLE S3: Horizon Sensitivity (Full-Struct, GRU, cross-driver)")
    print(f"{'='*95}")

    horizons = [1, 3, 5]
    header = f"{'Horizon':>10} | {'n':>1} | {'T2 AUC-ROC':>20} | {'T2 AUPRC':>20} | {'T3 AUC-ROC':>20} | {'T3 AUPRC':>20}"
    print(header)
    print("-" * len(header))

    for h in horizons:
        t2_roc, n2 = get_metric(agg, "task2", "Full-Struct", "gru", "cross_driver", h, "auc_roc")
        t2_prc, _ = get_metric(agg, "task2", "Full-Struct", "gru", "cross_driver", h, "auprc")
        t3_roc, n3 = get_metric(agg, "task3", "Full-Struct", "gru", "cross_driver", h, "auc_roc")
        t3_prc, _ = get_metric(agg, "task3", "Full-Struct", "gru", "cross_driver", h, "auprc")
        n = max(n2, n3)
        if n == 0:
            continue
        print(f"{'h=' + str(h) + 's':>10} | {n:>1} | {t2_roc:>20} | {t2_prc:>20} | {t3_roc:>20} | {t3_prc:>20}")


def print_table_vlm(agg):
    """VLM zero-shot baseline comparison table."""
    print(f"\n{'='*110}")
    print("TABLE VLM: Zero-Shot VLM Baselines (cross-driver, h=3s, no training)")
    print(f"{'='*110}")

    header = (f"{'Model':<20} | {'Modality':<18} | {'T1 Macro-F1':>12} | "
              f"{'T2 F1':>10} | {'T2 AUC-ROC':>10} | {'T3 F1':>10} | {'T3 AUC-ROC':>10}")
    print(header)
    print("-" * len(header))

    vlm_models = [("gpt4o", "GPT-4o"), ("gemini", "Gemini 2.5 Flash")]
    vlm_modalities = [
        "VLM-Text", "VLM-Front", "VLM-Cabin",
        "VLM-Front+Cabin", "VLM-Full", "VLM-All",
    ]

    for model_key, model_name in vlm_models:
        for mod in vlm_modalities:
            # VLM runs have n_seeds=1 (single seed=42), so use _mean columns directly
            t1, _ = get_metric(agg, "task1", mod, model_key, "cross_driver", 3, "macro_f1")
            t2_f1, _ = get_metric(agg, "task2", mod, model_key, "cross_driver", 3, "f1")
            t2_roc, _ = get_metric(agg, "task2", mod, model_key, "cross_driver", 3, "auc_roc")
            t3_f1, _ = get_metric(agg, "task3", mod, model_key, "cross_driver", 3, "f1")
            t3_roc, n = get_metric(agg, "task3", mod, model_key, "cross_driver", 3, "auc_roc")
            if n == 0:
                continue
            # Single-seed: strip "+/- 0.0000" for cleaner output
            t1 = t1.split(" +/-")[0] if "+/-" in t1 else t1
            t2_f1 = t2_f1.split(" +/-")[0] if "+/-" in t2_f1 else t2_f1
            t2_roc = t2_roc.split(" +/-")[0] if "+/-" in t2_roc else t2_roc
            t3_f1 = t3_f1.split(" +/-")[0] if "+/-" in t3_f1 else t3_f1
            t3_roc = t3_roc.split(" +/-")[0] if "+/-" in t3_roc else t3_roc
            print(f"{model_name:<20} | {mod:<18} | {t1:>12} | "
                  f"{t2_f1:>10} | {t2_roc:>10} | {t3_f1:>10} | {t3_roc:>10}")
        # Separator between models
        if model_key != vlm_models[-1][0]:
            print("-" * len(header))

    print(f"\nN=350(T1), 300(T2/T3). Same samples across all configs. No training.")
    print(f"F1 from hard predictions; AUC-ROC from self-reported confidence.")


def print_detailed_task1(agg):
    """Print per-class F1 for Task 1 key configs."""
    print(f"\n{'='*95}")
    print("DETAILED: Task 1 Per-Class F1 (GRU, cross-driver, h=3s)")
    print(f"{'='*95}")

    classes = ["Accelerating", "Braking", "CarFollowing", "Cruising",
               "LaneChange", "Stopped", "Turning"]
    configs = ["Full-Struct", "Full-Multimodal"]

    for cfg in configs:
        sub = agg[(agg["task"] == "task1") & (agg["modality"] == cfg) &
                  (agg["model"] == "gru") & (agg["split"] == "cross_driver") &
                  (agg["horizon"] == 3)]
        if len(sub) == 0:
            continue
        print(f"\n  {cfg}:")
        for cls in classes:
            mean_col = f"test_f1_{cls}_mean"
            std_col = f"test_f1_{cls}_std"
            if mean_col in sub.columns and not pd.isna(sub[mean_col].values[0]):
                print(f"    {cls:<15}: {fmt(sub[mean_col].values[0], sub[std_col].values[0])}")


def main():
    results = load_all_results()
    if not results:
        print("No results found. Run experiments first.")
        return

    print(f"Loaded {len(results)} individual run results.")

    agg = aggregate_seeds(results)
    agg.to_csv(RESULTS_DIR / "aggregated_results.csv", index=False)
    print(f"Saved aggregated results to {RESULTS_DIR / 'aggregated_results.csv'}")

    # Summary counts
    tasks_done = agg.groupby(["task", "modality", "model", "split", "horizon"]).size()
    print(f"Unique experiment configs: {len(tasks_done)}")

    # Main tables (Phase 1 + 2)
    print_table1(agg)
    print_table2(agg)

    # Supplementary tables (Phase 3)
    print_table_s1(agg)
    print_table_s2(agg)
    print_table_s3(agg)

    # VLM baselines
    print_table_vlm(agg)

    # Per-class detail
    print_detailed_task1(agg)


if __name__ == "__main__":
    main()
