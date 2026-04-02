"""
metrics.py — Evaluation metrics for PassingCtrl benchmark.

Task 1: Accuracy, Macro-F1, per-class F1
Tasks 2 & 3: AUC-ROC, AUPRC, F1 (optimal threshold), Precision@Recall=0.8
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve,
)

from config import IDX2LABEL, NUM_CLASSES_TASK1


def evaluate_task1(y_true, y_pred_logits):
    """Evaluate Task 1 (7-class action classification).

    Args:
        y_true: np.array [N] int64 — ground truth class indices
        y_pred_logits: np.array [N, 7] float — model logits

    Returns:
        dict with all metrics
    """
    y_pred = y_pred_logits.argmax(axis=1)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0,
                            labels=list(range(NUM_CLASSES_TASK1)))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES_TASK1)))

    result = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "confusion_matrix": cm.tolist(),
    }
    for i in range(NUM_CLASSES_TASK1):
        result[f"f1_{IDX2LABEL[i]}"] = float(per_class_f1[i])

    return result


def find_optimal_f1_threshold(y_true, y_scores):
    """Find threshold that maximizes F1 on given data.

    Args:
        y_true: np.array [N] binary
        y_scores: np.array [N] float — predicted probabilities

    Returns:
        (best_threshold, best_f1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    # F1 = 2 * P * R / (P + R)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1_vals = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    # precision_recall_curve returns one extra element; thresholds has len = len(precision) - 1
    f1_vals = f1_vals[:-1]
    if len(f1_vals) == 0:
        return 0.5, 0.0
    best_idx = np.argmax(f1_vals)
    return float(thresholds[best_idx]), float(f1_vals[best_idx])


def precision_at_recall(y_true, y_scores, target_recall=0.8):
    """Compute precision at a target recall level.

    Args:
        y_true: np.array [N] binary
        y_scores: np.array [N] float — predicted probabilities
        target_recall: float

    Returns:
        float — precision at the given recall, or 0.0 if unreachable
    """
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    # recall is decreasing; find first index where recall <= target_recall
    valid = recall >= target_recall
    if not valid.any():
        return 0.0
    # Among all points with recall >= target, pick the one with highest precision
    return float(precision[valid].max())


def evaluate_binary(y_true, y_scores, threshold=None):
    """Evaluate binary prediction (Tasks 2 & 3).

    Args:
        y_true: np.array [N] binary (0/1)
        y_scores: np.array [N] float — predicted probabilities (after sigmoid)
        threshold: float or None — if None, uses optimal F1 threshold from data

    Returns:
        dict with all metrics, including the threshold used
    """
    y_true = y_true.astype(np.int32)

    # AUC-ROC
    try:
        auc_roc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc_roc = 0.5  # single class in batch

    # AUPRC
    try:
        auprc = average_precision_score(y_true, y_scores)
    except ValueError:
        auprc = 0.0

    # Optimal F1 threshold
    if threshold is None:
        threshold, _ = find_optimal_f1_threshold(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(np.int32)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    # Precision at 80% recall
    p_at_r80 = precision_at_recall(y_true, y_scores, target_recall=0.8)

    return {
        "auc_roc": float(auc_roc),
        "auprc": float(auprc),
        "f1": float(f1),
        "accuracy": float(acc),
        "threshold": float(threshold),
        "precision_at_recall_0.8": float(p_at_r80),
        "n_pos": int(y_true.sum()),
        "n_neg": int((1 - y_true).sum()),
    }
