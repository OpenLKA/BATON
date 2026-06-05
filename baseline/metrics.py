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


def evaluate_event_level(route_ids, end_times, event_times, y_true, y_scores,
                         horizon, threshold=None):
    """Event-level evaluation for Tasks 2 & 3 (B3).

    The 0.5 s-stride sampling produces many overlapping windows per transition
    event, so sample-level AUPRC/AUROC over-weights events with more windows.
    This aggregates to ONE prediction per event:

      - Positives: grouped by (route_id, nearest_event_time); event score = MAX
        over its windows. One positive unit per real handover/takeover event.
      - Negatives: each route's no-event windows are binned into non-overlapping
        `horizon`-length time bins; bin score = MAX within the bin. One negative
        unit per bin.

    Args:
        route_ids:   array[N] str   — per-sample route_uid
        end_times:   array[N] float — per-sample window end time (sec)
        event_times: array[N] float — nearest_event_time (NaN for negatives)
        y_true:      array[N] {0,1}
        y_scores:    array[N] float — predicted probability
        horizon:     float          — prediction horizon (sec), negative bin width
        threshold:   float or None  — decision threshold (None → optimal-F1 here)

    Returns:
        dict with event-level auc_roc/auprc/f1/threshold, per-event detection
        recall, median lead-time (sec), and unit counts.
    """
    route_ids = np.asarray(route_ids)
    end_times = np.asarray(end_times, dtype=np.float64)
    event_times = np.asarray(event_times, dtype=np.float64)
    y_true = np.asarray(y_true).astype(np.int32)
    y_scores = np.asarray(y_scores, dtype=np.float64)

    # --- aggregate to event-level units ---
    # group_key -> {label, max_score, lead_windows: [(end_time, score)], event_time}
    groups = {}
    for i in range(len(y_true)):
        if y_true[i] == 1 and np.isfinite(event_times[i]):
            key = ("pos", route_ids[i], round(float(event_times[i]), 2))
            ev_t = float(event_times[i])
            label = 1
        else:
            # negative: bin by non-overlapping horizon-length window of end_time
            bin_idx = int(np.floor(end_times[i] / max(horizon, 1e-6)))
            key = ("neg", route_ids[i], bin_idx)
            ev_t = np.nan
            label = 0
        g = groups.get(key)
        if g is None:
            groups[key] = {"label": label, "max_score": y_scores[i],
                           "event_time": ev_t,
                           "ends": [end_times[i]], "scores": [y_scores[i]]}
        else:
            g["max_score"] = max(g["max_score"], y_scores[i])
            g["ends"].append(end_times[i])
            g["scores"].append(y_scores[i])

    ev_true = np.array([g["label"] for g in groups.values()], dtype=np.int32)
    ev_score = np.array([g["max_score"] for g in groups.values()], dtype=np.float64)

    if threshold is None:
        threshold, _ = find_optimal_f1_threshold(ev_true, ev_score)

    try:
        auc_roc = roc_auc_score(ev_true, ev_score)
    except ValueError:
        auc_roc = 0.5
    try:
        auprc = average_precision_score(ev_true, ev_score)
    except ValueError:
        auprc = 0.0

    ev_pred = (ev_score >= threshold).astype(np.int32)
    f1 = f1_score(ev_true, ev_pred, zero_division=0)
    p_at_r80 = precision_at_recall(ev_true, ev_score, target_recall=0.8)

    # per-event detection recall = fraction of positive events whose max score crosses threshold
    pos_groups = [g for g in groups.values() if g["label"] == 1]
    detected = 0
    lead_times = []
    for g in pos_groups:
        ends = np.array(g["ends"])
        scores = np.array(g["scores"])
        fired = scores >= threshold
        if fired.any():
            detected += 1
            # earliest (smallest end_time) window that fires → max anticipation
            first_end = ends[fired].min()
            if np.isfinite(g["event_time"]):
                lead_times.append(g["event_time"] - first_end)
    det_recall = detected / len(pos_groups) if pos_groups else 0.0
    median_lead = float(np.median(lead_times)) if lead_times else 0.0

    return {
        "event_auc_roc": float(auc_roc),
        "event_auprc": float(auprc),
        "event_f1": float(f1),
        "event_threshold": float(threshold),
        "event_precision_at_recall_0.8": float(p_at_r80),
        "event_detection_recall": float(det_recall),
        "event_median_lead_time_s": median_lead,
        "n_events_pos": int(ev_true.sum()),
        "n_events_neg": int((1 - ev_true).sum()),
    }
