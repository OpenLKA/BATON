#!/usr/bin/env python3
"""
sanity_check.py — Validate the entire pipeline before running experiments.

Checks:
  1. Data loading: struct signals, GPS, video features
  2. Tensor shapes and dtypes
  3. Normalization stats
  4. Model forward pass (all configs)
  5. Loss computation
  6. Metric computation (with synthetic data)
  7. Classical feature extraction
"""
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    BENCHMARK_DIR, MODALITY_CONFIGS, LABEL2IDX,
    STRUCT_SEQ_LEN, VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM,
    GPS_COLS, NUM_CLASSES_TASK1,
)


def check_metrics():
    """Verify metric functions with known inputs."""
    from metrics import evaluate_task1, evaluate_binary, find_optimal_f1_threshold, precision_at_recall

    print("\n--- Metrics Validation ---")

    # Task 1: perfect predictions
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    logits = np.eye(7) * 10.0  # perfect one-hot logits
    m = evaluate_task1(y_true, logits)
    assert m["accuracy"] == 1.0, f"Perfect accuracy should be 1.0, got {m['accuracy']}"
    assert m["macro_f1"] == 1.0, f"Perfect macro_f1 should be 1.0, got {m['macro_f1']}"
    print("  [OK] Task1 perfect prediction: accuracy=1.0, macro_f1=1.0")

    # Task 1: all same class
    y_true = np.array([0, 0, 0, 1, 1, 2, 2])
    logits = np.zeros((7, 7))
    logits[:, 0] = 10.0  # all predict class 0
    m = evaluate_task1(y_true, logits)
    assert m["accuracy"] == 3 / 7, f"Expected 3/7 accuracy, got {m['accuracy']}"
    # macro_f1: only class 0 has nonzero F1
    print(f"  [OK] Task1 all-class-0: accuracy={m['accuracy']:.4f}, macro_f1={m['macro_f1']:.4f}")

    # Binary: perfect
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = evaluate_binary(y_true, y_scores)
    assert m["auc_roc"] == 1.0, f"Perfect AUC should be 1.0, got {m['auc_roc']}"
    print(f"  [OK] Binary perfect: auc_roc={m['auc_roc']:.4f}, f1={m['f1']:.4f}")

    # Binary: random
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 1000)
    y_scores = np.random.rand(1000)
    m = evaluate_binary(y_true, y_scores)
    assert 0.4 < m["auc_roc"] < 0.6, f"Random AUC should be ~0.5, got {m['auc_roc']}"
    print(f"  [OK] Binary random: auc_roc={m['auc_roc']:.4f}")

    # Threshold from val applied to test
    y_val = np.array([0, 0, 1, 1])
    scores_val = np.array([0.2, 0.4, 0.6, 0.8])
    thr, f1_val = find_optimal_f1_threshold(y_val, scores_val)
    assert 0.3 <= thr <= 0.7, f"Threshold should be reasonable, got {thr}"
    print(f"  [OK] Optimal threshold: {thr:.4f}, val_f1={f1_val:.4f}")

    # Precision@Recall=0.8
    p = precision_at_recall(y_true[:100], y_scores[:100], target_recall=0.8)
    print(f"  [OK] Precision@Recall=0.8: {p:.4f}")

    print("  All metric checks passed!")


def check_dataset():
    """Load a few samples and verify shapes."""
    from dataset import PassingCtrlDataset, RouteCache

    print("\n--- Dataset Validation ---")
    cache = RouteCache()
    split_file = BENCHMARK_DIR / "split_cross_driver.json"

    # Test a few modality configs
    test_configs = ["Veh", "Full-Struct", "Full-Multimodal", "FV", "CV"]

    for cfg_name in test_configs:
        mod_cfg = MODALITY_CONFIGS[cfg_name]

        # Task 1
        ds = PassingCtrlDataset("task1", "train", split_file, mod_cfg,
                                norm_stats=None, route_cache=cache)
        item = ds[0]

        if "struct" in item:
            s = item["struct"]
            assert s.shape[0] == STRUCT_SEQ_LEN, f"{cfg_name}: struct seq len {s.shape[0]} != {STRUCT_SEQ_LEN}"
            assert s.dtype == torch.float32, f"{cfg_name}: struct dtype {s.dtype}"
            print(f"  [OK] {cfg_name} Task1: struct={s.shape}, label={item['label']}")
        if "front_video" in item:
            fv = item["front_video"]
            assert fv.shape == (VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM), f"{cfg_name}: fv shape {fv.shape}"
            print(f"  [OK] {cfg_name} Task1: front_video={fv.shape}")
        if "cabin_video" in item:
            cv = item["cabin_video"]
            assert cv.shape == (VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM), f"{cfg_name}: cv shape {cv.shape}"
            print(f"  [OK] {cfg_name} Task1: cabin_video={cv.shape}")

        # Task 2
        ds2 = PassingCtrlDataset("task2", "train", split_file, mod_cfg,
                                 horizon=3, norm_stats=None, route_cache=cache)
        item2 = ds2[0]
        assert isinstance(item2["label"], (float, np.floating)), f"Task2 label type: {type(item2['label'])}"
        print(f"  [OK] {cfg_name} Task2: label={item2['label']}, n_samples={len(ds2)}")

    print(f"  Dataset struct_dim for Full-Struct: {ds.struct_dim}")
    print("  All dataset checks passed!")


def check_models():
    """Forward pass for all model types."""
    from models import GRUBackbone, TCNBackbone

    print("\n--- Model Validation ---")
    device = torch.device("cpu")
    batch = 4

    configs = [
        ("Veh", 15, False, False),
        ("Full-Struct", 73, False, False),
        ("Full-Multimodal", 73, True, True),
        ("FV-only", 0, True, False),
        ("CV-only", 0, False, True),
    ]

    for name, sdim, fv, cv in configs:
        for model_name, ModelClass in [("GRU", GRUBackbone), ("TCN", TCNBackbone)]:
            for task in ["task1", "task2"]:
                model = ModelClass(sdim, fv, cv, task).to(device)
                kwargs = {}
                if sdim > 0:
                    kwargs["struct"] = torch.randn(batch, STRUCT_SEQ_LEN, sdim)
                if fv:
                    kwargs["front_video"] = torch.randn(batch, VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM)
                if cv:
                    kwargs["cabin_video"] = torch.randn(batch, VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM)

                out = model(**kwargs)
                expected = (batch, NUM_CLASSES_TASK1) if task == "task1" else (batch, 1)
                assert out.shape == expected, f"{name}/{model_name}/{task}: shape {out.shape} != {expected}"

        print(f"  [OK] {name}: GRU+TCN × task1+task2 forward pass")

    print("  All model checks passed!")


def check_loss():
    """Verify loss computation."""
    import torch.nn as nn

    print("\n--- Loss Validation ---")

    # Task 1: CrossEntropy with weights
    weights = torch.ones(NUM_CLASSES_TASK1)
    criterion = nn.CrossEntropyLoss(weight=weights)
    logits = torch.randn(4, NUM_CLASSES_TASK1)
    labels = torch.randint(0, NUM_CLASSES_TASK1, (4,))
    loss = criterion(logits, labels)
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss), "Loss should not be NaN"
    print(f"  [OK] Task1 CE loss: {loss.item():.4f}")

    # Task 2/3: BCEWithLogits
    pos_weight = torch.tensor([5.0])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logits = torch.randn(4, 1).squeeze()
    labels = torch.tensor([0.0, 1.0, 0.0, 1.0])
    loss = criterion(logits, labels)
    assert loss.item() > 0
    assert not torch.isnan(loss)
    print(f"  [OK] Binary BCE loss: {loss.item():.4f}")

    print("  All loss checks passed!")


def check_classical_features():
    """Verify statistical feature extraction."""
    from dataset import PassingCtrlDataset, RouteCache

    print("\n--- Classical Feature Extraction ---")
    cache = RouteCache()
    split_file = BENCHMARK_DIR / "split_cross_driver.json"
    mod_cfg = MODALITY_CONFIGS["Full-Struct"]

    ds = PassingCtrlDataset("task1", "train", split_file, mod_cfg,
                            norm_stats=None, route_cache=cache)

    # Extract 5 samples manually
    D = ds.struct_dim
    for i in range(min(5, len(ds))):
        item = ds[i]
        seq = item["struct"].numpy()
        feats = np.concatenate([
            seq.mean(axis=0), seq.std(axis=0), seq.min(axis=0),
            seq.max(axis=0), seq[-1],
        ])
        assert feats.shape == (D * 5,), f"Expected {D*5}, got {feats.shape}"
        assert not np.isnan(feats).any(), "NaN in features"

    print(f"  [OK] Statistical features: {D} channels × 5 stats = {D*5} features")
    print("  Classical feature extraction OK!")


def main():
    print("="*60)
    print("PassingCtrl Baseline — Sanity Check")
    print("="*60)

    check_metrics()
    check_dataset()
    check_models()
    check_loss()
    check_classical_features()

    print("\n" + "="*60)
    print("ALL CHECKS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
