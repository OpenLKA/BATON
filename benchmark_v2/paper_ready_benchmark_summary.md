# PassingCtrl: Benchmark Summary (Paper-Ready)

PassingCtrl is a multimodal benchmark for bidirectional driver–ADAS control handover,
comprising **565 driving routes** (162.1 hours)
from **150 drivers** across **99 vehicle models**.

## ADAS Control Definition
We define ADAS-active using OR-logic over two CAN-bus signals:
ADAS\_active = (cc\_latActive = 1) ∨ (cruiseState\_enabled = 1),
capturing any form of automated lateral or longitudinal control.
To suppress transient CAN noise, we apply a 1.0-second debounce filter
and require a minimum 2.0-second gap between consecutive events.

## Event Statistics
| | Count |
|---|---|
| Activation events (human → ADAS) | 1800 |
| Takeover events (ADAS → human) | 1793 |
| Total handover events | 3593 |

## Benchmark Tasks

| Task | Description | Samples (h=3s) | Metric |
|---|---|---|---|
| T1: Action Understanding | Classify driver/ADAS actions | 1161794 | Accuracy, Macro-F1 |
| T2: Activation Prediction | Predict ADAS engagement | 65223 | AUC-ROC, F1 |
| T3: Takeover Prediction | Predict human takeover | 91255 | AUC-ROC, F1 |

## Action Taxonomy (7 classes)
- **Cruising**: 29.1%
- **Stopped**: 19.2%
- **CarFollowing**: 19.1%
- **Braking**: 11.0%
- **Turning**: 10.9%
- **Accelerating**: 9.4%
- **LaneChange**: 1.3%

## Dataset Splits
We evaluate under three split protocols:
- **Cross-driver** (primary): 104 / 26 / 20 drivers
- **Cross-vehicle** (secondary): 59 / 18 / 22 vehicle models
- **Random** (baseline): 395 / 84 / 86 routes

## Multimodal Composition
Each sample provides access to 8 synchronized modalities:
vehicle dynamics (100 Hz), IMU (100 Hz), planning (20 Hz), radar (20 Hz),
driver monitoring (20 Hz), GPS (10 Hz), forward camera (20 fps), and driver camera (20 fps).

---
*PassingCtrl Benchmark v1 — 2026-06-02*
