# BATON — Revision Results (reproducible)

All numbers are leak-safe, cross-driver, horizon 3 s. Reproduce with the scripts in
`baseline/` (e.g. `run_*.sh`, `train_nn.py`, `train_classical.py`, `diag_vjepa2_fusion.py`,
`lead_time_analysis.py`, `summarize_*.py`). Video features are V-JEPA2 ViT-L; the
extraction scripts are in `data_processing/`.

## 1. Removing the automation-control variables (leakage check)

We remove the four label-defining / controller-internal automation variables
(`cc_latActive`, `cruiseState_enabled`, `cs_longControlState`, `actuators_accel`).
Sample / event AUPRC (XGBoost):

| Task | Full input | Leak-safe input |
|---|---|---|
| Handover | 0.360 / 0.292 | 0.222 / 0.091 |
| Takeover | 0.520 / 0.428 | 0.478 / 0.385 |

Verification: removed and retained field sets are disjoint; strongest retained single
feature AUPRC 0.247 vs 0.332 for the removed `cc_latActive`; reconstruction probe fails.

## 2. Multimodal fusion vs. the strong tabular baseline (10 seeds, leak-safe)

Sample-level AUPRC, mean ± s.d.:

| Task | XGBoost (signals) | V-JEPA2 video | Multimodal fusion |
|---|---|---|---|
| Handover | 0.246 ± .007 | 0.241 ± .007 | **0.264 ± .008** |
| Takeover | 0.441 ± .005 | 0.307 ± .010 | 0.440 ± .004 |

The frozen video alone is well above chance (takeover 0.307 = 2.5×, handover 0.240),
so it carries an independent signal; fusion beats the tabular baseline on handover and
matches it on takeover.

## 3. Action-recognition label validity (Task 1)

| Evidence | Result |
|---|---|
| Rule-free ablation (remove 9 rule signals) | Macro-F1 0.577 (≫ chance 0.14) |
| Front-video only | Macro-F1 0.592 |
| Threshold sensitivity (±20–30%) | 0.3–7.5% of labels change |
| Blind human validation (500 windows) | 96.0% agreement, κ = 0.953 |

## Large files (Google Drive)

Files exceeding GitHub's 100 MB limit are hosted on Google Drive
(https://drive.google.com/drive/folders/1AAU5AtZzTyCrHloQY-9JzJ9P3uZvG_R-):
`benchmark_v2/task1_action_samples.csv` and the V-JEPA2 video features (`data/`).
