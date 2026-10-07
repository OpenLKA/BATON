# Reproducing the BATON Baselines

This guide covers how to obtain the data and reproduce the structured-signal
baseline results. See `LICENSE` for terms: dataset/labels are CC BY-NC 4.0,
code is MIT.

## 1. Access tiers

| Tier | Where | Contents |
|------|-------|----------|
| GitHub package | [OpenLKA/BATON](https://github.com/OpenLKA/BATON) (this repo) | Benchmark CSVs, splits, event labels, and all baseline / processing code (`benchmark_v2/`, `baseline/`, `data_processing/`, `annotation_kit/`) |
| Sample corpus | [HF: HenryYHW/BATON-Sample](https://huggingface.co/datasets/HenryYHW/BATON-Sample) | 43 public routes, all modalities (video, CAN, IMU, GPS, ...) — no access request needed |
| Full corpus | [HF: HenryYHW/BATON](https://huggingface.co/datasets/HenryYHW/BATON) | Full route corpus (managed access — request on the HF page) |

```bash
# Code + benchmark tables
git lfs install
git clone https://github.com/OpenLKA/BATON.git

# Sample corpus (43 routes, no gating)
git clone https://huggingface.co/datasets/HenryYHW/BATON-Sample

# Full corpus (after access is granted)
python -c "from huggingface_hub import snapshot_download; \
           snapshot_download('HenryYHW/BATON', repo_type='dataset', local_dir='./BATON_raw')"
```

Note: `benchmark_v2/task1_action_samples.csv` (~172 MB) exceeds the GitHub
file-size limit and is distributed separately (see the README download link).
Verify all benchmark tables against `benchmark_v2/CHECKSUMS.sha256`:

```bash
cd benchmark_v2 && sha256sum -c CHECKSUMS.sha256
```

## 2. Point the code at the raw data

The raw-route root is defined in `baseline/paths.py`:

```python
BATON_ROOT = Path("/home/henry/Desktop/Drive/BATON")   # <-- edit this
```

Set `BATON_ROOT` to the directory containing the downloaded route bundles
(layout: `<car_model>/<dongle_id>/<route_id>/<segment_id>/{*.csv, *.mp4, metadata.json}`).
All other paths (`benchmark_v2/`, `baseline/cache/`) are resolved relative to
the repo automatically via `baseline/config.py`.

## 3. Build the 50 Hz structured-signal cache

```bash
python baseline/preprocess.py
```

This resamples the per-route CSVs (vehicle dynamics, planning, radar, driver
state, IMU) onto a regular 50 Hz grid and writes numpy caches to
`baseline/cache/struct_50hz/` plus GPS context to `baseline/cache/gps_per_route/`.
Expect roughly 6 GB of cache for the full corpus. This step is required before
any training command.

## 4. Train the structured-signal baselines

Classical models (logistic regression / XGBoost):

```bash
# Task 3 (takeover anticipation), leakage-safe structured features
python baseline/train_classical.py --task task3 --modality Safe-Full-Struct \
    --model xgb --split cross_driver --horizon 3 --seed 42

# Task 2 (activation anticipation)
python baseline/train_classical.py --task task2 --modality Safe-Full-Struct \
    --model xgb --split cross_driver --horizon 3 --seed 42

# Task 1 (action recognition)
python baseline/train_classical.py --task task1 --modality Safe-Full-Struct \
    --model lr --split cross_driver --seed 42
```

Neural sequence models:

```bash
python baseline/train_nn.py --task task3 --modality Safe-Full-Struct \
    --model gru --split cross_driver --horizon 3 --seed 42
```

Key flags (see `argparse` in `baseline/train_classical.py` / `baseline/train_nn.py`
for the full list):

- `--task` `{task1, task2, task3}`
- `--model` — classical: `{lr, xgb}`; NN: `{gru, tcn, transformer, rghbtq, dirghbtq}`
- `--modality` — feature set from `MODALITY_CONFIGS` in `baseline/config.py`
  (e.g. `Safe-Full-Struct`, `Safe-Full-Struct+GPS`, `Full-Struct`, `Drv`, `IMU`)
- `--split` `{cross_driver, cross_vehicle, random}`
- `--horizon` `{1, 3, 5}` (seconds; anticipation tasks)
- `--seed` (paper results use seeds 42/123/7)
- `--t3-suffix _antsafe` — use the anticipation-safe Task 3 sample table
  (`task3_takeover_samples_h3_antsafe.csv`)

Results (metrics JSON) are written under `baseline/results*/`;
`baseline/collect_results.py` aggregates them into summary tables.

Video (V-JEPA 2 / CLIP) and pose modalities additionally require precomputed
feature caches (`data/vjepa2_*_video_features/`, `wm_hybrid/cache/`), which are
produced from the raw videos by the scripts in `data_processing/` and
`wm_hybrid/` and are not included in this repository.

## 5. Planned: public benchmark-ready tier

A benchmark-ready public tier — precomputed 50 Hz structured tensors plus
video and pose features for the 565 benchmark routes, sufficient to reproduce
all baselines without downloading raw video — is planned for the camera-ready
release.

## 6. License

Dataset and labels: CC BY-NC 4.0
(https://creativecommons.org/licenses/by-nc/4.0/legalcode).
Code: MIT. See `LICENSE`.
