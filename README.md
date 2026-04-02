# BATON: A Multimodal Benchmark for Bidirectional Automation Transition Observation in Naturalistic Driving

**BATON** is a real-world multimodal benchmark for studying bidirectional driver–automation control transitions. It synchronizes front-view video, in-cabin video, decoded CAN bus signals, radar-based lead-vehicle interaction, and GPS-derived route context from naturalistic daily driving.

## Dataset

- **380 routes** from **127 drivers** across **84 vehicle models**
- **136.6 hours** of real-world driving
- **2,892 control-transition events** (1,460 handovers + 1,432 takeovers)
- Multimodal: front video, cabin video, CAN, radar, driver monitoring, IMU, GPS

**Data access:**
- Sample release: [HuggingFace (sample)](https://huggingface.co/datasets/HenryYHW/PassingCtrl-Sample)
- Full dataset: [HuggingFace (managed access)](https://huggingface.co/datasets/HenryYHW/PassingCtrl)

## Benchmark Tasks

| Task | Description | Samples | Primary Metric |
|------|-------------|---------|----------------|
| Task 1 | Driving action understanding (7-class) | 979,809 | Macro-F1 |
| Task 2 | Handover prediction (Human→DA) | 56,564 (h=3s) | AUPRC |
| Task 3 | Takeover prediction (DA→Human) | 71,079 (h=3s) | AUPRC |

## Repository Structure

```
BATON/
├── benchmark/               # Benchmark data and generation code
│   ├── generate_benchmark.py    # Full benchmark construction pipeline
│   ├── routes.csv               # Route metadata (379 routes)
│   ├── action_labels.csv        # 1Hz action labels
│   ├── task1_action_samples.csv # Task 1 samples
│   ├── task2_activation_samples_h{1,3,5}.csv  # Task 2 samples
│   ├── task3_takeover_samples_h{1,3,5}.csv    # Task 3 samples
│   ├── split_cross_driver.json  # Primary evaluation split
│   ├── split_cross_vehicle.json # Cross-vehicle split
│   ├── split_random.json        # Random split
│   └── benchmark_protocol.md    # Detailed protocol specification
├── baseline/                # Baseline training and evaluation code
│   ├── config.py                # Paths, modality definitions, hyperparameters
│   ├── dataset.py               # PyTorch dataset for all tasks
│   ├── models.py                # GRU and TCN architectures with gated fusion
│   ├── metrics.py               # Evaluation metrics
│   ├── train_nn.py              # Neural network training (GRU/TCN)
│   ├── train_classical.py       # XGBoost and LR training
│   ├── run_vlm.py               # Zero-shot VLM baselines (Gemini/GPT-4o)
│   ├── vlm_prompts.py           # VLM prompt construction
│   └── ...
└── data_processing/         # Feature extraction scripts
    ├── extract_front_video_features.py   # EfficientNet-B0 front video
    ├── extract_cabin_video_features.py   # EfficientNet-B0 cabin video
    ├── extract_clip_features.py          # CLIP ViT-B/32 features
    ├── video_utils.py                    # Shared video decoding utilities
    └── gps_semantic_enrichment.py        # GPS → road context features
```

## Quick Start

### 1. Download data
Download the dataset from HuggingFace and place route folders under the dataset root.

### 2. Preprocess signals
```bash
cd baseline
python preprocess.py
```

### 3. Extract video features
```bash
cd data_processing
python extract_front_video_features.py
python extract_cabin_video_features.py
```

### 4. Train baselines
```bash
cd baseline
# GRU on all modalities
python train_nn.py --task task1 --modality Full-All --model gru --seed 42

# XGBoost on structured signals
python train_classical.py --task task2 --model xgb --seed 42

# Zero-shot VLM
python run_vlm.py --model gemini --task task1
```

## Evaluation Protocol

- **Main split:** Cross-driver (disjoint drivers across train/val/test)
- **Main horizon:** 3 seconds
- **Input window:** 5 seconds
- **Seeds:** 42, 123, 7 (report 3-seed average)

## Citation

If you use BATON in your research, please cite:

```bibtex
@inproceedings{wang2026baton,
  title={BATON: A Multimodal Benchmark for Bidirectional Automation Transition Observation in Naturalistic Driving},
  author={Wang, Yuhang and Xu, Yiyao and Yang, Chaoyun and Li, Lingyao and Sun, Jingran and Zhou, Hao},
  booktitle={Proceedings of the 34th ACM International Conference on Multimedia},
  year={2026}
}
```

## License

This project is released for academic research purposes.
