#!/usr/bin/env python3
"""
pca_video_features.py — PCA dimensionality reduction on video features.

Fits PCA on cross_driver training split, transforms ALL routes.
Reduces 1280-d EfficientNet features to 128-d.

Usage:
  python3 pca_video_features.py
"""
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pca")

# ── Paths ──
DATA_DIR = Path("/home/henry/Desktop/Drive/HMI/data")
BENCHMARK_DIR = Path("/home/henry/Desktop/Drive/HMI/benchmark")

FRONT_SRC = DATA_DIR / "front_video_features"
CABIN_SRC = DATA_DIR / "cabin_video_features"
FRONT_DST = DATA_DIR / "pca128_front_video_features"
CABIN_DST = DATA_DIR / "pca128_cabin_video_features"

N_COMPONENTS = 128
BATCH_SIZE = 10000  # for IncrementalPCA


def get_train_routes():
    """Get training routes from cross_driver split."""
    with open(BENCHMARK_DIR / "split_cross_driver.json") as f:
        split = json.load(f)
    return set(split["train_routes"])


def fit_pca(src_dir, train_routes, label=""):
    """Fit IncrementalPCA on training routes only."""
    logger.info(f"Fitting PCA({N_COMPONENTS}) on {label} training features...")
    pca = IncrementalPCA(n_components=N_COMPONENTS)

    # Collect training features in batches
    buffer = []
    n_frames = 0
    files = sorted(src_dir.glob("*.npz"))

    for f in tqdm(files, desc=f"  Loading {label} train", unit="route"):
        rid = f.stem.replace("__", "/")
        if rid not in train_routes:
            continue
        d = np.load(f)
        feats = d["features"].astype(np.float32)  # [T, 1280]
        buffer.append(feats)
        n_frames += len(feats)

        # Partial fit when buffer is large enough
        if n_frames >= BATCH_SIZE:
            batch = np.concatenate(buffer, axis=0)
            pca.partial_fit(batch)
            buffer = []
            n_frames = 0

    # Fit remaining
    if buffer:
        batch = np.concatenate(buffer, axis=0)
        pca.partial_fit(batch)

    explained = pca.explained_variance_ratio_.sum()
    logger.info(f"  PCA fit: {N_COMPONENTS} components explain {explained*100:.1f}% variance")
    return pca


def transform_all(src_dir, dst_dir, pca, label=""):
    """Transform ALL routes (train + val + test) using fitted PCA."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(src_dir.glob("*.npz"))

    for f in tqdm(files, desc=f"  Transforming {label}", unit="route"):
        dst_path = dst_dir / f.name
        if dst_path.exists():
            continue
        d = np.load(f)
        timestamps = d["timestamps"]
        feats = d["features"].astype(np.float32)  # [T, 1280]

        # PCA transform
        reduced = pca.transform(feats).astype(np.float16)  # [T, 128]

        np.savez_compressed(dst_path,
                            timestamps=timestamps,
                            features=reduced)

    logger.info(f"  Saved {len(files)} files to {dst_dir}")


def main():
    train_routes = get_train_routes()
    logger.info(f"Training routes: {len(train_routes)}")

    # Front video
    if FRONT_SRC.exists():
        pca_front = fit_pca(FRONT_SRC, train_routes, "front")
        transform_all(FRONT_SRC, FRONT_DST, pca_front, "front")
        # Save PCA model
        np.savez(FRONT_DST / "_pca_model.npz",
                 components=pca_front.components_,
                 mean=pca_front.mean_,
                 variance=pca_front.explained_variance_ratio_)

    # Cabin video
    if CABIN_SRC.exists():
        pca_cabin = fit_pca(CABIN_SRC, train_routes, "cabin")
        transform_all(CABIN_SRC, CABIN_DST, pca_cabin, "cabin")
        np.savez(CABIN_DST / "_pca_model.npz",
                 components=pca_cabin.components_,
                 mean=pca_cabin.mean_,
                 variance=pca_cabin.explained_variance_ratio_)

    logger.info("Done!")


if __name__ == "__main__":
    main()
