"""
dataset.py — Unified PyTorch dataset for PassingCtrl benchmark.

Loads pre-cached 50Hz numpy files (from preprocess.py).
Uses direct array slicing — no runtime interpolation.
Single-process cache with eager pre-loading (num_workers=0).
"""
import json
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from config import (
    BENCHMARK_DIR, DATASET_ROOT, DATA_DIR,
    FRONT_VIDEO_DIR, CABIN_VIDEO_DIR,
    PCA_FRONT_VIDEO_DIR, PCA_CABIN_VIDEO_DIR,
    CLIP_FRONT_VIDEO_DIR, CLIP_CABIN_VIDEO_DIR,
    STRUCT_GROUPS, GPS_COLS,
    LABEL2IDX, RESAMPLE_HZ, STRUCT_SEQ_LEN,
    VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM, VIDEO_FEATURE_DIM_PCA,
    VIDEO_FEATURE_DIM_CLIP, CACHE_DIR,
)

logger = logging.getLogger("baseline")

STRUCT_CACHE_DIR = CACHE_DIR / f"struct_{RESAMPLE_HZ}hz"
GPS_CACHE_DIR = CACHE_DIR / "gps_per_route"


class RouteCache:
    """Loads pre-cached 50Hz numpy data per route.

    Designed for num_workers=0 (single process). Call preload() once
    before training to eagerly load all route data into memory.
    Total memory: ~2GB struct + ~120MB GPS + ~1.7GB video (if used).
    """

    def __init__(self, use_pca=False, use_clip=False):
        self._struct_cache = {}   # route_id → {csv_name: (t_start, step, data, cols)} or None
        self._gps_cache = {}      # route_id → (ts, arr) or None
        self._video_cache = {}    # (camera, route_id) → (timestamps, features) or None
        self.use_pca = use_pca
        self.use_clip = use_clip
        if use_clip:
            self.video_feature_dim = VIDEO_FEATURE_DIM_CLIP
        elif use_pca:
            self.video_feature_dim = VIDEO_FEATURE_DIM_PCA
        else:
            self.video_feature_dim = VIDEO_FEATURE_DIM

    def preload(self, route_ids, load_gps=False, load_front_video=False,
                load_cabin_video=False):
        """Eagerly load all route data into cache before training."""
        unique_routes = sorted(set(route_ids))
        logger.info(f"Pre-loading {len(unique_routes)} routes...")

        for rid in tqdm(unique_routes, desc="Loading routes", unit="route"):
            self._load_struct_npz(rid)
            if load_gps:
                self._load_gps(rid)
            if load_front_video:
                self._load_video("front", rid)
            if load_cabin_video:
                self._load_video("cabin", rid)

        n_struct = sum(1 for v in self._struct_cache.values() if v is not None)
        logger.info(f"Pre-loaded: {n_struct} struct, "
                    f"{sum(1 for v in self._gps_cache.values() if v is not None)} gps, "
                    f"{sum(1 for v in self._video_cache.values() if v is not None)} video")

    # ──────────────────────────────────────────────
    # STRUCTURED SIGNALS (direct slice, no interp)
    # ──────────────────────────────────────────────

    def _load_struct_npz(self, route_id):
        if route_id in self._struct_cache:
            return self._struct_cache[route_id]

        npz_path = STRUCT_CACHE_DIR / (route_id.replace("/", "__") + ".npz")
        if not npz_path.exists():
            self._struct_cache[route_id] = None
            return None

        data = np.load(npz_path)
        route_data = {}
        for csv_name in ["vehicle_dynamics.csv", "planning.csv", "radar.csv",
                         "driver_state.csv", "imu.csv"]:
            t_key = f"{csv_name}__t_start"
            s_key = f"{csv_name}__step"
            d_key = f"{csv_name}__data"
            c_key = f"{csv_name}__cols"
            if d_key in data:
                route_data[csv_name] = (
                    float(data[t_key]),
                    float(data[s_key]),
                    data[d_key],        # [T, D] float32, already on 50Hz grid
                    list(data[c_key]),
                )

        self._struct_cache[route_id] = route_data
        return route_data

    def load_struct_signals(self, route_id, source_csv, columns, start, end):
        """Load structured signals via direct array slice (no interp).

        Returns: np.array [STRUCT_SEQ_LEN, len(columns)], float32
        """
        route_data = self._load_struct_npz(route_id)
        if route_data is None or source_csv not in route_data:
            return np.zeros((STRUCT_SEQ_LEN, len(columns)), dtype=np.float32)

        t_start, step, data, cached_cols = route_data[source_csv]

        # Direct index calculation — O(1), no search
        idx_start = int(round((start - t_start) / step))
        idx_start = max(0, idx_start)
        idx_end = idx_start + STRUCT_SEQ_LEN

        n_total = data.shape[0]
        if idx_start >= n_total:
            return np.zeros((STRUCT_SEQ_LEN, len(columns)), dtype=np.float32)

        # Slice (may need padding if near end of route)
        actual_end = min(idx_end, n_total)
        sliced = data[idx_start:actual_end]  # [<=SEQ_LEN, D_cached]

        # Map requested columns to cached column indices
        col_map = {c: i for i, c in enumerate(cached_cols)}
        out = np.zeros((STRUCT_SEQ_LEN, len(columns)), dtype=np.float32)
        n_valid = sliced.shape[0]
        for i, col in enumerate(columns):
            if col in col_map:
                out[:n_valid, i] = sliced[:, col_map[col]]

        return out

    # ──────────────────────────────────────────────
    # GPS CONTEXT (interp needed — GPS is 10Hz, not 50Hz)
    # ──────────────────────────────────────────────

    def _load_gps(self, route_id):
        """Load GPS data for a route (called by preload or lazily)."""
        if route_id in self._gps_cache:
            return
        npz_path = GPS_CACHE_DIR / (route_id.replace("/", "__") + ".npz")
        if npz_path.exists():
            d = np.load(npz_path)
            self._gps_cache[route_id] = (d["ts"], d["arr"])
        else:
            self._gps_cache[route_id] = None

    def _load_video(self, camera, route_id):
        """Load video features for a route (called by preload or lazily)."""
        cache_key = (camera, route_id)
        if cache_key in self._video_cache:
            return
        if self.use_clip:
            vid_dir = CLIP_FRONT_VIDEO_DIR if camera == "front" else CLIP_CABIN_VIDEO_DIR
        elif self.use_pca:
            vid_dir = PCA_FRONT_VIDEO_DIR if camera == "front" else PCA_CABIN_VIDEO_DIR
        else:
            vid_dir = FRONT_VIDEO_DIR if camera == "front" else CABIN_VIDEO_DIR
        fname = route_id.replace("/", "__") + ".npz"
        fpath = vid_dir / fname
        if not fpath.exists():
            self._video_cache[cache_key] = None
        else:
            d = np.load(fpath)
            self._video_cache[cache_key] = (
                d["timestamps"].astype(np.float32),
                d["features"],  # float16
            )

    def load_gps_context(self, route_id, start, end):
        """Load GPS context features, resample from 10Hz to 50Hz grid.

        Returns: np.array [STRUCT_SEQ_LEN, len(GPS_COLS)], float32
        """
        n_cols = len(GPS_COLS)
        self._load_gps(route_id)
        cached = self._gps_cache[route_id]
        if cached is None:
            return np.zeros((STRUCT_SEQ_LEN, n_cols), dtype=np.float32)

        ts, arr = cached

        # Resample GPS (10Hz) to 50Hz grid matching struct signals
        grid = np.arange(start, end, 1.0 / RESAMPLE_HZ, dtype=np.float32)
        n_grid = min(len(grid), STRUCT_SEQ_LEN)

        out = np.zeros((STRUCT_SEQ_LEN, n_cols), dtype=np.float32)
        for c in range(n_cols):
            out[:n_grid, c] = np.interp(grid[:n_grid], ts, arr[:, c],
                                        left=arr[0, c], right=arr[-1, c])
        return out

    # ──────────────────────────────────────────────
    # VIDEO FEATURES (2fps, unchanged)
    # ──────────────────────────────────────────────

    def load_video_features(self, camera, route_id, start, end):
        """Load pre-extracted video features for a time window.

        Returns: np.array [VIDEO_SEQ_LEN, VIDEO_FEATURE_DIM], float32
        """
        cache_key = (camera, route_id)
        self._load_video(camera, route_id)
        cached = self._video_cache[cache_key]
        vdim = self.video_feature_dim
        if cached is None:
            return np.zeros((VIDEO_SEQ_LEN, vdim), dtype=np.float32)

        timestamps, features = cached
        mask = (timestamps >= start) & (timestamps < end)
        window = features[mask].astype(np.float32)

        out = np.zeros((VIDEO_SEQ_LEN, vdim), dtype=np.float32)
        n = min(len(window), VIDEO_SEQ_LEN)
        if n > 0:
            out[:n] = window[:n]
        return out


# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════

class PassingCtrlDataset(Dataset):
    """Unified dataset for all three tasks."""

    def __init__(self, task, split, split_file, modality_config,
                 horizon=3, norm_stats=None, route_cache=None,
                 single_frame=False):
        self.task = task
        self.split = split
        self.modality_config = modality_config
        self.norm_stats = norm_stats
        self.cache = route_cache or RouteCache()
        self.single_frame = single_frame

        with open(split_file) as f:
            sp = json.load(f)
        split_routes = set(sp[f"{split}_routes"])

        if task == "task1":
            csv_path = BENCHMARK_DIR / "task1_action_samples.csv"
        elif task == "task2":
            csv_path = BENCHMARK_DIR / f"task2_activation_samples_h{horizon}.csv"
        elif task == "task3":
            csv_path = BENCHMARK_DIR / f"task3_takeover_samples_h{horizon}.csv"
        else:
            raise ValueError(f"Unknown task: {task}")

        df = pd.read_csv(csv_path)
        df = df[df["route_id"].isin(split_routes)].reset_index(drop=True)

        self.sample_ids = df["sample_id"].values
        self.route_ids = df["route_id"].values
        self.starts = df["start_time_sec"].values.astype(np.float32)
        self.ends = df["end_time_sec"].values.astype(np.float32)

        if task == "task1":
            self.labels = df["label"].map(LABEL2IDX).values.astype(np.int64)
        else:
            self.labels = df["label"].values.astype(np.float32)

        self._struct_sources = []
        for group_name in modality_config["struct"]:
            src_csv, cols = STRUCT_GROUPS[group_name]
            self._struct_sources.append((src_csv, cols))

        self.struct_dim = sum(len(cols) for _, cols in self._struct_sources)
        self.use_gps = modality_config["gps"]
        self.gps_dim = len(GPS_COLS) if self.use_gps else 0

        logger.info(f"Dataset: {task}/{split}, {len(self)} samples, "
                    f"struct_dim={self.struct_dim}, gps={self.use_gps}, "
                    f"fv={modality_config['front_video']}, "
                    f"cv={modality_config['cabin_video']}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        route_id = self.route_ids[idx]
        start = float(self.starts[idx])
        end = float(self.ends[idx])
        label = self.labels[idx]

        item = {"label": label}

        if self.struct_dim > 0:
            parts = []
            for src_csv, cols in self._struct_sources:
                arr = self.cache.load_struct_signals(route_id, src_csv, cols, start, end)
                parts.append(arr)

            struct = np.concatenate(parts, axis=1) if parts else np.empty((STRUCT_SEQ_LEN, 0))

            if self.norm_stats is not None and struct.shape[1] > 0:
                # norm_stats covers struct columns only (first struct_dim cols)
                s_mean = self.norm_stats["mean"][:self.struct_dim]
                s_std = self.norm_stats["std"][:self.struct_dim]
                struct = (struct - s_mean) / (s_std + 1e-8)

            if self.single_frame:
                struct = struct[-1:, :]  # [1, D] — last timestep only
            item["struct"] = torch.from_numpy(struct)

        if self.use_gps:
            gps = self.cache.load_gps_context(route_id, start, end)
            if self.norm_stats is not None and self.gps_dim > 0:
                g_mean = self.norm_stats["mean"][self.struct_dim:]
                g_std = self.norm_stats["std"][self.struct_dim:]
                gps = (gps - g_mean) / (g_std + 1e-8)
            if self.single_frame:
                gps = gps[-1:, :]  # [1, D]
            item["gps"] = torch.from_numpy(gps)

        if self.modality_config["front_video"]:
            fv = self.cache.load_video_features("front", route_id, start, end)
            if self.single_frame:
                fv = fv[-1:, :]  # [1, D]
            item["front_video"] = torch.from_numpy(fv)

        if self.modality_config["cabin_video"]:
            cv = self.cache.load_video_features("cabin", route_id, start, end)
            if self.single_frame:
                cv = cv[-1:, :]  # [1, D]
            item["cabin_video"] = torch.from_numpy(cv)

        return item


# ═══════════════════════════════════════════════════════════
# NORMALIZATION STATS
# ═══════════════════════════════════════════════════════════

def compute_norm_stats(task, split_file, modality_config, horizon=3,
                       max_samples=50000):
    """Compute per-feature mean/std from training split.

    Returns norm_stats with shape [struct_dim + gps_dim] so that
    struct and gps can each slice their portion.
    """
    cache = RouteCache()
    ds = PassingCtrlDataset(
        task=task, split="train", split_file=split_file,
        modality_config=modality_config, horizon=horizon,
        norm_stats=None, route_cache=cache,
    )

    total_dim = ds.struct_dim + ds.gps_dim
    if total_dim == 0:
        return None

    n = min(len(ds), max_samples)
    indices = np.random.RandomState(42).choice(len(ds), n, replace=False)

    # Pre-load routes for norm computation
    all_routes = list(set(ds.route_ids))
    cache.preload(all_routes, load_gps=modality_config["gps"])

    logger.info(f"Computing norm stats from {n} samples "
                f"(struct_dim={ds.struct_dim}, gps_dim={ds.gps_dim})...")
    running_sum = np.zeros(total_dim, dtype=np.float64)
    running_sq = np.zeros(total_dim, dtype=np.float64)

    for i, idx in enumerate(tqdm(indices, desc="norm stats", leave=False, unit="sample")):
        item = ds[int(idx)]
        parts = []
        if "struct" in item:
            parts.append(item["struct"].numpy())
        if "gps" in item:
            parts.append(item["gps"].numpy())
        if not parts:
            continue
        x = np.concatenate(parts, axis=1)  # [SEQ_LEN, total_dim]
        running_sum += x.mean(axis=0)
        running_sq += (x ** 2).mean(axis=0)

    mean = (running_sum / n).astype(np.float32)
    var = (running_sq / n - mean ** 2).clip(min=0)
    std = np.sqrt(var).astype(np.float32)

    return {"mean": mean, "std": std}
