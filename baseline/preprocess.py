#!/usr/bin/env python3
"""
preprocess.py — Pre-resample all structured signals to 50Hz numpy cache.

Converts raw CSVs to compact numpy files on a regular grid, enabling
direct array slicing at runtime (no interpolation needed).

Usage:
  python3 preprocess.py
"""
import json
import logging
import numpy as np
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    BENCHMARK_DIR, DATASET_ROOT, CACHE_DIR,
    STRUCT_GROUPS, GPS_COLS, GPS_CONTEXT_PATH, ROAD_TYPE_MAP,
    RESAMPLE_HZ,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("preprocess")

STRUCT_CACHE_DIR = CACHE_DIR / f"struct_{RESAMPLE_HZ}hz"
GPS_CACHE_DIR = CACHE_DIR / "gps_per_route"


def resample_csv(csv_path, cols):
    """Read a CSV, resample to RESAMPLE_HZ grid, return (t_start, step, data)."""
    df = pd.read_csv(csv_path)
    ts = df["time_s"].values.astype(np.float64)
    if len(ts) < 2:
        return None

    t_start = ts[0]
    t_end = ts[-1]
    step = 1.0 / RESAMPLE_HZ
    grid = np.arange(t_start, t_end + step * 0.5, step, dtype=np.float64)

    data = np.zeros((len(grid), len(cols)), dtype=np.float32)
    for i, col in enumerate(cols):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)
            v = np.nan_to_num(v, nan=0.0)
            data[:, i] = np.interp(grid, ts, v, left=v[0], right=v[-1]).astype(np.float32)

    return np.float32(t_start), np.float32(step), data


def preprocess_struct():
    """Pre-resample all structured signals per route."""
    STRUCT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    routes = pd.read_csv(BENCHMARK_DIR / "routes.csv")
    n_routes = len(routes)

    # Gather all columns per source CSV
    source_cols = {}
    for group_name, (src_csv, cols) in STRUCT_GROUPS.items():
        if src_csv not in source_cols:
            source_cols[src_csv] = []
        source_cols[src_csv].extend(cols)
    for k in source_cols:
        seen = set()
        source_cols[k] = [c for c in source_cols[k] if not (c in seen or seen.add(c))]

    logger.info(f"Pre-resampling structured signals to {RESAMPLE_HZ}Hz for {n_routes} routes")
    for csv_name, cols in source_cols.items():
        logger.info(f"  {csv_name}: {len(cols)} columns")

    t0 = time.time()
    ok, fail, skip = 0, 0, 0

    for idx, row in routes.iterrows():
        rid = row["route_id"]
        driver, rhash = rid.split("/")
        vm = row["vehicle_model"]
        base = DATASET_ROOT / vm / driver / rhash
        acm_dirs = sorted(base.glob("ACM_MM/route_*"))
        if not acm_dirs:
            fail += 1
            continue
        acm = acm_dirs[0]

        out_key = rid.replace("/", "__")
        out_path = STRUCT_CACHE_DIR / f"{out_key}.npz"
        if out_path.exists():
            skip += 1
            ok += 1
            continue

        arrays = {}
        for csv_name, cols in source_cols.items():
            csv_path = acm / csv_name
            if not csv_path.exists():
                continue
            result = resample_csv(csv_path, cols)
            if result is not None:
                t_start, step, data = result
                # Store metadata for direct slicing
                arrays[f"{csv_name}__t_start"] = np.array(t_start)
                arrays[f"{csv_name}__step"] = np.array(step)
                arrays[f"{csv_name}__data"] = data
                arrays[f"{csv_name}__cols"] = np.array(cols)

        if arrays:
            np.savez(out_path, **arrays)
            ok += 1
        else:
            fail += 1

        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            logger.info(f"  [{idx+1}/{n_routes}] {ok} ok ({skip} cached), {fail} fail, {elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"Struct done: {ok} ok, {fail} fail, {elapsed:.0f}s")

    total_size = sum(f.stat().st_size for f in STRUCT_CACHE_DIR.glob("*.npz"))
    logger.info(f"Struct cache size: {total_size/1e9:.2f} GB ({STRUCT_CACHE_DIR})")


def preprocess_gps():
    """Pre-split GPS context CSV into per-route numpy files."""
    GPS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(GPS_CACHE_DIR.glob("*.npz"))
    if len(existing) >= 300:
        logger.info(f"GPS cache already exists ({len(existing)} files), skipping")
        return

    logger.info("Loading global GPS context CSV...")
    gps_df = pd.read_csv(GPS_CONTEXT_PATH, low_memory=False)
    gps_df["road_type_enc"] = gps_df["road_type"].map(ROAD_TYPE_MAP).fillna(6).astype(float)

    for c in GPS_COLS:
        if c not in gps_df.columns:
            gps_df[c] = 0.0

    n_routes = 0
    for rid, sub in gps_df.groupby("route_id"):
        out_path = GPS_CACHE_DIR / (rid.replace("/", "__") + ".npz")
        if out_path.exists():
            n_routes += 1
            continue
        arr = sub[GPS_COLS].values.astype(np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        ts = sub["time_s"].values.astype(np.float32)
        np.savez(out_path, ts=ts, arr=arr)
        n_routes += 1

    del gps_df
    total_size = sum(f.stat().st_size for f in GPS_CACHE_DIR.glob("*.npz"))
    logger.info(f"GPS done: {n_routes} routes, {total_size/1e6:.0f} MB")


def main():
    logger.info("=" * 60)
    logger.info(f"PassingCtrl — Preprocessing ({RESAMPLE_HZ}Hz)")
    logger.info("=" * 60)

    preprocess_gps()
    preprocess_struct()

    logger.info("\nAll preprocessing done. Cache at: " + str(CACHE_DIR))


if __name__ == "__main__":
    main()
