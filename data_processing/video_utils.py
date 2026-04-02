"""
video_utils.py — Shared utilities for video feature extraction.
PassingCtrl Benchmark.
"""
import subprocess
import numpy as np
import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision import transforms

logger = logging.getLogger("video_extraction")

# ── Constants ──
FPS = 2
FRAME_SIZE = 224
FEATURE_DIM = 1280  # EfficientNet-B0
ENCODER_NAME = "efficientnet_b0"
BATCH_SIZE = 64

DATASET_ROOT = Path("/home/henry/Desktop/Drive/Dataset")
BENCHMARK_DIR = Path("/home/henry/Desktop/Drive/HMI/benchmark")
OUTPUT_DIR = Path("/home/henry/Desktop/Drive/HMI/data")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_encoder(device="cuda"):
    """Load frozen EfficientNet-B0 feature extractor."""
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)
    model.classifier = nn.Identity()  # remove classifier → 1280-d output
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    logger.info(f"Loaded {ENCODER_NAME} (frozen, {FEATURE_DIM}-d) on {device}")
    return model


def get_preprocess():
    """ImageNet normalization transform (applied to float tensor [0,1])."""
    return transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def load_route_index():
    """Load benchmark routes.csv and build route_id → filesystem mapping."""
    import pandas as pd
    routes = pd.read_csv(BENCHMARK_DIR / "routes.csv")
    index = {}
    for _, row in routes.iterrows():
        rid = row["route_id"]
        driver, rhash = rid.split("/")
        vm = row["vehicle_model"]
        base = DATASET_ROOT / vm / driver / rhash
        if not base.exists():
            logger.warning(f"Route dir not found: {base}")
            continue
        index[rid] = {
            "base_dir": base,
            "vehicle_model": vm,
            "driver_id": driver,
            "route_hash": rhash,
            "duration_sec": row["duration_sec"],
            "n_segments": row["n_segments"],
            "has_qcamera": row["has_qcamera"],
        }
    return index


def route_id_to_filename(route_id):
    """Convert route_id 'driver/hash' to safe filename 'driver__hash'."""
    return route_id.replace("/", "__")


def filename_to_route_id(filename):
    """Convert filename 'driver__hash.npz' back to route_id."""
    return filename.replace(".npz", "").replace("__", "/")


# ═══════════════════════════════════════════════════════
# VIDEO DECODING
# ═══════════════════════════════════════════════════════

def decode_video_pipe(video_path, fps=FPS, size=FRAME_SIZE, start_s=None, duration_s=None):
    """Decode video to numpy array via ffmpeg pipe.

    Uses NVDEC hardware acceleration for HEVC files when available.

    Args:
        video_path: Path to video file (.mp4, .ts, .hevc)
        fps: Output frame rate
        size: Output square size (width=height)
        start_s: Optional start time in seconds
        duration_s: Optional duration in seconds

    Returns:
        frames: np.ndarray [N, H, W, 3] uint8, or None on failure
        actual_duration: float, actual video duration decoded
    """
    video_path = str(video_path)
    # Detect HEVC content: raw .hevc files or MP4s produced from dcamera concat
    is_hevc = video_path.endswith(".hevc") or "dcamera" in video_path or "_tmp_cabin" in video_path

    cmd = ["ffmpeg", "-v", "error"]

    # Use NVDEC for HEVC content (8x faster than CPU decode for 1928x1208 dcamera)
    if is_hevc:
        cmd += ["-hwaccel", "cuda", "-c:v", "hevc_cuvid"]

    if start_s is not None:
        cmd += ["-ss", str(start_s)]

    cmd += ["-i", video_path]

    if duration_s is not None:
        cmd += ["-t", str(duration_s)]

    cmd += [
        "-vf", f"fps={fps},scale={size}:{size}",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
    ]

    # Scale timeout with expected output: base 120s + 1s per expected output frame
    # For a 6000s route at 2fps = 12000 frames → timeout ~12120s
    expected_frames = (duration_s or 7200) * fps  # default 2h if unknown
    timeout_s = max(120, int(120 + expected_frames * 0.5))

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[:200]
            logger.warning(f"ffmpeg error for {video_path}: {stderr}")
            return None, 0.0

        raw = proc.stdout
        frame_bytes = size * size * 3
        n_frames = len(raw) // frame_bytes

        if n_frames == 0:
            return None, 0.0

        # Trim to exact frame boundary
        raw = raw[:n_frames * frame_bytes]
        frames = np.frombuffer(raw, dtype=np.uint8).reshape(n_frames, size, size, 3)
        actual_duration = n_frames / fps
        return frames.copy(), actual_duration  # .copy() to make writable

    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timeout for {video_path}")
        return None, 0.0
    except Exception as e:
        logger.error(f"Decode error for {video_path}: {e}")
        return None, 0.0


def get_video_duration(video_path):
    """Get video duration in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video_path)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════
# SEGMENT TIMING (for dcamera)
# ═══════════════════════════════════════════════════════

def load_segment_timing(route_dir):
    """Load segment timing from metadata.json.

    Returns:
        list of dicts: [{'seg_num': int, 'start_s': float, 'end_s': float, 'duration_s': float}, ...]
        or None if metadata not found.
    """
    meta_files = list(Path(route_dir).glob("ACM_MM/*/metadata.json"))
    if not meta_files:
        logger.warning(f"No metadata.json in {route_dir}")
        return None

    with open(meta_files[0]) as f:
        meta = json.load(f)

    segments = []
    cumulative = 0.0
    for s in meta["segments"]:
        dur = s["duration_s"]
        segments.append({
            "seg_num": s["seg_num"],
            "start_s": cumulative,
            "end_s": cumulative + dur,
            "duration_s": dur,
        })
        cumulative += dur

    return segments


def concatenate_segments(segment_files, output_path, timeout=300):
    """Concatenate HEVC segment files into a single MP4 using ffmpeg concat demuxer.

    Args:
        segment_files: list of Path objects, ordered by segment number
        output_path: Path for output MP4
        timeout: ffmpeg timeout in seconds

    Returns:
        True on success, False on failure
    """
    if not segment_files:
        return False

    # Create concat list file
    list_path = output_path.with_suffix(".txt")
    try:
        with open(list_path, "w") as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")

        cmd = [
            "ffmpeg", "-v", "error", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(output_path)
        ]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)

        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")[:200]
            logger.warning(f"Concat failed: {stderr}")
            return False

        return output_path.exists() and output_path.stat().st_size > 1000

    except subprocess.TimeoutExpired:
        logger.error(f"Concat timeout for {output_path}")
        return False
    except Exception as e:
        logger.error(f"Concat error: {e}")
        return False
    finally:
        if list_path.exists():
            list_path.unlink()


# ═══════════════════════════════════════════════════════
# GPU FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════

def extract_features_from_frames(frames, model, device="cuda", batch_size=BATCH_SIZE):
    """Extract features from numpy frames using frozen encoder.

    Args:
        frames: np.ndarray [N, 224, 224, 3] uint8
        model: frozen encoder on device
        device: torch device
        batch_size: batch size for inference

    Returns:
        features: np.ndarray [N, FEATURE_DIM] float16
    """
    normalize = get_preprocess()
    n = len(frames)
    all_features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_np = frames[start:end]

        # Convert to float tensor [0, 1] and normalize
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
        batch = normalize(batch).to(device)

        with torch.no_grad():
            feats = model(batch)  # [B, FEATURE_DIM]

        all_features.append(feats.cpu().numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float16)
    return features


def save_route_features(output_path, timestamps, features):
    """Save route features to .npz file.

    Args:
        output_path: Path for .npz file
        timestamps: np.ndarray [T] float32 — time_s at each frame
        features: np.ndarray [T, D] float16 — per-frame features
    """
    np.savez_compressed(
        output_path,
        timestamps=timestamps.astype(np.float32),
        features=features.astype(np.float16),
    )


def validate_route_features(npz_path, expected_duration, tolerance_s=5.0):
    """Validate a saved feature file.

    Returns:
        dict with validation results
    """
    try:
        data = np.load(npz_path)
        ts = data["timestamps"]
        feats = data["features"]
    except Exception as e:
        return {"valid": False, "error": str(e)}

    result = {
        "valid": True,
        "n_frames": len(ts),
        "feature_dim": feats.shape[1] if feats.ndim == 2 else 0,
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "timestamps_monotonic": bool(np.all(np.diff(ts) >= 0)),
        "expected_frames": int(expected_duration * FPS),
        "dtype": str(feats.dtype),
    }

    if feats.ndim != 2 or feats.shape[1] != FEATURE_DIM:
        result["valid"] = False
        result["error"] = f"Bad feature shape: {feats.shape}"

    if not result["timestamps_monotonic"]:
        result["valid"] = False
        result["error"] = "Non-monotonic timestamps"

    # Check frame count is reasonable
    expected = expected_duration * FPS
    actual = len(ts)
    if expected > 0 and actual / expected < 0.5:
        result["warning"] = f"Low frame count: {actual} vs expected {expected:.0f}"

    return result
