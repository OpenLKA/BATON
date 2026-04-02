#!/usr/bin/env python3
"""
extract_front_video_features.py — Extract frozen EfficientNet-B0 features from front camera.

Source priority:
  1. qcamera.mp4 (concatenated in ACM_MM)
  2. raw qcamera.ts segments (concatenated on-the-fly)
  3. fcamera.hevc segments (only when qcamera unavailable)

Output: per-route .npz files in front_video_features/
Usage:  python3 extract_front_video_features.py [--smoke-test]
"""
import argparse
import logging
import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_utils import (
    get_encoder, load_route_index, route_id_to_filename,
    decode_video_pipe, get_video_duration, concatenate_segments,
    extract_features_from_frames, save_route_features, validate_route_features,
    load_segment_timing, OUTPUT_DIR, FPS, FEATURE_DIM, ENCODER_NAME,
)

logger = logging.getLogger("video_extraction")

FRONT_DIR = OUTPUT_DIR / "front_video_features"


def find_front_video_source(route_dir):
    """Determine the best front-camera source for this route.

    Returns:
        (source_type, source_path_or_files)
        source_type: 'qcamera_mp4' | 'qcamera_ts' | 'fcamera_hevc' | None
    """
    route_dir = Path(route_dir)

    # Priority 1: qcamera.mp4 in ACM_MM
    mp4_files = sorted(route_dir.glob("ACM_MM/*/qcamera.mp4"))
    if mp4_files:
        mp4 = mp4_files[0]
        if mp4.stat().st_size > 1000:
            return "qcamera_mp4", mp4

    # Priority 2: raw qcamera.ts segments
    ts_files = sorted(route_dir.glob("*--qcamera.ts"),
                      key=lambda f: int(f.name.split("--")[0]))
    valid_ts = [f for f in ts_files if f.stat().st_size > 500]
    if valid_ts:
        return "qcamera_ts", valid_ts

    # Priority 3: fcamera.hevc segments
    fc_files = sorted(route_dir.glob("*--fcamera.hevc"),
                      key=lambda f: int(f.name.split("--")[0]))
    valid_fc = [f for f in fc_files if f.stat().st_size > 1000]
    if valid_fc:
        return "fcamera_hevc", valid_fc

    return None, None


def extract_from_mp4(mp4_path, model, device, route_duration=None):
    """Extract features from a single MP4 file."""
    frames, actual_dur = decode_video_pipe(mp4_path, duration_s=route_duration)
    if frames is None or len(frames) == 0:
        return None, None, 0.0

    timestamps = np.arange(len(frames), dtype=np.float32) / FPS
    features = extract_features_from_frames(frames, model, device)
    return timestamps, features, actual_dur


def extract_from_segments(segment_files, model, device, route_dir, file_ext, route_duration=None):
    """Extract features from raw video segments.

    First attempts concatenation to temp file; falls back to per-segment extraction.
    """
    # Try concatenation first
    tmp_mp4 = FRONT_DIR / "_tmp_front_concat.mp4"
    concat_timeout = max(300, len(segment_files) * 15)
    concat_ok = concatenate_segments(segment_files, tmp_mp4, timeout=concat_timeout)

    if concat_ok:
        logger.info(f"  Concat OK ({len(segment_files)} segments → {tmp_mp4.stat().st_size/1e6:.1f} MB)")
        try:
            ts, feats, dur = extract_from_mp4(tmp_mp4, model, device, route_duration=route_duration)
            return ts, feats, dur
        except Exception as e:
            logger.warning(f"  Concat extract failed: {e}, falling back to per-segment")
        finally:
            if tmp_mp4.exists():
                tmp_mp4.unlink()

    # Fallback: per-segment extraction
    logger.info(f"  Falling back to per-segment extraction ({len(segment_files)} segments)")
    seg_timing = load_segment_timing(route_dir)

    all_timestamps = []
    all_features = []
    cumulative_time = 0.0

    for i, seg_file in enumerate(segment_files):
        seg_num = int(seg_file.name.split("--")[0])

        # Get segment start time from metadata
        if seg_timing:
            match = [s for s in seg_timing if s["seg_num"] == seg_num]
            if match:
                cumulative_time = match[0]["start_s"]

        frames, seg_dur = decode_video_pipe(seg_file)
        if frames is None or len(frames) == 0:
            # Advance time by ~60s for missing segment
            if seg_timing and match:
                cumulative_time = match[0]["end_s"]
            else:
                cumulative_time += 60.0
            continue

        seg_ts = cumulative_time + np.arange(len(frames), dtype=np.float32) / FPS
        seg_feats = extract_features_from_frames(frames, model, device)

        all_timestamps.append(seg_ts)
        all_features.append(seg_feats)

        cumulative_time += len(frames) / FPS

    if not all_timestamps:
        return None, None, 0.0

    timestamps = np.concatenate(all_timestamps)
    features = np.concatenate(all_features)
    total_dur = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0

    # Clean up temp file if it exists
    if tmp_mp4.exists():
        tmp_mp4.unlink()

    return timestamps, features, total_dur


def process_route(route_id, info, model, device):
    """Process a single route for front video features.

    Returns:
        dict with: source_type, n_frames, duration_s, success, error
    """
    route_dir = info["base_dir"]
    out_file = FRONT_DIR / f"{route_id_to_filename(route_id)}.npz"

    result = {
        "route_id": route_id,
        "source_type": None,
        "n_frames": 0,
        "duration_s": 0.0,
        "success": False,
        "error": None,
    }

    # Find source
    source_type, source = find_front_video_source(route_dir)
    result["source_type"] = source_type

    if source_type is None:
        result["error"] = "no_front_video"
        logger.warning(f"  No front video for {route_id}")
        return result

    try:
        rdur = info["duration_sec"]
        if source_type == "qcamera_mp4":
            timestamps, features, dur = extract_from_mp4(source, model, device, route_duration=rdur)
        else:
            timestamps, features, dur = extract_from_segments(
                source, model, device, route_dir,
                ".ts" if source_type == "qcamera_ts" else ".hevc",
                route_duration=rdur,
            )

        if timestamps is None or features is None:
            result["error"] = "decode_failed"
            return result

        save_route_features(out_file, timestamps, features)

        result["n_frames"] = len(timestamps)
        result["duration_s"] = float(timestamps[-1]) if len(timestamps) > 0 else 0.0
        result["success"] = True

        # Validate
        val = validate_route_features(out_file, info["duration_sec"])
        if not val["valid"]:
            result["warning"] = val.get("error", "validation_failed")
        if "warning" in val:
            result["warning"] = val["warning"]

    except Exception as e:
        result["error"] = str(e)[:100]
        logger.error(f"  Error processing {route_id}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract front video features")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Process only a few test routes")
    parser.add_argument("--device", default="cuda", help="torch device")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / "front_video_extraction.log"),
        ]
    )

    FRONT_DIR.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("Front Video Feature Extraction — PassingCtrl")
    logger.info(f"Encoder: {ENCODER_NAME} ({FEATURE_DIM}-d), FPS: {FPS}")
    logger.info("=" * 70)

    # Load model
    model = get_encoder(args.device)

    # Load route index
    route_index = load_route_index()
    logger.info(f"Routes indexed: {len(route_index)}")

    # Resume: find already-processed routes
    done = set()
    for f in FRONT_DIR.glob("*.npz"):
        rid = filename_to_route_id(f.stem)
        done.add(f.stem)  # use filename stem for matching
    logger.info(f"Already processed: {len(done)} routes")

    # Filter routes to process
    to_process = {}
    for rid, info in route_index.items():
        fname = route_id_to_filename(rid)
        if fname not in done:
            to_process[rid] = info

    if args.smoke_test:
        # Select diverse test routes
        test_routes = select_smoke_test_routes(route_index, to_process)
        to_process = {rid: route_index[rid] for rid in test_routes if rid in route_index}
        logger.info(f"Smoke test: {len(to_process)} routes selected")

    logger.info(f"Routes to process: {len(to_process)}")

    if not to_process:
        logger.info("Nothing to process. All routes done.")
        return

    # Process routes
    t0 = time.time()
    results = []
    n_ok, n_fail = 0, 0

    for i, (rid, info) in enumerate(to_process.items()):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(f"[{i+1}/{len(to_process)}] {rid[:45]}  "
                        f"({n_ok} ok, {n_fail} fail)  ETA: {eta:.0f}min")

        logger.info(f"  Processing: {rid}")
        result = process_route(rid, info, model, args.device)
        results.append(result)

        if result["success"]:
            n_ok += 1
            src = result["source_type"]
            logger.info(f"  OK: {result['n_frames']} frames, {result['duration_s']:.1f}s, source={src}")
        else:
            n_fail += 1
            logger.warning(f"  FAIL: {result['error']}")

    elapsed = time.time() - t0
    logger.info(f"\nDone: {n_ok} ok, {n_fail} fail, {elapsed/60:.1f} min")

    # Source distribution
    from collections import Counter
    src_counts = Counter(r["source_type"] for r in results if r["success"])
    logger.info(f"Source distribution: {dict(src_counts)}")

    # Save results log
    import json
    log_path = OUTPUT_DIR / "front_video_extraction_results.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {log_path}")


def select_smoke_test_routes(full_index, pending):
    """Select a diverse set of routes for smoke testing."""
    selected = []

    # Pick routes by source type
    for rid, info in full_index.items():
        if len(selected) >= 6:
            break
        route_dir = info["base_dir"]
        src_type, _ = find_front_video_source(route_dir)

        if src_type == "qcamera_mp4" and sum(1 for r in selected if "mp4" in str(full_index.get(r, {}).get("_src", ""))) < 3:
            info["_src"] = "mp4"
            selected.append(rid)
        elif src_type == "qcamera_ts" and "ts" not in str([full_index.get(r, {}).get("_src", "") for r in selected]):
            info["_src"] = "ts"
            selected.append(rid)
        elif src_type == "fcamera_hevc" and "fc" not in str([full_index.get(r, {}).get("_src", "") for r in selected]):
            info["_src"] = "fc"
            selected.append(rid)

    # Ensure at least 3 qcamera_mp4 routes if possible
    if len(selected) < 3:
        for rid in list(pending.keys())[:3]:
            if rid not in selected:
                selected.append(rid)

    return selected[:8]


def filename_to_route_id(stem):
    """Helper for resume matching."""
    return stem.replace("__", "/")


if __name__ == "__main__":
    main()
