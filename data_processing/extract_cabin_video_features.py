#!/usr/bin/env python3
"""
extract_cabin_video_features.py — Extract frozen EfficientNet-B0 features from cabin camera.

Strategy:
  1. Locate dcamera.hevc segments using metadata.json timing
  2. Concatenate segments into temporary MP4
  3. Decode and extract features from concatenated file
  4. Delete temporary file immediately after extraction
  5. Fallback: segment-by-segment extraction if concatenation fails

Output: per-route .npz files in cabin_video_features/
Usage:  python3 extract_cabin_video_features.py [--smoke-test]
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
    decode_video_pipe, concatenate_segments, load_segment_timing,
    extract_features_from_frames, save_route_features, validate_route_features,
    OUTPUT_DIR, FPS, FEATURE_DIM, ENCODER_NAME,
)

logger = logging.getLogger("video_extraction")

CABIN_DIR = OUTPUT_DIR / "cabin_video_features"
TMP_DIR = OUTPUT_DIR / "_tmp_cabin"


def find_dcamera_segments(route_dir):
    """Find all dcamera.hevc segments, ordered by segment number.

    Returns:
        list of (seg_num, Path) sorted by seg_num, or empty list
    """
    route_dir = Path(route_dir)
    files = list(route_dir.glob("*--dcamera.hevc"))

    segments = []
    for f in files:
        try:
            seg_num = int(f.name.split("--")[0])
            if f.stat().st_size > 1000:  # skip empty/corrupt segments
                segments.append((seg_num, f))
        except (ValueError, OSError):
            continue

    segments.sort(key=lambda x: x[0])
    return segments


def extract_via_concatenation(dcam_segments, seg_timing, model, device, route_id, route_duration=None):
    """Concatenate all dcamera segments into temp MP4, then extract.

    Returns:
        (timestamps, features) or (None, None) on failure
    """
    TMP_DIR.mkdir(exist_ok=True)
    safe_name = route_id_to_filename(route_id)
    tmp_mp4 = TMP_DIR / f"{safe_name}.mp4"

    seg_files = [f for _, f in dcam_segments]

    # Scale concat timeout with number of segments
    concat_timeout = max(300, len(seg_files) * 15)

    try:
        ok = concatenate_segments(seg_files, tmp_mp4, timeout=concat_timeout)
        if not ok:
            logger.warning(f"  Concatenation failed for {route_id}")
            return None, None

        size_mb = tmp_mp4.stat().st_size / 1e6
        logger.info(f"  Concat OK: {len(seg_files)} segments → {size_mb:.1f} MB")

        # Decode full concatenated video
        frames, actual_dur = decode_video_pipe(tmp_mp4, duration_s=route_duration)
        if frames is None or len(frames) == 0:
            logger.warning(f"  Decode failed after concat for {route_id}")
            return None, None

        timestamps = np.arange(len(frames), dtype=np.float32) / FPS
        features = extract_features_from_frames(frames, model, device)

        return timestamps, features

    except Exception as e:
        logger.error(f"  Concat+extract error for {route_id}: {e}")
        return None, None

    finally:
        # Always clean up temp file
        if tmp_mp4.exists():
            tmp_mp4.unlink()
            logger.debug(f"  Cleaned up temp file: {tmp_mp4}")
        # Clean up concat list file if it exists
        txt = tmp_mp4.with_suffix(".txt")
        if txt.exists():
            txt.unlink()


def extract_via_segments(dcam_segments, seg_timing, model, device, route_id):
    """Extract features segment by segment (fallback).

    Returns:
        (timestamps, features) or (None, None) on failure
    """
    all_timestamps = []
    all_features = []

    for seg_num, seg_file in dcam_segments:
        # Get segment start time
        seg_start = 0.0
        if seg_timing:
            match = [s for s in seg_timing if s["seg_num"] == seg_num]
            if match:
                seg_start = match[0]["start_s"]
            else:
                # Estimate from segment number (~60s each)
                seg_start = seg_num * 60.0
        else:
            seg_start = seg_num * 60.0

        frames, seg_dur = decode_video_pipe(seg_file)
        if frames is None or len(frames) == 0:
            continue

        seg_ts = seg_start + np.arange(len(frames), dtype=np.float32) / FPS
        seg_feats = extract_features_from_frames(frames, model, device)

        all_timestamps.append(seg_ts)
        all_features.append(seg_feats)

    if not all_timestamps:
        return None, None

    timestamps = np.concatenate(all_timestamps)
    features = np.concatenate(all_features)
    return timestamps, features


def process_route(route_id, info, model, device):
    """Process a single route for cabin video features.

    Returns:
        dict with extraction results
    """
    route_dir = info["base_dir"]
    out_file = CABIN_DIR / f"{route_id_to_filename(route_id)}.npz"

    result = {
        "route_id": route_id,
        "method": None,
        "n_frames": 0,
        "n_segments": 0,
        "duration_s": 0.0,
        "success": False,
        "error": None,
    }

    # Find dcamera segments
    dcam_segments = find_dcamera_segments(route_dir)
    result["n_segments"] = len(dcam_segments)

    if not dcam_segments:
        result["error"] = "no_dcamera_segments"
        logger.warning(f"  No dcamera segments for {route_id}")
        return result

    # Load segment timing
    seg_timing = load_segment_timing(route_dir)

    try:
        # Try concatenation first
        timestamps, features = extract_via_concatenation(
            dcam_segments, seg_timing, model, device, route_id,
            route_duration=info["duration_sec"])

        if timestamps is not None:
            result["method"] = "concatenation"
        else:
            # Fallback to per-segment
            logger.info(f"  Falling back to per-segment extraction")
            timestamps, features = extract_via_segments(
                dcam_segments, seg_timing, model, device, route_id)
            if timestamps is not None:
                result["method"] = "per_segment"

        if timestamps is None or features is None:
            result["error"] = "extraction_failed"
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

    # Ensure cleanup
    for f in TMP_DIR.glob("*") if TMP_DIR.exists() else []:
        try:
            f.unlink()
        except Exception:
            pass

    return result


def select_smoke_test_routes(route_index):
    """Select diverse routes for smoke testing cabin video."""
    selected = []

    # Short route, medium route, long route
    by_dur = sorted(route_index.items(), key=lambda x: x[1]["duration_sec"])
    if len(by_dur) >= 3:
        selected.append(by_dur[5][0])       # short route (skip very shortest)
        selected.append(by_dur[len(by_dur)//2][0])  # medium
        selected.append(by_dur[-5][0])       # long route

    # Route with potentially problematic dcamera
    for rid, info in route_index.items():
        if rid in selected:
            continue
        segs = find_dcamera_segments(info["base_dir"])
        # Check for gaps or empty segments
        if segs:
            seg_nums = [s[0] for s in segs]
            expected = list(range(seg_nums[0], seg_nums[-1] + 1))
            if len(seg_nums) < len(expected):  # has gaps
                selected.append(rid)
                break

    # Ensure minimum count
    for rid in route_index:
        if len(selected) >= 5:
            break
        if rid not in selected:
            selected.append(rid)

    return selected[:6]


def main():
    parser = argparse.ArgumentParser(description="Extract cabin video features")
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
            logging.FileHandler(OUTPUT_DIR / "cabin_video_extraction.log"),
        ]
    )

    CABIN_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    logger.info("=" * 70)
    logger.info("Cabin Video Feature Extraction — PassingCtrl")
    logger.info(f"Encoder: {ENCODER_NAME} ({FEATURE_DIM}-d), FPS: {FPS}")
    logger.info("=" * 70)

    # Load model
    model = get_encoder(args.device)

    # Load route index
    route_index = load_route_index()
    logger.info(f"Routes indexed: {len(route_index)}")

    # Resume: find already-processed routes
    done = set()
    for f in CABIN_DIR.glob("*.npz"):
        done.add(f.stem)
    logger.info(f"Already processed: {len(done)} routes")

    # Filter
    to_process = {}
    for rid, info in route_index.items():
        fname = route_id_to_filename(rid)
        if fname not in done:
            to_process[rid] = info

    if args.smoke_test:
        test_routes = select_smoke_test_routes(route_index)
        # For smoke test, process even if already done (to verify)
        to_process = {rid: route_index[rid] for rid in test_routes if rid in route_index}
        # Remove existing files for smoke test routes to force re-extraction
        for rid in to_process:
            existing = CABIN_DIR / f"{route_id_to_filename(rid)}.npz"
            if existing.exists():
                existing.unlink()
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
        if (i + 1) % 5 == 0 or i == 0:
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
            logger.info(f"  OK: {result['n_frames']} frames, {result['duration_s']:.1f}s, "
                        f"method={result['method']}, segs={result['n_segments']}")
        else:
            n_fail += 1
            logger.warning(f"  FAIL: {result['error']}")

    elapsed = time.time() - t0
    logger.info(f"\nDone: {n_ok} ok, {n_fail} fail, {elapsed/60:.1f} min")

    # Method distribution
    from collections import Counter
    method_counts = Counter(r["method"] for r in results if r["success"])
    logger.info(f"Method distribution: {dict(method_counts)}")

    # Save results
    import json
    log_path = OUTPUT_DIR / "cabin_video_extraction_results.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {log_path}")

    # Clean up tmp dir
    if TMP_DIR.exists():
        for f in TMP_DIR.glob("*"):
            f.unlink()
        try:
            TMP_DIR.rmdir()
        except Exception:
            pass


if __name__ == "__main__":
    main()
