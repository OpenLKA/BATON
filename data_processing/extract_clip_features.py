#!/usr/bin/env python3
"""
extract_clip_features.py — Extract CLIP ViT-B/32 features from front & cabin cameras.

Reuses the same ffmpeg pipeline as EfficientNet extraction.
Output: per-route .npz files in clip_front_video_features/ and clip_cabin_video_features/

Usage:
  pip install open-clip-torch
  python3 extract_clip_features.py --camera front
  python3 extract_clip_features.py --camera cabin
"""
import argparse
import logging
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from video_utils import (
    load_route_index, route_id_to_filename,
    decode_video_pipe, concatenate_segments, load_segment_timing,
    save_route_features, OUTPUT_DIR, FPS,
)

logger = logging.getLogger("clip_extraction")

CLIP_FEATURE_DIM = 512  # ViT-B/32
CLIP_FRAME_SIZE = 224
BATCH_SIZE = 64


def get_clip_encoder(device="cuda"):
    """Load frozen CLIP ViT-B/32 visual encoder."""
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device,
        )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        logger.info(f"Loaded CLIP ViT-B/32 via open_clip ({CLIP_FEATURE_DIM}-d) on {device}")
        return model, preprocess, "open_clip"
    except ImportError:
        pass

    # Fallback: torchvision CLIP (if available via torch hub)
    try:
        import clip as clip_pkg
        model, preprocess = clip_pkg.load("ViT-B/32", device=device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        logger.info(f"Loaded CLIP ViT-B/32 via clip ({CLIP_FEATURE_DIM}-d) on {device}")
        return model, preprocess, "clip"
    except ImportError:
        pass

    raise RuntimeError(
        "Neither open_clip nor clip package found.\n"
        "Install one:  pip install open-clip-torch\n"
        "          or: pip install git+https://github.com/openai/CLIP.git"
    )


def extract_clip_features(frames, model, device, backend, batch_size=BATCH_SIZE):
    """Extract CLIP visual features from numpy frames.

    Args:
        frames: np.ndarray [N, 224, 224, 3] uint8
        model: CLIP model
        device: torch device
        backend: "open_clip" or "clip"
        batch_size: batch size

    Returns:
        features: np.ndarray [N, 512] float16
    """
    from torchvision import transforms
    # CLIP uses its own normalization
    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    n = len(frames)
    all_features = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_np = frames[start:end]

        # Convert to float tensor [0, 1] and normalize
        batch = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float() / 255.0
        batch = normalize(batch).to(device)

        with torch.no_grad():
            if backend == "open_clip":
                feats = model.encode_image(batch)
            else:
                feats = model.encode_image(batch)
            feats = feats.float()  # CLIP may output half

        all_features.append(feats.cpu().numpy())

    features = np.concatenate(all_features, axis=0).astype(np.float16)
    return features


def find_video_source(route_dir, camera):
    """Find video source for a camera (front or cabin)."""
    route_dir = Path(route_dir)

    if camera == "front":
        # Priority: qcamera.mp4 > qcamera.ts > fcamera.hevc
        mp4_files = sorted(route_dir.glob("ACM_MM/*/qcamera.mp4"))
        if mp4_files and mp4_files[0].stat().st_size > 1000:
            return "qcamera_mp4", mp4_files[0]

        ts_files = sorted(route_dir.glob("*--qcamera.ts"),
                          key=lambda f: int(f.name.split("--")[0]))
        valid_ts = [f for f in ts_files if f.stat().st_size > 500]
        if valid_ts:
            return "qcamera_ts", valid_ts

        fc_files = sorted(route_dir.glob("*--fcamera.hevc"),
                          key=lambda f: int(f.name.split("--")[0]))
        valid_fc = [f for f in fc_files if f.stat().st_size > 1000]
        if valid_fc:
            return "fcamera_hevc", valid_fc
    else:
        # Cabin: dcamera.hevc only
        dc_files = sorted(route_dir.glob("*--dcamera.hevc"),
                          key=lambda f: int(f.name.split("--")[0]))
        valid_dc = [f for f in dc_files if f.stat().st_size > 1000]
        if valid_dc:
            return "dcamera_hevc", valid_dc

    return None, None


def process_route(route_id, info, model, device, backend, camera, out_dir, tmp_dir):
    """Process a single route."""
    route_dir = info["base_dir"]
    out_file = out_dir / f"{route_id_to_filename(route_id)}.npz"

    if out_file.exists():
        return {"success": True, "skipped": True}

    src_type, source = find_video_source(route_dir, camera)
    if src_type is None:
        return {"success": False, "error": f"no_{camera}_video"}

    try:
        rdur = info["duration_sec"]
        if src_type.endswith("_mp4"):
            frames, dur = decode_video_pipe(source, fps=FPS, size=CLIP_FRAME_SIZE,
                                            duration_s=rdur)
        else:
            # Segments: concatenate first
            tmp_mp4 = tmp_dir / f"_tmp_{camera}_clip.mp4"
            concat_ok = concatenate_segments(source, tmp_mp4,
                                             timeout=max(300, len(source) * 15))
            if concat_ok:
                frames, dur = decode_video_pipe(tmp_mp4, fps=FPS, size=CLIP_FRAME_SIZE,
                                                duration_s=rdur)
                if tmp_mp4.exists():
                    tmp_mp4.unlink()
            else:
                # Per-segment fallback
                seg_timing = load_segment_timing(route_dir)
                all_frames = []
                all_ts = []
                cumtime = 0.0

                for seg_file in source:
                    seg_num = int(seg_file.name.split("--")[0])
                    if seg_timing:
                        match = [s for s in seg_timing if s["seg_num"] == seg_num]
                        if match:
                            cumtime = match[0]["start_s"]

                    seg_frames, seg_dur = decode_video_pipe(seg_file, fps=FPS,
                                                             size=CLIP_FRAME_SIZE)
                    if seg_frames is not None and len(seg_frames) > 0:
                        seg_ts = cumtime + np.arange(len(seg_frames), dtype=np.float32) / FPS
                        all_frames.append(seg_frames)
                        all_ts.append(seg_ts)
                    cumtime += (len(seg_frames) / FPS) if seg_frames is not None else 60.0

                if all_frames:
                    frames = np.concatenate(all_frames)
                    timestamps = np.concatenate(all_ts)
                    features = extract_clip_features(frames, model, device, backend)
                    save_route_features(out_file, timestamps, features)
                    return {"success": True, "n_frames": len(timestamps)}
                return {"success": False, "error": "no_frames"}

        if frames is None or len(frames) == 0:
            return {"success": False, "error": "decode_failed"}

        timestamps = np.arange(len(frames), dtype=np.float32) / FPS
        features = extract_clip_features(frames, model, device, backend)
        save_route_features(out_file, timestamps, features)
        return {"success": True, "n_frames": len(timestamps)}

    except Exception as e:
        return {"success": False, "error": str(e)[:100]}


def main():
    parser = argparse.ArgumentParser(description="Extract CLIP video features")
    parser.add_argument("--camera", required=True, choices=["front", "cabin"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(OUTPUT_DIR / f"clip_{args.camera}_extraction.log"),
        ],
    )

    out_dir = OUTPUT_DIR / f"clip_{args.camera}_video_features"
    out_dir.mkdir(exist_ok=True)
    tmp_dir = out_dir

    logger.info("=" * 60)
    logger.info(f"CLIP Feature Extraction — {args.camera} camera")
    logger.info(f"Model: ViT-B/32, {CLIP_FEATURE_DIM}-d, FPS: {FPS}")
    logger.info("=" * 60)

    model, preprocess, backend = get_clip_encoder(args.device)
    route_index = load_route_index()

    done = {f.stem for f in out_dir.glob("*.npz")}
    to_process = {rid: info for rid, info in route_index.items()
                  if route_id_to_filename(rid) not in done}

    logger.info(f"Routes: {len(route_index)} total, {len(done)} done, {len(to_process)} remaining")

    if not to_process:
        logger.info("All routes already processed.")
        return

    t0 = time.time()
    n_ok, n_fail, n_skip = 0, 0, 0

    for i, (rid, info) in enumerate(to_process.items()):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate / 60 if rate > 0 else 0
            logger.info(f"[{i+1}/{len(to_process)}] ({n_ok} ok, {n_fail} fail) ETA: {eta:.0f}min")

        result = process_route(rid, info, model, args.device, backend,
                               args.camera, out_dir, tmp_dir)
        if result.get("skipped"):
            n_skip += 1
        elif result["success"]:
            n_ok += 1
        else:
            n_fail += 1
            logger.warning(f"  FAIL {rid}: {result.get('error', '?')}")

    elapsed = time.time() - t0
    logger.info(f"\nDone: {n_ok} ok, {n_fail} fail, {n_skip} skip, {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
