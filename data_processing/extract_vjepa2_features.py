#!/usr/bin/env python3
"""
extract_vjepa2_features.py — Frozen V-JEPA2 ViT-L clip-embedding extractor.

For each route-bundle video (front qcamera.mp4 / cabin dcamera.mp4), decode short
clips at a fixed stride, encode each with a FROZEN V-JEPA2 ViT-L/256 (1024-d), and
mean-pool the encoder tokens to one embedding per clip. Saves a per-route sequence
{timestamps[N], features[N,1024]} in the SAME npz cache contract as the EfficientNet
extractor, so it drops into RouteCache/dataset unchanged (dim switches via --use-vjepa).

Each clip summarizes a `clip_span`-second window ENDING at its timestamp `t` (recent
past → causal for prediction). Stride 0.5s, clip_frames 16 → ~10 clips per 5s window
(matching VIDEO_SEQ_LEN=10).

Usage:
  python3 extract_vjepa2_features.py --camera front
  python3 extract_vjepa2_features.py --camera cabin --limit 3   # smoke
"""
import argparse, logging, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from config import VJEPA2_FRONT_VIDEO_DIR, VJEPA2_CABIN_VIDEO_DIR  # noqa: E402
from paths import discover_segments, route_uid, npz_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vjepa2")

CKPT = "facebook/vjepa2-vitl-fpc64-256"
CROP = 256
CLIP_FRAMES = 16          # frames per clip (divisible by tubelet=2); native-fidelity
CLIP_SPAN = 2.0           # seconds each clip spans (8 fps within clip)
STRIDE = 0.5              # seconds between clip timestamps (matches sample stride)
BATCH_CLIPS = 64          # clips per model forward — tuned to fill RTX 5090 32GB VRAM
DECORD_THREADS = 12       # decord CPU decode threads (decode is the bottleneck)
FEATURE_DIM = 1024


def get_model():
    from transformers import VJEPA2Model, VJEPA2VideoProcessor
    proc = VJEPA2VideoProcessor.from_pretrained(CKPT)
    model = VJEPA2Model.from_pretrained(CKPT, torch_dtype=torch.bfloat16).to("cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, proc


def resolve_video(seg_dir, camera):
    p = seg_dir / ("qcamera.mp4" if camera == "front" else "dcamera.mp4")
    return p if p.exists() and p.stat().st_size > 1000 else None


@torch.no_grad()
def extract_route(video_path, model, proc):
    """Return (timestamps[N], features[N,1024] float16) or (None, None)."""
    import decord
    decord.bridge.set_bridge("native")
    try:
        vr = decord.VideoReader(str(video_path), ctx=decord.cpu(0),
                                width=CROP, height=CROP, num_threads=DECORD_THREADS)
    except Exception as e:
        logger.warning(f"decode-open failed {video_path}: {e}")
        return None, None
    n = len(vr)
    fps = float(vr.get_avg_fps()) or 20.0
    dur = n / fps
    if dur < CLIP_SPAN + STRIDE:
        return None, None

    clip_ts = np.arange(CLIP_SPAN, dur, STRIDE)              # window-end times
    # per-clip frame indices (CLIP_FRAMES samples spanning [t-span, t])
    rel = np.linspace(-CLIP_SPAN, 0.0, CLIP_FRAMES)
    feats = np.zeros((len(clip_ts), FEATURE_DIM), dtype=np.float16)

    for b0 in range(0, len(clip_ts), BATCH_CLIPS):
        bts = clip_ts[b0:b0 + BATCH_CLIPS]
        # frame indices for this batch of clips
        idx_per_clip = [np.clip(np.round((t + rel) * fps).astype(int), 0, n - 1)
                        for t in bts]
        uniq = np.unique(np.concatenate(idx_per_clip))
        frames = vr.get_batch(list(uniq)).asnumpy()          # [U, H, W, 3] uint8
        pos = {int(u): k for k, u in enumerate(uniq)}
        clips = [frames[[pos[int(i)] for i in idxs]] for idxs in idx_per_clip]  # each [16,H,W,3]
        inp = proc(videos=clips, return_tensors="pt")["pixel_values_videos"]
        inp = inp.to("cuda", torch.bfloat16)
        out = model(pixel_values_videos=inp, skip_predictor=True)
        pooled = out.last_hidden_state.float().mean(1).cpu().numpy()  # [b,1024]
        feats[b0:b0 + len(bts)] = pooled.astype(np.float16)

    return clip_ts.astype(np.float32), feats


def main():
    global BATCH_CLIPS, DECORD_THREADS, CLIP_FRAMES
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", required=True, choices=["front", "cabin"])
    ap.add_argument("--limit", type=int, default=0, help="process only first N routes (smoke)")
    ap.add_argument("--shard", type=int, default=0, help="this worker's shard index")
    ap.add_argument("--num-shards", type=int, default=1, help="total parallel workers")
    ap.add_argument("--batch", type=int, default=BATCH_CLIPS, help="clips per forward")
    ap.add_argument("--threads", type=int, default=DECORD_THREADS, help="decord decode threads")
    ap.add_argument("--clip-frames", type=int, default=CLIP_FRAMES,
                    help="frames per clip (front=16 native; cabin=8 for speed)")
    args = ap.parse_args()
    BATCH_CLIPS = args.batch
    DECORD_THREADS = args.threads
    CLIP_FRAMES = args.clip_frames

    out_dir = VJEPA2_FRONT_VIDEO_DIR if args.camera == "front" else VJEPA2_CABIN_VIDEO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    segs = discover_segments()
    if args.limit:
        segs = segs[:args.limit]
    segs = segs[args.shard::args.num_shards]  # this worker's route shard
    logger.info(f"V-JEPA2 [{args.camera}] shard {args.shard}/{args.num_shards}: "
                f"{len(segs)} routes (clip_frames={CLIP_FRAMES}, batch={BATCH_CLIPS}, "
                f"threads={DECORD_THREADS}) → {out_dir}")
    model, proc = get_model()

    ok = miss = skip = 0
    t0 = time.time()
    for i, seg in enumerate(segs):
        uid = route_uid(seg)
        out_path = out_dir / f"{npz_key(uid)}.npz"
        if out_path.exists():
            skip += 1; ok += 1; continue
        vp = resolve_video(seg, args.camera)
        if vp is None:
            miss += 1; continue
        ts, feats = extract_route(vp, model, proc)
        if ts is None:
            miss += 1; continue
        np.savez_compressed(out_path, timestamps=ts, features=feats)
        ok += 1
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            logger.info(f"[{i+1}/{len(segs)}] ok={ok} miss={miss} skip={skip} "
                        f"{el:.0f}s ({el/max(i+1,1):.1f}s/route)")
    logger.info(f"DONE [{args.camera}]: ok={ok} miss={miss} skip={skip} "
                f"in {time.time()-t0:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
