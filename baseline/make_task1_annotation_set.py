#!/usr/bin/env python3
"""
make_task1_annotation_set.py — Build a human-annotation kit for Task 1 label validation.

Class-balanced stratified sample of Task-1 windows; extracts a short front-view clip per
window (niced, single-thread ffmpeg so it never competes with the V-JEPA2 extraction);
writes annotation_kit/{manifest.js, sample.csv} and (clips/<idx>.mp4).

Usage:
  python3 make_task1_annotation_set.py --manifest-only      # instant: manifest+sample.csv
  python3 make_task1_annotation_set.py --clips --limit 10   # preview clips (validate HTML)
  python3 make_task1_annotation_set.py --clips              # full 500 clips (run post-front)
"""
import argparse, csv, json, os, random, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, TASK1_LABELS
from paths import BATON_ROOT

KIT = Path(__file__).resolve().parent.parent / "annotation_kit"
CLIPS = KIT / "clips"
N_TOTAL = 500
PER_ROUTE_CAP = 2          # ≤2 windows per route for diversity
SEED = 42


def build_sample():
    rows = [r for r in csv.DictReader(open(BENCHMARK_DIR / "task1_action_samples.csv"))
            if r.get("has_qcamera") == "1"]
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    rng = random.Random(SEED)
    per_class = -(-N_TOTAL // len(TASK1_LABELS))  # ceil
    picked = []
    for lab in TASK1_LABELS:
        pool = by_label.get(lab, [])
        rng.shuffle(pool)
        seen = defaultdict(int)
        chosen = []
        for r in pool:
            if seen[r["route_id"]] >= PER_ROUTE_CAP:
                continue
            seen[r["route_id"]] += 1
            chosen.append(r)
            if len(chosen) >= per_class:
                break
        picked.extend(chosen)
    rng.shuffle(picked)
    picked = picked[:N_TOTAL]
    return picked


def write_manifest(sample):
    KIT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, r in enumerate(sample):
        manifest.append({
            "idx": i,
            "sample_id": r["sample_id"],
            "route_id": r["route_id"],
            "start": round(float(r["start_time_sec"]), 2),
            "end": round(float(r["end_time_sec"]), 2),
            "rule_label": r["label"],
            "adas_state": r.get("current_adas_state", ""),
            "clip": f"clips/{i:04d}.mp4",
        })
    (KIT / "manifest.js").write_text("const MANIFEST = " + json.dumps(manifest, indent=1) + ";\n")
    with open(KIT / "sample.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "sample_id", "route_id", "start", "end", "rule_label", "adas_state"])
        for m in manifest:
            w.writerow([m["idx"], m["sample_id"], m["route_id"], m["start"], m["end"],
                        m["rule_label"], m["adas_state"]])
    # class balance report
    from collections import Counter
    c = Counter(m["rule_label"] for m in manifest)
    print(f"manifest: {len(manifest)} windows | per-class: {dict(c)}")
    return manifest


def extract_clips(manifest, limit=0):
    CLIPS.mkdir(parents=True, exist_ok=True)
    items = manifest[:limit] if limit else manifest
    ok = miss = skip = 0
    t0 = time.time()
    for m in items:
        out = CLIPS / f"{m['idx']:04d}.mp4"
        if out.exists() and out.stat().st_size > 1000:
            skip += 1; ok += 1; continue
        qcam = BATON_ROOT / m["route_id"] / "qcamera.mp4"
        if not qcam.exists():
            miss += 1; continue
        dur = max(m["end"] - m["start"], 1.0)
        ffmpeg = "/usr/bin/ffmpeg" if Path("/usr/bin/ffmpeg").exists() else "ffmpeg"
        cmd = ["nice", "-n", "19", ffmpeg, "-v", "error", "-y",
               "-ss", str(m["start"]), "-t", str(dur), "-i", str(qcam),
               "-vf", "scale=320:-2", "-an", "-threads", "1",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
               "-pix_fmt", "yuv420p", str(out)]
        try:
            subprocess.run(cmd, check=True, timeout=120)
            ok += 1
        except Exception as e:
            miss += 1
            print(f"  clip {m['idx']} failed: {e}")
        if (ok + miss) % 25 == 0:
            print(f"  clips {ok}/{len(items)} ({skip} cached, {miss} miss) {time.time()-t0:.0f}s")
    print(f"clips done: {ok} ok, {miss} miss, {skip} cached in {time.time()-t0:.0f}s → {CLIPS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--clips", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="only first N clips (preview)")
    args = ap.parse_args()

    sample = build_sample()
    manifest = write_manifest(sample)
    if args.clips and not args.manifest_only:
        extract_clips(manifest, limit=args.limit)


if __name__ == "__main__":
    main()
