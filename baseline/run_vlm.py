#!/usr/bin/env python3
"""
run_vlm.py — Zero-shot VLM baselines for PassingCtrl benchmark.

Providers: OpenAI GPT-4o + Google Gemini 2.5 Flash
Modalities: Text, Front, Cabin, Text+Front, Text+Cabin, Full, All

Memory-safe: designed to run alongside GPU training (~500MB peak).

Usage:
  python3 run_vlm.py                          # all 42 configs
  python3 run_vlm.py --model gpt4o            # GPT-4o only
  python3 run_vlm.py --model gemini --task task1
  python3 run_vlm.py --dry-run
"""
import argparse
import asyncio
import gc
import json
import logging
import numpy as np
import os
import pandas as pd
import re
import subprocess
import sys
import time
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR, DATASET_ROOT, CACHE_DIR, RESULTS_DIR, LABEL2IDX
from metrics import evaluate_task1, evaluate_binary
from vlm_prompts import (
    build_text_context, build_gps_context, preload_gps_routes,
    build_openai_messages, build_gemini_contents, get_gemini_system_instruction,
    parse_task1_response, parse_binary_response,
    SYSTEM_PROMPT, TASK_PROMPTS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vlm")

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

GEMINI_KEY_PATH = Path("/home/henry/Desktop/Drive/HMI/GEMINI_API.txt")
VLM_CACHE = CACHE_DIR / "vlm"
FRAMES_CACHE = VLM_CACHE / "frames"
SAMPLES_CACHE = VLM_CACHE / "samples"
RESPONSES_CACHE = VLM_CACHE / "responses"

SEED = 42
MIN_GAP_SEC = 5.0
TASKS = ["task1", "task2", "task3"]
SAMPLE_N = {"task1": 350, "task2": 300, "task3": 300}

MODALITIES = [
    "VLM-Text", "VLM-Front", "VLM-Cabin",
    "VLM-Front+Cabin", "VLM-Full", "VLM-All",
]

# What each modality needs — aligned with trained baseline ablation:
#   Veh(CAN) → VLM-Text | FV → VLM-Front | CV → VLM-Cabin
#   FV+CV → VLM-Front+Cabin | FV+CV+CAN → VLM-Full | FV+CV+CAN+GPS → VLM-All
MOD_NEEDS = {
    "VLM-Text":         {"text": True,  "gps": False, "front": False, "cabin": False},
    "VLM-Front":        {"text": False, "gps": False, "front": True,  "cabin": False},
    "VLM-Cabin":        {"text": False, "gps": False, "front": False, "cabin": True},
    "VLM-Front+Cabin":  {"text": False, "gps": False, "front": True,  "cabin": True},
    "VLM-Full":         {"text": True,  "gps": False, "front": True,  "cabin": True},
    "VLM-All":          {"text": True,  "gps": True,  "front": True,  "cabin": True},
}

MODELS = {
    "gpt4o": {"provider": "openai", "model_id": "gpt-4o", "display": "GPT-4o"},
    "gemini": {"provider": "gemini", "model_id": "gemini-2.5-flash", "display": "Gemini 2.5 Flash"},
}

FRAME_OFFSETS = [0.5, 2.5, 4.5]  # seconds into the 5s window

# Memory safety thresholds (MB)
MEM_PAUSE_THRESHOLD = 3000   # pause when MemAvailable < 3GB
MEM_RESUME_THRESHOLD = 5000  # resume when MemAvailable > 5GB
MEM_CHECK_INTERVAL = 10      # seconds between checks while paused


# ═══════════════════════════════════════════════════════════
# MEMORY SAFETY
# ═══════════════════════════════════════════════════════════

def _get_mem_available_mb():
    """Read MemAvailable from /proc/meminfo (Linux only)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024  # kB -> MB
    except Exception:
        return 99999  # assume safe if can't read


def _set_low_priority():
    """Set process to lowest CPU/IO priority. Protects training from contention."""
    try:
        os.nice(19)  # lowest CPU priority
    except OSError:
        pass
    try:
        # ionice: class 3 = idle (only uses IO when no other process needs it)
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())],
                       capture_output=True, timeout=5)
    except Exception:
        pass


def _set_oom_score():
    """Set high OOM score so Linux kills us first, not the training process."""
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write("500\n")  # range -1000..1000, higher = killed first
    except Exception:
        pass


def memory_guard():
    """Block until system memory is above safe threshold."""
    avail = _get_mem_available_mb()
    if avail >= MEM_PAUSE_THRESHOLD:
        return
    logger.warning(f"⚠ MemAvailable={avail:.0f}MB < {MEM_PAUSE_THRESHOLD}MB, "
                   f"pausing until > {MEM_RESUME_THRESHOLD}MB...")
    gc.collect()
    while True:
        time.sleep(MEM_CHECK_INTERVAL)
        avail = _get_mem_available_mb()
        if avail >= MEM_RESUME_THRESHOLD:
            logger.info(f"✓ MemAvailable={avail:.0f}MB, resuming")
            return
        # Still low — log every 60s
        logger.warning(f"  still waiting... MemAvailable={avail:.0f}MB")


async def async_memory_guard():
    """Async version: yield to event loop while waiting for memory."""
    avail = _get_mem_available_mb()
    if avail >= MEM_PAUSE_THRESHOLD:
        return
    logger.warning(f"⚠ MemAvailable={avail:.0f}MB < {MEM_PAUSE_THRESHOLD}MB, pausing...")
    gc.collect()
    while True:
        await asyncio.sleep(MEM_CHECK_INTERVAL)
        avail = _get_mem_available_mb()
        if avail >= MEM_RESUME_THRESHOLD:
            logger.info(f"✓ MemAvailable={avail:.0f}MB, resuming")
            return


# ═══════════════════════════════════════════════════════════
# SAMPLING
# ═══════════════════════════════════════════════════════════

def load_split_test_routes():
    split = json.load(open(BENCHMARK_DIR / "split_cross_driver.json"))
    return set(split["test_routes"])


def subsample_task(task, n_total):
    """Stratified subsample of test set. Returns DataFrame."""
    cache_path = SAMPLES_CACHE / f"{task}_samples.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path)

    SAMPLES_CACHE.mkdir(parents=True, exist_ok=True)
    test_routes = load_split_test_routes()

    if task == "task1":
        df = pd.read_csv(BENCHMARK_DIR / "task1_action_samples.csv")
    elif task == "task2":
        df = pd.read_csv(BENCHMARK_DIR / "task2_activation_samples_h3.csv")
    else:
        df = pd.read_csv(BENCHMARK_DIR / "task3_takeover_samples_h3.csv")

    df = df[df["route_id"].isin(test_routes)]
    df = df[df["has_qcamera"] == 1]

    rng = np.random.RandomState(SEED)

    if task == "task1":
        # 50 per class
        n_per_class = n_total // 7
        selected = []
        for label in sorted(df["label"].unique()):
            sub = df[df["label"] == label].sort_values(["route_id", "start_time_sec"])
            # Enforce min gap
            filtered = _enforce_gap(sub)
            if len(filtered) >= n_per_class:
                chosen = filtered.sample(n=n_per_class, random_state=rng)
            else:
                chosen = filtered
            selected.append(chosen)
        result = pd.concat(selected, ignore_index=True)
    else:
        # Stratified binary
        pos = df[df["label"] == 1].sort_values(["route_id", "start_time_sec"])
        neg = df[df["label"] == 0].sort_values(["route_id", "start_time_sec"])
        pos_filtered = _enforce_gap(pos)
        neg_filtered = _enforce_gap(neg)
        pos_rate = df["label"].mean()
        n_pos = max(int(n_total * pos_rate), 20)
        n_neg = n_total - n_pos
        pos_chosen = pos_filtered.sample(n=min(n_pos, len(pos_filtered)), random_state=rng)
        neg_chosen = neg_filtered.sample(n=min(n_neg, len(neg_filtered)), random_state=rng)
        result = pd.concat([pos_chosen, neg_chosen], ignore_index=True)

    result = result.sample(frac=1, random_state=rng).reset_index(drop=True)
    result.to_csv(cache_path, index=False)
    logger.info(f"Subsampled {task}: {len(result)} samples -> {cache_path}")
    return result


def _enforce_gap(df):
    """Remove overlapping samples (min 5s gap within same route)."""
    kept = []
    last_time = {}
    for _, row in df.iterrows():
        rid = row["route_id"]
        t = row["start_time_sec"]
        if rid not in last_time or (t - last_time[rid]) >= MIN_GAP_SEC:
            kept.append(row)
            last_time[rid] = t
    return pd.DataFrame(kept)


# ═══════════════════════════════════════════════════════════
# FRAME EXTRACTION
# ═══════════════════════════════════════════════════════════

_routes_df = None
def _get_routes():
    global _routes_df
    if _routes_df is None:
        _routes_df = pd.read_csv(BENCHMARK_DIR / "routes.csv")
    return _routes_df


def _get_acm_dir(route_id):
    routes = _get_routes()
    row = routes[routes["route_id"] == route_id]
    if len(row) == 0:
        return None
    vm = row["vehicle_model"].values[0]
    driver, rhash = route_id.split("/")
    base = DATASET_ROOT / vm / driver / rhash
    acm_dirs = sorted(base.glob("ACM_MM/route_*"))
    return acm_dirs[0] if acm_dirs else None


def _get_segment_info(route_id):
    """Return (route_base_dir, seg_start) for dcamera segment mapping."""
    routes = _get_routes()
    row = routes[routes["route_id"] == route_id]
    if len(row) == 0:
        return None, None
    vm = row["vehicle_model"].values[0]
    driver, rhash = route_id.split("/")
    base = DATASET_ROOT / vm / driver / rhash
    acm_dirs = sorted(base.glob("ACM_MM/route_*"))
    if not acm_dirs:
        return base, 0
    meta_path = acm_dirs[0] / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        seg_range = meta.get("segment_range", [0, 0])
        return base, seg_range[0]
    return base, 0


def extract_front_frames(sample_id, route_id, start_time):
    """Extract 3 front camera frames. Returns list of paths or None."""
    out_dir = FRAMES_CACHE / "front" / sample_id
    paths = [out_dir / f"frame_{i}.jpg" for i in range(3)]
    if all(p.exists() for p in paths):
        return paths

    acm = _get_acm_dir(route_id)
    if acm is None:
        return None
    qcam = acm / "qcamera.mp4"
    if not qcam.exists():
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, offset in enumerate(FRAME_OFFSETS):
        t = start_time + offset
        out = paths[i]
        if out.exists():
            continue
        cmd = [
            "nice", "-n", "19", "ffmpeg", "-v", "quiet",
            "-ss", f"{t:.3f}", "-i", str(qcam),
            "-frames:v", "1", "-q:v", "3", str(out), "-y",
        ]
        subprocess.run(cmd, timeout=30)
    return paths if all(p.exists() for p in paths) else None


def extract_cabin_frames(sample_id, route_id, start_time):
    """Extract 3 cabin camera frames from dcamera.hevc segments. Returns list of paths or None."""
    out_dir = FRAMES_CACHE / "cabin" / sample_id
    paths = [out_dir / f"frame_{i}.jpg" for i in range(3)]
    if all(p.exists() for p in paths):
        return paths

    route_base, seg_start = _get_segment_info(route_id)
    if route_base is None:
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, offset in enumerate(FRAME_OFFSETS):
        t = start_time + offset
        seg_num = seg_start + int(t // 60)
        seek = t % 60
        hevc_path = route_base / f"{seg_num}--dcamera.hevc"
        out = paths[i]
        if out.exists():
            continue
        if not hevc_path.exists():
            return None
        cmd = [
            "nice", "-n", "19", "ffmpeg", "-v", "quiet",
            "-ss", f"{seek:.3f}", "-i", str(hevc_path),
            "-frames:v", "1", "-vf", "scale=964:604", "-q:v", "3",
            str(out), "-y",
        ]
        subprocess.run(cmd, timeout=30)
    return paths if all(p.exists() for p in paths) else None


def extract_all_frames(samples_dict):
    """Pre-extract all needed frames for all tasks."""
    logger.info("Extracting video frames...")
    all_samples = []
    for task, df in samples_dict.items():
        for _, row in df.iterrows():
            all_samples.append(row)

    # Deduplicate by (route_id, start_time_sec)
    seen = set()
    unique = []
    for row in all_samples:
        key = (row["route_id"], row["start_time_sec"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    n_front_ok, n_cabin_ok, n_fail = 0, 0, 0
    for i, row in enumerate(tqdm(unique, desc="Extracting frames", unit="sample")):
        if i % 20 == 0:
            memory_guard()
        sid = row["sample_id"]
        rid = row["route_id"]
        t0 = row["start_time_sec"]
        front = extract_front_frames(sid, rid, t0)
        cabin = extract_cabin_frames(sid, rid, t0)
        if front:
            n_front_ok += 1
        if cabin:
            n_cabin_ok += 1
        if not front and not cabin:
            n_fail += 1

    logger.info(f"Frames: {n_front_ok} front, {n_cabin_ok} cabin, {n_fail} failed")


# ═══════════════════════════════════════════════════════════
# API INFERENCE
# ═══════════════════════════════════════════════════════════

_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.AsyncOpenAI()
    return _openai_client


async def call_openai(messages, model_id="gpt-4o", semaphore=None):
    """Call OpenAI API. Returns response text."""
    client = _get_openai_client()
    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=50,
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"OpenAI error: {e}")
            return None


def _call_gemini_sync(parts, task, gemini_client, model_id="gemini-2.5-flash"):
    """Synchronous Gemini call with 429 retry (avoids aiohttp DNS bug)."""
    from google.genai import types
    for attempt in range(5):
        try:
            resp = gemini_client.models.generate_content(
                model=model_id,
                contents=parts,
                config=types.GenerateContentConfig(
                    system_instruction=get_gemini_system_instruction(task),
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    temperature=0.0,
                    max_output_tokens=50,
                ),
            )
            return resp.text.strip() if resp.text else None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                wait = min(30 * (attempt + 1), 120)
                logger.info(f"  Rate limited, waiting {wait}s (attempt {attempt+1}/5)")
                time.sleep(wait)
                continue
            logger.warning(f"Gemini error: {e}")
            return None
    logger.warning("Gemini: 5 retries exhausted")
    return None


async def call_gemini(parts, task, gemini_client, model_id="gemini-2.5-flash", semaphore=None):
    """Call Gemini API via sync client in thread executor (avoids aiohttp DNS issues)."""
    async with semaphore:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, _call_gemini_sync, parts, task, gemini_client, model_id
        )


async def run_inference_batch(task, modality, model_key, samples_df, rpm=None):
    """Run inference for one (task, modality, model) config."""
    model_info = MODELS[model_key]
    needs = MOD_NEEDS[modality]
    provider = model_info["provider"]

    run_name = f"{task}_{modality}_{model_key}_cross_driver_h3_s{SEED}"
    result_path = RESULTS_DIR / run_name / "results.json"
    if result_path.exists():
        logger.info(f"  [SKIP] {run_name}")
        return True

    # Response cache
    resp_dir = RESPONSES_CACHE / model_key / f"{task}_{modality}"
    resp_dir.mkdir(parents=True, exist_ok=True)

    # Setup client (low concurrency to limit memory)
    if provider == "openai":
        concurrency = 3
        default_rpm = 60
    else:
        from google import genai
        gemini_key = GEMINI_KEY_PATH.read_text().strip()
        gemini_client = genai.Client(api_key=gemini_key)
        concurrency = 10
        default_rpm = 200

    if rpm is None:
        rpm = default_rpm
    delay = 60.0 / rpm
    sem = asyncio.Semaphore(concurrency)

    results = []
    n_parse_fail = 0

    async def process_one(idx, row):
        nonlocal n_parse_fail
        sid = row["sample_id"]
        rid = row["route_id"]
        t0 = row["start_time_sec"]
        t1 = row["end_time_sec"]

        # Check response cache
        resp_path = resp_dir / f"{sid}.json"
        if resp_path.exists():
            with open(resp_path) as f:
                cached = json.load(f)
            return cached

        # Build inputs
        text_ctx = build_text_context(rid, t0, t1) if needs["text"] else None
        gps_ctx = build_gps_context(rid, t0, t1) if needs["gps"] else None
        front_frames = None
        cabin_frames = None
        if needs["front"]:
            front_frames = extract_front_frames(sid, rid, t0)
        if needs["cabin"]:
            cabin_frames = extract_cabin_frames(sid, rid, t0)

        # Make API call
        resp_text = None
        for attempt in range(2):
            if provider == "openai":
                msgs = build_openai_messages(task, text_ctx, gps_ctx, front_frames, cabin_frames)
                resp_text = await call_openai(msgs, model_info["model_id"], sem)
            else:
                parts = build_gemini_contents(task, text_ctx, gps_ctx, front_frames, cabin_frames)
                resp_text = await call_gemini(parts, task, gemini_client, model_info["model_id"], sem)

            if resp_text is not None:
                break
            await asyncio.sleep(delay)

        # Parse
        parsed = None
        if resp_text:
            if task == "task1":
                label = parse_task1_response(resp_text)
                if label:
                    parsed = {"label": label}
                else:
                    n_parse_fail += 1
            else:
                result = parse_binary_response(resp_text)
                if result:
                    parsed = {"prediction": result[0], "confidence": result[1]}
                else:
                    n_parse_fail += 1
                    parsed = {"prediction": 0, "confidence": 0.5}

        entry = {
            "sample_id": sid,
            "response": resp_text,
            "parsed": parsed,
        }
        # Save response cache
        with open(resp_path, "w") as f:
            json.dump(entry, f)
        return entry

    # Process with rate limiting + memory guard
    entries = []
    for idx, (_, row) in enumerate(tqdm(
        samples_df.iterrows(), total=len(samples_df),
        desc=f"  {model_key}/{modality}", unit="call",
    )):
        if idx % 20 == 0:
            await async_memory_guard()
        entry = await process_one(idx, row)
        entries.append(entry)
        if not (resp_dir / f"{row['sample_id']}.json").stat().st_mtime < time.time() - 1:
            # Only delay for new calls (not cached)
            await asyncio.sleep(delay)

    # Compute metrics
    compute_and_save_metrics(task, modality, model_key, samples_df, entries, n_parse_fail)
    return True


def compute_and_save_metrics(task, modality, model_key, samples_df, entries, n_parse_fail):
    """Compute metrics and save results.json."""
    run_name = f"{task}_{modality}_{model_key}_cross_driver_h3_s{SEED}"
    run_dir = RESULTS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if task == "task1":
        y_true = []
        y_logits = []
        n_valid = 0
        for entry, (_, row) in zip(entries, samples_df.iterrows()):
            if entry["parsed"] is None:
                continue
            true_label = LABEL2IDX[row["label"]]
            pred_label = LABEL2IDX.get(entry["parsed"]["label"])
            if pred_label is None:
                continue
            y_true.append(true_label)
            # One-hot logits
            logit = np.zeros(7, dtype=np.float32)
            logit[pred_label] = 1.0
            y_logits.append(logit)
            n_valid += 1

        if n_valid == 0:
            logger.warning(f"No valid predictions for {run_name}")
            return

        y_true = np.array(y_true)
        y_logits = np.stack(y_logits)
        test_metrics = evaluate_task1(y_true, y_logits)
    else:
        y_true = []
        y_scores = []
        for entry, (_, row) in zip(entries, samples_df.iterrows()):
            if entry["parsed"] is None:
                continue
            y_true.append(int(row["label"]))
            pred = entry["parsed"]["prediction"]
            conf = entry["parsed"]["confidence"]
            # Convert to positive-class score
            score = conf if pred == 1 else (1.0 - conf)
            y_scores.append(score)

        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        test_metrics = evaluate_binary(y_true, y_scores)

    # Remove confusion matrix (not JSON serializable as-is)
    test_metrics = {k: v for k, v in test_metrics.items() if k != "confusion_matrix"}

    result = {
        "run_name": run_name,
        "task": task,
        "modality": modality,
        "model": model_key,
        "split": "cross_driver",
        "horizon": 3,
        "seed": SEED,
        "best_epoch": 0,
        "best_val_metric": 0.0,
        "n_params": 0,
        "val_metrics": {},
        "test_metrics": test_metrics,
        "vlm_meta": {
            "n_samples": len(samples_df),
            "n_valid": len(y_true),
            "n_parse_failures": n_parse_fail,
            "provider": MODELS[model_key]["provider"],
            "model_id": MODELS[model_key]["model_id"],
        },
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"  Saved: {run_name} (valid={len(y_true)}, fail={n_parse_fail})")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

async def run_all_async(tasks, models, modalities, samples_dict, rpm_overrides):
    """Run all configs sequentially (model by model, task by task)."""
    for model_key in models:
        model_info = MODELS[model_key]
        rpm = rpm_overrides.get(model_key)
        logger.info(f"\n{'━'*60}")
        logger.info(f"  {model_info['display']} ({model_info['model_id']})")
        logger.info(f"{'━'*60}")

        for task in tasks:
            samples_df = samples_dict[task]
            for modality in modalities:
                await run_inference_batch(task, modality, model_key, samples_df, rpm)
                gc.collect()  # free CSV DataFrames from build_text_context


def _get_rss_mb():
    """Current process RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0


def main():
    parser = argparse.ArgumentParser(description="Run VLM baselines")
    parser.add_argument("--task", choices=TASKS, help="Single task (default: all)")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Single model (default: all)")
    parser.add_argument("--modality", choices=MODALITIES, help="Single modality (default: all)")
    parser.add_argument("--gpt4o-rpm", type=int, default=60)
    parser.add_argument("--gemini-rpm", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tasks = [args.task] if args.task else TASKS
    models = [args.model] if args.model else list(MODELS.keys())
    modalities = [args.modality] if args.modality else MODALITIES

    if not args.dry_run:
        # Safety: lowest CPU/IO priority, high OOM score
        _set_low_priority()
        _set_oom_score()
        logger.info("Safety: nice=19, ionice=idle, oom_score_adj=500")
        memory_guard()  # wait if system is already under pressure

    # Subsample one task at a time (task1 CSV is 119MB, load and discard)
    logger.info("Step 1: Subsampling test set...")
    samples_dict = {}
    for task in tasks:
        samples_dict[task] = subsample_task(task, SAMPLE_N[task])
        logger.info(f"  {task}: {len(samples_dict[task])} samples")
        gc.collect()

    n_configs = len(tasks) * len(models) * len(modalities)
    n_calls = sum(len(samples_dict[t]) for t in tasks) * len(models) * len(modalities)
    logger.info(f"\nConfigs: {n_configs}, Total API calls: {n_calls}")
    logger.info(f"Process RSS: {_get_rss_mb():.0f} MB")

    if args.dry_run:
        for model_key in models:
            for task in tasks:
                for mod in modalities:
                    run_name = f"{task}_{mod}_{model_key}_cross_driver_h3_s{SEED}"
                    done = (RESULTS_DIR / run_name / "results.json").exists()
                    status = "DONE" if done else "TODO"
                    print(f"  [{status}] {run_name}")
        return

    # Extract frames (one ffmpeg at a time, nice -n 19)
    logger.info("\nStep 2: Extracting frames...")
    extract_all_frames(samples_dict)
    logger.info(f"Process RSS after frames: {_get_rss_mb():.0f} MB")

    # Pre-load GPS for test routes if VLM-All is requested (chunked read, ~50MB)
    if "VLM-All" in modalities:
        all_route_ids = set()
        for df in samples_dict.values():
            all_route_ids.update(df["route_id"].unique())
        logger.info(f"\nStep 2.5: Pre-loading GPS for {len(all_route_ids)} routes (chunked)...")
        preload_gps_routes(all_route_ids)
        gc.collect()
        logger.info(f"Process RSS after GPS: {_get_rss_mb():.0f} MB")

    # Run inference
    logger.info("\nStep 3: Running inference...")
    t0 = time.time()
    rpm_overrides = {"gpt4o": args.gpt4o_rpm, "gemini": args.gemini_rpm}
    asyncio.run(run_all_async(tasks, models, modalities, samples_dict, rpm_overrides))
    elapsed = time.time() - t0

    logger.info(f"\n{'━'*60}")
    logger.info(f"  DONE in {elapsed/3600:.1f}h  (RSS: {_get_rss_mb():.0f} MB)")
    logger.info(f"{'━'*60}")


if __name__ == "__main__":
    main()
