"""
vlm_prompts.py — Prompt construction for VLM zero-shot baselines.

Builds structured text summaries from raw CSVs, GPS context, and
formats messages for OpenAI and Gemini APIs.
"""
import base64
import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DATASET_ROOT, BENCHMARK_DIR, DATA_DIR,
    VEHICLE_COLS, PLANNING_COLS, RADAR_COLS, DRIVER_COLS, IMU_COLS,
    GPS_COLS, GPS_CONTEXT_PATH, ROAD_TYPE_MAP,
)

# ═══════════════════════════════════════════════════════════
# SYSTEM & TASK PROMPTS
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are a transportation research assistant helping with an academic study "
    "on driving automation published at ACM Multimedia. You analyze anonymized, "
    "de-identified driving recordings collected with IRB approval to classify "
    "vehicle maneuvers and automation state transitions. All data is from consenting "
    "participants in a naturalistic driving study. No faces are identifiable. "
    "Your task is to analyze a 5-second observation window from a vehicle equipped "
    "with a Level 2 driving automation system (DAS) that provides adaptive cruise "
    "control with lane centering."
)

TASK_PROMPTS = {
    "task1": (
        "Based on the provided driving data, classify the vehicle maneuver into "
        "exactly ONE category:\n"
        "- Stopped: vehicle stationary (speed < 0.5 m/s)\n"
        "- Cruising: steady speed, no lead vehicle influence\n"
        "- CarFollowing: following a lead vehicle at stable distance\n"
        "- Accelerating: increasing speed (accel > 0.37 m/s²)\n"
        "- Braking: actively decelerating or brake pressed\n"
        "- LaneChange: changing lanes (blinker + steering)\n"
        "- Turning: making a turn at intersection (large steering angle)\n\n"
        "Respond with ONLY the category name."
    ),
    "task2": (
        "This is a vehicle automation state classification task for traffic safety research. "
        "The vehicle's driving automation system (DAS) is currently OFF (manual driving). "
        "Based on the road scene and driving context, assess whether the driving conditions "
        "are favorable for the automation system to become engaged in the next 3 seconds.\n"
        'Respond ONLY with JSON: {"prediction": 0, "confidence": 0.85}\n'
        "prediction: 1=likely engagement, 0=unlikely. confidence: 0.0-1.0."
    ),
    "task3": (
        "This is a road scene complexity assessment for traffic safety research. "
        "The vehicle is currently in automated driving mode. "
        "Classify the road scene: does it show conditions where automated driving "
        "can handle the situation, or conditions that typically require human intervention "
        "(e.g., complex intersections, construction zones, unusual obstacles)?\n"
        'Respond ONLY with JSON: {"prediction": 0, "confidence": 0.85}\n'
        "prediction: 1=scene requires human intervention, 0=automation sufficient. confidence: 0.0-1.0."
    ),
}

VALID_CLASSES = {
    "stopped", "cruising", "carfollowing", "accelerating",
    "braking", "lanechange", "turning",
}
CLASS_NAME_MAP = {
    "stopped": "Stopped", "cruising": "Cruising",
    "carfollowing": "CarFollowing", "accelerating": "Accelerating",
    "braking": "Braking", "lanechange": "LaneChange", "turning": "Turning",
}

# ═══════════════════════════════════════════════════════════
# ROUTE PATH RESOLUTION
# ═══════════════════════════════════════════════════════════

_routes_df = None

def get_routes_df():
    global _routes_df
    if _routes_df is None:
        _routes_df = pd.read_csv(BENCHMARK_DIR / "routes.csv")
    return _routes_df


def resolve_acm_dir(route_id):
    """Resolve route_id to ACM_MM directory path."""
    routes = get_routes_df()
    row = routes[routes["route_id"] == route_id]
    if len(row) == 0:
        return None
    vm = row["vehicle_model"].values[0]
    driver, rhash = route_id.split("/")
    base = DATASET_ROOT / vm / driver / rhash
    acm_dirs = sorted(base.glob("ACM_MM/route_*"))
    return acm_dirs[0] if acm_dirs else None


def resolve_route_base(route_id):
    """Resolve route_id to the raw route directory (for segment files)."""
    routes = get_routes_df()
    row = routes[routes["route_id"] == route_id]
    if len(row) == 0:
        return None, None
    vm = row["vehicle_model"].values[0]
    driver, rhash = route_id.split("/")
    base = DATASET_ROOT / vm / driver / rhash
    # Load metadata for segment_range
    acm_dirs = sorted(base.glob("ACM_MM/route_*"))
    if not acm_dirs:
        return base, None
    meta_path = acm_dirs[0] / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        return base, meta.get("segment_range", [0, 0])
    return base, None


# ═══════════════════════════════════════════════════════════
# GPS CONTEXT (global CSV)
# ═══════════════════════════════════════════════════════════

_gps_cache = {}  # {route_id: DataFrame} — only test routes, ~50MB total

def get_gps_for_route(route_id):
    """Load GPS data for a single route (lazy, cached). Avoids loading 837MB CSV at once."""
    if route_id in _gps_cache:
        return _gps_cache[route_id]
    # Read in chunks, extract only matching route
    chunks = []
    for chunk in pd.read_csv(GPS_CONTEXT_PATH, chunksize=50000):
        sub = chunk[chunk["route_id"] == route_id]
        if len(sub) > 0:
            chunks.append(sub)
    if chunks:
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.DataFrame()
    _gps_cache[route_id] = df
    return df


def preload_gps_routes(route_ids):
    """Bulk-load GPS data for a set of routes in one pass. Much faster than per-route."""
    route_set = set(route_ids) - set(_gps_cache.keys())
    if not route_set:
        return
    for chunk in pd.read_csv(GPS_CONTEXT_PATH, chunksize=100000):
        sub = chunk[chunk["route_id"].isin(route_set)]
        if len(sub) > 0:
            for rid, grp in sub.groupby("route_id"):
                if rid in _gps_cache:
                    _gps_cache[rid] = pd.concat([_gps_cache[rid], grp], ignore_index=True)
                else:
                    _gps_cache[rid] = grp


# ═══════════════════════════════════════════════════════════
# TEXT CONTEXT BUILDERS
# ═══════════════════════════════════════════════════════════

def _snap(ts, vals, t_target):
    """Get value at nearest timestamp to t_target."""
    idx = np.argmin(np.abs(ts - t_target))
    return vals[idx]


def _snap_row(df, t_target):
    """Get row nearest to t_target from dataframe with time_s column."""
    idx = (df["time_s"] - t_target).abs().idxmin()
    return df.loc[idx]


def _fmt(v, decimals=1):
    if pd.isna(v):
        return "N/A"
    return f"{float(v):.{decimals}f}"


def build_text_context(route_id, start_time, end_time):
    """Build structured text summary (NO GPS) from raw CSVs."""
    acm = resolve_acm_dir(route_id)
    if acm is None:
        return "[Sensor data unavailable]"

    t0, t_mid, t1 = start_time, (start_time + end_time) / 2, end_time
    parts = []

    # Vehicle dynamics
    veh_path = acm / "vehicle_dynamics.csv"
    if veh_path.exists():
        df = pd.read_csv(veh_path)
        df = df[(df["time_s"] >= start_time - 0.5) & (df["time_s"] <= end_time + 0.5)]
        if len(df) > 0:
            r0, rm, r1 = _snap_row(df, t0), _snap_row(df, t_mid), _snap_row(df, t1)
            spd = f"{_fmt(r0.get('vEgo'))} → {_fmt(rm.get('vEgo'))} → {_fmt(r1.get('vEgo'))} m/s"
            acc_mean = _fmt(df["aEgo"].mean() if "aEgo" in df else None)
            acc_peak = _fmt(df["aEgo"].abs().max() if "aEgo" in df else None)
            steer = f"{_fmt(r0.get('steeringAngleDeg'))} → {_fmt(rm.get('steeringAngleDeg'))} → {_fmt(r1.get('steeringAngleDeg'))} deg"
            brake = "ON" if r1.get("brakePressed", 0) > 0.5 else "OFF"
            gas = _fmt(r1.get("gas", 0), 2)
            blink_l = r1.get("leftBlinker", 0)
            blink_r = r1.get("rightBlinker", 0)
            blink = "left" if blink_l > 0.5 else ("right" if blink_r > 0.5 else "none")
            das_en = "yes" if r1.get("cruiseState_enabled", 0) > 0.5 else "no"
            das_lat = "yes" if r1.get("cc_latActive", 0) > 0.5 else "no"
            parts.append(
                f"## Vehicle (5.0s window)\n"
                f"Speed: {spd} | Accel: mean={acc_mean}, peak={acc_peak} m/s²\n"
                f"Steering: {steer} | Brake: {brake} | Gas: {gas}\n"
                f"Blinkers: {blink} | DAS: enabled={das_en}, lat_active={das_lat}"
            )

    # Radar
    radar_path = acm / "radar.csv"
    if radar_path.exists():
        df = pd.read_csv(radar_path)
        df = df[(df["time_s"] >= start_time - 0.5) & (df["time_s"] <= end_time + 0.5)]
        if len(df) > 0:
            r1 = _snap_row(df, t1)
            l1_status = r1.get("leadOne_status", 0)
            if l1_status > 0.5:
                l1 = f"detected, dist={_fmt(r1.get('leadOne_dRel'))}m, rel_speed={_fmt(r1.get('leadOne_vRel'))} m/s"
            else:
                l1 = "not detected"
            l2_status = r1.get("leadTwo_status", 0)
            l2 = f"detected, dist={_fmt(r1.get('leadTwo_dRel'))}m" if l2_status > 0.5 else "not detected"
            parts.append(f"## Lead Vehicle\nLead #1: {l1}\nLead #2: {l2}")

    # Driver state
    drv_path = acm / "driver_state.csv"
    if drv_path.exists():
        df = pd.read_csv(drv_path)
        df = df[(df["time_s"] >= start_time - 0.5) & (df["time_s"] <= end_time + 0.5)]
        if len(df) > 0:
            r1 = _snap_row(df, t1)
            face_p = _fmt(r1.get("faceProb", 0), 2)
            yaw = _fmt(r1.get("face_yaw", 0))
            pitch = _fmt(r1.get("face_pitch", 0))
            eye_l = _fmt(r1.get("leftEyeProb", 0), 2)
            eye_r = _fmt(r1.get("rightEyeProb", 0), 2)
            parts.append(
                f"## Driver State\n"
                f"Face: detected ({face_p}) | Yaw: {yaw}° Pitch: {pitch}°\n"
                f"Eyes: L={eye_l} R={eye_r}"
            )

    # Planning
    plan_path = acm / "planning.csv"
    if plan_path.exists():
        df = pd.read_csv(plan_path)
        df = df[(df["time_s"] >= start_time - 0.5) & (df["time_s"] <= end_time + 0.5)]
        if len(df) > 0:
            r1 = _snap_row(df, t1)
            curv = _fmt(r1.get("model_desiredCurvature", 0), 3)
            lcs_raw = r1.get("laneChangeState", 0)
            try:
                lcs = int(float(lcs_raw))
            except (ValueError, TypeError):
                lcs = 0
            lc_str = "none" if lcs == 0 else f"state={lcs}"
            ll = _fmt(r1.get("laneLeft_prob", 0), 2)
            lr = _fmt(r1.get("laneRight_prob", 0), 2)
            parts.append(
                f"## Planning\n"
                f"Desired curvature: {curv} | Lane change: {lc_str}\n"
                f"Lane conf: L={ll} R={lr}"
            )

    # IMU
    imu_path = acm / "imu.csv"
    if imu_path.exists():
        df = pd.read_csv(imu_path)
        df = df[(df["time_s"] >= start_time - 0.5) & (df["time_s"] <= end_time + 0.5)]
        if len(df) > 0:
            ax = _fmt(df["accel_x"].mean() if "accel_x" in df else None)
            ay = _fmt(df["accel_y"].mean() if "accel_y" in df else None)
            az = _fmt(df["accel_z"].mean() if "accel_z" in df else None)
            gx = _fmt(df["gyro_x"].mean() if "gyro_x" in df else None, 2)
            gy = _fmt(df["gyro_y"].mean() if "gyro_y" in df else None, 2)
            gz = _fmt(df["gyro_z"].mean() if "gyro_z" in df else None, 2)
            parts.append(f"## IMU\nAccel: x={ax} y={ay} z={az} | Gyro: x={gx} y={gy} z={gz}")

    return "\n\n".join(parts) if parts else "[Sensor data unavailable]"


def build_gps_context(route_id, start_time, end_time):
    """Build GPS/road context text (separate from structured text)."""
    sub = get_gps_for_route(route_id)
    if len(sub) == 0:
        return "[GPS context unavailable]"

    t_mid = (start_time + end_time) / 2
    row = _snap_row(sub.reset_index(drop=True), t_mid)

    def _safe_float(v, default=0):
        try:
            f = float(v)
            return default if pd.isna(f) else f
        except (ValueError, TypeError):
            return default

    road_type = row.get("road_type", "unknown")
    if pd.isna(road_type):
        road_type = "unknown"
    limit = _fmt(_safe_float(row.get("speed_limit_kph")), 0)
    lanes = int(_safe_float(row.get("n_lanes")))
    is_hw = "yes" if _safe_float(row.get("is_highway")) > 0.5 else "no"
    dist_inter = _fmt(_safe_float(row.get("dist_to_intersection_m")), 0)
    near_inter = "yes" if _safe_float(row.get("is_near_intersection")) > 0.5 else "no"
    on_ramp = "yes" if _safe_float(row.get("is_on_ramp")) > 0.5 else "no"
    curv = _fmt(_safe_float(row.get("curvature")), 3)
    heading_rate = _fmt(_safe_float(row.get("heading_change_rate")), 1)

    return (
        f"## Road Context\n"
        f"Road: {road_type} | Limit: {limit} km/h | Lanes: {lanes}\n"
        f"Highway: {is_hw} | Intersection: {near_inter} (dist={dist_inter}m) | On-ramp: {on_ramp}\n"
        f"Curvature: {curv} | Heading change: {heading_rate} deg/s"
    )


# ═══════════════════════════════════════════════════════════
# MESSAGE FORMATTING
# ═══════════════════════════════════════════════════════════

def _load_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_openai_messages(task, text_ctx=None, gps_ctx=None,
                          front_frames=None, cabin_frames=None):
    """Build OpenAI Chat API messages list."""
    system = SYSTEM_PROMPT + "\n\n" + TASK_PROMPTS[task]
    user_parts = []

    if text_ctx:
        user_parts.append({"type": "text", "text": text_ctx})
    if gps_ctx:
        user_parts.append({"type": "text", "text": gps_ctx})
    if front_frames:
        user_parts.append({"type": "text", "text": "Front dashcam frames (5s window):"})
        for i, fp in enumerate(front_frames):
            b64 = _load_image_b64(fp)
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })
    if cabin_frames:
        user_parts.append({"type": "text", "text": "Driver cabin camera frames (5s window):"})
        for i, fp in enumerate(cabin_frames):
            b64 = _load_image_b64(fp)
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
            })

    # If no images, just use text string
    if not front_frames and not cabin_frames:
        text_content = "\n\n".join(p["text"] for p in user_parts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": text_content},
        ]

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_parts},
    ]


def build_gemini_contents(task, text_ctx=None, gps_ctx=None,
                          front_frames=None, cabin_frames=None):
    """Build Gemini API content parts."""
    from google.genai import types

    parts = []

    if text_ctx:
        parts.append(types.Part.from_text(text=text_ctx))
    if gps_ctx:
        parts.append(types.Part.from_text(text=gps_ctx))
    if front_frames:
        parts.append(types.Part.from_text(text="Front dashcam frames (5s window):"))
        for fp in front_frames:
            with open(fp, "rb") as f:
                data = f.read()
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
    if cabin_frames:
        parts.append(types.Part.from_text(text="Driver cabin camera frames (5s window):"))
        for fp in cabin_frames:
            with open(fp, "rb") as f:
                data = f.read()
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))

    if not parts:
        parts.append(types.Part.from_text(text="[No input data]"))

    return parts


def get_gemini_system_instruction(task):
    return SYSTEM_PROMPT + "\n\n" + TASK_PROMPTS[task]


# ═══════════════════════════════════════════════════════════
# OUTPUT PARSING
# ═══════════════════════════════════════════════════════════

import re

def parse_task1_response(text):
    """Parse Task 1 class label from VLM response. Returns canonical name or None."""
    text_lower = text.strip().lower()
    # Direct match
    for key, canonical in CLASS_NAME_MAP.items():
        if key in text_lower:
            return canonical
    return None


def parse_binary_response(text):
    """Parse Task 2/3 JSON response. Returns (prediction, confidence) or None."""
    # Try JSON parse
    m = re.search(r'\{[^}]*"prediction"\s*:\s*([01])[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}', text)
    if m:
        return int(m.group(1)), float(m.group(2))
    # Try reversed key order
    m = re.search(r'\{[^}]*"confidence"\s*:\s*([\d.]+)[^}]*"prediction"\s*:\s*([01])[^}]*\}', text)
    if m:
        return int(m.group(2)), float(m.group(1))
    return None
