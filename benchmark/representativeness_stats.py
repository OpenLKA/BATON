#!/usr/bin/env python3
"""Representativeness / coverage statistics for the frozen benchmark subset.

Outputs (printed + benchmark_v2/representativeness_stats.json):
  1. lat x long engagement four-state taxonomy: time share per state and
     transition subtype counts (which flag toggled at each OR-transition)
  2. driver--vehicle bipartite stats (driver == dongle/device identity)
  3. per-driver / per-model transition and hours concentration
  4. software version, device type, route duration distributions
  5. approximate day/night share (UTC timestamp + longitude -> local solar time)
  6. DMS (driver_state) NaN fraction and pose coverage, split by day/night

Read-only over benchmark_v2/, baseline/cache/struct_50hz/, raw metadata.json/gps.csv.
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

BATON = Path("/home/henry/Desktop/Drive/BATON_Benchmark/BATON")
BENCH = BATON / "benchmark_v2"
STRUCT = BATON / "baseline" / "cache" / "struct_50hz"
POSE = BATON / "wm_hybrid" / "cache" / "pose_features_full.parquet"

routes = pd.read_csv(BENCH / "routes.csv")
index = pd.read_csv(BENCH / "route_index.csv")
out = {}

# ---------- 1. four-state taxonomy + transition subtypes ----------
state_time = np.zeros(4)          # [lat0long0, lat0long1, lat1long0, lat1long1]
sub_counts = Counter()            # activation/deactivation x which-flag
n_missing = 0
for rid in routes["route_id"]:
    p = STRUCT / (rid.replace("/", "__") + ".npz")
    if not p.exists():
        n_missing += 1
        continue
    d = np.load(p)
    cols = list(d["vehicle_dynamics.csv__cols"])
    X = d["vehicle_dynamics.csv__data"]
    step = float(d["vehicle_dynamics.csv__step"])
    lat = np.nan_to_num(X[:, cols.index("cc_latActive")]) > 0.5
    lon = np.nan_to_num(X[:, cols.index("cruiseState_enabled")]) > 0.5
    s = lat.astype(int) * 2 + lon.astype(int)
    state_time += np.bincount(s, minlength=4) * step
    orv = (lat | lon).astype(int)
    tr = np.flatnonzero(np.diff(orv))
    for i in tr:
        kind = "activation" if orv[i + 1] == 1 else "deactivation"
        if kind == "activation":
            which = ("both" if lat[i + 1] and lon[i + 1]
                     else "lat_only" if lat[i + 1] else "long_only")
        else:
            which = ("both" if lat[i] and lon[i]
                     else "lat_only" if lat[i] else "long_only")
        sub_counts[f"{kind}:{which}"] += 1
tot = state_time.sum()
out["four_state"] = {
    "hours": {"lat0_long0": state_time[0] / 3600, "lat0_long1": state_time[1] / 3600,
              "lat1_long0": state_time[2] / 3600, "lat1_long1": state_time[3] / 3600},
    "share": {k: v / tot for k, v in zip(
        ["lat0_long0", "lat0_long1", "lat1_long0", "lat1_long1"], state_time)},
    "raw_or_transition_subtypes": dict(sub_counts),
    "n_routes_missing_cache": n_missing,
    "note": "raw un-debounced OR-transitions; benchmark events apply the debounce/merge filters",
}

# ---------- 2. driver--vehicle bipartite ----------
g = index.groupby("dongle_id")["car_model"].nunique()
g2 = index.groupby("car_model")["dongle_id"].nunique()
out["bipartite"] = {
    "n_devices": int(index["dongle_id"].nunique()),
    "n_vehicle_models": int(index["car_model"].nunique()),
    "devices_spanning_multiple_models": int((g > 1).sum()),
    "models_with_multiple_devices": int((g2 > 1).sum()),
    "max_devices_per_model": int(g2.max()),
    "driver_equals_dongle": True,
}

# ---------- 3. concentration ----------
per_drv = routes.groupby("driver_id").agg(
    hours=("duration_sec", lambda s: s.sum() / 3600),
    transitions=("n_activations", "sum"))
per_drv["transitions"] += routes.groupby("driver_id")["n_takeovers"].sum()
per_mod = routes.groupby("vehicle_model")["duration_sec"].sum() / 3600
def topshare(s, k):
    s = np.sort(np.asarray(s, dtype=float))[::-1]
    return float(s[:k].sum() / max(s.sum(), 1e-9))
out["concentration"] = {
    "hours_top5_driver_share": topshare(per_drv["hours"], 5),
    "hours_top20_driver_share": topshare(per_drv["hours"], 20),
    "transitions_top5_driver_share": topshare(per_drv["transitions"], 5),
    "median_hours_per_driver": float(per_drv["hours"].median()),
    "max_hours_one_driver": float(per_drv["hours"].max()),
    "hours_top5_model_share": topshare(per_mod, 5),
    "drivers_with_lt_30min": int((per_drv["hours"] < 0.5).sum()),
}

# ---------- 4. version / device / duration ----------
out["op_version_top"] = routes["op_version"].astype(str).value_counts().head(8).to_dict()
out["device_type"] = routes["device_type"].astype(str).value_counts().to_dict()
dur_min = routes["duration_sec"] / 60
out["route_duration_min"] = {"median": float(dur_min.median()),
                             "p10": float(dur_min.quantile(.1)),
                             "p90": float(dur_min.quantile(.9)),
                             "max": float(dur_min.max())}

# ---------- 5. day/night (approx local solar time) ----------
uid2path = dict(zip(index["route_id"] if "route_id" in index else [], []))
day_hours = night_hours = 0.0
route_daynight = {}
for _, r in index.iterrows():
    seg = Path(r["abs_path"])
    meta = seg / "metadata.json"
    ts_ns, lon_deg = None, None
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            ts_ns = None  # group_t_start_ns is openpilot boot-relative monotonic time, not wall-clock; use the GPS unixTimestamp below
        except Exception:
            pass
    gps = seg / "gps.csv"
    if gps.exists():
        try:
            gd = pd.read_csv(gps, nrows=5)
            lon_col = next((c for c in gd.columns if "lon" in c.lower()), None)
            ts_col = next((c for c in gd.columns if "unixTimestamp" in c), None)
            if lon_col is not None and gd[lon_col].notna().any():
                lon_deg = float(gd[lon_col].dropna().iloc[0])
            if ts_ns is None and ts_col is not None and gd[ts_col].notna().any():
                ts_ns = float(gd[ts_col].dropna().iloc[0]) * 1e6  # ms -> ns
        except Exception:
            pass
    if ts_ns is None or lon_deg is None or not np.isfinite(lon_deg) or ts_ns <= 0:
        continue
    utc_h = (ts_ns / 1e9 % 86400) / 3600
    local_h = (utc_h + lon_deg / 15.0) % 24
    is_day = 7.0 <= local_h < 19.0
    dur_h = float(r["duration_s"]) / 3600
    if is_day:
        day_hours += dur_h
    else:
        night_hours += dur_h
    key = f'{r["car_model"]}/{r["dongle_id"]}/{r["route_id"]}/{r["segment_id"]}'
    route_daynight[key] = is_day
covered = day_hours + night_hours
out["day_night"] = {"day_hours": day_hours, "night_hours": night_hours,
                    "day_share_of_localizable": day_hours / max(covered, 1e-9),
                    "hours_with_time_and_lon": covered,
                    "n_routes_localizable": len(route_daynight)}

# ---------- 6. DMS / pose coverage by day/night ----------
dms_nan = {"day": [], "night": []}
for rid in routes["route_id"]:
    p = STRUCT / (rid.replace("/", "__") + ".npz")
    if not p.exists() or rid not in route_daynight:
        continue
    d = np.load(p)
    X = d["driver_state.csv__data"]
    frac = float(np.isnan(X).all(axis=1).mean())
    dms_nan["day" if route_daynight[rid] else "night"].append(frac)
pose_zero = {"day": [], "night": []}
if POSE.exists():
    pf = pd.read_parquet(POSE)
    num = pf.select_dtypes("number")
    zero_by_uid = ((num.drop(columns=["t"], errors="ignore") == 0).all(axis=1)
                   .groupby(pf["uid"]).mean())
    for rid, is_day in route_daynight.items():
        u = rid.replace("/", "__")
        if u in zero_by_uid.index:
            pose_zero["day" if is_day else "night"].append(float(zero_by_uid[u]))
out["coverage_by_daynight"] = {
    "dms_allnan_rowfrac_day_mean": float(np.mean(dms_nan["day"])) if dms_nan["day"] else None,
    "dms_allnan_rowfrac_night_mean": float(np.mean(dms_nan["night"])) if dms_nan["night"] else None,
    "pose_zero_rowfrac_day_mean": float(np.mean(pose_zero["day"])) if pose_zero["day"] else None,
    "pose_zero_rowfrac_night_mean": float(np.mean(pose_zero["night"])) if pose_zero["night"] else None,
    "n_day_routes": len(dms_nan["day"]), "n_night_routes": len(dms_nan["night"]),
}

o = BENCH / "representativeness_stats.json"
o.write_text(json.dumps(out, indent=2, default=float))
print(json.dumps(out, indent=2, default=float))
print(f"[done] wrote {o}", file=sys.stderr)
