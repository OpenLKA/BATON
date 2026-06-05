"""
paths.py — Single source of truth for locating BATON route data on disk.

New BATON layout (current):
    <BATON_ROOT>/<car_model>/<dongle_id>/<route_id>/<segment_id>/{metadata.json, *.csv, *.mp4}

where <segment_id> is a bundle folder named like "route_0-60". Each bundle folder
is the ATOMIC unit (per-bundle): it holds one concatenated, continuous-timeline
set of CSVs + videos + metadata.json. (The old layout had an extra "ACM_MM/" level
between <route_id> and the bundle folder; that level is gone.)

Canonical key:
    route_uid = "<car_model>/<dongle_id>/<route_id>/<segment_id>"   (path relative to BATON_ROOT)

All scripts should resolve route data through resolve_route_dir(route_uid) instead
of reconstructing paths or globbing for "ACM_MM/route_*".
"""
import json
from pathlib import Path

BATON_ROOT = Path("/home/henry/Desktop/Drive/BATON")

# Test/non-real fingerprints to exclude from the benchmark.
EXCLUDE_CAR_MODELS = {"MOCK"}

# Per-bundle CSV / video files expected inside a segment dir.
STRUCT_CSV_NAMES = [
    "vehicle_dynamics.csv", "planning.csv", "radar.csv",
    "driver_state.csv", "imu.csv", "localization.csv", "gps.csv",
]


def discover_segments(root=BATON_ROOT, exclude=EXCLUDE_CAR_MODELS):
    """Return sorted list of segment-bundle dirs (each = parent of a metadata.json).

    Layout depth: <root>/<car_model>/<dongle_id>/<route_id>/<segment_id>/metadata.json
    """
    root = Path(root)
    seg_dirs = []
    for meta in root.glob("*/*/*/*/metadata.json"):
        seg = meta.parent
        car_model = seg.relative_to(root).parts[0]
        if car_model in exclude:
            continue
        seg_dirs.append(seg)
    return sorted(seg_dirs)


def route_uid(seg_dir, root=BATON_ROOT):
    """Canonical key for a segment-bundle dir: path relative to BATON_ROOT."""
    return str(Path(seg_dir).resolve().relative_to(Path(root).resolve()))


def resolve_route_dir(uid, root=BATON_ROOT):
    """Map a route_uid back to its on-disk segment-bundle directory."""
    return Path(root) / uid


def uid_parts(uid):
    """Split a route_uid into (car_model, dongle_id, route_id, segment_id)."""
    parts = uid.split("/")
    if len(parts) != 4:
        raise ValueError(f"route_uid must have 4 parts (car/dongle/route/segment): {uid!r}")
    return parts[0], parts[1], parts[2], parts[3]


def npz_key(uid):
    """Filesystem-safe key for npz/per-route caches."""
    return uid.replace("/", "__")


def load_meta(seg_dir):
    """Load a segment-bundle's metadata.json."""
    with open(Path(seg_dir) / "metadata.json") as f:
        return json.load(f)


def build_route_index(out_csv, root=BATON_ROOT, exclude=EXCLUDE_CAR_MODELS):
    """Write route_index.csv mapping every route_uid to its path + summary stats.

    Columns: route_uid, abs_path, car_model, dongle_id, route_id, segment_id,
             duration_s, n_inner_segments
    Returns the number of rows written.
    """
    import csv as _csv

    seg_dirs = discover_segments(root, exclude)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow([
            "route_uid", "abs_path", "car_model", "dongle_id", "route_id",
            "segment_id", "duration_s", "n_inner_segments",
        ])
        for seg in seg_dirs:
            uid = route_uid(seg, root)
            car_model, dongle_id, rid, segment_id = uid_parts(uid)
            try:
                meta = load_meta(seg)
                dur = meta.get("total_duration_s", 0.0) or 0.0
                n_inner = meta.get("n_segments", 0)
            except Exception:
                dur, n_inner = 0.0, 0
            w.writerow([
                uid, str(seg), car_model, dongle_id, rid, segment_id,
                f"{dur:.3f}", n_inner,
            ])
            n += 1
    return n


if __name__ == "__main__":
    segs = discover_segments()
    total = 0.0
    for s in segs:
        try:
            total += load_meta(s).get("total_duration_s", 0.0) or 0.0
        except Exception:
            pass
    print(f"segments (ex-{sorted(EXCLUDE_CAR_MODELS)}): {len(segs)}")
    print(f"total duration: {total/3600:.1f} h")
    # quick resolve sanity
    if segs:
        uid = route_uid(segs[0])
        print(f"example uid : {uid}")
        print(f"resolves to : {resolve_route_dir(uid)} (exists={resolve_route_dir(uid).exists()})")
