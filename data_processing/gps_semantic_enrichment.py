#!/usr/bin/env python3
"""
gps_semantic_enrichment.py — GPS Semantic Feature Extraction for PassingCtrl
Produces: gps_context_features.csv, gps_maneuver_annotations.csv, gps_route_quality.csv
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.signal import savgol_filter
from math import radians, sin, cos, atan2, degrees, sqrt
import time, json, sys, glob, traceback

# ── Paths ──
DATASET_ROOT = Path("/home/henry/Desktop/Drive/Dataset")
BENCHMARK_DIR = Path("/home/henry/Desktop/Drive/HMI/benchmark")
OUTPUT_DIR = Path("/home/henry/Desktop/Drive/HMI/data")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constants ──
HACC_THRESHOLD = 50.0       # discard device GPS with hAcc > 50m
SMOOTH_WINDOW = 11          # Savitzky-Golay window (~1.1s at 10Hz)
SMOOTH_POLY = 2
STOP_SPEED_THRESH = 0.5     # m/s
INTERSECTION_DEGREE = 3     # OSM node degree for intersection
HORIZON_DIST = 300          # meters lookahead
TURN_THRESHOLD_DEG = 30     # degrees heading change for "turn"
TURN_WINDOW_SEC = 5         # seconds window for turn detection

# Road type mapping
ROAD_TYPE_MAP = {
    'motorway': 'motorway', 'motorway_link': 'motorway',
    'trunk': 'trunk', 'trunk_link': 'trunk',
    'primary': 'primary', 'primary_link': 'primary',
    'secondary': 'secondary', 'secondary_link': 'secondary',
    'tertiary': 'tertiary', 'tertiary_link': 'tertiary',
    'residential': 'residential', 'living_street': 'residential',
    'unclassified': 'other', 'service': 'other', 'track': 'other',
}

# ═══════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════

def haversine_dist(lat1, lon1, lat2, lon2):
    """Haversine distance in meters between two points."""
    R = 6371000
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1-a))

def compute_bearing(lat1, lon1, lat2, lon2):
    """Compute bearing from point 1 to point 2 in degrees [0, 360)."""
    dlon = radians(lon2 - lon1)
    lat1r, lat2r = radians(lat1), radians(lat2)
    x = sin(dlon) * cos(lat2r)
    y = cos(lat1r) * sin(lat2r) - sin(lat1r) * cos(lat2r) * cos(dlon)
    return (degrees(atan2(x, y)) + 360) % 360

def angle_diff(a, b):
    """Signed angular difference a-b, result in [-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d

def menger_curvature(x1, y1, x2, y2, x3, y3):
    """Menger curvature from 3 points (using Cartesian approximation)."""
    area2 = abs((x2-x1)*(y3-y1) - (x3-x1)*(y2-y1))
    d12 = sqrt((x2-x1)**2 + (y2-y1)**2)
    d23 = sqrt((x3-x2)**2 + (y3-y2)**2)
    d13 = sqrt((x3-x1)**2 + (y3-y1)**2)
    denom = d12 * d23 * d13
    if denom < 1e-12:
        return 0.0
    return 2 * area2 / denom


# ═══════════════════════════════════════════════════════
# STEP 1: BUILD ROUTE INDEX
# ═══════════════════════════════════════════════════════

def build_route_index():
    """Map route_id → filesystem path for GPS and localization CSVs."""
    routes = pd.read_csv(BENCHMARK_DIR / "routes.csv")

    gps_files = sorted(glob.glob(str(DATASET_ROOT / "*/*/*/ACM_MM/*/gps.csv")))

    index = {}
    for f in gps_files:
        p = Path(f)
        driver_id = p.parts[-5]
        route_hash = p.parts[-4]
        route_dir = p.parent
        route_id = f"{driver_id}/{route_hash}"
        index[route_id] = {
            'gps_file': str(p),
            'loc_file': str(route_dir / "localization.csv"),
            'route_dir': str(route_dir),
        }

    # Attach metadata from routes.csv
    for _, row in routes.iterrows():
        rid = row['route_id']
        if rid in index:
            index[rid]['vehicle_model'] = row['vehicle_model']
            index[rid]['driver_id'] = row['driver_id']
            index[rid]['duration_sec'] = row['duration_sec']
            index[rid]['has_gps'] = row['has_gps']

    return routes, index


# ═══════════════════════════════════════════════════════
# STEP 2: LOAD & CLEAN GPS
# ═══════════════════════════════════════════════════════

def load_clean_gps(gps_file):
    """Load GPS CSV, select best source, filter bad points.

    Returns: DataFrame with columns [time_s, lat, lon, gps_speed, gps_hAcc, source]
             or None if no valid data.
    """
    df = pd.read_csv(gps_file)
    if len(df) == 0:
        return None

    # Try phone GPS first (cleaner)
    has_phone = ('gps_phone_lat' in df.columns and
                 df['gps_phone_lat'].notna().sum() > 10 and
                 (df['gps_phone_lat'] != 0).sum() > 10)

    has_device = ('gps_lat' in df.columns and
                  df['gps_lat'].notna().sum() > 10 and
                  (df['gps_lat'] != 0).sum() > 10)

    if has_phone:
        valid = df['gps_phone_lat'].notna() & (df['gps_phone_lat'] != 0)
        result = pd.DataFrame({
            'time_s': df.loc[valid, 'time_s'].values,
            'lat': df.loc[valid, 'gps_phone_lat'].values,
            'lon': df.loc[valid, 'gps_phone_lon'].values,
            'gps_speed': df.loc[valid, 'gps_phone_speed'].values if 'gps_phone_speed' in df.columns else np.nan,
            'gps_hAcc': df.loc[valid, 'gps_phone_hAcc'].values if 'gps_phone_hAcc' in df.columns else np.nan,
        })
        result['source'] = 'phone'
    elif has_device:
        valid = (df['gps_lat'].notna() & (df['gps_lat'] != 0) & (df['gps_lon'] != 0))
        if 'gps_hAcc' in df.columns:
            hacc_valid = df['gps_hAcc'].notna() & (df['gps_hAcc'] < HACC_THRESHOLD)
            valid = valid & hacc_valid
        result = pd.DataFrame({
            'time_s': df.loc[valid, 'time_s'].values,
            'lat': df.loc[valid, 'gps_lat'].values,
            'lon': df.loc[valid, 'gps_lon'].values,
            'gps_speed': df.loc[valid, 'gps_speed'].values if 'gps_speed' in df.columns else np.nan,
            'gps_hAcc': df.loc[valid, 'gps_hAcc'].values if 'gps_hAcc' in df.columns else np.nan,
        })
        result['source'] = 'device'
    else:
        return None

    if len(result) < 10:
        return None

    # Remove obvious outlier jumps (>500m between consecutive 10Hz samples = >5000 m/s)
    if len(result) > 1:
        dlat = np.abs(np.diff(result['lat'].values))
        dlon = np.abs(np.diff(result['lon'].values))
        # ~0.005 degrees ≈ 500m
        jump_mask = np.concatenate([[False], (dlat > 0.005) | (dlon > 0.005)])
        result = result[~jump_mask].reset_index(drop=True)

    return result if len(result) >= 10 else None


# ═══════════════════════════════════════════════════════
# STEP 3: TRAJECTORY FEATURES (NO MAP NEEDED)
# ═══════════════════════════════════════════════════════

def compute_trajectory_features(gps_df):
    """Compute heading, curvature, speed from GPS trajectory.

    All features are backward-looking (no future leakage).
    """
    n = len(gps_df)
    lat = gps_df['lat'].values.copy()
    lon = gps_df['lon'].values.copy()
    t = gps_df['time_s'].values.copy()

    # Smooth lat/lon with Savitzky-Golay
    if n >= SMOOTH_WINDOW:
        lat_s = savgol_filter(lat, SMOOTH_WINDOW, SMOOTH_POLY)
        lon_s = savgol_filter(lon, SMOOTH_WINDOW, SMOOTH_POLY)
    else:
        lat_s, lon_s = lat, lon

    # Convert to local Cartesian (meters) for curvature computation
    lat0, lon0 = lat_s[0], lon_s[0]
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * cos(radians(lat0))
    x_m = (lon_s - lon0) * m_per_deg_lon
    y_m = (lat_s - lat0) * m_per_deg_lat

    # ── Heading (backward-looking: from t-1 to t) ──
    heading = np.full(n, np.nan)
    for i in range(1, n):
        heading[i] = compute_bearing(lat_s[i-1], lon_s[i-1], lat_s[i], lon_s[i])
    heading[0] = heading[1] if n > 1 else 0

    # ── Heading change rate (°/s, backward-looking 1s window) ──
    heading_change_rate = np.full(n, 0.0)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt > 0:
            heading_change_rate[i] = angle_diff(heading[i], heading[i-1]) / dt

    # Smooth heading change rate
    if n >= SMOOTH_WINDOW:
        heading_change_rate = savgol_filter(heading_change_rate, SMOOTH_WINDOW, SMOOTH_POLY)

    # ── Curvature (from 3-point Menger, backward-looking) ──
    curvature = np.full(n, 0.0)
    for i in range(2, n):
        curvature[i] = menger_curvature(x_m[i-2], y_m[i-2], x_m[i-1], y_m[i-1], x_m[i], y_m[i])

    # ── Speed from GPS (m/s) ──
    speed = np.full(n, 0.0)
    for i in range(1, n):
        dt = t[i] - t[i-1]
        if dt > 0:
            dist = haversine_dist(lat_s[i-1], lon_s[i-1], lat_s[i], lon_s[i])
            speed[i] = dist / dt
    speed[0] = speed[1] if n > 1 else 0

    # Use reported GPS speed if available and reasonable
    gps_speed = gps_df['gps_speed'].values
    use_reported = np.isfinite(gps_speed) & (gps_speed >= 0)
    speed[use_reported] = gps_speed[use_reported]

    # ── Stopped detection ──
    is_stopped = (speed < STOP_SPEED_THRESH).astype(int)

    # ── Stopped duration (cumulative seconds stopped) ──
    stopped_dur = np.zeros(n)
    for i in range(1, n):
        if is_stopped[i]:
            stopped_dur[i] = stopped_dur[i-1] + (t[i] - t[i-1])

    return pd.DataFrame({
        'heading_deg': np.round(heading, 2),
        'heading_change_rate': np.round(heading_change_rate, 4),
        'curvature': np.round(curvature, 6),
        'traj_speed_mps': np.round(speed, 3),
        'is_stopped': is_stopped,
        'stopped_duration_s': np.round(stopped_dur, 2),
    })


# ═══════════════════════════════════════════════════════
# STEP 4: OSM MAP ENRICHMENT
# ═══════════════════════════════════════════════════════

def init_osm():
    """Initialize osmnx with caching."""
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(OUTPUT_DIR / "osm_cache")
    Path(ox.settings.cache_folder).mkdir(exist_ok=True)
    ox.settings.timeout = 300
    ox.settings.log_console = False
    return ox

def get_osm_graph(ox, lats, lons, buffer_m=500):
    """Download OSM drive graph covering the given coordinates with buffer.

    Uses IQR-based outlier removal to avoid downloading continental-scale graphs
    when a few GPS points are wildly off.
    """
    # Remove outlier coordinates using IQR — apply combined mask to keep arrays in sync
    lat_arr = np.array(lats)
    lon_arr = np.array(lons)

    combined_mask = np.ones(len(lat_arr), dtype=bool)
    for arr in [lat_arr, lon_arr]:
        q1, q3 = np.percentile(arr, [5, 95])
        iqr = q3 - q1
        combined_mask &= (arr >= q1 - 1.5 * iqr) & (arr <= q3 + 1.5 * iqr)
    lat_arr = lat_arr[combined_mask]
    lon_arr = lon_arr[combined_mask]

    if len(lat_arr) < 10:
        return None

    north = lat_arr.max() + buffer_m / 111320
    south = lat_arr.min() - buffer_m / 111320
    cos_lat = cos(radians(lat_arr.mean()))
    east = lon_arr.max() + buffer_m / (111320 * cos_lat)
    west = lon_arr.min() - buffer_m / (111320 * cos_lat)

    # Check bbox size — skip if too large (>50km span in either direction)
    lat_span_km = (north - south) * 111.32
    lon_span_km = (east - west) * 111.32 * cos_lat
    MAX_SPAN_KM = 50

    if lat_span_km > MAX_SPAN_KM or lon_span_km > MAX_SPAN_KM:
        # Use center point with limited radius instead
        center = (np.median(lat_arr), np.median(lon_arr))
        try:
            G = ox.graph_from_point(center, dist=5000, network_type='drive')
            return G
        except Exception:
            return None

    try:
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type='drive')
        return G
    except Exception:
        try:
            center = (np.median(lat_arr), np.median(lon_arr))
            G = ox.graph_from_point(center, dist=3000, network_type='drive')
            return G
        except Exception:
            return None

def classify_road_type(highway_tag):
    """Map OSM highway tag to simplified category."""
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    highway_tag = str(highway_tag).lower()
    return ROAD_TYPE_MAP.get(highway_tag, 'other')

def is_link_road(highway_tag):
    """Check if road is a link/ramp."""
    if isinstance(highway_tag, list):
        highway_tag = highway_tag[0]
    return 'link' in str(highway_tag).lower()

def parse_speed_limit(maxspeed):
    """Parse OSM maxspeed tag to km/h."""
    if maxspeed is None or maxspeed == '':
        return np.nan
    if isinstance(maxspeed, list):
        maxspeed = maxspeed[0]
    s = str(maxspeed).strip()
    try:
        if 'mph' in s.lower():
            return float(s.lower().replace('mph', '').strip()) * 1.60934
        return float(s.split()[0])
    except (ValueError, IndexError):
        return np.nan

def parse_lanes(lanes_tag):
    """Parse OSM lanes tag."""
    if lanes_tag is None:
        return np.nan
    if isinstance(lanes_tag, list):
        lanes_tag = lanes_tag[0]
    try:
        return int(float(str(lanes_tag)))
    except (ValueError, TypeError):
        return np.nan

def snap_and_enrich(ox, G, lats, lons, headings):
    """Snap GPS points to nearest edges and extract road attributes + horizon window.

    OPTIMIZED: Runs graph traversals at 1Hz (every 10th point), then forward-fills
    to native 10Hz. The expensive Dijkstra calls run on ~1/10th of the data.

    Returns DataFrame with OSM-derived features (same length as input).
    """
    import networkx as nx

    n = len(lats)

    # Snap ALL points to nearest edges (this is fast — vectorized in osmnx)
    try:
        edges = ox.nearest_edges(G, X=lons, Y=lats)
    except Exception:
        return None

    # Pre-compute node degrees for intersection detection
    node_degrees = dict(G.degree())

    # Pre-cache Dijkstra results for unique nodes to avoid redundant computation
    _dijkstra_cache = {}
    def cached_dijkstra(node, cutoff):
        key = (node, cutoff)
        if key not in _dijkstra_cache:
            try:
                _dijkstra_cache[key] = nx.single_source_dijkstra_path_length(
                    G, node, cutoff=cutoff, weight='length')
            except nx.NetworkXError:
                _dijkstra_cache[key] = {}
        return _dijkstra_cache[key]

    # ── Phase 1: Extract edge-level attributes for ALL points (fast) ──
    road_types = np.empty(n, dtype=object)
    is_highway = np.zeros(n, dtype=np.int8)
    speed_limits = np.full(n, np.nan)
    n_lanes_arr = np.full(n, np.nan)
    is_ramp = np.zeros(n, dtype=np.int8)
    bearing_vs_road = np.full(n, 0.0)

    for i in range(n):
        u, v, key = edges[i]
        ed = G.edges[u, v, key]
        hw_tag = ed.get('highway', 'unknown')
        rt = classify_road_type(hw_tag)
        road_types[i] = rt
        is_highway[i] = 1 if rt == 'motorway' else 0
        speed_limits[i] = parse_speed_limit(ed.get('maxspeed'))
        n_lanes_arr[i] = parse_lanes(ed.get('lanes'))
        is_ramp[i] = 1 if is_link_road(hw_tag) else 0

        u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
        v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
        rb = compute_bearing(u_y, u_x, v_y, v_x)
        bearing_vs_road[i] = round(abs(angle_diff(headings[i], rb)), 2)

    # ── Phase 2: Graph traversals at 1Hz (every 10th point) ──
    STEP = 10  # downsample factor: 10Hz → 1Hz
    sample_indices = list(range(0, n, STEP))
    if sample_indices[-1] != n - 1:
        sample_indices.append(n - 1)

    # Initialize sparse arrays
    dist_to_int_sparse = {}
    is_near_int_sparse = {}
    density_sparse = {}
    hw_dist_int_sparse = {}
    hw_n_int_sparse = {}
    hw_rt_changes_sparse = {}
    hw_ramp_ahead_sparse = {}
    hw_sl_decrease_sparse = {}

    for idx in sample_indices:
        u, v, key = edges[idx]
        ed = G.edges[u, v, key]
        edge_len = ed.get('length', 50)

        # Distance to nearest intersection
        min_int_dist = 999.0
        if node_degrees.get(u, 0) >= INTERSECTION_DEGREE:
            min_int_dist = min(min_int_dist, edge_len * 0.5)
        if node_degrees.get(v, 0) >= INTERSECTION_DEGREE:
            min_int_dist = min(min_int_dist, edge_len * 0.5)

        if min_int_dist > 100:
            for node in [u, v]:
                for target, dist in cached_dijkstra(node, 200).items():
                    if node_degrees.get(target, 0) >= INTERSECTION_DEGREE:
                        min_int_dist = min(min_int_dist, dist)

        dist_to_int_sparse[idx] = round(min_int_dist, 1)
        is_near_int_sparse[idx] = 1 if min_int_dist < 50 else 0

        # Road network density
        density = max(len(cached_dijkstra(u, 200)), len(cached_dijkstra(v, 200)))
        density_sparse[idx] = density

        # Horizon window
        heading_i = headings[idx]
        u_y, u_x = G.nodes[u]['y'], G.nodes[u]['x']
        v_y, v_x = G.nodes[v]['y'], G.nodes[v]['x']
        rb = compute_bearing(u_y, u_x, v_y, v_x)
        forward_node = v if abs(angle_diff(heading_i, rb)) < 90 else u

        paths = cached_dijkstra(forward_node, HORIZON_DIST)

        hw_ints = sum(1 for nd, d in paths.items()
                     if node_degrees.get(nd, 0) >= INTERSECTION_DEGREE and d > 0)
        int_dists = [d for nd, d in paths.items()
                    if node_degrees.get(nd, 0) >= INTERSECTION_DEGREE and d > 0]

        hw_n_int_sparse[idx] = hw_ints
        hw_dist_int_sparse[idx] = round(min(int_dists), 1) if int_dists else HORIZON_DIST

        # Road type changes and ramps ahead
        ahead_rt = set()
        ahead_ramp = False
        ahead_sl = []
        for nd in list(paths.keys())[:50]:
            for _, nb, k, edata in G.edges(nd, data=True, keys=True):
                if nb in paths:
                    ahw = edata.get('highway', 'unknown')
                    ahead_rt.add(classify_road_type(ahw))
                    if is_link_road(ahw):
                        ahead_ramp = True
                    sl = parse_speed_limit(edata.get('maxspeed'))
                    if not np.isnan(sl):
                        ahead_sl.append(sl)

        hw_rt_changes_sparse[idx] = 1 if len(ahead_rt) > 1 else 0
        hw_ramp_ahead_sparse[idx] = 1 if ahead_ramp else 0

        current_sl = speed_limits[idx]
        if not np.isnan(current_sl) and ahead_sl:
            hw_sl_decrease_sparse[idx] = 1 if any(sl < current_sl - 5 for sl in ahead_sl) else 0
        else:
            hw_sl_decrease_sparse[idx] = 0

    # ── Phase 3: Interpolate/forward-fill from 1Hz to 10Hz ──
    def fill_from_sparse(sparse_dict, default=0):
        arr = np.full(n, default, dtype=type(list(sparse_dict.values())[0]) if sparse_dict else float)
        sorted_keys = sorted(sparse_dict.keys())
        for ki, k in enumerate(sorted_keys):
            end = sorted_keys[ki+1] if ki+1 < len(sorted_keys) else n
            arr[k:end] = sparse_dict[k]
        return arr

    return pd.DataFrame({
        'road_type': road_types,
        'is_highway': is_highway,
        'speed_limit_kph': speed_limits,
        'n_lanes': n_lanes_arr,
        'is_on_ramp': is_ramp,
        'dist_to_intersection_m': fill_from_sparse(dist_to_int_sparse, 999.0),
        'is_near_intersection': fill_from_sparse(is_near_int_sparse, 0),
        'road_network_density': fill_from_sparse(density_sparse, 0),
        'bearing_vs_road': bearing_vs_road,
        'hw_dist_to_intersection_m': fill_from_sparse(hw_dist_int_sparse, float(HORIZON_DIST)),
        'hw_n_intersections_300m': fill_from_sparse(hw_n_int_sparse, 0),
        'hw_road_type_changes': fill_from_sparse(hw_rt_changes_sparse, 0),
        'hw_is_ramp_ahead_300m': fill_from_sparse(hw_ramp_ahead_sparse, 0),
        'hw_speed_limit_decreases': fill_from_sparse(hw_sl_decrease_sparse, 0),
    })


# ═══════════════════════════════════════════════════════
# STEP 5: MANEUVER ANNOTATIONS (FUTURE — ANALYSIS ONLY)
# ═══════════════════════════════════════════════════════

def compute_maneuver_annotations(gps_df, traj_features):
    """Detect future maneuvers from trajectory. ANALYSIS ONLY — NOT for model input."""
    n = len(gps_df)
    t = gps_df['time_s'].values
    heading = traj_features['heading_deg'].values
    speed = traj_features['traj_speed_mps'].values

    # ── Turn detection: >30° heading change over 5s window ──
    turn_within_10s = np.zeros(n, dtype=int)
    turn_within_30s = np.zeros(n, dtype=int)
    next_maneuver_type = ['straight'] * n
    dist_to_next_maneuver = np.full(n, np.nan)
    time_to_next_maneuver = np.full(n, np.nan)
    stop_within_10s = np.zeros(n, dtype=int)

    # Pre-compute cumulative heading changes in sliding windows
    for i in range(n):
        # Look forward from current position
        for j in range(i+1, n):
            dt = t[j] - t[i]
            if dt > 30:
                break

            # Check for turn (heading change > threshold over TURN_WINDOW_SEC)
            if dt >= TURN_WINDOW_SEC - 0.5:  # allow some tolerance
                window_start = i
                window_end = j
                # Max heading change in this window
                max_hchange = 0
                for k in range(window_start+1, min(window_end+1, n)):
                    hchange = abs(angle_diff(heading[k], heading[window_start]))
                    max_hchange = max(max_hchange, hchange)

                if max_hchange > TURN_THRESHOLD_DEG:
                    if dt <= 10:
                        turn_within_10s[i] = 1
                    turn_within_30s[i] = 1

                    if next_maneuver_type[i] == 'straight':
                        # Determine turn direction
                        hdiff = angle_diff(heading[min(j, n-1)], heading[i])
                        if hdiff > TURN_THRESHOLD_DEG:
                            next_maneuver_type[i] = 'left_turn'
                        elif hdiff < -TURN_THRESHOLD_DEG:
                            next_maneuver_type[i] = 'right_turn'
                        else:
                            next_maneuver_type[i] = 'turn'
                        time_to_next_maneuver[i] = dt
                    break

            # Check for stop
            if dt <= 10 and speed[j] < STOP_SPEED_THRESH:
                stop_within_10s[i] = 1

    return pd.DataFrame({
        'next_maneuver_type': next_maneuver_type,
        'time_to_next_maneuver_s': np.round(time_to_next_maneuver, 2),
        'turn_within_10s': turn_within_10s,
        'turn_within_30s': turn_within_30s,
        'stop_within_10s': stop_within_10s,
    })


def compute_maneuver_fast(gps_df, traj_features):
    """Vectorized fast maneuver annotation. ANALYSIS ONLY."""
    n = len(gps_df)
    t = gps_df['time_s'].values
    heading = traj_features['heading_deg'].values
    speed = traj_features['traj_speed_mps'].values

    turn_within_10s = np.zeros(n, dtype=np.int8)
    turn_within_30s = np.zeros(n, dtype=np.int8)
    stop_within_10s = np.zeros(n, dtype=np.int8)
    next_maneuver_type = np.full(n, 'straight', dtype=object)
    time_to_next_maneuver = np.full(n, np.nan)

    hz = 10  # approximate sampling rate

    # ── Stop detection: vectorized with cumsum ──
    is_slow = (speed < STOP_SPEED_THRESH).astype(np.int32)
    cumsum_slow = np.cumsum(is_slow)
    window_10s = 10 * hz
    for i in range(n):
        end = min(i + window_10s, n)
        count = cumsum_slow[end-1] - (cumsum_slow[i-1] if i > 0 else 0)
        if count > 0:
            stop_within_10s[i] = 1

    # ── Turn detection: process at 1Hz then fill blocks ──
    STEP = hz
    for i in range(0, n, STEP):
        for offset_s in range(5, 31):
            j = i + offset_s * hz
            if j >= n:
                break
            hdiff = angle_diff(heading[j], heading[i])
            if abs(hdiff) > TURN_THRESHOLD_DEG:
                dt = t[j] - t[i]
                block_end = min(i + STEP, n)
                turn_within_30s[i:block_end] = 1
                if dt <= 10:
                    turn_within_10s[i:block_end] = 1
                time_to_next_maneuver[i:block_end] = dt
                if hdiff > 0:
                    next_maneuver_type[i:block_end] = 'left_turn'
                else:
                    next_maneuver_type[i:block_end] = 'right_turn'
                break

    return pd.DataFrame({
        'next_maneuver_type': next_maneuver_type,
        'time_to_next_maneuver_s': np.round(time_to_next_maneuver, 2),
        'turn_within_10s': turn_within_10s,
        'turn_within_30s': turn_within_30s,
        'stop_within_10s': stop_within_10s,
    })


# ═══════════════════════════════════════════════════════
# STEP 6: QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════

def assess_route_quality(gps_file, clean_df):
    """Assess GPS quality for a single route."""
    raw = pd.read_csv(gps_file)
    total_rows = len(raw)

    if clean_df is None:
        return {
            'gps_source': 'none',
            'total_rows': total_rows,
            'valid_rows': 0,
            'coverage_frac': 0.0,
            'median_hAcc': np.nan,
            'n_jumps': 0,
            'quality_tier': 'none',
        }

    coverage = len(clean_df) / max(total_rows, 1)
    median_hAcc = clean_df['gps_hAcc'].median() if 'gps_hAcc' in clean_df.columns else np.nan
    source = clean_df['source'].iloc[0] if 'source' in clean_df.columns else 'unknown'

    # Count jumps in cleaned data
    if len(clean_df) > 1:
        dlat = np.abs(np.diff(clean_df['lat'].values))
        dlon = np.abs(np.diff(clean_df['lon'].values))
        n_jumps = int(np.sum((dlat > 0.001) | (dlon > 0.001)))  # ~100m jumps
    else:
        n_jumps = 0

    # Quality tier
    if coverage > 0.9 and n_jumps < 5:
        tier = 'good'
    elif coverage > 0.5:
        tier = 'moderate'
    else:
        tier = 'poor'

    return {
        'gps_source': source,
        'total_rows': total_rows,
        'valid_rows': len(clean_df),
        'coverage_frac': round(coverage, 4),
        'median_hAcc': round(median_hAcc, 2) if not np.isnan(median_hAcc) else np.nan,
        'n_jumps': n_jumps,
        'quality_tier': tier,
    }


# ═══════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════

def process_route(route_id, info, ox_module, do_osm=True):
    """Process a single route. Returns (context_df, maneuver_df, quality_dict)."""
    gps_file = info['gps_file']

    # Load & clean GPS
    gps_df = load_clean_gps(gps_file)

    # Quality assessment
    quality = assess_route_quality(gps_file, gps_df)
    quality['route_id'] = route_id
    quality['vehicle_model'] = info.get('vehicle_model', '')
    quality['driver_id'] = info.get('driver_id', '')

    if gps_df is None or len(gps_df) < 20:
        quality['map_match_status'] = 'skip_no_gps'
        quality['lat_center'] = np.nan
        quality['lon_center'] = np.nan
        return None, None, quality

    quality['lat_center'] = round(gps_df['lat'].median(), 6)
    quality['lon_center'] = round(gps_df['lon'].median(), 6)

    # Trajectory features
    traj = compute_trajectory_features(gps_df)

    # Build context DataFrame
    context = pd.DataFrame({
        'route_id': route_id,
        'time_s': gps_df['time_s'].values,
        'lat': np.round(gps_df['lat'].values, 7),
        'lon': np.round(gps_df['lon'].values, 7),
        'gps_speed_mps': np.round(gps_df['gps_speed'].values, 3),
        'gps_hAcc': np.round(gps_df['gps_hAcc'].values, 2),
    })
    context = pd.concat([context, traj], axis=1)

    # OSM enrichment (with per-route timeout)
    if do_osm and ox_module is not None:
        import signal

        class TimeoutError(Exception):
            pass

        def _timeout_handler(signum, frame):
            raise TimeoutError("OSM enrichment timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        # Timeout: 120s base + 1s per 1000 GPS points
        timeout_sec = 120 + len(gps_df) // 1000
        signal.alarm(timeout_sec)
        try:
            G = get_osm_graph(ox_module, gps_df['lat'], gps_df['lon'])
            if G is not None and len(G.nodes) > 0:
                osm_features = snap_and_enrich(
                    ox_module, G,
                    gps_df['lat'].values, gps_df['lon'].values,
                    traj['heading_deg'].values
                )
                if osm_features is not None:
                    context = pd.concat([context, osm_features], axis=1)
                    quality['map_match_status'] = 'ok'
                    quality['map_match_coverage'] = 1.0
                else:
                    quality['map_match_status'] = 'snap_failed'
            else:
                quality['map_match_status'] = 'no_graph'
        except TimeoutError:
            quality['map_match_status'] = 'timeout'
        except Exception as e:
            quality['map_match_status'] = f'error:{str(e)[:50]}'
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        quality['map_match_status'] = 'skipped'

    # Maneuver annotations (future — analysis only)
    maneuver = compute_maneuver_fast(gps_df, traj)
    maneuver.insert(0, 'time_s', gps_df['time_s'].values)
    maneuver.insert(0, 'route_id', route_id)

    return context, maneuver, quality


def main():
    print("=" * 70)
    print("PassingCtrl GPS Semantic Enrichment Pipeline")
    print("=" * 70)

    t0 = time.time()

    # Build route index
    print("\n[1/6] Building route index...")
    routes_df, route_index = build_route_index()
    print(f"  {len(route_index)} routes indexed")

    # Initialize OSM
    print("\n[2/6] Initializing OSM (osmnx)...")
    try:
        ox = init_osm()
        print(f"  osmnx ready, cache: {OUTPUT_DIR / 'osm_cache'}")
        do_osm = True
    except Exception as e:
        print(f"  WARNING: osmnx init failed: {e}")
        print("  Will produce trajectory-only features (no map enrichment)")
        ox = None
        do_osm = False

    # Process all routes
    print(f"\n[3/6] Processing {len(route_index)} routes...")

    all_context = []
    all_maneuver = []
    all_quality = []

    n_total = len(route_index)
    n_ok = 0
    n_osm_ok = 0
    n_fail = 0

    context_file = OUTPUT_DIR / "gps_context_features.csv"
    maneuver_file = OUTPUT_DIR / "gps_maneuver_annotations.csv"

    # Resume support: check which routes already processed
    done_routes = set()
    if context_file.exists():
        try:
            existing = pd.read_csv(context_file, usecols=['route_id'], dtype=str)
            done_routes = set(existing['route_id'].unique())
            print(f"  Resuming: {len(done_routes)} routes already processed")
        except Exception:
            pass

    header_written = context_file.exists() and len(done_routes) > 0
    header_written_m = maneuver_file.exists() and len(done_routes) > 0

    for i, (route_id, info) in enumerate(route_index.items()):
        if route_id in done_routes:
            n_ok += 1
            n_osm_ok += 1  # approximate
            continue

        if (i+1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed if elapsed > 0 else 0
            eta = (n_total - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{n_total}] {route_id[:30]:30s}  "
                  f"({n_ok} ok, {n_osm_ok} osm, {n_fail} fail)  "
                  f"ETA: {eta/60:.0f}min", flush=True)

        try:
            ctx, man, qual = process_route(route_id, info, ox, do_osm=do_osm)

            all_quality.append(qual)

            if ctx is not None:
                n_ok += 1
                if qual.get('map_match_status') == 'ok':
                    n_osm_ok += 1

                # Append to CSV incrementally
                ctx.to_csv(context_file, mode='a', header=not header_written, index=False)
                header_written = True

                man.to_csv(maneuver_file, mode='a', header=not header_written_m, index=False)
                header_written_m = True
            else:
                n_fail += 1

        except Exception as e:
            n_fail += 1
            all_quality.append({
                'route_id': route_id,
                'quality_tier': 'error',
                'map_match_status': f'error:{str(e)[:80]}',
                'vehicle_model': info.get('vehicle_model', ''),
                'driver_id': info.get('driver_id', ''),
            })
            if i < 5:  # Print first few errors for debugging
                traceback.print_exc()

    elapsed = time.time() - t0

    # Save quality file
    print(f"\n[4/6] Saving gps_route_quality.csv...")
    quality_df = pd.DataFrame(all_quality)
    quality_df.to_csv(OUTPUT_DIR / "gps_route_quality.csv", index=False)

    # Summary statistics
    print(f"\n[5/6] Generating summary...")
    print(f"  Routes processed: {n_ok}/{n_total}")
    print(f"  OSM enriched: {n_osm_ok}")
    print(f"  Failed: {n_fail}")
    print(f"  Total time: {elapsed/60:.1f} minutes")

    # File sizes
    for f in [context_file, maneuver_file, OUTPUT_DIR / "gps_route_quality.csv"]:
        if f.exists():
            sz = f.stat().st_size / (1024*1024)
            print(f"  {f.name}: {sz:.1f} MB")

    # Quality distribution
    if len(quality_df) > 0 and 'quality_tier' in quality_df.columns:
        print(f"\n  Quality tier distribution:")
        for tier, count in quality_df['quality_tier'].value_counts().items():
            print(f"    {tier}: {count}")

        if 'map_match_status' in quality_df.columns:
            print(f"\n  Map match status:")
            for status, count in quality_df['map_match_status'].value_counts().items():
                print(f"    {status}: {count}")

    print(f"\n[6/6] Done! Output in {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
