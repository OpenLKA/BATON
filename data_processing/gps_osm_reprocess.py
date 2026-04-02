#!/usr/bin/env python3
"""
gps_osm_reprocess.py — Re-enrich routes that are missing OSM features.

Reads existing gps_context_features.csv, identifies routes where OSM columns
(road_type, dist_to_intersection_m, etc.) are all NaN, and re-processes them
with osmnx map matching. Overwrites the affected rows in place.

Also regenerates gps_route_quality.csv with accurate map_match_status.

Usage:
    python3 gps_osm_reprocess.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import time, sys, signal, traceback

# Import functions from the main enrichment script
# Force reimport to pick up bug fixes
sys.path.insert(0, str(Path(__file__).parent))
import importlib
import gps_semantic_enrichment as _mod
importlib.reload(_mod)
from gps_semantic_enrichment import (
    build_route_index, init_osm, get_osm_graph, snap_and_enrich,
    load_clean_gps, compute_trajectory_features, assess_route_quality,
    compute_maneuver_fast,
    DATASET_ROOT, BENCHMARK_DIR, OUTPUT_DIR
)

OSM_COLUMNS = [
    'road_type', 'is_highway', 'speed_limit_kph', 'n_lanes', 'is_on_ramp',
    'dist_to_intersection_m', 'is_near_intersection', 'road_network_density',
    'bearing_vs_road', 'hw_dist_to_intersection_m', 'hw_n_intersections_300m',
    'hw_road_type_changes', 'hw_is_ramp_ahead_300m', 'hw_speed_limit_decreases',
]

CONTEXT_COLUMNS = [
    'route_id', 'time_s', 'lat', 'lon', 'gps_speed_mps', 'gps_hAcc',
    'heading_deg', 'heading_change_rate', 'curvature', 'traj_speed_mps',
    'is_stopped', 'stopped_duration_s',
    'road_type', 'is_highway', 'speed_limit_kph', 'n_lanes', 'is_on_ramp',
    'dist_to_intersection_m', 'is_near_intersection', 'road_network_density',
    'bearing_vs_road', 'hw_dist_to_intersection_m', 'hw_n_intersections_300m',
    'hw_road_type_changes', 'hw_is_ramp_ahead_300m', 'hw_speed_limit_decreases',
]


def find_routes_missing_osm(context_file):
    """Find route_ids where all OSM columns are NaN."""
    print("Reading existing context features to find routes missing OSM...")
    df = pd.read_csv(context_file, usecols=['route_id'] + OSM_COLUMNS,
                     dtype={'route_id': str})

    # For each route, check if ALL OSM columns are entirely NaN/missing
    missing = []
    has_osm = []
    for rid, grp in df.groupby('route_id'):
        osm_data = grp[OSM_COLUMNS]
        if osm_data.isna().all().all() or (osm_data.apply(lambda x: x.astype(str) == '').all().all()):
            missing.append(rid)
        else:
            has_osm.append(rid)

    print(f"  Routes WITH OSM features: {len(has_osm)}")
    print(f"  Routes MISSING OSM features: {len(missing)}")
    return missing


def process_route_osm_only(route_id, info, ox_module):
    """Re-process a single route: load GPS, compute trajectory, do OSM enrichment.

    Returns (context_df, quality_dict) or (None, quality_dict) on failure.
    """
    gps_file = info['gps_file']
    gps_df = load_clean_gps(gps_file)

    quality = assess_route_quality(gps_file, gps_df)
    quality['route_id'] = route_id
    quality['vehicle_model'] = info.get('vehicle_model', '')
    quality['driver_id'] = info.get('driver_id', '')

    if gps_df is None or len(gps_df) < 20:
        quality['map_match_status'] = 'skip_no_gps'
        quality['lat_center'] = np.nan
        quality['lon_center'] = np.nan
        return None, quality

    quality['lat_center'] = round(gps_df['lat'].median(), 6)
    quality['lon_center'] = round(gps_df['lon'].median(), 6)

    # Trajectory features (needed for heading input to snap_and_enrich)
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

    # OSM enrichment with timeout
    class TimeoutError(Exception):
        pass

    def _timeout_handler(signum, frame):
        raise TimeoutError("OSM enrichment timed out")

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
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

    # Ensure all expected columns exist
    for col in CONTEXT_COLUMNS:
        if col not in context.columns:
            context[col] = np.nan
    context = context[CONTEXT_COLUMNS]

    return context, quality


def main():
    print("=" * 70)
    print("GPS OSM Re-enrichment Pipeline")
    print("=" * 70)

    context_file = OUTPUT_DIR / "gps_context_features.csv"
    if not context_file.exists():
        print("ERROR: gps_context_features.csv not found. Run gps_semantic_enrichment.py first.")
        sys.exit(1)

    # Step 1: Find routes missing OSM
    missing_routes = find_routes_missing_osm(context_file)
    if not missing_routes:
        print("\nAll routes already have OSM features. Nothing to do.")
        sys.exit(0)

    # Step 2: Init osmnx
    print("\nInitializing osmnx...")
    try:
        ox = init_osm()
        print(f"  osmnx ready, cache: {OUTPUT_DIR / 'osm_cache'}")
    except Exception as e:
        print(f"FATAL: Cannot import osmnx: {e}")
        print("Install with: pip install osmnx")
        sys.exit(1)

    # Step 3: Build route index
    print("\nBuilding route index...")
    routes_df, route_index = build_route_index()

    # Filter to only missing routes that exist in the index
    missing_set = set(missing_routes)
    to_process = {k: v for k, v in route_index.items() if k in missing_set}
    print(f"  {len(to_process)} routes to re-process (of {len(missing_routes)} missing)")

    # Step 4: Read the full existing CSV into memory
    print("\nLoading existing context features into memory...")
    existing_df = pd.read_csv(context_file, dtype={'route_id': str})
    print(f"  {len(existing_df)} rows loaded, {existing_df['route_id'].nunique()} routes")

    # Step 5: Process missing routes
    print(f"\nProcessing {len(to_process)} routes with OSM enrichment...")
    t0 = time.time()
    n_ok = 0
    n_osm_ok = 0
    n_fail = 0
    all_quality = []

    for i, (route_id, info) in enumerate(to_process.items()):
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(to_process)}] {route_id[:35]:35s}  "
                  f"({n_ok} ok, {n_osm_ok} osm, {n_fail} fail)  "
                  f"ETA: {eta/60:.0f}min", flush=True)

        try:
            ctx, qual = process_route_osm_only(route_id, info, ox)
            all_quality.append(qual)

            if ctx is not None:
                n_ok += 1
                if qual.get('map_match_status') == 'ok':
                    n_osm_ok += 1

                # Replace rows for this route in the existing DataFrame
                mask = existing_df['route_id'] == route_id
                existing_df = existing_df[~mask]
                existing_df = pd.concat([existing_df, ctx], ignore_index=True)
            else:
                n_fail += 1

        except Exception as e:
            n_fail += 1
            all_quality.append({
                'route_id': route_id,
                'quality_tier': 'error',
                'map_match_status': f'error:{str(e)[:80]}',
            })
            if i < 5:
                traceback.print_exc()

    elapsed = time.time() - t0
    print(f"\n  Re-processing done: {n_ok} ok, {n_osm_ok} OSM enriched, {n_fail} failed")
    print(f"  Time: {elapsed/60:.1f} minutes")

    # Step 6: Sort and save the updated context features
    print("\nSaving updated gps_context_features.csv...")
    # Ensure column order
    for col in CONTEXT_COLUMNS:
        if col not in existing_df.columns:
            existing_df[col] = np.nan
    existing_df = existing_df[CONTEXT_COLUMNS]
    existing_df.to_csv(context_file, index=False)
    sz = context_file.stat().st_size / (1024*1024)
    print(f"  Saved: {len(existing_df)} rows, {sz:.1f} MB")

    # Step 7: Regenerate quality file (combining old quality with new)
    print("\nRegenerating gps_route_quality.csv...")
    old_quality_file = OUTPUT_DIR / "gps_route_quality.csv"
    if old_quality_file.exists():
        old_q = pd.read_csv(old_quality_file, dtype={'route_id': str})
        # Remove routes we just re-processed
        reprocessed_ids = set(q['route_id'] for q in all_quality if 'route_id' in q)
        old_q = old_q[~old_q['route_id'].isin(reprocessed_ids)]
        new_q = pd.DataFrame(all_quality)
        quality_df = pd.concat([old_q, new_q], ignore_index=True)
    else:
        quality_df = pd.DataFrame(all_quality)
    quality_df.to_csv(old_quality_file, index=False)
    print(f"  Saved: {len(quality_df)} routes")

    # Step 8: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Re-check OSM coverage
    osm_check = existing_df.groupby('route_id')[OSM_COLUMNS[:1]].apply(
        lambda g: g.notna().any().any())
    print(f"  Routes with OSM features: {osm_check.sum()}/{len(osm_check)}")
    print(f"  Routes without OSM: {(~osm_check).sum()}/{len(osm_check)}")

    if 'map_match_status' in quality_df.columns:
        print(f"\n  Map match status:")
        for status, count in quality_df['map_match_status'].value_counts().items():
            print(f"    {status}: {count}")

    print(f"\n  Output: {context_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
