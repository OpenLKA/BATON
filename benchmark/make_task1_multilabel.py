#!/usr/bin/env python3
"""
make_task1_multilabel.py — Multi-label variant of Task-1 action recognition.

Motivation (mock-review): the single-label Task-1 pipeline applies a strict
priority order (Stopped > LaneChange > Turning > Braking > Accelerating >
CarFollowing > Cruising) per second, which collapses co-occurring actions
(e.g. braking-while-turning). This script re-evaluates ALL per-second rule
conditions independently (exact same thresholds / signal logic as
generate_benchmark.assign_action_1hz — only the priority/exclusivity and the
Cruising default are removed) and aggregates to the SAME 5 s windows as
benchmark_v2/task1_action_samples.csv via PER-CLASS majority:
class c is positive for a window iff >=50% of the window's 1 Hz seconds
satisfy rule c. Cruising = no class positive (implicit, no column).

Output: benchmark_v2/task1_action_samples_multilabel.csv
  = all columns of task1_action_samples.csv (same rows, same order, same
    sample_id/route_id/start/end so splits and feature caches align)
  + 6 binary columns ml_Stopped, ml_LaneChange, ml_Turning, ml_Braking,
    ml_Accelerating, ml_CarFollowing.

Also writes benchmark_v2/task1_multilabel_stats.json with per-class positive
rates, multi-label fraction, and agreement of the reconstructed priority
label with the published single label (alignment sanity check).

Read-only w.r.t. existing benchmark files.
"""
import csv
import json
import sys
import time
from collections import Counter
from itertools import groupby
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import generate_benchmark as gb  # noqa: E402  (reuses rule fns + thresholds)

BENCH_DIR = gb.BENCH_DIR  # benchmark_v2/
IN_CSV = BENCH_DIR / "task1_action_samples.csv"
OUT_CSV = BENCH_DIR / "task1_action_samples_multilabel.csv"
STATS_JSON = BENCH_DIR / "task1_multilabel_stats.json"
ROUTE_INDEX = BENCH_DIR / "route_index.csv"

# Priority order of the original taxonomy — kept as the fixed column order.
ML_CLASSES = ["Stopped", "LaneChange", "Turning", "Braking",
              "Accelerating", "CarFollowing"]


def multihot_1hz(vego, aego, steer, bp, lcs, ls, ld, bl, br):
    """Per-second multi-hot over the 6 non-default classes.

    EXACTLY the rule conditions of generate_benchmark.assign_action_1hz,
    evaluated independently (no priority, no Cruising default).
    Returns np.array [n_sec, 6] uint8, column order = ML_CLASSES.
    """
    n = len(vego)
    out = np.zeros((n, 6), dtype=np.uint8)
    for i in range(n):
        v = vego[i]
        a = aego[i]
        s = abs(steer[i])
        out[i, 0] = 1 if v < gb.STOPPED_SPEED else 0
        out[i, 1] = 1 if (lcs[i] > 0 or ((bl[i] > 0 or br[i] > 0) and s > 5)) else 0
        out[i, 2] = 1 if s > gb.TURN_THRESH else 0
        out[i, 3] = 1 if (a < gb.BRAKE_THRESH or bp[i] > 0) else 0
        out[i, 4] = 1 if a > gb.ACCEL_THRESH else 0
        out[i, 5] = 1 if (ls[i] > 0 and ld[i] < gb.LEAD_DIST_THRESH
                          and abs(a) < 1.0) else 0
    return out


def priority_label(hot_row):
    """Reconstruct the original single label from a multi-hot second."""
    for name, v in zip(ML_CLASSES, hot_row):
        if v:
            return name
    return "Cruising"


def main():
    t0 = time.time()

    # route_uid -> on-disk path (emitted by generate_benchmark pass 1)
    route_path = {}
    with open(ROUTE_INDEX) as f:
        for row in csv.DictReader(f):
            route_path[row["route_uid"]] = row["abs_path"]

    in_f = open(IN_CSV, newline="")
    reader = csv.reader(in_f)
    header = next(reader)
    ml_cols = [f"ml_{c}" for c in ML_CLASSES]

    out_f = open(OUT_CSV, "w", newline="")
    writer = csv.writer(out_f)
    writer.writerow(header + ml_cols)

    i_route = header.index("route_id")
    i_start = header.index("start_time_sec")
    i_label = header.index("label")

    n_rows = 0
    n_agree = 0
    pos_counts = Counter()          # class -> n positive windows
    npos_hist = Counter()           # n positive classes per window -> count
    single_label_dist = Counter()   # original label distribution (check)
    combo_counts = Counter()        # frozenset of positive classes -> count
    n_routes = 0
    mismatched_windows = 0
    tail_fallback_windows = 0
    empty_fallback_windows = 0

    for route_id, rows_iter in groupby(reader, key=lambda r: r[i_route]):
        rows = list(rows_iter)
        n_routes += 1
        rdir = route_path.get(route_id)
        if rdir is None:
            raise RuntimeError(f"route {route_id} missing from route_index.csv")

        data = gb.load_route_data(Path(rdir))
        if data is None or len(data["times"]) < 10:
            raise RuntimeError(f"route {route_id}: raw data unavailable")

        times = data["times"]
        resampled = gb.resample_to_1hz(
            times, {"vego": data["vego"], "aego": data["aego"],
                    "steer": data["steer"], "brake_pressed": data["brake_pressed"],
                    "blinker_l": data["blinker_l"], "blinker_r": data["blinker_r"]},
            data["lcs_times"], data["lcs_vals"],
            data["lead_times"], data["lead_status"], data["lead_drel"],
            data["blinker_l"], data["blinker_r"],
        )
        if not resampled[0]:
            raise RuntimeError(f"route {route_id}: empty 1Hz resample")

        (r_times, r_vego, r_aego, r_steer, r_bp,
         r_lcs, r_ls, r_ld, r_bl, r_br) = resampled

        hots = multihot_1hz(r_vego, r_aego, r_steer, r_bp,
                            r_lcs, r_ls, r_ld, r_bl, r_br)
        # Sanity: the priority-collapsed per-second labels must reproduce the
        # published pipeline exactly (same rules, priority re-applied).
        pri = [priority_label(h) for h in hots]
        # The published pass 2 read timestamps back from action_labels.csv where
        # they were written as "%.1f" — replicate that rounding for bit-exact
        # window membership.
        sec_times = np.asarray([float(f"{x:.1f}") for x in r_times])
        cum = np.concatenate([np.zeros((1, 6), dtype=np.int64),
                              np.cumsum(hots, axis=0, dtype=np.int64)])

        # Regenerate windows with the ORIGINAL float accumulation so window
        # membership matches task1_action_samples.csv bit-for-bit.
        t_start = times[0]
        t_end = times[-1]
        def emit(row, lo, hi, check_agreement):
            nonlocal n_rows, n_agree
            n_sec = hi - lo
            counts = cum[hi] - cum[lo]                  # [6]
            pos = (2 * counts >= n_sec).astype(int)     # >=50% of seconds

            if check_agreement:
                # Original majority label reconstruction (validation only)
                maj = Counter(pri[lo:hi]).most_common(1)[0][0]
                if maj == row[i_label]:
                    n_agree += 1
            single_label_dist[row[i_label]] += 1

            for c, p in zip(ML_CLASSES, pos):
                if p:
                    pos_counts[c] += 1
            npos = int(pos.sum())
            npos_hist[npos] += 1
            if npos >= 2:
                combo_counts[tuple(c for c, p in zip(ML_CLASSES, pos) if p)] += 1

            writer.writerow(list(row) + [int(p) for p in pos])
            n_rows += 1

        row_i = 0
        t = t_start + gb.INPUT_WINDOW
        while t <= t_end and row_i < len(rows):
            w_start = t - gb.INPUT_WINDOW
            w_end = t
            lo = int(np.searchsorted(sec_times, w_start, side="left"))
            hi = int(np.searchsorted(sec_times, w_end, side="left"))
            if hi <= lo:  # empty window — skipped by the original generator too
                t += gb.STRIDE
                continue
            row = rows[row_i]
            row_i += 1

            # Alignment check against the published CSV row
            if row[i_start] != f"{w_start:.2f}":
                mismatched_windows += 1
                if mismatched_windows <= 5:
                    print(f"  WARN start mismatch {route_id}: "
                          f"csv={row[i_start]} regen={w_start:.2f}")

            emit(row, lo, hi, check_agreement=True)
            t += gb.STRIDE

        # Tail fallback: raw route data can be marginally shorter today than at
        # benchmark-generation time, leaving a few published windows at the very
        # end of a route that the regen loop no longer reaches. Label them from
        # whatever 1Hz seconds still fall inside the published window (epsilon
        # absorbs the 2-decimal rounding of the stored start time).
        EPS = 0.05
        for row in rows[row_i:]:
            tail_fallback_windows += 1
            w_start = float(row[i_start])
            w_end = w_start + gb.INPUT_WINDOW
            lo = int(np.searchsorted(sec_times, w_start - EPS, side="left"))
            hi = int(np.searchsorted(sec_times, w_end - EPS, side="left"))
            if hi <= lo:
                # No current-data seconds left in this published window at all —
                # fall back to the published single label as a one-hot vector
                # (Cruising -> all-zero), the best available estimate.
                empty_fallback_windows += 1
                single_label_dist[row[i_label]] += 1
                pos = [1 if c == row[i_label] else 0 for c in ML_CLASSES]
                for c, p in zip(ML_CLASSES, pos):
                    if p:
                        pos_counts[c] += 1
                npos_hist[sum(pos)] += 1
                writer.writerow(list(row) + pos)
                n_rows += 1
            else:
                emit(row, lo, hi, check_agreement=False)

        if n_routes % 50 == 0:
            print(f"  [{n_routes} routes] rows={n_rows} "
                  f"agree={n_agree/max(n_rows,1):.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    in_f.close()
    out_f.close()

    frac_multi = sum(v for k, v in npos_hist.items() if k >= 2) / n_rows
    frac_none = npos_hist.get(0, 0) / n_rows
    stats = {
        "n_windows": n_rows,
        "n_routes": n_routes,
        "class_order": ML_CLASSES,
        "per_class_positive_rate": {c: pos_counts[c] / n_rows for c in ML_CLASSES},
        "per_class_positive_count": {c: pos_counts[c] for c in ML_CLASSES},
        "n_positive_classes_histogram": {str(k): v for k, v in sorted(npos_hist.items())},
        "frac_windows_multilabel_ge2": frac_multi,
        "frac_windows_no_class_cruising": frac_none,
        "top_cooccurrence_combos": {" + ".join(k): v for k, v in
                                    combo_counts.most_common(15)},
        "priority_label_agreement_with_published":
            n_agree / max(n_rows - tail_fallback_windows, 1),
        "window_start_mismatches": mismatched_windows,
        "tail_fallback_windows": tail_fallback_windows,
        "empty_fallback_windows_all_zero": empty_fallback_windows,
        "aggregation": "class positive iff >=50% of window's 1Hz seconds satisfy its rule",
        "source_samples": str(IN_CSV),
        "elapsed_s": time.time() - t0,
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=2)

    print("\n══════ Task-1 multi-label statistics ══════")
    print(f"windows: {n_rows}  (routes: {n_routes})")
    print(f"priority-label agreement with published CSV: "
          f"{n_agree/max(n_rows - tail_fallback_windows,1):.4f}")
    print(f"window-start mismatches: {mismatched_windows}")
    print(f"tail-fallback windows: {tail_fallback_windows} "
          f"(all-zero: {empty_fallback_windows})")
    print("per-class positive rate:")
    for c in ML_CLASSES:
        print(f"  {c:>14}: {pos_counts[c]/n_rows*100:6.2f}%  ({pos_counts[c]})")
    print(f"windows with >=2 positive labels: {frac_multi*100:.2f}%")
    print(f"windows with 0 positive labels (Cruising): {frac_none*100:.2f}%")
    print("top co-occurrence combos:")
    for k, v in combo_counts.most_common(10):
        print(f"  {' + '.join(k)}: {v} ({v/n_rows*100:.2f}%)")
    print(f"\nWrote {OUT_CSV}")
    print(f"Wrote {STATS_JSON}")
    print(f"Elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
