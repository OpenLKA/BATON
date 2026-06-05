#!/usr/bin/env python3
"""
make_anticipation_safe_t3.py (Exp 2) — window-buffer anticipation protocol for takeover.

In openpilot the human disengages by physically overriding (brake / gas / steering). Those
driver-input fields are the *trigger* of the takeover label, so a 5 s window that already
contains the override action leaks the label. This script makes T3 positives strictly
ANTICIPATORY: for each takeover event it finds the override onset and keeps only positive
windows that END at least BUFFER seconds BEFORE that onset (so the override is not in the
window). Negatives are unchanged.

Output: benchmark_v2/task3_takeover_samples_h3_antsafe.csv
Usage:  python3 make_anticipation_safe_t3.py [--buffer 1.0 --search 3.0 --horizon 3]
"""
import argparse, csv, sys
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from config import BENCHMARK_DIR
from dataset import RouteCache

OVERRIDE = ["brakePressed", "steeringPressed", "gasPressed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", type=float, default=1.0, help="window must end ≥ buffer s before override onset")
    ap.add_argument("--search", type=float, default=3.0, help="look back this many s for the override onset")
    ap.add_argument("--horizon", type=int, default=3)
    args = ap.parse_args()

    src = BENCHMARK_DIR / f"task3_takeover_samples_h{args.horizon}.csv"
    out = BENCHMARK_DIR / f"task3_takeover_samples_h{args.horizon}_antsafe.csv"
    rows = list(csv.DictReader(open(src)))
    header = rows[0].keys()
    pos = [r for r in rows if r["label"] == "1"]
    neg = [r for r in rows if r["label"] == "0"]

    # group positives by (route, event)
    by_ev = defaultdict(list)
    for r in pos:
        by_ev[(r["route_id"], round(float(r["nearest_event_time"]), 2))].append(r)

    cache = RouteCache()
    cache.preload(sorted({r["route_id"] for r in pos}))

    def override_onset(route_id, ev_t):
        """First time in [ev_t-search, ev_t] where any override==1; else ev_t (system-initiated)."""
        rd = cache._load_struct_npz(route_id)
        if rd is None or "vehicle_dynamics.csv" not in rd:
            return ev_t, False
        t0, step, data, cols = rd["vehicle_dynamics.csv"]
        cols = [str(c) for c in cols]
        idx = [cols.index(c) for c in OVERRIDE if c in cols]
        if not idx:
            return ev_t, False
        i_lo = max(0, int(round((ev_t - args.search - t0) / step)))
        i_hi = min(data.shape[0] - 1, int(round((ev_t - t0) / step)))
        if i_hi <= i_lo:
            return ev_t, False
        seg = data[i_lo:i_hi + 1][:, idx]            # [n, |override|]
        fired = (seg >= 0.5).any(axis=1)
        if not fired.any():
            return ev_t, False                        # no override → system-initiated, no leak
        onset_i = i_lo + int(np.argmax(fired))
        return t0 + onset_i * step, True

    kept, dropped, n_driver, n_system = [], 0, 0, 0
    for (rid, ev_t), samples in by_ev.items():
        onset, is_driver = override_onset(rid, ev_t)
        n_driver += int(is_driver); n_system += int(not is_driver)
        cutoff = (onset - args.buffer) if is_driver else ev_t  # system-initiated: keep standard
        for r in samples:
            if float(r["end_time_sec"]) <= cutoff:
                kept.append(r)
            else:
                dropped += 1

    allrows = kept + neg
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(header))
        w.writeheader(); w.writerows(allrows)

    print(f"events: {len(by_ev)} ({n_driver} driver-override, {n_system} system-initiated)")
    print(f"positives: {len(pos)} -> {len(kept)} kept ({dropped} dropped as in/near-override)")
    print(f"negatives: {len(neg)} (unchanged)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
