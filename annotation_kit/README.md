# BATON Task 1 — Human Annotation Kit (500 windows)

Validates whether the **rule-generated** Task-1 action labels match human judgment
(addresses reviewers' "rule reconstruction" concern). Class-balanced: ~71 windows per
class including 71 LaneChange.

## How to run

```bash
cd annotation_kit
python3 -m http.server 8000      # avoids file:// video issues
# open http://localhost:8000 in a browser
```

(If `clips/` isn't fully populated yet, the full 500 clips are extracted right after the
V-JEPA2 front feature-extraction finishes — a ~4-minute job. A few preview clips are
present now so you can try the interface.)

## How to label

Watch each clip (front road view of the **ego vehicle**, autoplay+loop) and press the key
for what the ego vehicle is doing:

| key | class | meaning |
|---|---|---|
| 1 | Cruising | steady speed; no notable accel/brake/turn; no close lead |
| 2 | CarFollowing | following a lead vehicle at a steady gap |
| 3 | Stopped | vehicle stationary |
| 4 | Turning | turning at intersection / curve (clear steering) |
| 5 | Braking | slowing / braking |
| 6 | Accelerating | speeding up |
| 7 | LaneChange | lateral lane change (often with blinker) |

- **1–7** = label + auto-advance · **←/→** = prev/next · **R** = replay · **B** = blind toggle.
- The **rule label is shown** beside the clip and turns green (match) / red (differ) after
  you choose. Press **B** to hide it and label a *blind* subset (recommended for ~100
  windows so we can also report an unbiased agreement number).
- Progress + your labels are **saved automatically** (browser localStorage); you can stop
  and resume anytime (it reopens at the first unlabeled window).

## When done (or partway)

Click **⤓ Download annotations.csv** and send me the file. Then:

```bash
python3 ../baseline/task1_agreement.py annotations.csv
```

prints overall accuracy, **Cohen's κ**, and per-class F1 (incl. LaneChange) — the numbers
that go into the rebuttal.
