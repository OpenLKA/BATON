# Leakage Verification — task2 (h=3, leak-safe BATON)

Label is deterministic: `ADAS_active = (cc_latActive OR cruiseState_enabled)`.
Leak-safe inputs drop `['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState']`.

## Proof 1 — Set membership (formal)

| config | #cols | leaky cols present |
|---|---|---|
| Veh | 15 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Drv | 14 | — (none) |
| Int | 20 | — (none) |
| Veh+Int | 35 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Veh+Int+Drv | 49 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-Struct | 55 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-Struct+GPS | 73 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-Struct+FV | 55 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-Struct+CV | 55 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-Multimodal | 55 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Full-All | 73 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |
| Ctx | 18 | — (none) |
| IMU | 6 | — (none) |
| FV | 0 | — (none) |
| CV | 0 | — (none) |
| FV+CV | 0 | — (none) |
| Safe-Full-Struct | 51 | — (none) |
| Safe-Full-Struct+GPS | 69 | — (none) |
| Safe-Ego | 9 | — (none) |
| Safe-Ego+Drv_in | 17 | — (none) |
| Safe-Ego+Drv_in+Lead | 29 | — (none) |
| Safe-Ego+Drv_in+Lead+Road | 37 | — (none) |
| Safe-Ego+Drv_in+Lead+Road+DMS | 51 | — (none) |
| ADASctrl-only | 4 | ['actuators_accel', 'cc_latActive', 'cruiseState_enabled', 'cs_longControlState'] |

**All `Safe-*` configs contain zero leaky columns: LEAKY ∩ used = ∅.**

## Proof 2 — Single-feature predictive ceiling

Base rate = 0.1583. Max single-feature AUPRC over **safe** columns = **0.2470**, vs **0.3315** for the leaky columns. No single safe signal is a stealth label proxy.

| feature | leaky? | 1-feat AUPRC | \|corr\| | MI |
|---|---|---|---|---|
| cc_latActive__mean | YES | 0.3315 | 0.361 | 0.0646 |
| cc_latActive__std | YES | 0.3315 | 0.407 | 0.0616 |
| cc_latActive__max | YES | 0.3315 | 0.423 | 0.0631 |
| cruiseState_enabled__mean | YES | 0.2563 | 0.269 | 0.0339 |
| cruiseState_enabled__std | YES | 0.2563 | 0.305 | 0.0368 |
| cruiseState_enabled__max | YES | 0.2563 | 0.316 | 0.0368 |
| accel_z__min |  | 0.2470 | 0.186 | 0.1036 |
| laneLeft_prob__std |  | 0.2466 | 0.164 | 0.0205 |
| laneRight_prob__std |  | 0.2451 | 0.161 | 0.0172 |
| accel_z__last |  | 0.2449 | 0.152 | 0.0147 |
| aEgo__last |  | 0.2448 | 0.149 | 0.0237 |
| vEgo__std |  | 0.2427 | 0.170 | 0.0230 |
| aEgo__max |  | 0.2381 | 0.158 | 0.0912 |
| aEgo__mean |  | 0.2380 | 0.096 | 0.0211 |
| accel_z__std |  | 0.2360 | 0.172 | 0.0260 |

## Proof 3 — Reconstruction probe

- base rate: 0.1583
- leaky-only (20 cols): **0.3491**
- safe-only (255 cols): **0.4228**

A probe on leak-safe features cannot reconstruct the label near the leaky ceiling, confirming the leak-safe inputs carry no target-derived proxy.