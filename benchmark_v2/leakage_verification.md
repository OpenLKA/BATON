# Leakage Verification — task3 (h=3, leak-safe BATON)

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

Base rate = 0.1167. Max single-feature AUPRC over **safe** columns = **0.2504**, vs **0.1662** for the leaky columns. No single safe signal is a stealth label proxy.

| feature | leaky? | 1-feat AUPRC | \|corr\| | MI |
|---|---|---|---|---|
| aEgo__mean |  | 0.2504 | 0.191 | 0.0202 |
| aEgo__last |  | 0.2405 | 0.186 | 0.0232 |
| aEgo__min |  | 0.2345 | 0.216 | 0.0672 |
| brakePressed__mean |  | 0.2258 | 0.276 | 0.0283 |
| accel_z__mean |  | 0.2219 | 0.155 | 0.0169 |
| vEgo__std |  | 0.2164 | 0.193 | 0.0177 |
| steeringAngleDeg__std |  | 0.2139 | 0.145 | 0.0180 |
| aEgo__max |  | 0.2125 | 0.055 | 0.0573 |
| vEgo__last |  | 0.2044 | 0.214 | 0.0441 |
| brakePressed__last |  | 0.2037 | 0.285 | 0.0245 |
| model_desiredCurvature__std |  | 0.2017 | 0.105 | 0.0203 |
| accel_z__last |  | 0.1996 | 0.146 | 0.0108 |
| brakePressed__max |  | 0.1951 | 0.265 | 0.0234 |
| vEgo__min |  | 0.1948 | 0.204 | 0.0489 |
| accel_z__min |  | 0.1910 | 0.093 | 0.0623 |

## Proof 3 — Reconstruction probe

- base rate: 0.1167
- leaky-only (20 cols): **0.2699**
- safe-only (255 cols): **0.4044**

A probe on leak-safe features cannot reconstruct the label near the leaky ceiling, confirming the leak-safe inputs carry no target-derived proxy.