"""
config.py — Constants, paths, modality definitions for PassingCtrl baselines.
"""
from pathlib import Path
from paths import BATON_ROOT  # single source of truth for raw-data location

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
# Raw BATON dataset (new layout: <car_model>/<dongle_id>/<route_id>/<segment_id>/).
DATASET_ROOT = BATON_ROOT
# Regenerated benchmark (v2) — kept separate from the published v1 for before/after.
REPO_DIR = Path(__file__).resolve().parent.parent  # /home/henry/Desktop/Drive/Benchmark/BATON
BENCHMARK_DIR = REPO_DIR / "benchmark_v2"
DATA_DIR = REPO_DIR / "data"
BASELINE_DIR = REPO_DIR / "baseline"
CACHE_DIR = BASELINE_DIR / "cache"
RESULTS_DIR = BASELINE_DIR / "results"

FRONT_VIDEO_DIR = DATA_DIR / "front_video_features"
CABIN_VIDEO_DIR = DATA_DIR / "cabin_video_features"
PCA_FRONT_VIDEO_DIR = DATA_DIR / "pca128_front_video_features"
PCA_CABIN_VIDEO_DIR = DATA_DIR / "pca128_cabin_video_features"
VIDEO_FEATURE_DIM_PCA = 128
CLIP_FRONT_VIDEO_DIR = DATA_DIR / "clip_front_video_features"
CLIP_CABIN_VIDEO_DIR = DATA_DIR / "clip_cabin_video_features"
VIDEO_FEATURE_DIM_CLIP = 512
VJEPA2_FRONT_VIDEO_DIR = DATA_DIR / "vjepa2_front_video_features"
VJEPA2_CABIN_VIDEO_DIR = DATA_DIR / "vjepa2_cabin_video_features"
VIDEO_FEATURE_DIM_VJEPA2 = 1024
GPS_CONTEXT_PATH = DATA_DIR / "gps_context_features.csv"

# ═══════════════════════════════════════════════════════════
# TASK DEFINITIONS
# ═══════════════════════════════════════════════════════════
TASK1_LABELS = [
    "Accelerating", "Braking", "CarFollowing", "Cruising",
    "LaneChange", "Stopped", "Turning",
]
LABEL2IDX = {l: i for i, l in enumerate(TASK1_LABELS)}
IDX2LABEL = {i: l for l, i in LABEL2IDX.items()}
NUM_CLASSES_TASK1 = 7

# ═══════════════════════════════════════════════════════════
# SIGNAL COLUMNS PER SOURCE CSV
# ═══════════════════════════════════════════════════════════

VEHICLE_COLS = [
    "vEgo", "aEgo", "steeringAngleDeg", "steeringTorque", "steeringPressed",
    "gas", "gasPressed", "brake", "brakePressed",
    "cruiseState_enabled", "cc_latActive",
    "leftBlinker", "rightBlinker",
    "actuators_accel", "cs_longControlState",
]

PLANNING_COLS = [
    "model_desiredCurvature", "model_desiredAcceleration",
    "laneLeft_prob", "laneRight_prob", "laneLeft_y", "laneRight_y",
    "laneChangeState", "hasLead",
]

RADAR_COLS = [
    "leadOne_status", "leadOne_dRel", "leadOne_vRel", "leadOne_aRel",
    "leadOne_yRel", "leadOne_vLead",
    "leadTwo_status", "leadTwo_dRel", "leadTwo_vRel", "leadTwo_aRel",
    "leadTwo_yRel", "leadTwo_vLead",
]

DRIVER_COLS = [
    "face_yaw", "face_pitch", "face_roll", "face_pos_x", "face_pos_y",
    "faceProb", "leftEyeProb", "rightEyeProb",
    "leftBlinkProb", "rightBlinkProb",
    "sunglassesProb", "occludedProb",
    "readyProb_1", "notReadyProb_1",
]

IMU_COLS = [
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
]

GPS_COLS = [
    "gps_speed_mps", "heading_deg", "heading_change_rate", "curvature",
    "is_stopped", "stopped_duration_s",
    "road_type_enc", "is_highway", "speed_limit_kph", "n_lanes",
    "is_on_ramp", "dist_to_intersection_m", "is_near_intersection",
    "road_network_density", "bearing_vs_road",
    "hw_dist_to_intersection_m", "hw_n_intersections_300m",
    "hw_is_ramp_ahead_300m",
]

ROAD_TYPE_MAP = {
    "motorway": 0, "trunk": 1, "primary": 2, "secondary": 3,
    "tertiary": 4, "residential": 5, "other": 6,
}

# ═══════════════════════════════════════════════════════════
# MODALITY GROUPS — maps group name → (source_csv, columns)
# ═══════════════════════════════════════════════════════════

# source_csv is the filename inside each route's ACM_MM/route_*/  directory
STRUCT_GROUPS = {
    "Veh": ("vehicle_dynamics.csv", VEHICLE_COLS),
    "Int_plan": ("planning.csv", PLANNING_COLS),
    "Int_radar": ("radar.csv", RADAR_COLS),
    "Drv": ("driver_state.csv", DRIVER_COLS),
    "IMU": ("imu.csv", IMU_COLS),
    # GPS is loaded separately (single global CSV, not per-route file)
}

# ═══════════════════════════════════════════════════════════
# MODALITY CONFIGS — which groups to include for each experiment
# ═══════════════════════════════════════════════════════════

MODALITY_CONFIGS = {
    # Single-modality
    "Veh":              {"struct": ["Veh"],                                         "gps": False, "front_video": False, "cabin_video": False},
    "Drv":              {"struct": ["Drv"],                                         "gps": False, "front_video": False, "cabin_video": False},
    "Int":              {"struct": ["Int_plan", "Int_radar"],                       "gps": False, "front_video": False, "cabin_video": False},
    # Incremental struct
    "Veh+Int":          {"struct": ["Veh", "Int_plan", "Int_radar"],               "gps": False, "front_video": False, "cabin_video": False},
    "Veh+Int+Drv":      {"struct": ["Veh", "Int_plan", "Int_radar", "Drv"],        "gps": False, "front_video": False, "cabin_video": False},
    # Full-Struct = all structured signals, NO GPS
    "Full-Struct":      {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": False, "front_video": False, "cabin_video": False},
    # GPS added as separate branch
    "Full-Struct+GPS":  {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": True,  "front_video": False, "cabin_video": False},
    # Video combos (no GPS)
    "Full-Struct+FV":   {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": False, "front_video": True,  "cabin_video": False},
    "Full-Struct+CV":   {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": False, "front_video": False, "cabin_video": True},
    "Full-Multimodal":  {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": False, "front_video": True,  "cabin_video": True},
    # GPS added last = truly full modality
    "Full-All":         {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": True,  "front_video": True,  "cabin_video": True},
    # Supplementary single-modality
    "Ctx":              {"struct": [],                                              "gps": True,  "front_video": False, "cabin_video": False},
    "IMU":              {"struct": ["IMU"],                                         "gps": False, "front_video": False, "cabin_video": False},
    "FV":               {"struct": [],                                              "gps": False, "front_video": True,  "cabin_video": False},
    "CV":               {"struct": [],                                              "gps": False, "front_video": False, "cabin_video": True},
    "FV+CV":            {"struct": [],                                              "gps": False, "front_video": True,  "cabin_video": True},
}

# ═══════════════════════════════════════════════════════════
# LEAKAGE-SAFE INPUT SET + HIERARCHICAL CAN TAXONOMY (B2)
# ═══════════════════════════════════════════════════════════
# Labels for Tasks 2/3 are defined as ADAS_active = (cc_latActive==1) OR
# (cruiseState_enabled==1). These two flags — plus the openpilot internal
# longitudinal-control state and commanded acceleration — are direct label
# proxies / DA-internal precursors and MUST be removed for a leakage-safe input.
LEAKY_COLS = {
    "cc_latActive", "cruiseState_enabled",   # label-defining flags
    "cs_longControlState", "actuators_accel", # DA-internal control state & command
}

# vehicle_dynamics columns with the leaky flags removed.
VEHICLE_COLS_SAFE = [c for c in VEHICLE_COLS if c not in LEAKY_COLS]

# --- Layered CAN sub-groups (each maps to ONE source CSV) -------------------
# Compose these in MODALITY_CONFIGS["struct"]; the dataset concatenates groups.
STRUCT_GROUPS.update({
    # leakage-safe full vehicle_dynamics
    "Veh_safe":     ("vehicle_dynamics.csv", VEHICLE_COLS_SAFE),
    # ego kinematics (what the car is doing) — physics, leak-safe
    "Veh_ego":      ("vehicle_dynamics.csv", ["vEgo", "aEgo", "steeringAngleDeg"]),
    # human driver inputs — leak-safe
    "Veh_drvinput": ("vehicle_dynamics.csv", ["gas", "gasPressed", "brake", "brakePressed",
                                              "steeringTorque", "steeringPressed",
                                              "leftBlinker", "rightBlinker"]),
    # turn-signal intent only (no override) — for the anticipation/no-override setting
    "Veh_blinker":  ("vehicle_dynamics.csv", ["leftBlinker", "rightBlinker"]),
    # ADAS control flags — LEAKY, reference-only upper bound
    "Veh_adasctrl": ("vehicle_dynamics.csv", ["cc_latActive", "cruiseState_enabled",
                                              "cs_longControlState", "actuators_accel"]),
})

# All struct columns referenced above are subsets of the existing groups' union,
# so the 50Hz npz cache built by preprocess.py already contains them — no extra
# preprocessing is required to run any leakage-safe / ablation config.

MODALITY_CONFIGS.update({
    # Leakage-safe counterpart of Full-Struct (the headline anti-leakage input set).
    "Safe-Full-Struct":   {"struct": ["Veh_safe", "Int_plan", "Int_radar", "Drv", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "Safe-Full-Struct+GPS": {"struct": ["Veh_safe", "Int_plan", "Int_radar", "Drv", "IMU"],
                           "gps": True,  "front_video": False, "cabin_video": False},
    # Add-one-group ablation ladder over the leak-safe base.
    "Safe-Ego":           {"struct": ["Veh_ego", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "Safe-Ego+Drv_in":    {"struct": ["Veh_ego", "IMU", "Veh_drvinput"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "Safe-Ego+Drv_in+Lead": {"struct": ["Veh_ego", "IMU", "Veh_drvinput", "Int_radar"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "Safe-Ego+Drv_in+Lead+Road": {"struct": ["Veh_ego", "IMU", "Veh_drvinput", "Int_radar", "Int_plan"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "Safe-Ego+Drv_in+Lead+Road+DMS": {"struct": ["Veh_ego", "IMU", "Veh_drvinput", "Int_radar", "Int_plan", "Drv"],
                           "gps": False, "front_video": False, "cabin_video": False},
    # LEAKY reference: how trivially the ADAS-control flags alone predict the label.
    "ADASctrl-only":      {"struct": ["Veh_adasctrl"],
                           "gps": False, "front_video": False, "cabin_video": False},
    # Hierarchical leak-safe multimodal input for RG-HBT-Q: the 6 leak-safe CAN
    # semantic categories (kept SEPARATE so the model can fuse them by category)
    # + front and cabin video. Leak-safe by construction (excludes Veh_adasctrl).
    # Struct order fixes the per-category split [3, 8, 12, 8, 14, 6].
    "Hier-Safe-MM":       {"struct": ["Veh_ego", "Veh_drvinput", "Int_radar",
                                      "Int_plan", "Drv", "IMU"],
                           "gps": False, "front_video": True, "cabin_video": True},
    "Hier-Safe-MM+GPS":   {"struct": ["Veh_ego", "Veh_drvinput", "Int_radar",
                                      "Int_plan", "Drv", "IMU"],
                           "gps": True, "front_video": True, "cabin_video": True},
    # Driver-input-guided fusion input (DI-RG-HBT-Q): driver-input is the hub, the
    # rest of the leak-safe CAN is auxiliary context. Driver-monitoring (Drv/DMS) is
    # REMOVED from the inputs. Order puts Veh_drvinput first for readability (the
    # model locates it by name, not position). Struct dims [8,3,12,8,6] = 37.
    "DI-Safe-MM":         {"struct": ["Veh_drvinput", "Veh_ego", "Int_radar",
                                      "Int_plan", "IMU"],
                           "gps": False, "front_video": True, "cabin_video": True},
    # Structured-only counterpart of DI-Safe-MM (no video, no DMS) — the MATCHED
    # leak-safe CAN input for XGBoost so the tree baseline and the DI model see the
    # exact same controller signals (DMS removed per the latest spec).
    "DI-Safe-Struct":     {"struct": ["Veh_drvinput", "Veh_ego", "Int_radar",
                                      "Int_plan", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    # Further-reduced: also drop the planning/road-geometry group (Int_plan, incl.
    # the automation's planner outputs). Structured-only, no DMS, no planning.
    "DI-Safe-Struct-noPlan": {"struct": ["Veh_drvinput", "Veh_ego", "Int_radar", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    # ANTICIPATION / no-override setting: remove the 6 driver-override fields (the
    # in-window disengagement trigger), keep blinker intent. Lowers the tree's
    # spike advantage -> predict-before-the-driver-acts. Struct dims [3,2,12,8,6]=31.
    "DI-Safe-Struct-noOver": {"struct": ["Veh_ego", "Veh_blinker", "Int_radar",
                                         "Int_plan", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "DI-Safe-MM-noOver":  {"struct": ["Veh_blinker", "Veh_ego", "Int_radar",
                                      "Int_plan", "IMU"],
                           "gps": False, "front_video": True, "cabin_video": True},
})

# ═══════════════════════════════════════════════════════════
# V-JEPA2 7-TYPE ABLATION CONFIGS (T3)
# ═══════════════════════════════════════════════════════════
# Seven leak-safe input TYPES. RoadGeom uses planning lane-geometry (Int_plan)
# rather than GPS, since GPS context features are not extracted for benchmark_v2.
ABLATION_TYPES = {
    "FrontVideo": {"front_video": True},
    "CabinVideo": {"cabin_video": True},
    "Ego":        {"struct": ["Veh_ego", "IMU"]},
    "DrvInput":   {"struct": ["Veh_drvinput"]},
    "Lead":       {"struct": ["Int_radar"]},
    "DMS":        {"struct": ["Drv"]},
    "RoadGeom":   {"struct": ["Int_plan"]},
}


def _compose_types(type_names):
    cfg = {"struct": [], "gps": False, "front_video": False, "cabin_video": False}
    for t in type_names:
        spec = ABLATION_TYPES[t]
        cfg["struct"] += spec.get("struct", [])
        cfg["front_video"] = cfg["front_video"] or spec.get("front_video", False)
        cfg["cabin_video"] = cfg["cabin_video"] or spec.get("cabin_video", False)
    return cfg


_ALL_TYPES = list(ABLATION_TYPES)
# Full (all 7), leave-one-out (drop each), single-modality (each alone)
MODALITY_CONFIGS["VJEPA-Full"] = _compose_types(_ALL_TYPES)
for _t in _ALL_TYPES:
    MODALITY_CONFIGS[f"VJEPA-no-{_t}"] = _compose_types([x for x in _ALL_TYPES if x != _t])
    MODALITY_CONFIGS[f"VJEPA-only-{_t}"] = _compose_types([_t])

# ═══════════════════════════════════════════════════════════
# TRAINING DEFAULTS
# ═══════════════════════════════════════════════════════════
RESAMPLE_HZ = 50
INPUT_WINDOW_SEC = 5.0
STRUCT_SEQ_LEN = int(INPUT_WINDOW_SEC * RESAMPLE_HZ)  # 250
VIDEO_FPS = 2
VIDEO_SEQ_LEN = int(INPUT_WINDOW_SEC * VIDEO_FPS)  # 10
VIDEO_FEATURE_DIM = 1280

GRU_HIDDEN = 256
GRU_LAYERS_STRUCT = 2
GRU_LAYERS_VIDEO = 1
GRU_LAYERS_GPS = 1
FUSION_DIM = 256
DROPOUT = 0.3

BATCH_SIZE = 2048
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 7
NUM_WORKERS = 4
SEEDS = [42, 123, 7]

MAX_CLASS_WEIGHT = 10.0
MAX_POS_WEIGHT = 10.0
LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 3

# ═══════════════════════════════════════════════════════════
# TASK 1 RULE-FREE ABLATION (A) — exclude label-defining signals
# ═══════════════════════════════════════════════════════════
# Task-1 rules use: vEgo, aEgo, steeringAngleDeg, brakePressed, laneChangeState,
# leftBlinker, rightBlinker, leadOne_status, leadOne_dRel. A "rule-free" input drops
# exactly these; if Task 1 still beats chance, the actions are recoverable from
# independent evidence (not just rule reconstruction).
TASK1_RULE_COLS = {"vEgo", "aEgo", "steeringAngleDeg", "brakePressed", "laneChangeState",
                   "leftBlinker", "rightBlinker", "leadOne_status", "leadOne_dRel"}

STRUCT_GROUPS.update({
    "Veh_t1free":   ("vehicle_dynamics.csv", [c for c in VEHICLE_COLS if c not in TASK1_RULE_COLS]),
    "Plan_t1free":  ("planning.csv",         [c for c in PLANNING_COLS if c not in TASK1_RULE_COLS]),
    "Radar_t1free": ("radar.csv",            [c for c in RADAR_COLS if c not in TASK1_RULE_COLS]),
})

MODALITY_CONFIGS.update({
    # full structured (repro ~0.910) — reuse existing "Full-Struct"
    "T1-RuleFree-Struct": {"struct": ["Veh_t1free", "Plan_t1free", "Radar_t1free", "Drv", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
    "T1-Drv-only":        {"struct": ["Drv"], "gps": False, "front_video": False, "cabin_video": False},
    "T1-IMU-only":        {"struct": ["IMU"], "gps": False, "front_video": False, "cabin_video": False},
})

# ═══════════════════════════════════════════════════════════
# INCREMENTAL-WITHHOLD ABLATION (which fields to withhold, and the difference)
# ═══════════════════════════════════════════════════════════
# Candidate leaky fields, by category:
#   label-defining flags : cc_latActive, cruiseState_enabled   (define ADAS_active)
#   DA-internal/command  : cs_longControlState, actuators_accel (precursor candidates)
#   DA-planner outputs   : model_desiredCurvature, model_desiredAcceleration
_FLAGS = {"cc_latActive", "cruiseState_enabled"}
_CS    = {"cs_longControlState"}
_ACT   = {"actuators_accel"}
_PLANNER_OUT = {"model_desiredCurvature", "model_desiredAcceleration"}

STRUCT_GROUPS.update({
    "Veh_noflags":      ("vehicle_dynamics.csv", [c for c in VEHICLE_COLS if c not in _FLAGS]),
    "Veh_noflags_nocs": ("vehicle_dynamics.csv", [c for c in VEHICLE_COLS if c not in (_FLAGS | _CS)]),
    "Veh_noactuator":   ("vehicle_dynamics.csv", [c for c in VEHICLE_COLS if c not in _ACT]),
    "Plan_noout":       ("planning.csv",         [c for c in PLANNING_COLS if c not in _PLANNER_OUT]),
})

# All configs share the same plan/radar/drv/imu tail; only the vehicle (and, last, planner) set changes.
MODALITY_CONFIGS.update({
    "WH-Full":           {"struct": ["Veh", "Int_plan", "Int_radar", "Drv", "IMU"],            "gps": False, "front_video": False, "cabin_video": False},
    "WH-noFlags":        {"struct": ["Veh_noflags", "Int_plan", "Int_radar", "Drv", "IMU"],    "gps": False, "front_video": False, "cabin_video": False},
    "WH-noFlags-noCS":   {"struct": ["Veh_noflags_nocs", "Int_plan", "Int_radar", "Drv", "IMU"],"gps": False, "front_video": False, "cabin_video": False},
    "WH-noActuator":     {"struct": ["Veh_noactuator", "Int_plan", "Int_radar", "Drv", "IMU"], "gps": False, "front_video": False, "cabin_video": False},
    "WH-Safe":           {"struct": ["Veh_safe", "Int_plan", "Int_radar", "Drv", "IMU"],       "gps": False, "front_video": False, "cabin_video": False},
    "WH-Safe-noPlanner": {"struct": ["Veh_safe", "Plan_noout", "Int_radar", "Drv", "IMU"],     "gps": False, "front_video": False, "cabin_video": False},
})

# ── Takeover residual-leak probe: also withhold driver-OVERRIDE fields ──
# In openpilot the human disengages by braking / gas / steering-override, so these driver
# inputs are the *mechanism* of takeover (near-deterministic trigger). "Anticipation-safe"
# additionally withholds them, keeping only signals not under the override action itself.
_OVERRIDE = {"brakePressed", "gasPressed", "steeringPressed", "steeringTorque", "brake", "gas"}
STRUCT_GROUPS.update({
    "Veh_noctrl_noover": ("vehicle_dynamics.csv",
        [c for c in VEHICLE_COLS if c not in (LEAKY_COLS | _OVERRIDE)]),
})
MODALITY_CONFIGS.update({
    "WH-Safe-noOverride": {"struct": ["Veh_noctrl_noover", "Int_plan", "Int_radar", "Drv", "IMU"],
                           "gps": False, "front_video": False, "cabin_video": False},
})
