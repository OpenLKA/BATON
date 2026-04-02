"""
config.py — Constants, paths, modality definitions for PassingCtrl baselines.
"""
from pathlib import Path

# ═══════════════════════════════════════════════════════════
# PATHS
# ═══════════════════════════════════════════════════════════
DATASET_ROOT = Path("/home/henry/Desktop/Drive/Dataset")
BENCHMARK_DIR = Path("/home/henry/Desktop/Drive/HMI/benchmark")
DATA_DIR = Path("/home/henry/Desktop/Drive/HMI/data")
BASELINE_DIR = Path("/home/henry/Desktop/Drive/HMI/baseline")
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
