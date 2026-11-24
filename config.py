"""GA-SLNet experiment configuration."""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(ROOT, "data", "raw")
PROC_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "output")
FIG_DIR = os.path.join(ROOT, "figs")
for _d in (RAW_DIR, PROC_DIR, OUT_DIR, FIG_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------- data ----------------
DATES = ["2020-01-01", "2020-07-01", "2020-12-01"]
TRAIN_DATES = ["2020-01-01", "2020-07-01"]
EVAL_DATE = "2020-12-01"

REGIONS = {
    "Gulf":     dict(lat=(24.0, 31.5),  lon=(-98.0, -80.5)),
    "Atlantic": dict(lat=(24.0, 45.0),  lon=(-82.0, -64.0)),
    "Pacific":  dict(lat=(32.0, 49.0),  lon=(-126.0, -116.0)),
}
TRAIN_REGIONS = ["Gulf", "Atlantic"]
CROSS_REGION = "Pacific"

SOG_MIN, SOG_MAX = 0.6, 40.0      # knot, filter noise / impossible speed
TRIP_MAX_GAP_MIN = 60.0           # a silence longer than this starts a new trip
MIN_POINTS = 40                   # min reports in a usable trip
MIN_STEP_SEC = 20.0               # thin duplicate / sub-20s reports

T_OBS_MIN = 45.0                  # observation window
HORIZONS_MIN = [5, 10, 15, 20, 30]
MAX_OBS_STEPS = 64                # cap on reports fed to the model
WINDOW_STRIDE_MIN = 20.0
MAX_WINDOWS_PER_TRIP = 3

# ---------------- model ----------------
N_FEAT = 8
N_ATTR = 12
HIDDEN = 64
N_MODES = 3
K_EXPERTS = 3
DROPOUT = 0.1

# ---------------- training ----------------
SEED = 2020
BATCH = 256
EPOCHS = 30
LR = 2e-3
WD = 1e-4
GRAD_CLIP = 1.0
USE_CV_ANCHOR = False             # CV anchor tested, did not help
GAP_AUG_P = 0.25                  # curriculum-style gap augmentation while training
GAP_AUG_MAX_MIN = 30.0

# ---------------- evaluation ----------------
GAP_SWEEP = [0, 5, 15, 30, 60]    # minutes of injected observation blackout
EVAL_SEED = 7
TARGET_SIGMA_KM = 0.05            # floor on predictive sigma
