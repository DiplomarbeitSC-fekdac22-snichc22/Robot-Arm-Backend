import os
import math
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# Paths / env
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", "./best.pt" if (BASE_DIR / "best.pt").exists() else "models/best.pt")
MODEL_PATH = str((BASE_DIR / MODEL_PATH).resolve())

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
ROTATE_CAMERA_180 = os.getenv("ROTATE_CAMERA_180", "false").lower() == "true"

CAMERA_CONFIGS = {}
for side, default_index in (("left", 0), ("right", 1)):
    prefix = f"CAMERA_{side.upper()}_"
    lens_position = os.getenv(prefix + "LENS_POSITION")
    CAMERA_CONFIGS[side] = {
        "index": int(os.getenv(prefix + "INDEX", str(default_index))),
        "enabled": os.getenv(prefix + "ENABLED", "true").lower() == "true",
        "width": int(os.getenv(prefix + "WIDTH", str(CAMERA_WIDTH))),
        "height": int(os.getenv(prefix + "HEIGHT", str(CAMERA_HEIGHT))),
        "rotate_180": os.getenv(prefix + "ROTATE_180", str(ROTATE_CAMERA_180)).lower() == "true",
        "lens_position": float(lens_position) if lens_position else None,
    }
    camera = CAMERA_CONFIGS[side]
    if camera["index"] < 0 or min(camera["width"], camera["height"]) <= 0:
        raise ValueError(f"Invalid {side} camera index or resolution")
    if camera["lens_position"] is not None and (
        not math.isfinite(camera["lens_position"]) or camera["lens_position"] < 0
    ):
        raise ValueError(f"Invalid {side} camera lens position")

MAX_FRAME_AGE_SECONDS = float(os.getenv("MAX_FRAME_AGE_SECONDS", "5.0"))
STEREO_MAX_TIME_SKEW_SECONDS = float(os.getenv("STEREO_MAX_TIME_SKEW_SECONDS", "0.10"))
if not all(math.isfinite(value) and value > 0 for value in (
    MAX_FRAME_AGE_SECONDS, STEREO_MAX_TIME_SKEW_SECONDS,
)):
    raise ValueError("Frame age and stereo time limits must be finite and positive")

STATIC_DIR = BASE_DIR / "static"
CROPS_DIR = STATIC_DIR / "crops"
FRAMES_DIR = STATIC_DIR / "frames"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)
