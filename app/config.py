import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------
# Paths / env
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", "./best.pt")
MODEL_PATH = str((BASE_DIR / MODEL_PATH).resolve())

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
ROTATE_CAMERA_180 = os.getenv("ROTATE_CAMERA_180", "false").lower() == "true"

STATIC_DIR = BASE_DIR / "static"
CROPS_DIR = STATIC_DIR / "crops"
FRAMES_DIR = STATIC_DIR / "frames"
CROPS_DIR.mkdir(parents=True, exist_ok=True)
FRAMES_DIR.mkdir(parents=True, exist_ok=True)