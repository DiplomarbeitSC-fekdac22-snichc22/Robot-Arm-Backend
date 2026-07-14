import os
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

import cv2
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from picamera2 import Picamera2
from libcamera import controls, Transform

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

MODEL_PATH = os.getenv("MODEL_PATH", "./best.pt")
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.30"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
ROTATE_CAMERA_180=os.getenv("ROTATE_CAMERA_180", "false").lower() == "true"

MODEL_PATH = str((BASE_DIR / MODEL_PATH).resolve())

STATIC_DIR = BASE_DIR / "static"
CROPS_DIR = STATIC_DIR / "crops"
CROPS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FastAPI setup
# --------------------------------------------------

app = FastAPI(title="Robot Arm Vision Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------
# Set up Model
# --------------------------------------------------

print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("Model classes:", model.names)


# --------------------------------------------------
# Set up Camera
# --------------------------------------------------

print("Starting Raspberry Pi Camera Module 3...")
camera = Picamera2()

camera_transform = Transform(
    hflip=1 if ROTATE_CAMERA_180 else 0,
    vflip=1 if ROTATE_CAMERA_180 else 0,
)

camera_config = camera.create_video_configuration(
    main={
        "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
        "format": "RGB888",
    },
    transform=camera_transform,
)

camera.configure(camera_config)
camera.start()

# Camera autofocus
try:
    camera.set_controls({
        "AfMode": controls.AfModeEnum.Continuous
    })
except Exception as error:
    print("Autofocus warning:", error)

time.sleep(1)

print("Pi camera started.")

camera_lock = threading.Lock()


# --------------------------------------------------
# OLD WORKING CAMERA FUNCTIONS
# Same logic as the prototype.
# --------------------------------------------------

def read_camera():
    with camera_lock:
        frame = camera.capture_array()

    return frame


def make_jpeg(frame):
    success, buffer = cv2.imencode(".jpg", frame)

    if not success:
        return None

    return buffer.tobytes()


def send_frame(jpeg):
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
    )


# --------------------------------------------------
# Detection helper functions
# --------------------------------------------------

def run_detection(frame):
    return model.predict(
        source=frame,
        imgsz=320,
        conf=0.35,
        iou=0.45,
        max_det=10,
        verbose=False,
    )[0]


def parse_detections(result):
    detections = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "id": str(uuid.uuid4()),
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 3),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            },
            "center": {
                "x": round((x1 + x2) / 2, 2),
                "y": round((y1 + y2) / 2, 2),
            },
        })

    return detections


# --------------------------------------------------
# OLD WORKING STREAMS
# --------------------------------------------------

def raw_stream():
    while True:
        frame = read_camera()

        jpeg = make_jpeg(frame)

        if jpeg is None:
            continue

        yield send_frame(jpeg)

def detection_stream():
    while True:
        frame = read_camera()

        result = run_detection(frame)
        frame_with_boxes = result.plot()

        jpeg = make_jpeg(frame_with_boxes)

        if jpeg is None:
            continue

        yield send_frame(jpeg)

# --------------------------------------------------
# Endpoints
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "name": "Robot Arm Vision Backend",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_path": MODEL_PATH,
        "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT,
        "confidence": YOLO_CONF,
        "rotate_camera_180": ROTATE_CAMERA_180,
    }


@app.get("/video")
def video():
    return StreamingResponse(
        raw_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detect")
def detect():
    return StreamingResponse(
        detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections")
def detections():
    frame = read_camera()
    result = run_detection(frame)

    return {
        "timestamp": datetime.now().isoformat(),
        "detections": parse_detections(result),
    }


@app.get("/objects")
def objects(request: Request):
    frame = read_camera()
    result = run_detection(frame)
    detections = parse_detections(result)

    height, width = frame.shape[:2]
    objects_list = []

    for detection in detections:
        x1 = int(detection["bbox"]["x1"])
        y1 = int(detection["bbox"]["y1"])
        x2 = int(detection["bbox"]["x2"])
        y2 = int(detection["bbox"]["y2"])

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        filename = f"{detection['class_name']}_{uuid.uuid4().hex}.jpg"
        crop_path = CROPS_DIR / filename

        # Same as prototype: no color conversion
        cv2.imwrite(str(crop_path), crop)

        crop_url = str(request.base_url).rstrip("/") + f"/static/crops/{filename}"

        objects_list.append({
            **detection,
            "crop_url": crop_url,
            "timestamp": datetime.now().isoformat(),
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "objects": objects_list,
    }


@app.on_event("shutdown")
def shutdown():
    camera.stop()