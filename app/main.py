import uuid
from datetime import datetime

import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import raw_stream, read_camera, stop_camera
from .config import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    CROPS_DIR,
    FRAMES_DIR,
    MODEL_PATH,
    ROTATE_CAMERA_180,
    STATIC_DIR,
    YOLO_CONF,
)
from .detection import detection_stream, parse_detections, run_detection

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

    # Save the full frame once for this request; all objects below share it.
    frame_filename = f"frame_{uuid.uuid4().hex}.jpg"
    frame_path = FRAMES_DIR / frame_filename
    cv2.imwrite(str(frame_path), frame)
    frame_url = str(request.base_url).rstrip("/") + f"/static/frames/{frame_filename}"

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
            "frame_url": frame_url,
            "timestamp": datetime.now().isoformat(),
        })

    return {
        "timestamp": datetime.now().isoformat(),
        "objects": objects_list,
    }


@app.on_event("shutdown")
def shutdown():
    stop_camera()