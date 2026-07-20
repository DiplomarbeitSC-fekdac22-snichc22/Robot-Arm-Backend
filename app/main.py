import uuid

import cv2
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import raw_stream, stop_camera
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
from .detection import (
    detection_stream,
    get_detection_snapshot,
    stop_detection,
)


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

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


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
    """
    Raw camera MJPEG stream.

    All connected clients receive frames from the same camera
    capture thread.
    """
    return StreamingResponse(
        raw_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detect")
def detect():
    """
    YOLO-annotated MJPEG stream.

    All connected clients receive results from the same
    background detection thread.
    """
    return StreamingResponse(
        detection_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/detections")
def detections():
    """
    Return the latest cached YOLO detection result.

    This endpoint does not run another inference.
    """
    snapshot = get_detection_snapshot()

    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail="Detection result is not available yet",
        )

    return {
        "timestamp": snapshot["timestamp"],
        "detections": snapshot["detections"],
    }


@app.get("/objects")
def objects(request: Request):
    """
    Return the latest detected objects and save their crops.

    The detections and source frame come from the same cached
    detection result, so bounding boxes match the saved image.
    """
    snapshot = get_detection_snapshot()

    if snapshot is None:
        raise HTTPException(
            status_code=503,
            detail="Detection result is not available yet",
        )

    frame = snapshot["frame"]
    detections = snapshot["detections"]
    timestamp = snapshot["timestamp"]

    height, width = frame.shape[:2]
    objects_list = []

    # Save the full detection frame once. All objects from this
    # request reference the same frame.
    frame_filename = f"frame_{uuid.uuid4().hex}.jpg"
    frame_path = FRAMES_DIR / frame_filename

    frame_saved = cv2.imwrite(str(frame_path), frame)

    if not frame_saved:
        raise HTTPException(
            status_code=500,
            detail="Could not save detection frame",
        )

    base_url = str(request.base_url).rstrip("/")
    frame_url = f"{base_url}/static/frames/{frame_filename}"

    for detection in detections:
        x1 = int(detection["bbox"]["x1"])
        y1 = int(detection["bbox"]["y1"])
        x2 = int(detection["bbox"]["x2"])
        y2 = int(detection["bbox"]["y2"])

        # Keep coordinates inside the image.
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        # Ignore invalid or zero-sized bounding boxes.
        if x2 <= x1 or y2 <= y1:
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        # Use the class ID in the filename so unusual class names
        # cannot accidentally create invalid paths.
        crop_filename = (
            f"class_{detection['class_id']}_"
            f"{uuid.uuid4().hex}.jpg"
        )
        crop_path = CROPS_DIR / crop_filename

        crop_saved = cv2.imwrite(str(crop_path), crop)

        if not crop_saved:
            continue

        crop_url = f"{base_url}/static/crops/{crop_filename}"

        objects_list.append({
            **detection,
            "crop_url": crop_url,
            "frame_url": frame_url,
            "timestamp": timestamp,
        })

    return {
        "timestamp": timestamp,
        "objects": objects_list,
    }


# --------------------------------------------------
# Shutdown
# --------------------------------------------------

@app.on_event("shutdown")
def shutdown():
    # Stop inference first because it consumes camera frames.
    stop_detection()
    stop_camera()