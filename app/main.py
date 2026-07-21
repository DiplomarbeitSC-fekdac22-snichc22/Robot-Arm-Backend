import logging
import uuid

import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .camera import (
    camera_is_running,
    raw_stream,
    stop_camera,
)
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
from .errors import (
    AppError,
    CameraUnavailableError,
    DetectionUnavailableError,
    ImageSaveError,
)


# --------------------------------------------------
# Logging
# --------------------------------------------------

logger = logging.getLogger(__name__)


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
# Exception handlers
# --------------------------------------------------

@app.exception_handler(AppError)
async def app_error_handler(
    request: Request,
    error: AppError,
):
    logger.warning(
        "%s %s failed: %s",
        request.method,
        request.url.path,
        error.code,
    )

    return JSONResponse(
        status_code=error.status_code,
        content={
            "status": "error",
            "code": error.code,
            "message": error.message,
            "movement_allowed": False,
            "movement_started": False,
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    error: Exception,
):
    logger.error(
        "Unexpected error during %s %s",
        request.method,
        request.url.path,
        exc_info=(
            type(error),
            error,
            error.__traceback__,
        ),
    )

    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "code": "INTERNAL_ERROR",
            "message": "An unexpected backend error occurred",
            "movement_allowed": False,
            "movement_started": False,
        },
    )


# --------------------------------------------------
# Helpers
# --------------------------------------------------

def require_detection_snapshot(timeout=5.0):
    try:
        snapshot = get_detection_snapshot(timeout=timeout)
    except (RuntimeError, TimeoutError) as error:
        logger.exception(
            "Could not get detection snapshot"
        )

        raise DetectionUnavailableError() from error

    if snapshot is None:
        raise DetectionUnavailableError()

    return snapshot


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
    camera_available = camera_is_running()

    return {
        "status": (
            "ok"
            if camera_available
            else "degraded"
        ),
        "camera_available": camera_available,
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
    """
    if not camera_is_running():
        raise CameraUnavailableError()

    return StreamingResponse(
        raw_stream(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.get("/detect")
def detect():
    """
    YOLO-annotated MJPEG stream.
    """
    if not camera_is_running():
        raise CameraUnavailableError()

    # Wait briefly for the detection worker to produce its
    # first result before opening the stream.
    require_detection_snapshot(timeout=2.0)

    return StreamingResponse(
        detection_stream(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.get("/detections")
def detections():
    """
    Return the latest cached YOLO detections.
    """
    snapshot = require_detection_snapshot()

    return {
        "timestamp": snapshot["timestamp"],
        "detections": snapshot["detections"],
    }


@app.get("/objects")
def objects(request: Request):
    """
    Return the latest detected objects and save their crops.
    """
    snapshot = require_detection_snapshot()

    frame = snapshot["frame"]
    detections = snapshot["detections"]
    timestamp = snapshot["timestamp"]

    height, width = frame.shape[:2]
    objects_list = []

    # Save the full frame once for all detected objects.
    frame_filename = (
        f"frame_{uuid.uuid4().hex}.jpg"
    )
    frame_path = FRAMES_DIR / frame_filename

    frame_saved = cv2.imwrite(
        str(frame_path),
        frame,
    )

    if not frame_saved:
        raise ImageSaveError()

    base_url = str(request.base_url).rstrip("/")

    frame_url = (
        f"{base_url}/static/frames/"
        f"{frame_filename}"
    )

    for detection in detections:
        try:
            bbox = detection["bbox"]

            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            logger.warning(
                "Ignoring malformed detection: %r",
                detection,
            )
            continue

        # Keep coordinates inside the image.
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        # Ignore invalid or zero-sized boxes.
        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Ignoring invalid bounding box: %r",
                bbox,
            )
            continue

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            logger.warning(
                "Ignoring empty crop for detection: %r",
                detection,
            )
            continue

        class_id = detection.get(
            "class_id",
            "unknown",
        )

        crop_filename = (
            f"class_{class_id}_"
            f"{uuid.uuid4().hex}.jpg"
        )
        crop_path = CROPS_DIR / crop_filename

        crop_saved = cv2.imwrite(
            str(crop_path),
            crop,
        )

        if not crop_saved:
            logger.warning(
                "Could not save crop: %s",
                crop_path,
            )
            continue

        crop_url = (
            f"{base_url}/static/crops/"
            f"{crop_filename}"
        )

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
    try:
        stop_detection()
    except Exception:
        logger.exception(
            "Could not stop detection worker"
        )

    try:
        stop_camera()
    except Exception:
        logger.exception(
            "Could not stop camera"
        )