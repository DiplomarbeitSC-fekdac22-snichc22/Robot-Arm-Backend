from __future__ import annotations

import logging
import uuid

import cv2
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
from .kinematics_bridge import (
    KinematicsBridgeError,
    run_hardware_pick,
    run_scan_pose,
)
from .localization import (
    FIXED_TARGET_X_MM,
    LocalizationError,
    localize_detection,
)


logger = logging.getLogger(__name__)


# Requested coordinate corrections.
Y_COORDINATE_OFFSET_MM = 20.0
Z_COORDINATE_SCALE = 0.9


app = FastAPI(
    title="Robot Arm Vision Backend",
)

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


class KinematicsTargetRequest(BaseModel):
    object_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)

    y_mm: float = Field(
        ge=0.0,
        le=500.0,
    )

    z_mm: float = Field(
        ge=-180.0,
        le=180.0,
    )


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
            "message": (
                "An unexpected backend error occurred"
            ),
            "movement_allowed": False,
            "movement_started": False,
        },
    )


def require_detection_snapshot(
    timeout: float = 5.0,
):
    try:
        snapshot = get_detection_snapshot(
            timeout=timeout,
        )

    except (
        RuntimeError,
        TimeoutError,
    ) as error:
        logger.exception(
            "Could not get detection snapshot"
        )

        raise DetectionUnavailableError() from error

    if snapshot is None:
        raise DetectionUnavailableError()

    return snapshot


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
        "rotate_camera_180": (
            ROTATE_CAMERA_180
        ),
    }


@app.get("/video")
def video():
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
    if not camera_is_running():
        raise CameraUnavailableError()

    require_detection_snapshot(
        timeout=2.0,
    )

    return StreamingResponse(
        detection_stream(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        ),
    )


@app.get("/detections")
def detections():
    snapshot = require_detection_snapshot()

    return {
        "timestamp": snapshot["timestamp"],
        "detections": snapshot["detections"],
    }


@app.get("/objects")
def objects(request: Request):
    snapshot = require_detection_snapshot()

    frame = snapshot["frame"]
    detections = snapshot["detections"]
    timestamp = snapshot["timestamp"]

    height, width = frame.shape[:2]

    objects_list: list[dict] = []

    frame_filename = (
        f"frame_{uuid.uuid4().hex}.jpg"
    )

    frame_path = (
        FRAMES_DIR / frame_filename
    )

    frame_saved = cv2.imwrite(
        str(frame_path),
        frame,
    )

    if not frame_saved:
        raise ImageSaveError()

    base_url = str(
        request.base_url
    ).rstrip("/")

    frame_url = (
        f"{base_url}/static/frames/"
        f"{frame_filename}"
    )

    for detection in detections:
        try:
            bbox = detection["bbox"]

            x1 = int(float(bbox["x1"]))
            y1 = int(float(bbox["y1"]))
            x2 = int(float(bbox["x2"]))
            y2 = int(float(bbox["y2"]))

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

        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))

        if x2 <= x1 or y2 <= y1:
            logger.warning(
                "Ignoring invalid bounding box: %r",
                bbox,
            )
            continue

        try:
            robot_coordinates = localize_detection(
                frame.shape,
                detection,
            )

            # Requested calibration corrections:
            # Y = calculated Y + 20 mm
            # Z = calculated Z * 0.9
            robot_coordinates["y_mm"] = round(
                float(
                    robot_coordinates["y_mm"]
                )
                + Y_COORDINATE_OFFSET_MM,
                1,
            )

            robot_coordinates["z_mm"] = round(
                float(
                    robot_coordinates["z_mm"]
                )
                * Z_COORDINATE_SCALE,
                1,
            )

        except LocalizationError as error:
            logger.warning(
                "Could not localize %s detection: %s",
                detection.get(
                    "class_name",
                    "unknown",
                ),
                error,
            )

            # Do not return selectable objects that
            # cannot be sent to Kinematics.
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

        crop_path = (
            CROPS_DIR / crop_filename
        )

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
            "robot_coordinates": (
                robot_coordinates
            ),
            "crop_url": crop_url,
            "frame_url": frame_url,
            "timestamp": timestamp,
        })

    return {
        "timestamp": timestamp,
        "objects": objects_list,
    }


@app.post("/kinematics/target")
def send_kinematics_target(
    payload: KinematicsTargetRequest,
):
    target = {
        "x_mm": FIXED_TARGET_X_MM,
        "y_mm": round(
            float(payload.y_mm),
            1,
        ),
        "z_mm": round(
            float(payload.z_mm),
            1,
        ),
    }

    logger.info(
        "Kinematics target requested: "
        "object=%s class=%s target=%r",
        payload.object_id,
        payload.class_name,
        target,
    )

    try:
        result = run_hardware_pick(
            x_mm=target["x_mm"],
            y_mm=target["y_mm"],
            z_mm=target["z_mm"],
        )

    except KinematicsBridgeError as error:
        logger.error(
            "Kinematics hardware run failed: %s",
            error,
        )

        return JSONResponse(
            status_code=502,
            content={
                "status": "error",
                "code": "KINEMATICS_FAILED",
                "message": str(error),
                "movement_allowed": False,
                "movement_started": False,
            },
        )

    logger.info(
        "Kinematics hardware run completed: "
        "object=%s target=%r",
        payload.object_id,
        target,
    )

    return {
        "status": "accepted",
        "object_id": payload.object_id,
        "class_name": payload.class_name,
        "target": target,
        "kinematics": result,
    }


@app.post("/kinematics/scan-pose")
def move_to_scan_pose():
    try:
        return {
            "status": "completed",
            "kinematics": run_scan_pose(),
        }

    except KinematicsBridgeError as error:
        raise HTTPException(
            status_code=503,
            detail=str(error),
        ) from error


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