import logging
import json
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Literal
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, Request as UrlRequest, build_opener

import cv2
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .camera import camera_is_running, camera_status, make_jpeg, raw_stream, read_camera, start_cameras, stop_camera
from .config import (
    CAMERA_HEIGHT, CAMERA_WIDTH, CROPS_DIR, FRAMES_DIR, MODEL_PATH,
    ROTATE_CAMERA_180, STATIC_DIR, STEREO_MAX_TIME_SKEW_SECONDS, YOLO_CONF,
)
from .detection import (
    detection_status, detection_stream, get_detection_snapshot,
    get_detection_snapshots, start_detection, stop_detection,
)
from .errors import AppError, CameraUnavailableError, DetectionUnavailableError, ImageSaveError
from .localization import get_calibration_status, localize_stereo


logger = logging.getLogger(__name__)
CameraId = Literal["left", "right"]
_mode = "manual"
_image_lock = threading.Lock()
_image_cache = {}


@asynccontextmanager
async def lifespan(app):
    global _mode
    _mode = "manual"
    start_cameras()
    start_detection()
    try:
        yield
    finally:
        stop_detection()
        stop_camera()


app = FastAPI(title="Robot Arm Vision Backend", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class KinematicsTargetRequest(BaseModel):
    object_id: str = Field(min_length=1)
    class_name: str = Field(min_length=1)
    y_mm: float = Field(ge=0.0, le=500.0)
    z_mm: float = Field(ge=-180.0, le=180.0)


class OperationModeRequest(BaseModel):
    mode: Literal["manual", "automatic"]


@app.exception_handler(AppError)
async def app_error_handler(request: Request, error: AppError):
    logger.warning("%s %s failed: %s", request.method, request.url.path, error.code)
    return JSONResponse(status_code=error.status_code, content={
        "status": "error", "code": error.code, "message": error.message,
        "movement_allowed": False, "movement_started": False,
    })


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, error: Exception):
    logger.error("Unexpected error during %s %s", request.method, request.url.path,
                 exc_info=(type(error), error, error.__traceback__))
    return JSONResponse(status_code=500, content={
        "status": "error", "code": "INTERNAL_ERROR",
        "message": "An unexpected backend error occurred",
        "movement_allowed": False, "movement_started": False,
    })


def require_detection_snapshot(timeout=5.0, camera_id="left"):
    snapshot = get_detection_snapshot(timeout=timeout, camera_id=camera_id)
    if snapshot is None:
        raise DetectionUnavailableError()
    return snapshot


@app.get("/")
def root():
    return {"status": "running", "name": "Robot Arm Vision Backend"}


@app.get("/health")
def health():
    cameras = camera_status()
    detection = detection_status()
    return {
        "status": "ok" if all((c["available"] and detection["cameras"][side]) or not c["enabled"]
                              for side, c in cameras.items())
                  and detection["available"] else "degraded",
        "camera_available": cameras["left"]["available"], "cameras": cameras,
        "detection": detection, "calibration": get_calibration_status(),
        "model_path": MODEL_PATH, "camera_width": CAMERA_WIDTH,
        "camera_height": CAMERA_HEIGHT, "confidence": YOLO_CONF,
        "rotate_camera_180": ROTATE_CAMERA_180,
    }


@app.get("/video")
@app.get("/video/{camera_id}")
@app.get("/calibration/video/{camera_id}")
def video(camera_id: CameraId = "left"):
    if not camera_is_running(camera_id):
        raise CameraUnavailableError()
    return StreamingResponse(raw_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/detect")
@app.get("/detect/{camera_id}")
def detect(camera_id: CameraId = "left"):
    if not camera_is_running(camera_id):
        raise CameraUnavailableError()
    require_detection_snapshot(timeout=2.0, camera_id=camera_id)
    return StreamingResponse(detection_stream(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/snapshot")
def snapshot(camera_id: CameraId = "left"):
    if not camera_is_running(camera_id):
        raise CameraUnavailableError()
    try:
        jpeg = make_jpeg(read_camera(camera_id=camera_id))
    except RuntimeError as error:
        raise CameraUnavailableError() from error
    if jpeg is None:
        raise ImageSaveError()
    return Response(jpeg, media_type="image/jpeg")


@app.get("/detections")
def detections(camera_id: CameraId = "left"):
    snapshot = require_detection_snapshot(camera_id=camera_id)
    return {"timestamp": snapshot["timestamp"], "camera_id": camera_id,
            "detections": snapshot["detections"]}


def _save_image(path, frame):
    temporary = path.with_name(f".{path.stem}.tmp.jpg")
    if not cv2.imwrite(str(temporary), frame):
        raise ImageSaveError()
    temporary.replace(path)


def _object_images(side, snapshot):
    with _image_lock:
        sequence = snapshot["sequence"]
        if side in _image_cache and _image_cache[side][0] == sequence:
            return _image_cache[side][1]
        frame = snapshot["frame"]
        height, width = frame.shape[:2]
        _save_image(FRAMES_DIR / f"{side}.jpg", frame)
        images = {}
        for index, detection in enumerate(snapshot["detections"]):
            try:
                values = [float(detection["bbox"][key]) for key in ("x1", "y1", "x2", "y2")]
                if not all(math.isfinite(value) for value in values):
                    continue
                x1, y1, x2, y2 = map(int, values)
                x1, x2 = max(0, x1), min(width, x2)
                y1, y2 = max(0, y1), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    continue
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            _save_image(CROPS_DIR / f"{side}_{index}.jpg", frame[y1:y2, x1:x2])
            images[detection["id"]] = f"/static/crops/{side}_{index}.jpg?v={sequence}"
        result = (f"/static/frames/{side}.jpg?v={sequence}", images)
        _image_cache[side] = (sequence, result)
        return result


@app.get("/objects")
@app.get("/detections/objects")
def objects(request: Request, camera_id: CameraId | None = None):
    snapshots = get_detection_snapshots(timeout=2.0)
    objects_list = localize_stereo(snapshots.get("left"), snapshots.get("right"),
                                   max_time_skew_s=STEREO_MAX_TIME_SKEW_SECONDS)
    image_urls = {side: _object_images(side, snapshot) for side, snapshot in snapshots.items()}
    base_url = str(request.base_url).rstrip("/")
    for item in objects_list:
        side = item["camera_ids"][0]
        observation = snapshots[side]
        frame_url, crops = image_urls[side]
        crop_url = crops.get(item["observation_ids"][0])
        item.update({"camera_id": side, "timestamp": observation["timestamp"],
                     "frame_url": base_url + frame_url,
                     "crop_url": base_url + crop_url if crop_url else None})
    if camera_id is not None:
        objects_list = [item for item in objects_list if camera_id in item["camera_ids"]]
    return {"timestamp": max((s["timestamp"] for s in snapshots.values()), default=None),
            "objects": objects_list, "cameras": camera_status(),
            "calibration": get_calibration_status()}


@app.get("/calibration/status")
def calibration_status():
    return get_calibration_status()


@app.get("/operation-mode")
def operation_mode():
    return _operation_status()


def _controller_request(mode=None):
    url = os.getenv("ROBOT_CONTROLLER_URL", "http://127.0.0.1:8765").rstrip("/")
    parsed = urlsplit(url)
    if (parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
        raise ValueError("ROBOT_CONTROLLER_URL must point to the local controller API")
    request = UrlRequest(url + ("/mode" if mode is not None else "/status"),
                         data=json.dumps({"mode": mode}).encode() if mode is not None else None,
                         headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=0.5) as response:
            contents = response.read(16385)
    except HTTPError as error:
        if mode is None or error.code != 409:
            raise
        status = _controller_request()
        status["mode_error"] = "Controller rejected the mode change; release controls and retry"
        return status
    if len(contents) > 16384:
        raise ValueError("Controller response is too large")
    status = json.loads(contents)
    if not isinstance(status, dict) or status.get("mode") not in ("manual", "automatic"):
        raise ValueError("Invalid controller status")
    return status


def _operation_status(requested_mode=None):
    global _mode
    try:
        controller = _controller_request(requested_mode)
        _mode = controller["mode"]
        return {"mode": _mode, "requested_mode": requested_mode or _mode,
                "controller_mode": _mode, "controller_connected": True,
                "controller": controller, "movement_allowed": False,
                "reason": controller.get("mode_error") or "Automatic pickup is not integrated; bounding-box centres are not validated grasp points"}
    except (OSError, ValueError, TypeError) as error:
        return {"mode": _mode, "requested_mode": _mode, "controller_mode": None,
                "controller_connected": False, "controller": None,
                "movement_allowed": False,
                "reason": "Controller unavailable; selection has not been applied. Automatic pickup remains disabled.",
                "controller_error": str(error)}


@app.post("/operation-mode")
def set_operation_mode(payload: OperationModeRequest):
    global _mode
    _mode = payload.mode
    return _operation_status(payload.mode)


def movement_unavailable():
    return JSONResponse(status_code=409, content={
        "status": "error", "code": "MOVEMENT_NOT_PERMITTED",
        "message": operation_mode()["reason"],
        "movement_allowed": False, "movement_started": False,
    })


@app.post("/kinematics/target")
def send_kinematics_target(payload: KinematicsTargetRequest):
    return movement_unavailable()


@app.post("/kinematics/scan-pose")
def move_to_scan_pose():
    return movement_unavailable()
