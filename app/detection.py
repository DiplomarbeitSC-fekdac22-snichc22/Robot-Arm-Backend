import copy
import threading
import time
import uuid

from .camera import camera_is_running, make_jpeg, send_frame, wait_for_frame_packet
from .config import CAMERA_CONFIGS, MAX_FRAME_AGE_SECONDS, MODEL_PATH, YOLO_CONF


model = None
_model_lock = threading.Lock()
_condition = threading.Condition()
_running = threading.Event()
_thread = None
_error = "Detection has not started"
_snapshots = {}
_jpegs = {}


def run_detection(frame):
    with _model_lock:
        if model is None:
            raise RuntimeError(_error)
        return model.predict(source=frame, imgsz=320, conf=YOLO_CONF, iou=0.45,
                             max_det=10, verbose=False)[0]


def parse_detections(result, camera_id="left"):
    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "id": str(uuid.uuid4()), "camera_id": camera_id,
            "class_id": class_id, "class_name": model.names[class_id],
            "confidence": round(float(box.conf[0]), 3),
            "bbox": dict(zip(("x1", "y1", "x2", "y2"),
                             (round(value, 2) for value in (x1, y1, x2, y2)))),
            "center": {"x": round((x1 + x2) / 2, 2), "y": round((y1 + y2) / 2, 2)},
        })
    return detections


def _detection_loop():
    global _error
    sequences = {side: -1 for side in CAMERA_CONFIGS}
    try:
        while _running.is_set():
            # Capture both observations before running the shared model.
            packets = {side: wait_for_frame_packet(sequences[side], timeout=0.1, camera_id=side)
                       for side in CAMERA_CONFIGS if camera_is_running(side)}
            snapshots, jpegs = {}, {}
            for side, packet in packets.items():
                if packet is None or not _running.is_set():
                    continue
                sequences[side] = packet["sequence"]
                try:
                    result = run_detection(packet["frame"])
                    detections = parse_detections(result, side)
                    jpeg = make_jpeg(result.plot())
                    if jpeg is None:
                        raise RuntimeError("Could not encode detection frame")
                    snapshots[side] = {**packet, "detections": detections}
                    jpegs[side] = jpeg
                    _error = None
                except Exception as error:
                    _error = str(error)
                    print(f"Camera {side} detection error:", error)
            with _condition:
                _snapshots.clear()
                _jpegs.clear()
                if _running.is_set():
                    _snapshots.update(snapshots)
                    _jpegs.update(jpegs)
                _condition.notify_all()
            if not snapshots:
                time.sleep(0.1)
    finally:
        _running.clear()
        with _condition:
            _condition.notify_all()


def start_detection():
    global model, _thread, _error
    if _running.is_set():
        return
    try:
        if model is None:
            from ultralytics import YOLO
            model = YOLO(MODEL_PATH)
        _running.set()
        _error = None
        _thread = threading.Thread(target=_detection_loop, name="yolo-detection", daemon=True)
        _thread.start()
    except Exception as error:
        _error = str(error)
        print("Detection unavailable:", error)


def _fresh(side):
    snapshot = _snapshots.get(side)
    return snapshot is not None and camera_is_running(side) and (
        time.monotonic() - snapshot["captured_at"] <= MAX_FRAME_AGE_SECONDS
    )


def detection_status():
    return {"available": _running.is_set(), "error": _error,
            "cameras": {side: _fresh(side) for side in CAMERA_CONFIGS}}


def get_detection_snapshots(timeout=0.0):
    with _condition:
        _condition.wait_for(lambda: any(_fresh(side) for side in CAMERA_CONFIGS)
                            or not _running.is_set(), timeout=timeout)
        return {side: {**snapshot, "detections": copy.deepcopy(snapshot["detections"])}
                for side, snapshot in _snapshots.items() if _fresh(side)}


def get_detection_snapshot(timeout=10.0, camera_id="left"):
    with _condition:
        _condition.wait_for(lambda: _fresh(camera_id) or not _running.is_set(), timeout=timeout)
        snapshot = _snapshots.get(camera_id) if _fresh(camera_id) else None
        if snapshot is None:
            return None
        return {**snapshot, "frame": snapshot["frame"].copy(),
                "detections": copy.deepcopy(snapshot["detections"])}


def detection_stream(camera_id="left"):
    sequence = -1
    while _running.is_set() and camera_is_running(camera_id):
        with _condition:
            available = _condition.wait_for(
                lambda: not _running.is_set() or not camera_is_running(camera_id) or (
                    _fresh(camera_id) and _snapshots[camera_id]["sequence"] != sequence
                ), timeout=1.0,
            )
            if not available or not _fresh(camera_id):
                continue
            sequence = _snapshots[camera_id]["sequence"]
            jpeg = _jpegs[camera_id]
        yield send_frame(jpeg)


def stop_detection():
    _running.clear()
    with _condition:
        _condition.notify_all()
    if _thread is not None:
        _thread.join(timeout=10)
    with _condition:
        _snapshots.clear()
        _jpegs.clear()
