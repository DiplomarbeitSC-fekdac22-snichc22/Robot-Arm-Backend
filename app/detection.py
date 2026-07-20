import copy
import threading
import time
import uuid
from datetime import datetime

from ultralytics import YOLO

from .camera import (
    camera_is_running,
    make_jpeg,
    send_frame,
    wait_for_frame,
)
from .config import MODEL_PATH, YOLO_CONF


print("Loading YOLO model...")
model = YOLO(MODEL_PATH)
print("YOLO model loaded.")
print("Model classes:", model.names)


# Prevent accidental simultaneous use of the same model.
_model_lock = threading.Lock()

_detection_condition = threading.Condition()
_detection_running = threading.Event()

_latest_detection_sequence = -1
_latest_annotated_jpeg = None
_latest_detections = []
_latest_detection_frame = None
_latest_detection_timestamp = None


def run_detection(frame):
    with _model_lock:
        return model.predict(
            source=frame,
            imgsz=320,
            conf=YOLO_CONF,
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


def _detection_loop():
    global _latest_detection_sequence
    global _latest_annotated_jpeg
    global _latest_detections
    global _latest_detection_frame
    global _latest_detection_timestamp

    camera_sequence = -1

    try:
        while _detection_running.is_set():
            item = wait_for_frame(camera_sequence)

            if item is None:
                if not camera_is_running():
                    break

                continue

            camera_sequence, frame = item

            try:
                result = run_detection(frame)
                detections = parse_detections(result)
                annotated_frame = result.plot()
                annotated_jpeg = make_jpeg(annotated_frame)

                if annotated_jpeg is None:
                    continue

                timestamp = datetime.now().isoformat()

                with _detection_condition:
                    _latest_detection_sequence = camera_sequence
                    _latest_annotated_jpeg = annotated_jpeg
                    _latest_detections = detections
                    _latest_detection_frame = frame
                    _latest_detection_timestamp = timestamp
                    _detection_condition.notify_all()

            except Exception as error:
                print("Detection error:", error)
                time.sleep(0.1)

    finally:
        _detection_running.clear()

        with _detection_condition:
            _detection_condition.notify_all()


def detection_stream():
    sequence = -1

    while _detection_running.is_set():
        with _detection_condition:
            available = _detection_condition.wait_for(
                lambda: (
                    not _detection_running.is_set()
                    or (
                        _latest_annotated_jpeg is not None
                        and _latest_detection_sequence != sequence
                    )
                ),
                timeout=1.0,
            )

            if not _detection_running.is_set():
                break

            if not available or _latest_annotated_jpeg is None:
                continue

            sequence = _latest_detection_sequence
            jpeg = _latest_annotated_jpeg

        yield send_frame(jpeg)


def get_detection_snapshot(timeout=10.0):
    """Return detections and their corresponding camera frame."""
    with _detection_condition:
        available = _detection_condition.wait_for(
            lambda: (
                _latest_detection_frame is not None
                or not _detection_running.is_set()
            ),
            timeout=timeout,
        )

        if not available or _latest_detection_frame is None:
            return None

        return {
            "timestamp": _latest_detection_timestamp,
            "detections": copy.deepcopy(_latest_detections),
            "frame": _latest_detection_frame.copy(),
        }


def stop_detection():
    if not _detection_running.is_set():
        return

    _detection_running.clear()

    with _detection_condition:
        _detection_condition.notify_all()

    _detection_thread.join(timeout=3)


_detection_running.set()

_detection_thread = threading.Thread(
    target=_detection_loop,
    name="yolo-detection",
    daemon=True,
)
_detection_thread.start()