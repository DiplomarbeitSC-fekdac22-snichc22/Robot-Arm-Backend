import threading
import time
from datetime import datetime, timezone

import cv2

from .config import CAMERA_CONFIGS, MAX_FRAME_AGE_SECONDS


_condition = threading.Condition()
_cameras = {
    side: {**config, "camera": None, "running": threading.Event(),
           "thread": None, "packet": None, "jpeg": None, "sequence": 0,
           "error": "Camera has not started"}
    for side, config in CAMERA_CONFIGS.items()
}


def make_jpeg(frame):
    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buffer.tobytes() if success else None


def send_frame(jpeg):
    return b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"


def _release_camera(state):
    camera = state["camera"]
    if camera is None:
        return
    try:
        camera.stop()
    except Exception as error:
        print("Camera stop warning:", error)
    try:
        camera.close()
    except Exception as error:
        print("Camera close warning:", error)
    state["camera"] = None


def _capture_loop(side):
    state = _cameras[side]
    try:
        while state["running"].is_set():
            request = state["camera"].capture_request()
            try:
                frame = request.make_array("main")
                metadata = request.get_metadata()
            finally:
                request.release()
            captured_at = time.monotonic()
            sensor_timestamp = metadata.get("SensorTimestamp")
            if isinstance(sensor_timestamp, (int, float)) and sensor_timestamp > 0:
                delay = time.clock_gettime(time.CLOCK_BOOTTIME) - sensor_timestamp / 1e9
                if delay < 0 or delay > MAX_FRAME_AGE_SECONDS:
                    raise RuntimeError("Invalid or stale sensor timestamp")
                captured_at -= delay
            jpeg = make_jpeg(frame)
            if jpeg is None:
                raise RuntimeError("Could not encode camera frame")
            with _condition:
                state["sequence"] += 1
                state["packet"] = {
                    "camera_id": side, "camera_index": state["index"],
                    "rotation_degrees": 180 if state["rotate_180"] else 0,
                    "focus_locked": state["lens_position"] is not None,
                    "lens_position": metadata.get("LensPosition"),
                    "timestamp_source": "sensor" if isinstance(sensor_timestamp, (int, float))
                                        and sensor_timestamp > 0 else "capture_completion",
                    "captured_at": captured_at,
                    "timestamp": datetime.fromtimestamp(
                        time.time() - (time.monotonic() - captured_at), timezone.utc,
                    ).isoformat(),
                    "sequence": state["sequence"], "frame": frame,
                }
                state["jpeg"] = jpeg
                _condition.notify_all()
    except Exception as error:
        state["error"] = str(error)
        print(f"Camera {side} capture error:", error)
    finally:
        state["running"].clear()
        with _condition:
            _condition.notify_all()
        _release_camera(state)


def start_cameras():
    used_indices = set()
    for side, state in _cameras.items():
        if state["running"].is_set():
            used_indices.add(state["index"])
            continue
        if not state["enabled"]:
            state["error"] = "Camera disabled in configuration"
            continue
        if state["index"] in used_indices:
            state["error"] = "Camera index already assigned to another camera"
            continue
        used_indices.add(state["index"])
        state["packet"] = state["jpeg"] = None
        try:
            from libcamera import Transform, controls
            from picamera2 import Picamera2

            state["camera"] = Picamera2(camera_num=state["index"])
            camera = state["camera"]
            camera.configure(camera.create_video_configuration(
                main={"size": (state["width"], state["height"]), "format": "RGB888"},
                transform=Transform(hflip=state["rotate_180"], vflip=state["rotate_180"]),
            ))
            if state["lens_position"] is not None:
                camera.set_controls({"AfMode": controls.AfModeEnum.Manual,
                                     "LensPosition": state["lens_position"]})
            else:
                try:
                    camera.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                except Exception as error:
                    print(f"Camera {side} autofocus warning:", error)
            camera.start()
            state["error"] = None
            state["running"].set()
            state["thread"] = threading.Thread(target=_capture_loop, args=(side,),
                                                 name=f"camera-{side}", daemon=True)
            state["thread"].start()
        except Exception as error:
            state["error"] = str(error)
            state["running"].clear()
            _release_camera(state)


def camera_is_running(camera_id="left"):
    state = _cameras[camera_id]
    packet = state["packet"]
    return state["running"].is_set() and packet is not None and (
        time.monotonic() - packet["captured_at"] <= MAX_FRAME_AGE_SECONDS
    )


def camera_status():
    return {
        side: {**CAMERA_CONFIGS[side], "available": camera_is_running(side),
               "error": state["error"] if not state["running"].is_set()
               else (None if camera_is_running(side) else "Camera frames are stale")}
        for side, state in _cameras.items()
    }


def wait_for_frame_packet(after_sequence=-1, timeout=1.0, camera_id="left"):
    state = _cameras[camera_id]
    with _condition:
        available = _condition.wait_for(
            lambda: not state["running"].is_set() or (
                state["packet"] is not None and state["sequence"] != after_sequence
            ), timeout=timeout,
        )
        if not available or not camera_is_running(camera_id) or state["packet"] is None:
            return None
        return dict(state["packet"])


def wait_for_frame(after_sequence=-1, timeout=1.0, camera_id="left"):
    packet = wait_for_frame_packet(after_sequence, timeout, camera_id)
    return (packet["sequence"], packet["frame"].copy()) if packet else None


def wait_for_raw_jpeg(after_sequence=-1, timeout=1.0, camera_id="left"):
    state = _cameras[camera_id]
    with _condition:
        packet = wait_for_frame_packet(after_sequence, timeout, camera_id)
        return (state["sequence"], state["jpeg"]) if packet else None


def read_camera(timeout=2.0, camera_id="left"):
    packet = wait_for_frame_packet(timeout=timeout, camera_id=camera_id)
    if packet is None:
        raise RuntimeError("No camera frame is available")
    return packet["frame"].copy()


def raw_stream(camera_id="left"):
    sequence = -1
    while camera_is_running(camera_id):
        item = wait_for_raw_jpeg(sequence, camera_id=camera_id)
        if item is not None:
            sequence, jpeg = item
            yield send_frame(jpeg)


def stop_camera():
    for state in _cameras.values():
        state["running"].clear()
    with _condition:
        _condition.notify_all()
    for state in _cameras.values():
        if state["camera"] is not None:
            try:
                state["camera"].stop()
            except Exception as error:
                print("Camera stop warning:", error)
        if state["thread"] is not None:
            state["thread"].join(timeout=3)
        if state["thread"] is None or not state["thread"].is_alive():
            _release_camera(state)
