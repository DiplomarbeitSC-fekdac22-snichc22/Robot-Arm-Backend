import threading
import time

import cv2
from libcamera import Transform, controls
from picamera2 import Picamera2

from .config import CAMERA_HEIGHT, CAMERA_WIDTH, ROTATE_CAMERA_180


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

try:
    camera.set_controls({
        "AfMode": controls.AfModeEnum.Continuous,
    })
except Exception as error:
    print("Autofocus warning:", error)

time.sleep(1)
print("Pi camera started.")


# --------------------------------------------------
# Shared latest-frame state
# --------------------------------------------------

_frame_condition = threading.Condition()
_capture_running = threading.Event()

_latest_frame = None
_latest_raw_jpeg = None
_frame_sequence = 0


def make_jpeg(frame):
    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 80],
    )

    if not success:
        return None

    return buffer.tobytes()


def send_frame(jpeg):
    return (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n"
        + jpeg
        + b"\r\n"
    )


def _capture_loop():
    global _latest_frame
    global _latest_raw_jpeg
    global _frame_sequence

    try:
        while _capture_running.is_set():
            frame = camera.capture_array()
            jpeg = make_jpeg(frame)

            if jpeg is None:
                continue

            with _frame_condition:
                _latest_frame = frame
                _latest_raw_jpeg = jpeg
                _frame_sequence += 1
                _frame_condition.notify_all()

    except Exception as error:
        print("Camera capture error:", error)

    finally:
        _capture_running.clear()

        with _frame_condition:
            _frame_condition.notify_all()


def camera_is_running():
    return _capture_running.is_set()


def wait_for_frame(after_sequence=-1, timeout=1.0):
    """Used by the detection worker.

    Returns only the newest frame. If inference is slower than the camera,
    intermediate frames are automatically skipped.
    """
    with _frame_condition:
        available = _frame_condition.wait_for(
            lambda: (
                not _capture_running.is_set()
                or (
                    _latest_frame is not None
                    and _frame_sequence != after_sequence
                )
            ),
            timeout=timeout,
        )

        if (
            not available
            or not _capture_running.is_set()
            or _latest_frame is None
            or _frame_sequence == after_sequence
        ):
            return None

        return _frame_sequence, _latest_frame.copy()


def wait_for_raw_jpeg(after_sequence=-1, timeout=1.0):
    with _frame_condition:
        available = _frame_condition.wait_for(
            lambda: (
                not _capture_running.is_set()
                or (
                    _latest_raw_jpeg is not None
                    and _frame_sequence != after_sequence
                )
            ),
            timeout=timeout,
        )

        if (
            not available
            or not _capture_running.is_set()
            or _latest_raw_jpeg is None
            or _frame_sequence == after_sequence
        ):
            return None

        # bytes are immutable, so no copy is necessary.
        return _frame_sequence, _latest_raw_jpeg


def read_camera(timeout=2.0):
    """Return a copy of the newest captured frame."""
    with _frame_condition:
        available = _frame_condition.wait_for(
            lambda: (
                _latest_frame is not None
                or not _capture_running.is_set()
            ),
            timeout=timeout,
        )

        if not available or _latest_frame is None:
            raise RuntimeError("No camera frame is available")

        return _latest_frame.copy()


def raw_stream():
    sequence = -1

    while _capture_running.is_set():
        item = wait_for_raw_jpeg(sequence)

        if item is None:
            continue

        sequence, jpeg = item
        yield send_frame(jpeg)


def stop_camera():
    if not _capture_running.is_set():
        return

    _capture_running.clear()

    with _frame_condition:
        _frame_condition.notify_all()

    _capture_thread.join(timeout=2)
    camera.stop()


_capture_running.set()

_capture_thread = threading.Thread(
    target=_capture_loop,
    name="camera-capture",
    daemon=True,
)
_capture_thread.start()