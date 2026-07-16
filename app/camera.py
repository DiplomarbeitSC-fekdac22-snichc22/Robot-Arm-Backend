import threading
import time

import cv2
from libcamera import Transform, controls
from picamera2 import Picamera2

from .config import CAMERA_HEIGHT, CAMERA_WIDTH, ROTATE_CAMERA_180

# --------------------------------------------------
# Camera setup
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
# Frame helpers
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
# Raw (no detection) MJPEG stream
# --------------------------------------------------

def raw_stream():
    while True:
        frame = read_camera()

        jpeg = make_jpeg(frame)

        if jpeg is None:
            continue

        yield send_frame(jpeg)


def stop_camera():
    camera.stop()