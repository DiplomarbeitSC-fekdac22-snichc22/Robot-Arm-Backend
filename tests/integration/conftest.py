import os

import pytest

from app.camera import camera_is_running, start_cameras, stop_camera
from app.detection import detection_status, get_detection_snapshot, start_detection, stop_detection


@pytest.fixture(scope="session", autouse=True)
def real_hardware():
    if os.getenv("RUN_HARDWARE_TESTS") != "1":
        pytest.skip("Run on the Pi with RUN_HARDWARE_TESTS=1 and the backend stopped")
    start_cameras()
    start_detection()
    try:
        assert get_detection_snapshot(timeout=30) is not None, detection_status()
        assert camera_is_running(), "Left camera did not start"
        yield
    finally:
        stop_detection()
        stop_camera()
