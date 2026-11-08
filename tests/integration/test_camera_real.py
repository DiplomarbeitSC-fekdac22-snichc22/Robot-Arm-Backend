import numpy as np

from app.camera import read_camera


def test_real_camera_returns_an_image():
    frame = read_camera()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.size > 0


def test_real_camera_returns_color_image():
    frame = read_camera()

    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_real_camera_image_is_not_completely_black():
    frame = read_camera()

    assert frame.mean() > 1


def test_configured_second_camera_has_frame_and_detections():
    import pytest
    from app.config import CAMERA_CONFIGS
    from app.detection import get_detection_snapshot

    if not CAMERA_CONFIGS["right"]["enabled"]:
        pytest.skip("Second camera disabled for single-camera deployment")
    snapshot = get_detection_snapshot(timeout=30, camera_id="right")
    assert snapshot is not None, "Second camera has no fresh detections"
    assert snapshot["camera_index"] == CAMERA_CONFIGS["right"]["index"]
    assert snapshot["frame"].size > 0
    assert snapshot["timestamp_source"] == "sensor"
