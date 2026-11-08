import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from app import camera, detection


def wait_until(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@pytest.fixture
def cameras(monkeypatch):
    opened = {}

    class FakeCamera:
        def __init__(self, camera_num):
            self.index = camera_num
            self.stopped = threading.Event()
            self.closed = False
            self.fail_capture = False
            opened[camera_num] = self

        def create_video_configuration(self, **kwargs):
            return kwargs

        def configure(self, config):
            self.config = config

        def set_controls(self, values):
            self.controls = values

        def start(self):
            pass

        def capture_request(self):
            self.stopped.wait(0.01)
            if self.stopped.is_set() or self.fail_capture:
                raise RuntimeError("Camera disconnected")
            return SimpleNamespace(
                make_array=lambda name: np.full((24, 32, 3), 10 + self.index, dtype=np.uint8),
                get_metadata=lambda: {"LensPosition": 1.0}, release=lambda: None,
            )

        def stop(self):
            self.stopped.set()

        def close(self):
            self.closed = True

    configs = {side: {"enabled": True, "index": index, "width": 32, "height": 24,
                      "rotate_180": False, "lens_position": 1.0}
               for side, index in (("left", 2), ("right", 5))}
    states = {side: {**config, "camera": None, "running": threading.Event(), "thread": None,
                     "packet": None, "jpeg": None, "sequence": 0, "error": "Not started"}
              for side, config in configs.items()}
    monkeypatch.setattr(camera, "_cameras", states)
    monkeypatch.setattr(camera, "CAMERA_CONFIGS", configs)
    monkeypatch.setitem(sys.modules, "picamera2", SimpleNamespace(Picamera2=FakeCamera))
    monkeypatch.setitem(sys.modules, "libcamera", SimpleNamespace(
        Transform=lambda **kwargs: kwargs,
        controls=SimpleNamespace(AfModeEnum=SimpleNamespace(Manual=0, Continuous=2)),
    ))
    yield opened, configs
    detection.stop_detection()
    camera.stop_camera()


def test_independent_selection_capture_and_shutdown(cameras):
    opened, _ = cameras
    camera.start_cameras()
    wait_until(lambda: all(camera.camera_is_running(side) for side in ("left", "right")))
    assert set(opened) == {2, 5}
    assert camera.read_camera()[0, 0, 0] == 12
    assert camera.read_camera(camera_id="right")[0, 0, 0] == 15
    packet = camera.wait_for_frame_packet(camera_id="right")
    assert packet["camera_index"] == 5
    assert packet["lens_position"] == 1
    assert packet["captured_at"] <= time.monotonic()
    assert packet["rotation_degrees"] == 0
    assert next(camera.raw_stream("right")).startswith(b"--frame\r\nContent-Type: image/jpeg")
    camera.stop_camera()
    assert all(cam.closed for cam in opened.values())
    assert not camera.camera_is_running()
    with pytest.raises(RuntimeError):
        camera.read_camera()


def test_single_camera_keeps_legacy_defaults(cameras):
    opened, _ = cameras
    camera._cameras["right"]["enabled"] = False
    camera.start_cameras()
    wait_until(camera.camera_is_running)
    assert set(opened) == {2}
    sequence, frame = camera.wait_for_frame()
    assert sequence > 0 and frame.shape == (24, 32, 3)
    assert not camera.camera_is_running("right")


def test_second_initialization_failure_does_not_stop_first(cameras, monkeypatch):
    opened, _ = cameras
    factory = sys.modules["picamera2"].Picamera2

    def open_camera(camera_num):
        if camera_num == 5:
            raise RuntimeError("Missing second camera")
        return factory(camera_num)

    monkeypatch.setattr(sys.modules["picamera2"], "Picamera2", open_camera)
    camera.start_cameras()
    wait_until(camera.camera_is_running)
    assert not camera.camera_is_running("right")
    assert "Missing second camera" in camera.camera_status()["right"]["error"]
    assert not opened[2].closed


def test_duplicate_selection_rejected(cameras):
    opened, _ = cameras
    camera._cameras["right"]["index"] = 2
    camera.start_cameras()
    wait_until(camera.camera_is_running)
    assert set(opened) == {2}
    assert "already assigned" in camera.camera_status()["right"]["error"]


def test_disconnected_camera_does_not_return_old_frame(cameras):
    opened, _ = cameras
    camera.start_cameras()
    wait_until(lambda: camera.camera_is_running("right"))
    opened[5].fail_capture = True
    wait_until(lambda: not camera.camera_is_running("right"))
    assert camera.wait_for_frame_packet(camera_id="right") is None
    wait_until(lambda: opened[5].closed)
    assert camera.camera_is_running("left")


def test_no_first_frame_or_stale_frame_is_unavailable(cameras):
    camera._cameras["left"]["running"].set()
    assert not camera.camera_is_running()
    camera._cameras["left"]["packet"] = {"captured_at": time.monotonic() - 20}
    assert not camera.camera_is_running()


def test_both_streams_use_one_model(cameras, monkeypatch):
    model = MagicMock()
    model.predict.side_effect = lambda **kwargs: [SimpleNamespace(boxes=[], plot=lambda: kwargs["source"])]
    monkeypatch.setattr(detection, "model", model)
    camera.start_cameras()
    wait_until(lambda: all(camera.camera_is_running(side) for side in ("left", "right")))
    detection.start_detection()
    wait_until(lambda: len(detection.get_detection_snapshots()) == 2)
    snapshots = detection.get_detection_snapshots()
    assert snapshots["left"]["camera_index"] == 2
    assert snapshots["right"]["camera_index"] == 5
    assert snapshots["left"]["detections"] == []
    assert model.predict.call_count >= 2
    detection.stop_detection()
    assert detection.get_detection_snapshot(timeout=0) is None


def test_model_failure_leaves_cameras_available(cameras, monkeypatch):
    camera.start_cameras()
    wait_until(camera.camera_is_running)
    monkeypatch.setattr(detection, "model", None)
    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(
        YOLO=MagicMock(side_effect=RuntimeError("Missing model file"))))
    detection.start_detection()
    assert not detection.detection_status()["available"]
    assert "Missing model file" in detection.detection_status()["error"]
    assert camera.camera_is_running()


def test_model_inference_is_serialized(monkeypatch):
    active = 0
    peak = 0

    def predict(**kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        time.sleep(0.02)
        active -= 1
        return [object()]

    monkeypatch.setattr(detection, "model", SimpleNamespace(predict=predict))
    threads = [threading.Thread(target=detection.run_detection, args=(np.zeros((2, 2, 3)),)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)
    assert peak == 1


def test_per_camera_environment_configuration(monkeypatch):
    import runpy
    from pathlib import Path

    for name, value in {
        "CAMERA_LEFT_INDEX": "4", "CAMERA_RIGHT_INDEX": "7",
        "CAMERA_LEFT_WIDTH": "640", "CAMERA_RIGHT_HEIGHT": "480",
        "CAMERA_LEFT_ROTATE_180": "true", "CAMERA_RIGHT_LENS_POSITION": "1.25",
        "CAMERA_RIGHT_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    config = runpy.run_path(str(Path(camera.__file__).with_name("config.py")))
    assert config["CAMERA_CONFIGS"]["left"]["index"] == 4
    assert config["CAMERA_CONFIGS"]["left"]["width"] == 640
    assert config["CAMERA_CONFIGS"]["left"]["rotate_180"] is True
    assert config["CAMERA_CONFIGS"]["right"]["index"] == 7
    assert config["CAMERA_CONFIGS"]["right"]["height"] == 480
    assert config["CAMERA_CONFIGS"]["right"]["lens_position"] == 1.25
    assert config["CAMERA_CONFIGS"]["right"]["enabled"] is False


@pytest.mark.parametrize("name,value", [
    ("MAX_FRAME_AGE_SECONDS", "nan"), ("STEREO_MAX_TIME_SKEW_SECONDS", "inf"),
    ("STEREO_MAX_TIME_SKEW_SECONDS", "0"), ("CAMERA_LEFT_INDEX", "-1"),
    ("CAMERA_RIGHT_LENS_POSITION", "nan"),
])
def test_invalid_camera_and_time_configuration_is_rejected(monkeypatch, name, value):
    import runpy
    from pathlib import Path

    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError):
        runpy.run_path(str(Path(camera.__file__).with_name("config.py")))
