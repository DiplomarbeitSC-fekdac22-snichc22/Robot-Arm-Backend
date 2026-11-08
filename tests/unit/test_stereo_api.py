import time
from unittest.mock import Mock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import kinematics_bridge, main


@pytest.fixture
def api(monkeypatch, tmp_path):
    monkeypatch.setenv("STEREO_CALIBRATION_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setattr(main, "_mode", "manual")
    monkeypatch.setattr(main, "_image_cache", {})
    for name in ("CROPS_DIR", "FRAMES_DIR"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(main, name, directory)
    snapshots = {}
    for index, side in enumerate(("left", "right")):
        detection = {"id": f"{side}-object", "class_id": 0, "class_name": "Ball", "confidence": 0.9,
                     "bbox": {"x1": 20, "y1": 10, "x2": 40, "y2": 30}, "center": {"x": 30, "y": 20}}
        snapshots[side] = {"camera_id": side, "camera_index": index, "sequence": index + 1,
                           "timestamp": f"2026-11-08T18:00:0{index}", "captured_at": time.monotonic(),
                           "frame": np.zeros((60, 80, 3), np.uint8), "detections": [detection]}
    cameras = {side: {"available": True, "enabled": True, "error": None, "index": index}
               for index, side in enumerate(snapshots)}
    monkeypatch.setattr(main, "camera_status", lambda: cameras)
    monkeypatch.setattr(main, "detection_status", lambda: {
        "available": True, "error": None, "cameras": {side: True for side in snapshots}})
    monkeypatch.setattr(main, "get_detection_snapshots", lambda timeout=0: snapshots)
    monkeypatch.setattr(main, "get_detection_snapshot", lambda timeout=0, camera_id="left": snapshots.get(camera_id))
    monkeypatch.setattr(main, "camera_is_running", lambda camera_id="left": True)
    monkeypatch.setattr(main, "read_camera", lambda camera_id="left": snapshots[camera_id]["frame"])
    monkeypatch.setattr(main, "raw_stream", lambda side="left": iter([f"raw-{side}".encode()]))
    monkeypatch.setattr(main, "detection_stream", lambda side="left": iter([f"detect-{side}".encode()]))
    return TestClient(main.app), snapshots


@pytest.mark.parametrize("path", ["/objects", "/detections/objects"])
def test_missing_calibration_keeps_both_observations_with_null_xyz(api, path):
    client, _ = api
    response = client.get(path)
    assert response.status_code == 200
    data = response.json()
    assert data["calibration"]["ready"] is False
    assert data["timestamp"] == "2026-11-08T18:00:01"
    assert len(data["objects"]) == 2
    for side, item in zip(("left", "right"), data["objects"]):
        assert item["id"] == f"{side}-object"
        assert item["camera_ids"] == [side] and item["camera_id"] == side
        assert item["observation_ids"] == [f"{side}-object"]
        assert item["position"] is None and item["robot_coordinates"] is None
        assert item["valid"] is False and item["grasp_valid"] is False
        assert item["point_type"] == "bbox_center" and item["unit"] == "mm"
        assert item["localization_error"] and item["status"] == "unavailable"
        assert item["crop_url"].startswith(f"http://testserver/static/crops/{side}_0.jpg")
        assert item["frame_url"].startswith(f"http://testserver/static/frames/{side}.jpg")


def test_stereo_object_response_preserves_geometry_and_source_metadata(api, monkeypatch):
    client, snapshots = api
    merged = {**snapshots["left"]["detections"][0], "camera_ids": ["left", "right"],
              "observation_ids": ["left-object", "right-object"], "position": {"x_mm": 120, "y_mm": 150, "z_mm": -20},
              "robot_coordinates": None, "valid": True, "status": "stereo_estimate", "coordinate_system": "robot_base",
              "unit": "mm", "point_type": "bbox_center", "grasp_valid": False, "reprojection_error_px": 0.1}
    localize = Mock(return_value=[merged])
    monkeypatch.setattr(main, "localize_stereo", localize)
    response = client.get("/objects?camera_id=right")
    assert response.status_code == 200
    item = response.json()["objects"][0]
    assert item["position"] == {"x_mm": 120, "y_mm": 150, "z_mm": -20}
    assert item["coordinate_system"] == "robot_base" and item["unit"] == "mm"
    assert item["camera_ids"] == ["left", "right"] and item["observation_ids"] == ["left-object", "right-object"]
    assert item["valid"] is True and item["grasp_valid"] is False and item["robot_coordinates"] is None
    assert item["crop_url"].startswith("http://testserver/static/crops/left_0.jpg")
    assert localize.call_args.args == (snapshots["left"], snapshots["right"])
    assert "max_time_skew_s" in localize.call_args.kwargs


def test_object_camera_filter_returns_only_selected_observations(api):
    client, _ = api
    objects = client.get("/objects?camera_id=right").json()["objects"]
    assert len(objects) == 1 and objects[0]["camera_ids"] == ["right"]


@pytest.mark.parametrize("path,side", [("/detections", "left"), ("/detections?camera_id=right", "right")])
def test_detection_selection_preserves_default_left_camera(api, path, side):
    client, _ = api
    response = client.get(path)
    assert response.status_code == 200
    assert response.json()["camera_id"] == side
    assert response.json()["detections"][0]["id"] == f"{side}-object"


@pytest.mark.parametrize("path,content", [
    ("/video", b"raw-left"), ("/video/right", b"raw-right"), ("/video?camera_id=right", b"raw-right"),
    ("/calibration/video/right", b"raw-right"), ("/detect", b"detect-left"),
    ("/detect/right", b"detect-right"), ("/detect?camera_id=right", b"detect-right"),
])
def test_raw_and_detection_stream_camera_selection(api, path, content):
    client, _ = api
    response = client.get(path)
    assert response.status_code == 200
    assert response.content == content
    assert response.headers["content-type"].startswith("multipart/x-mixed-replace")


@pytest.mark.parametrize("path", ["/video/right", "/detect/right", "/snapshot?camera_id=right"])
def test_unavailable_camera_reports_503_without_movement(api, monkeypatch, path):
    client, _ = api
    monkeypatch.setattr(main, "camera_is_running", lambda camera_id="left": False)
    response = client.get(path)
    assert response.status_code == 503
    assert response.json()["code"] == "CAMERA_UNAVAILABLE"
    assert response.json()["movement_allowed"] is False and response.json()["movement_started"] is False


def test_unavailable_detection_reports_503(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(main, "get_detection_snapshot", lambda **kwargs: None)
    response = client.get("/detections?camera_id=right")
    assert response.status_code == 503 and response.json()["code"] == "DETECTION_UNAVAILABLE"


@pytest.mark.parametrize("path", ["/video/third", "/detect/third", "/calibration/video/third",
                                  "/snapshot?camera_id=third", "/detections?camera_id=third", "/objects?camera_id=third"])
def test_invalid_camera_ids_are_rejected(api, path):
    client, _ = api
    assert client.get(path).status_code == 422


def test_health_and_calibration_status_explain_missing_files(api):
    client, _ = api
    health = client.get("/health")
    assert health.status_code == 200
    data = health.json()
    assert data["camera_available"] is True and set(data["cameras"]) == {"left", "right"}
    assert data["detection"]["available"] is True
    calibration = client.get("/calibration/status").json()
    assert calibration["ready"] is False and calibration["grasp_valid"] is False
    assert calibration["error"] and calibration["instruction"] and calibration["workflow"]
    assert data["calibration"] == calibration


@pytest.mark.parametrize("mode", ["manual", "automatic"])
def test_mode_changes_never_enable_or_spawn_movement(api, monkeypatch, mode):
    client, _ = api
    pick = Mock(side_effect=AssertionError("Must not call robot bridge"))
    scan = Mock(side_effect=AssertionError("Must not call robot bridge"))
    process = Mock(side_effect=AssertionError("Must not spawn hardware process"))
    monkeypatch.setattr(kinematics_bridge, "run_hardware_pick", pick)
    monkeypatch.setattr(kinematics_bridge, "run_scan_pose", scan)
    monkeypatch.setattr(kinematics_bridge.subprocess, "run", process)
    assert client.get("/operation-mode").json()["mode"] == "manual"
    response = client.post("/operation-mode", json={"mode": mode})
    assert response.status_code == 200 and response.json()["mode"] == mode
    assert response.json()["movement_allowed"] is False and response.json()["reason"]
    target = {"object_id": "left-object", "class_name": "Ball", "x_mm": 120, "y_mm": 150, "z_mm": -20}
    for path, body in (("/kinematics/target", target), ("/kinematics/scan-pose", {})):
        response = client.post(path, json=body)
        assert response.status_code == 409 and response.json()["code"] == "MOVEMENT_NOT_PERMITTED"
        assert response.json()["movement_started"] is False and response.json()["movement_allowed"] is False
    pick.assert_not_called()
    scan.assert_not_called()
    process.assert_not_called()
    assert client.post("/operation-mode", json={"mode": "unknown"}).status_code == 422


@pytest.mark.parametrize("query,side", [("", "left"), ("?camera_id=right", "right")])
def test_snapshot_uses_selected_camera(api, monkeypatch, query, side):
    client, snapshots = api
    read = Mock(return_value=snapshots[side]["frame"])
    monkeypatch.setattr(main, "read_camera", read)
    response = client.get(f"/snapshot{query}")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")
    read.assert_called_once_with(camera_id=side)


def test_health_marks_failed_second_camera_as_degraded(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(main, "camera_status", lambda: {
        "left": {"available": True, "enabled": True, "error": None},
        "right": {"available": False, "enabled": True, "error": "Camera disconnected"},
    })
    response = client.get("/health")
    assert response.status_code == 200 and response.json()["status"] == "degraded"
    assert response.json()["camera_available"] is True
    assert response.json()["cameras"]["right"]["error"] == "Camera disconnected"


def test_backend_restart_resets_to_manual_and_releases_resources(monkeypatch):
    monkeypatch.setattr(main, "_mode", "automatic")
    hooks = {}
    for name in ("start_cameras", "start_detection", "stop_detection", "stop_camera"):
        hooks[name] = Mock()
        monkeypatch.setattr(main, name, hooks[name])
    with TestClient(main.app) as client:
        assert client.get("/operation-mode").json()["mode"] == "manual"
        hooks["start_cameras"].assert_called_once()
        hooks["start_detection"].assert_called_once()
    hooks["stop_detection"].assert_called_once()
    hooks["stop_camera"].assert_called_once()
