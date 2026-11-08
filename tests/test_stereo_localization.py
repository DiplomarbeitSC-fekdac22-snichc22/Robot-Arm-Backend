import json
import time

import cv2
import numpy as np
import pytest

from app.localization import (LocalizationError, _triangulate, get_calibration_status,
                              load_stereo_calibration, localize_stereo)
from calibration.calibrate_intrinsics import calibrate
from calibration import calibrate_intrinsics as intrinsic_workflow
from calibration import stereo_auto_setup as setup


@pytest.fixture
def calibration():
    matrix = [[500., 0., 640.], [0., 500., 360.], [0., 0., 1.]]
    cameras = {}
    for index, side in enumerate(("left", "right")):
        cameras[f"{side}_camera"] = {
            "camera_matrix": matrix, "distortion_coefficients": [0.] * 5,
            "rotation_world_to_camera": np.eye(3).tolist(),
            "translation_world_to_camera_mm": [-80. - index * 100., -50., 500.],
            "image_size": {"width": 1280, "height": 720}, "camera_index": index,
            "rotation_degrees": 0, "lens_position": 1.5,
        }
    return {"intrinsics_calibrated": True, "shelf_layout_physically_validated": True,
            "coordinate_system": {"name": "robot_base", "units": "mm"},
            "workspace": {"x_min_mm": -100, "x_max_mm": 600, "y_min_mm": 0,
                          "y_max_mm": 500, "z_min_mm": -250, "z_max_mm": 250}, **cameras}


def pixel(point, pose):
    return cv2.projectPoints(np.asarray(point, float).reshape(1, 3),
                             cv2.Rodrigues(np.asarray(pose["rotation_world_to_camera"]))[0],
                             np.asarray(pose["translation_world_to_camera_mm"]),
                             np.asarray(pose["camera_matrix"]),
                             np.asarray(pose["distortion_coefficients"]))[0].ravel()


def snapshots(calibration, points=((120., 150., 0.),)):
    result = []
    captured_at = time.monotonic()
    for side in ("left", "right"):
        pose = calibration[f"{side}_camera"]
        detections = []
        for index, point in enumerate(points):
            x, y = pixel(point, pose)
            detections.append({"id": f"{side}-{index}", "class_name": "Ball", "confidence": 0.9,
                               "bbox": {"x1": x - 5, "x2": x + 5, "y1": y - 5, "y2": y + 5}})
        result.append({"camera_id": side, "camera_index": pose["camera_index"],
                       "rotation_degrees": pose["rotation_degrees"], "lens_position": pose["lens_position"], "focus_locked": True,
                       "frame": np.zeros((720, 1280, 3), np.uint8), "captured_at": captured_at, "timestamp_source": "sensor",
                       "detections": detections})
    return result


def test_known_robot_point_and_observation_metadata(calibration):
    objects = localize_stereo(*snapshots(calibration), calibration=calibration)
    assert len(objects) == 1
    item = objects[0]
    assert item["valid"] is True
    assert item["position"] == pytest.approx({"x_mm": 120., "y_mm": 150., "z_mm": 0.}, abs=1e-7)
    assert item["observation_ids"] == ["left-0", "right-0"]
    assert item["camera_ids"] == ["left", "right"]
    assert item["coordinate_system"] == "robot_base" and item["unit"] == "mm"
    assert item["robot_coordinates"] is None and item["grasp_valid"] is False


def test_distortion_and_rotated_translated_cameras(calibration):
    for pose in (calibration["left_camera"], calibration["right_camera"]):
        rotation = cv2.Rodrigues(np.array([0.05, -0.1, 0.02]))[0]
        center = -np.asarray(pose["translation_world_to_camera_mm"])
        pose["rotation_world_to_camera"] = rotation.tolist()
        pose["translation_world_to_camera_mm"] = (-rotation @ center).tolist()
        pose["distortion_coefficients"] = [0.1, -0.03, 0.001, -0.002, 0.005]
    result = localize_stereo(*snapshots(calibration, [(135., 120., -20.)]), calibration=calibration)
    assert result[0]["position"] == pytest.approx({"x_mm": 135., "y_mm": 120., "z_mm": -20.}, abs=1e-5)


def test_ambiguous_same_class_is_not_greedily_matched(calibration):
    objects = localize_stereo(*snapshots(calibration, [(100., 150., 0.), (140., 150., 0.)]), calibration=calibration)
    assert len(objects) == 4
    assert all(not o["valid"] and o["position"] is None and o["status"] == "ambiguous" for o in objects)


def test_same_class_distinct_epipolar_lines(calibration):
    objects = localize_stereo(*snapshots(calibration, [(100., 100., 0.), (140., 200., 0.)]), calibration=calibration)
    assert len(objects) == 2 and all(o["valid"] for o in objects)


@pytest.mark.parametrize("change", ["skew", "stale", "rotation", "index", "focus", "autofocus", "software_time", "size"])
def test_capture_configuration_and_timestamp_rejection(calibration, change):
    left, right = snapshots(calibration)
    if change == "skew":
        right["captured_at"] += 0.2
    elif change == "stale":
        left["captured_at"] -= 6
        right["captured_at"] -= 6
    elif change == "rotation":
        right["rotation_degrees"] = 180
    elif change == "index":
        right["camera_index"] = 0
    elif change == "focus":
        right["lens_position"] = 2
    elif change == "autofocus":
        right["focus_locked"] = False
    elif change == "software_time":
        right["timestamp_source"] = "capture_completion"
    else:
        right["frame"] = np.zeros((360, 640, 3), np.uint8)
    objects = localize_stereo(left, right, calibration=calibration)
    assert len(objects) == 2 and all(not o["valid"] and o["position"] is None for o in objects)


def test_single_camera_keeps_detection_without_fake_xyz(calibration):
    objects = localize_stereo(snapshots(calibration)[0], None, calibration=calibration)
    assert len(objects) == 1 and objects[0]["position"] is None and not objects[0]["valid"]


@pytest.mark.parametrize("change", ["class", "epipolar", "invalid_bbox"])
def test_invalid_matches_keep_null_coordinates(calibration, change):
    left, right = snapshots(calibration)
    if change == "class":
        right["detections"][0]["class_name"] = "Car"
    else:
        box = right["detections"][0]["bbox"]
        if change == "epipolar":
            box["y1"] += 50
            box["y2"] += 50
        else:
            box["x1"] = float("nan")
    assert all(o["position"] is None for o in localize_stereo(left, right, calibration=calibration))


def test_reprojection_error(calibration):
    left, right = snapshots(calibration)
    calibration.update(maximum_epipolar_error_px=100., maximum_reprojection_error_px=0.2)
    right["detections"][0]["bbox"]["y1"] += 4
    right["detections"][0]["bbox"]["y2"] += 4
    assert all(not o["valid"] for o in localize_stereo(left, right, calibration=calibration))


@pytest.mark.parametrize("point", [(120, 150, -600), (120, 150, 1e12), (700, 150, 0)])
def test_behind_camera_low_parallax_or_outside_workspace(calibration, point):
    lp, rp = calibration["left_camera"], calibration["right_camera"]
    with pytest.raises(LocalizationError):
        _triangulate(pixel(point, lp), pixel(point, rp), lp, rp, calibration)


def test_calibration_file_loading(calibration, tmp_path):
    path = tmp_path / "stereo.json"
    path.write_text(json.dumps(calibration))
    assert load_stereo_calibration(path)["coordinate_system"]["name"] == "robot_base"


@pytest.mark.parametrize("change", ["bootstrap", "unvalidated", "rotation", "units", "baseline", "focal", "nan", "focus", "threshold", "bounds", "singular_intrinsics"])
def test_invalid_calibration_file(calibration, tmp_path, change):
    if change == "bootstrap":
        calibration.pop("intrinsics_calibrated")
    elif change == "unvalidated":
        calibration["shelf_layout_physically_validated"] = False
    elif change == "rotation":
        calibration["left_camera"]["rotation_world_to_camera"][0][0] = -1
    elif change == "units":
        calibration["coordinate_system"]["units"] = "cm"
    elif change == "baseline":
        calibration["right_camera"]["translation_world_to_camera_mm"] = calibration["left_camera"]["translation_world_to_camera_mm"]
    elif change == "focal":
        calibration["left_camera"]["camera_matrix"][0][0] = -1
    elif change == "nan":
        calibration["left_camera"]["camera_matrix"][0][0] = float("nan")
    elif change == "focus":
        calibration["left_camera"]["lens_position"] = None
    elif change == "singular_intrinsics":
        calibration["left_camera"]["camera_matrix"][0][1] = 500
        calibration["left_camera"]["camera_matrix"][1][0] = 500
    elif change == "threshold":
        calibration["maximum_epipolar_error_px"] = float("nan")
    else:
        calibration["workspace"]["x_min_mm"] = 700
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(calibration))
    with pytest.raises(LocalizationError):
        load_stereo_calibration(path)


def test_missing_calibration_status(monkeypatch, tmp_path):
    monkeypatch.setenv("STEREO_CALIBRATION_PATH", str(tmp_path / "missing.json"))
    status = get_calibration_status()
    assert status["ready"] is False and status["workflow"] and status["error"]


def test_intrinsic_workflow_rejects_insufficient_views():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((6, 8), 25., 18., dictionary, np.arange(100, 124, dtype=np.int32))
    with pytest.raises(ValueError, match="12 sharp board views"):
        calibrate([np.zeros((720, 1280, 3), np.uint8)], board)


def test_intrinsic_workflow_with_rendered_board_views():
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
    board = cv2.aruco.CharucoBoard((6, 8), 25., 18., dictionary, np.arange(100, 124, dtype=np.int32))
    image = board.generateImage((600, 800))
    matrix = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
    corners = np.array([[0, 0, 0], [150, 0, 0], [150, 200, 0], [0, 200, 0]], float)
    source = np.array([[0, 0], [599, 0], [599, 799], [0, 799]], np.float32)
    images = []
    for index in range(16):
        rotation = np.array([-.35 + .23 * (index % 4), -.3 + .2 * (index // 4), .04 * (index % 3)])
        translation = np.array([-75. + (index % 3 - 1) * 20, -100. + (index % 2) * 20, 450. + (index % 4) * 40])
        target = cv2.projectPoints(corners, rotation, translation, matrix, None)[0].reshape(4, 2).astype(np.float32)
        images.append(cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, target), (1280, 720), borderValue=255))
    result = calibrate(images, board)
    assert result["intrinsics_calibrated"] and result["views"] == 16
    assert result["camera_matrix"][0][0] == pytest.approx(800, rel=.01)
    assert result["camera_matrix"][1][1] == pytest.approx(800, rel=.01)


def test_extrinsic_fit_reuses_real_intrinsics(monkeypatch):
    center = np.array([450., 250., -160.])
    forward = np.array([130., 260., 0.]) - center
    forward /= np.linalg.norm(forward)
    right = np.cross([0., 1., 0.], forward)
    right /= np.linalg.norm(right)
    rotation = np.array([right, np.cross(forward, right), forward])
    translation = -rotation @ center
    matrix = np.array([[800., 0., 640.], [0., 800., 360.], [0., 0., 1.]])
    distortion = np.array([.03, -.01, .001, -.002, 0.])
    centers = {20: [130., 140., -190.], 21: [130., 140., 190.],
               22: [130., 372., -190.], 23: [130., 372., 190.]}
    detected = {i: cv2.projectPoints(setup.marker_world_corners(c, 20.), cv2.Rodrigues(rotation)[0],
                                    translation, matrix, distortion)[0].reshape(4, 2) for i, c in centers.items()}
    layout = {"aruco": {"shelf_marker_size_mm": 20.}, "markers": {
        str(i): {"center_mm": c, "orientation": "upright", "facing": "-x"} for i, c in centers.items()}}
    intrinsics = {"image_size": {"width": 1280, "height": 720}, "camera_matrix": matrix.tolist(),
                  "distortion_coefficients": distortion.tolist()}
    monkeypatch.setattr(setup, "detect_shelf_markers", lambda frame: (detected, [], None))
    result = setup.estimate_camera_pose(np.zeros((720, 1280, 3), np.uint8), layout, -1, intrinsics)
    assert result.position_robot_mm.ravel() == pytest.approx(center, abs=1e-5)
    assert result.rotation_world_to_camera == pytest.approx(rotation, abs=1e-7)
    assert result.distortion.ravel() == pytest.approx(distortion)


def test_existing_calibration_and_capture_files_are_preserved(tmp_path):
    path = tmp_path / "existing.json"
    path.write_text("original calibration")
    with pytest.raises(FileExistsError):
        setup.save_json(path, {"replacement": True})
    with pytest.raises(FileExistsError):
        setup.save_image(path, np.zeros((10, 10, 3), np.uint8))
    assert path.read_text() == "original calibration"


@pytest.mark.parametrize("change", ["focus", "rms", "board"])
def test_invalid_intrinsic_inputs_fail_before_capture(monkeypatch, tmp_path, change):
    board = {"dictionary": "DICT_4X4_1000", "squares": [6, 8], "square_mm": 25., "marker_mm": 18.}
    if change == "board":
        board["square_mm"] = float("nan")
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps(board))
    arguments = ["calibrate_intrinsics.py", "--board", str(board_path), "--images", str(tmp_path / "images"),
                 "--output", str(tmp_path / "intrinsics.json"), "--camera-index", "0", "--capture", "20",
                 "--lens-position", "nan" if change == "focus" else "1.5"]
    if change == "rms":
        arguments += ["--maximum-rms-px", "nan"]
    monkeypatch.setattr("sys.argv", arguments)
    monkeypatch.setattr(intrinsic_workflow, "capture", lambda args: pytest.fail("Invalid inputs must not open cameras"))
    with pytest.raises(ValueError):
        intrinsic_workflow.main()
    assert not (tmp_path / "images").exists() and not (tmp_path / "intrinsics.json").exists()


@pytest.mark.parametrize("contents", ["null", "[]", "true", '{"left_camera":null}', "not JSON"])
def test_malformed_calibration_is_unavailable(tmp_path, contents):
    path = tmp_path / "broken.json"
    path.write_text(contents)
    with pytest.raises(LocalizationError):
        load_stereo_calibration(path)


def test_status_compares_runtime_configuration(calibration, monkeypatch, tmp_path):
    from app import config

    path = tmp_path / "stereo.json"
    path.write_text(json.dumps(calibration))
    monkeypatch.setenv("STEREO_CALIBRATION_PATH", str(path))
    cameras = {side: {"index": index, "enabled": True, "width": 1280, "height": 720,
                      "rotate_180": False, "lens_position": 1.5} for index, side in enumerate(("left", "right"))}
    monkeypatch.setattr(config, "CAMERA_CONFIGS", cameras)
    assert get_calibration_status()["ready"] is True
    cameras["right"]["lens_position"] = None
    status = get_calibration_status()
    assert status["ready"] is False and "Camera right configuration differs" in status["error"]


def test_api_triangulates_using_loaded_calibration(calibration, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import config, main

    path = tmp_path / "stereo.json"
    path.write_text(json.dumps(calibration))
    monkeypatch.setenv("STEREO_CALIBRATION_PATH", str(path))
    cameras = {side: {"index": index, "enabled": True, "width": 1280, "height": 720,
                      "rotate_180": False, "lens_position": 1.5}
               for index, side in enumerate(("left", "right"))}
    monkeypatch.setattr(config, "CAMERA_CONFIGS", cameras)
    observations = snapshots(calibration)
    for snapshot in observations:
        snapshot.update(timestamp="2026-11-08T18:00:00+00:00", sequence=1)
    monkeypatch.setattr(main, "get_detection_snapshots", lambda timeout: {
        snapshot["camera_id"]: snapshot for snapshot in observations})
    monkeypatch.setattr(main, "_image_cache", {})
    for name in ("CROPS_DIR", "FRAMES_DIR"):
        directory = tmp_path / name
        directory.mkdir()
        monkeypatch.setattr(main, name, directory)
    response = TestClient(main.app).get("/objects")
    assert response.status_code == 200
    data = response.json()
    assert data["calibration"]["ready"] is True
    assert len(data["objects"]) == 1
    item = data["objects"][0]
    assert item["position"] == pytest.approx({"x_mm": 120., "y_mm": 150., "z_mm": 0.}, abs=1e-7)
    assert item["valid"] is True and item["camera_ids"] == ["left", "right"]
    assert item["robot_coordinates"] is None and item["grasp_valid"] is False
