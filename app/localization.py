from __future__ import annotations

import json
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

POSE_PATH = (
    BASE_DIR
    / "calibration"
    / "camera_pose_centered_provisional.json"
)

FIXED_TARGET_X_MM = 230.0

Y_MIN_MM = 0.0
Y_MAX_MM = 500.0
Z_MIN_MM = -180.0
Z_MAX_MM = 180.0


class LocalizationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _load_pose() -> dict[str, Any]:
    try:
        return json.loads(
            POSE_PATH.read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as error:
        raise LocalizationError(
            f"Could not load camera pose: {POSE_PATH}"
        ) from error


def _scaled_camera_matrix(
    pose: dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    calibration_width = int(
        pose["image_size"]["width"]
    )

    calibration_height = int(
        pose["image_size"]["height"]
    )

    image_ratio = width / height
    calibration_ratio = (
        calibration_width / calibration_height
    )

    if abs(image_ratio - calibration_ratio) > 0.01:
        raise LocalizationError(
            "Camera frame has a different aspect ratio "
            "from the centered calibration"
        )

    scale_x = width / calibration_width
    scale_y = height / calibration_height

    matrix = np.asarray(
        pose["camera_matrix"],
        dtype=np.float64,
    ).copy()

    matrix[0, 0] *= scale_x
    matrix[0, 2] *= scale_x
    matrix[1, 1] *= scale_y
    matrix[1, 2] *= scale_y

    return matrix


def localize_detection(
    frame_shape: tuple[int, ...],
    detection: dict[str, Any],
) -> dict[str, float]:
    pose = _load_pose()

    height = int(frame_shape[0])
    width = int(frame_shape[1])

    try:
        bbox = detection["bbox"]

        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])

    except (KeyError, TypeError, ValueError) as error:
        raise LocalizationError(
            "Detection has an invalid bounding box"
        ) from error

    if x2 <= x1 or y2 <= y1:
        raise LocalizationError(
            "Detection bounding box is empty"
        )

    center_pixel = np.array(
        [
            (x1 + x2) / 2.0,
            (y1 + y2) / 2.0,
        ],
        dtype=np.float64,
    )

    camera_matrix = _scaled_camera_matrix(
        pose,
        width,
        height,
    )

    distortion = np.asarray(
        pose["distortion_coefficients"],
        dtype=np.float64,
    ).reshape(-1, 1)

    corrected = cv2.undistortPoints(
        center_pixel.reshape(1, 1, 2),
        camera_matrix,
        distortion,
    ).reshape(2)

    camera_ray = np.array(
        [
            corrected[0],
            corrected[1],
            1.0,
        ],
        dtype=np.float64,
    )

    rotation = np.asarray(
        pose["rotation_world_to_camera"],
        dtype=np.float64,
    )

    translation = np.asarray(
        pose["translation_world_to_camera_mm"],
        dtype=np.float64,
    ).reshape(3)

    camera_position = (
        -rotation.T @ translation
    )

    world_ray = rotation.T @ camera_ray
    world_ray /= np.linalg.norm(world_ray)

    denominator = float(world_ray[0])

    if abs(denominator) < 1e-9:
        raise LocalizationError(
            "Detection ray is parallel to X=230 mm"
        )

    distance = (
        FIXED_TARGET_X_MM
        - float(camera_position[0])
    ) / denominator

    if distance <= 0.0:
        raise LocalizationError(
            "X=230 mm intersection is behind camera"
        )

    point = (
        camera_position
        + distance * world_ray
    )

    y_mm = float(point[1])
    z_mm = float(point[2])

    if not Y_MIN_MM <= y_mm <= Y_MAX_MM:
        raise LocalizationError(
            f"Calculated Y={y_mm:.1f} mm "
            "is outside the enclosure"
        )

    if not Z_MIN_MM <= z_mm <= Z_MAX_MM:
        raise LocalizationError(
            f"Calculated Z={z_mm:.1f} mm "
            "is outside the enclosure"
        )

    return {
        "x_mm": FIXED_TARGET_X_MM,
        "y_mm": round(y_mm, 1),
        "z_mm": round(z_mm, 1),
    }


@lru_cache(maxsize=4)
def _read_stereo(path: str) -> dict[str, Any]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("intrinsics_calibrated") is not True:
            raise ValueError("Multi-view intrinsics are missing; bootstrap poses are provisional")
        if data.get("shelf_layout_physically_validated") is not True:
            raise ValueError("Measured shelf/robot alignment has not been validated")
        system = data["coordinate_system"]
        if system["name"] not in ("robot_base", "shelf") or system["units"] != "mm":
            raise ValueError("Calibration must declare shelf or robot_base coordinates in mm")
        for side in ("left", "right"):
            pose = data[f"{side}_camera"]
            pose.setdefault("image_size", data.get("image_size"))
            size = pose["image_size"]
            if min(int(size["width"]), int(size["height"])) <= 0:
                raise ValueError("Invalid calibration image size")
            if type(pose["camera_index"]) is not int or pose["camera_index"] < 0:
                raise ValueError("Calibration camera_index is missing or invalid")
            lens = float(pose["lens_position"])
            if not np.isfinite(lens) or lens < 0:
                raise ValueError("A finite locked-focus lens_position is required")
            pose["lens_position"] = lens
            if pose["rotation_degrees"] not in (0, 90, 180, 270):
                raise ValueError("Invalid calibration image rotation")
            matrix = np.asarray(pose["camera_matrix"], dtype=float)
            rotation = np.asarray(pose["rotation_world_to_camera"], dtype=float)
            translation = np.asarray(pose["translation_world_to_camera_mm"], dtype=float)
            distortion = np.asarray(pose["distortion_coefficients"], dtype=float)
            if (matrix.shape != (3, 3) or rotation.shape != (3, 3)
                    or translation.size != 3 or distortion.size not in (4, 5, 8, 12, 14)
                    or not all(np.isfinite(a).all() for a in (matrix, rotation, translation, distortion))):
                raise ValueError("Invalid camera matrices or distortion coefficients")
            if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
                    or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)
                    or not np.allclose(matrix[2], [0, 0, 1])
                    or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
                    or not np.isclose(np.linalg.det(rotation), 1, atol=1e-5)):
                raise ValueError("Invalid focal length or camera rotation")
        left, right = (_camera_arrays(data[f"{side}_camera"]) for side in ("left", "right"))
        baseline = np.linalg.norm(-left[2].T @ left[3] + right[2].T @ right[3])
        if not np.isfinite(baseline) or baseline < 1:
            raise ValueError("Stereo cameras have no usable baseline")
        if data["left_camera"]["camera_index"] == data["right_camera"]["camera_index"]:
            raise ValueError("Stereo calibration uses the same camera twice")
        for name, default in (("maximum_epipolar_error_px", 2.), ("maximum_reprojection_error_px", 2.),
                              ("minimum_ray_angle_degrees", 1.)):
            value = float(data.get(name, default))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            data[name] = value
        for axis in "xyz":
            lower, upper = (float(data["workspace"][f"{axis}_{bound}_mm"]) for bound in ("min", "max"))
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError("Measured workspace bounds must be finite and ordered")
            data["workspace"].update({f"{axis}_min_mm": lower, f"{axis}_max_mm": upper})
        return data
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError, AttributeError, cv2.error) as error:
        raise LocalizationError(f"Stereo calibration unavailable: {error}") from error


def load_stereo_calibration(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or os.getenv("STEREO_CALIBRATION_PATH", "calibration/stereo_calibration.json"))
    return _read_stereo(str(path if path.is_absolute() else BASE_DIR / path))


def get_calibration_status() -> dict[str, Any]:
    try:
        data = load_stereo_calibration()
        from .config import CAMERA_CONFIGS

        for side, config in CAMERA_CONFIGS.items():
            pose = data[f"{side}_camera"]
            size = pose["image_size"]
            if (not config["enabled"] or config["index"] != pose["camera_index"]
                    or (config["width"], config["height"]) != (size["width"], size["height"])
                    or (180 if config["rotate_180"] else 0) != pose["rotation_degrees"]
                    or config["lens_position"] is None
                    or not np.isclose(config["lens_position"], pose["lens_position"], rtol=0, atol=0.01)):
                raise LocalizationError(f"Camera {side} configuration differs from calibration; check index, resolution, rotation and locked focus")
        return {"valid": True, "ready": True, "status": "ready", "coordinate_system": data["coordinate_system"]["name"],
                "unit": "mm", "grasp_valid": False,
                "message": "Stereo bounding-box centres are estimates, not verified grasp points"}
    except LocalizationError as error:
        return {"valid": False, "ready": False, "status": "unavailable", "message": str(error),
                "error": str(error), "grasp_valid": False,
                "instruction": "Calibrate both fixed-focus lenses, then measure and validate the robot-frame marker layout.",
                "workflow": ["Run calibration/calibrate_intrinsics.py separately for each camera with your measured ChArUco board.",
                             "Set each intrinsics path in calibration/stereo_setup.toml and verify the shelf marker measurements.",
                             "Run calibration/run_stereo_setup.sh explicitly, verify independent distances, then restart the backend."]}


def _camera_arrays(pose):
    return (np.asarray(pose["camera_matrix"], dtype=float),
            np.asarray(pose["distortion_coefficients"], dtype=float),
            np.asarray(pose["rotation_world_to_camera"], dtype=float),
            np.asarray(pose["translation_world_to_camera_mm"], dtype=float).reshape(3))


def _check_snapshot(snapshot, pose):
    size = pose["image_size"]
    if tuple(snapshot["frame"].shape[:2]) != (size["height"], size["width"]):
        raise LocalizationError("Frame resolution differs from calibration; use the same sensor mode and crop")
    if (snapshot.get("camera_index") != pose["camera_index"]
            or snapshot.get("rotation_degrees") != pose["rotation_degrees"]):
        raise LocalizationError("Camera selection or rotation differs from calibration")
    lens = pose.get("lens_position")
    current_lens = snapshot.get("lens_position")
    if lens is not None and (snapshot.get("focus_locked") is not True
                            or current_lens is None or not np.isfinite(float(current_lens))
                            or abs(float(current_lens) - float(lens)) > 0.01):
        raise LocalizationError("Camera focus differs from calibrated locked focus")


def _detection_point(detection, frame_shape):
    box = detection["bbox"]
    x1, y1, x2, y2 = (float(box[k]) for k in ("x1", "y1", "x2", "y2"))
    height, width = frame_shape[:2]
    if (not np.isfinite([x1, y1, x2, y2]).all()
            or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height):
        raise LocalizationError("Invalid detection bounding box")
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2])


def _triangulate(left_pixel, right_pixel, left_pose, right_pose, calibration):
    kl, dl, rl, tl = _camera_arrays(left_pose)
    kr, dr, rr, tr = _camera_arrays(right_pose)
    ul = cv2.undistortPoints(left_pixel.reshape(1, 1, 2), kl, dl, P=kl).reshape(2)
    ur = cv2.undistortPoints(right_pixel.reshape(1, 1, 2), kr, dr, P=kr).reshape(2)
    relative_r = rr @ rl.T
    tx, ty, tz = tr - relative_r @ tl
    skew = np.array([[0, -tz, ty], [tz, 0, -tx], [-ty, tx, 0]])
    fundamental = np.linalg.inv(kr).T @ skew @ relative_r @ np.linalg.inv(kl)
    hl, hr = np.r_[ul, 1], np.r_[ur, 1]
    line_r, line_l = fundamental @ hl, fundamental.T @ hr
    denominator = min(np.linalg.norm(line_r[:2]), np.linalg.norm(line_l[:2]))
    if denominator < 1e-12:
        raise LocalizationError("Degenerate epipolar geometry")
    epi_error = abs(float(hr @ fundamental @ hl)) / denominator
    if epi_error > calibration.get("maximum_epipolar_error_px", 2.0):
        raise LocalizationError("Epipolar error exceeds limit")
    ray_l = rl.T @ np.linalg.solve(kl, hl)
    ray_r = rr.T @ np.linalg.solve(kr, hr)
    angle = np.degrees(np.arccos(np.clip(np.dot(ray_l, ray_r) /
                                      (np.linalg.norm(ray_l) * np.linalg.norm(ray_r)), -1, 1)))
    if min(angle, 180 - angle) < calibration.get("minimum_ray_angle_degrees", 1.0):
        raise LocalizationError("Insufficient triangulation angle")
    pl, pr = kl @ np.c_[rl, tl], kr @ np.c_[rr, tr]
    point_h = cv2.triangulatePoints(pl, pr, ul.reshape(2, 1), ur.reshape(2, 1)).ravel()
    if not np.isfinite(point_h).all() or abs(point_h[3]) < 1e-12:
        raise LocalizationError("Invalid triangulation")
    point = point_h[:3] / point_h[3]
    if min((rl @ point + tl)[2], (rr @ point + tr)[2]) <= 0:
        raise LocalizationError("Triangulated point is behind a camera")
    errors = []
    for k, d, r, t, pixel in ((kl, dl, rl, tl, left_pixel), (kr, dr, rr, tr, right_pixel)):
        projected, _ = cv2.projectPoints(point.reshape(1, 3), cv2.Rodrigues(r)[0], t, k, d)
        errors.append(float(np.linalg.norm(projected.ravel() - pixel)))
    error = max(errors)
    if not np.isfinite(error) or error > calibration.get("maximum_reprojection_error_px", 2.0):
        raise LocalizationError("Reprojection error exceeds limit")
    for axis, value in zip("xyz", point):
        workspace = calibration.get("workspace", {})
        if not workspace.get(f"{axis}_min_mm", -np.inf) <= value <= workspace.get(f"{axis}_max_mm", np.inf):
            raise LocalizationError("Triangulated point is outside the measured workspace")
    return {f"{axis}_mm": float(value) for axis, value in zip("xyz", point)}, error


def localize_stereo(left_snapshot, right_snapshot, calibration=None, max_time_skew_s=0.1):
    snapshots = [s for s in (left_snapshot, right_snapshot) if s is not None]
    objects = []
    by_side = {}
    for snapshot in snapshots:
        side = snapshot["camera_id"]
        by_side[side] = []
        for detection in snapshot["detections"]:
            item = {**detection, "camera_ids": [side], "observation_ids": [detection["id"]],
                    "position": None, "robot_coordinates": None, "coordinate_system": None, "unit": "mm",
                    "valid": False, "status": "unmatched", "point_type": "bbox_center", "grasp_valid": False}
            by_side[side].append(len(objects))
            objects.append(item)
    try:
        calibration = calibration if calibration is not None else load_stereo_calibration()
        for item in objects:
            item["coordinate_system"] = calibration["coordinate_system"]["name"]
        if left_snapshot is None or right_snapshot is None:
            raise LocalizationError("Both camera observations are required for stereo")
        for side, snapshot in (("left", left_snapshot), ("right", right_snapshot)):
            _check_snapshot(snapshot, calibration[f"{side}_camera"])
            if snapshot.get("timestamp_source") != "sensor":
                raise LocalizationError("Sensor capture timestamps are required for stereo coordinates")
        capture_times = [float(s["captured_at"]) for s in snapshots]
        if (not np.isfinite(capture_times).all() or not np.isfinite(max_time_skew_s) or max_time_skew_s <= 0
                or abs(capture_times[0] - capture_times[1]) > max_time_skew_s):
            raise LocalizationError("Camera capture timestamps differ too much")
        age = time.monotonic() - min(capture_times)
        if age > 5 or age < -0.1:
            raise LocalizationError("Camera observations are stale or have invalid timestamps")
    except (LocalizationError, KeyError, ValueError, TypeError, OverflowError, AttributeError, cv2.error) as error:
        for item in objects:
            item.update(status="unavailable", localization_error=str(error))
        return objects

    candidates = []
    for li, left in enumerate(left_snapshot["detections"]):
        for ri, right in enumerate(right_snapshot["detections"]):
            if left["class_name"].casefold() != right["class_name"].casefold():
                continue
            try:
                position, error = _triangulate(
                    _detection_point(left, left_snapshot["frame"].shape),
                    _detection_point(right, right_snapshot["frame"].shape),
                    calibration["left_camera"], calibration["right_camera"], calibration)
                candidates.append((li, ri, position, error))
            except (LocalizationError, KeyError, TypeError, ValueError, cv2.error):
                continue
    merged_right = set()
    for li, ri, position, error in candidates:
        left_index, right_index = by_side["left"][li], by_side["right"][ri]
        if sum(c[0] == li for c in candidates) != 1 or sum(c[1] == ri for c in candidates) != 1:
            objects[left_index]["status"] = objects[right_index]["status"] = "ambiguous"
            continue
        left, right = objects[left_index], objects[right_index]
        left.update(camera_ids=["left", "right"], observation_ids=left["observation_ids"] + right["observation_ids"],
                    confidence=min(left["confidence"], right["confidence"]), position=position,
                    valid=True, status="stereo_estimate", reprojection_error_px=error)
        merged_right.add(right_index)
    return [item for index, item in enumerate(objects) if index not in merged_right]
