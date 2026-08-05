from __future__ import annotations

import json
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
