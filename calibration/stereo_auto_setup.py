from __future__ import annotations

import argparse
import json
import math
import time
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


SHELF_MARKER_IDS = {20, 21, 22, 23}


@dataclass
class CameraPose:
    camera_matrix: np.ndarray
    distortion: np.ndarray
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    position_robot_mm: np.ndarray
    focal_length_px: float
    reprojection_rms_px: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "One-command two-camera setup: capture, fixed-marker pose "
            "calibration, YOLO detection, stereo matching and triangulation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("calibration/stereo_setup.toml"),
    )
    parser.add_argument(
        "--skip-capture",
        action="store_true",
        help="Reuse the configured raw left/right images.",
    )
    return parser.parse_args()


def resolve_path(repository_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repository_root / path


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        return tomllib.load(stream)


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2) + "\n",
        encoding="utf-8",
    )


def save_image(path: Path, image: np.ndarray, quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        str(path),
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise RuntimeError(f"Could not save image: {path}")


def capture_two_cameras(
    left_index: int,
    right_index: int,
    width: int,
    height: int,
    focus_wait_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from libcamera import controls
        from picamera2 import Picamera2
    except ImportError as error:
        raise RuntimeError(
            "Picamera2 is required. Run this capture step on the Raspberry Pi."
        ) from error

    camera_info = Picamera2.global_camera_info()
    required_index = max(left_index, right_index)
    if len(camera_info) <= required_index:
        raise RuntimeError(
            f"Two cameras were not found. Picamera2 reports "
            f"{len(camera_info)} camera(s): {camera_info}"
        )

    left_camera = Picamera2(camera_num=left_index)
    right_camera = Picamera2(camera_num=right_index)
    cameras = [left_camera, right_camera]

    try:
        for camera in cameras:
            configuration = camera.create_still_configuration(
                main={
                    "size": (width, height),
                    "format": "RGB888",
                }
            )
            camera.configure(configuration)
            camera.start()
            camera.set_controls(
                {"AfMode": controls.AfModeEnum.Continuous}
            )

        time.sleep(focus_wait_seconds)

        metadata = []
        for camera in cameras:
            camera_metadata = camera.capture_metadata()
            lens_position = camera_metadata.get("LensPosition")
            metadata.append(
                {
                    "lens_position": (
                        float(lens_position)
                        if lens_position is not None
                        else None
                    ),
                    "exposure_time": (
                        int(camera_metadata["ExposureTime"])
                        if camera_metadata.get("ExposureTime") is not None
                        else None
                    ),
                    "analogue_gain": (
                        float(camera_metadata["AnalogueGain"])
                        if camera_metadata.get("AnalogueGain") is not None
                        else None
                    ),
                }
            )

            if lens_position is not None:
                try:
                    camera.set_controls(
                        {
                            "AfMode": controls.AfModeEnum.Manual,
                            "LensPosition": float(lens_position),
                        }
                    )
                except Exception as error:
                    print(f"Focus-lock warning: {error}")

        left_frame = left_camera.capture_array("main")
        right_frame = right_camera.capture_array("main")

    except Exception as error:
        message = str(error)
        if (
            "busy" in message.casefold()
            or "acquire" in message.casefold()
        ):
            raise RuntimeError(
                "A camera is busy. Stop the backend service before running "
                "the stereo setup."
            ) from error
        raise

    finally:
        for camera in cameras:
            try:
                camera.stop()
            except Exception:
                pass
            try:
                camera.close()
            except Exception:
                pass

    return (
        left_frame,
        right_frame,
        {
            "left": metadata[0],
            "right": metadata[1],
            "picamera2_camera_info": [
                {
                    str(key): str(value)
                    for key, value in item.items()
                }
                for item in camera_info
            ],
        },
    )


def make_aruco_detector() -> cv2.aruco.ArucoDetector:
    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_50
    )
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


def detect_shelf_markers(
    image: np.ndarray,
) -> tuple[dict[int, np.ndarray], list[np.ndarray], np.ndarray | None]:
    detector = make_aruco_detector()
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None:
        return {}, corners, ids

    detected = {
        int(marker_id): marker_corners.reshape(4, 2)
        for marker_id, marker_corners in zip(ids.flatten(), corners)
        if int(marker_id) in SHELF_MARKER_IDS
    }
    return detected, corners, ids


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    if degrees == 0:
        return image
    if degrees == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if degrees == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if degrees == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {degrees}")


def orient_from_marker_layout(
    image: np.ndarray,
) -> tuple[np.ndarray, int]:
    candidates: list[tuple[float, int, np.ndarray]] = []

    for degrees in (0, 90, 180, 270):
        rotated = rotate_image(image, degrees)
        detected, _, _ = detect_shelf_markers(rotated)
        if set(detected) != SHELF_MARKER_IDS:
            continue

        centers = {
            marker_id: corners.mean(axis=0)
            for marker_id, corners in detected.items()
        }

        left_x = float((centers[20][0] + centers[22][0]) / 2.0)
        right_x = float((centers[21][0] + centers[23][0]) / 2.0)
        top_y = float((centers[20][1] + centers[21][1]) / 2.0)
        bottom_y = float((centers[22][1] + centers[23][1]) / 2.0)

        horizontal_separation = right_x - left_x
        vertical_separation = bottom_y - top_y
        if horizontal_separation <= 0 or vertical_separation <= 0:
            continue

        score = horizontal_separation + vertical_separation
        candidates.append((score, degrees, rotated))

    if not candidates:
        raise RuntimeError(
            "Could not orient the image: IDs 20, 21, 22 and 23 must all "
            "be visible."
        )

    _, degrees, oriented = max(candidates, key=lambda item: item[0])
    return oriented, degrees


def marker_world_corners(
    center_mm: list[float],
    marker_size_mm: float,
) -> np.ndarray:
    x, y, z = center_mm
    half = marker_size_mm / 2.0
    return np.array(
        [
            [x, y - half, z - half],
            [x, y - half, z + half],
            [x, y + half, z + half],
            [x, y + half, z - half],
        ],
        dtype=np.float64,
    )


def make_camera_matrix(
    focal_px: float,
    width: int,
    height: int,
) -> np.ndarray:
    return np.array(
        [
            [focal_px, 0.0, (width - 1) / 2.0],
            [0.0, focal_px, (height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def calculate_reprojection_rms(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    rotation_vector: np.ndarray,
    translation: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )
    residual = projected.reshape(-1, 2) - image_points
    return math.sqrt(
        float(np.mean(np.sum(residual * residual, axis=1)))
    )


def fit_pose(
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray] | None:
    success, rotation_vector, translation = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    camera_points = (
        rotation @ object_points.T + translation
    ).T
    if np.min(camera_points[:, 2]) <= 0:
        return None

    rms = calculate_reprojection_rms(
        object_points,
        image_points,
        camera_matrix,
        distortion,
        rotation_vector,
        translation,
    )
    position = -rotation.T @ translation
    return rms, rotation, translation, position


def estimate_camera_pose(
    image: np.ndarray,
    marker_layout: dict[str, Any],
    expected_z_sign: int,
) -> CameraPose:
    detected, _, _ = detect_shelf_markers(image)
    missing = SHELF_MARKER_IDS - set(detected)
    if missing:
        raise RuntimeError(
            f"Missing shelf markers: {sorted(missing)}"
        )

    marker_size_mm = float(
        marker_layout["aruco"]["shelf_marker_size_mm"]
    )
    marker_config = marker_layout["markers"]

    object_groups = []
    image_groups = []
    for marker_id in sorted(SHELF_MARKER_IDS):
        configuration = marker_config[str(marker_id)]
        if configuration["orientation"] != "upright":
            raise RuntimeError(
                f"Marker {marker_id} must be configured upright"
            )
        if configuration["facing"] != "-x":
            raise RuntimeError(
                f"Marker {marker_id} must be configured facing -x"
            )
        object_groups.append(
            marker_world_corners(
                configuration["center_mm"],
                marker_size_mm,
            )
        )
        image_groups.append(detected[marker_id])

    object_points = np.concatenate(object_groups).astype(np.float64)
    image_points = np.concatenate(image_groups).astype(np.float64)

    height, width = image.shape[:2]
    distortion = np.zeros((5, 1), dtype=np.float64)
    low = width * 0.45
    high = width * 2.0

    best: tuple[
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ] | None = None

    def test_focal(focal_px: float) -> None:
        nonlocal best
        camera_matrix = make_camera_matrix(
            focal_px,
            width,
            height,
        )
        result = fit_pose(
            object_points,
            image_points,
            camera_matrix,
            distortion,
        )
        if result is None:
            return

        rms, rotation, translation, position = result
        sign_is_correct = (
            float(position[2, 0]) * expected_z_sign > 0
        )
        score = rms if sign_is_correct else rms + 1000.0
        candidate = (
            score,
            focal_px,
            camera_matrix,
            rotation,
            translation,
            position,
        )
        if best is None or score < best[0]:
            best = candidate

    coarse_values = np.linspace(low, high, 201)
    for focal_px in coarse_values:
        test_focal(float(focal_px))

    if best is None:
        raise RuntimeError("Could not estimate camera pose")

    coarse_step = float(coarse_values[1] - coarse_values[0])
    fine_low = max(low, best[1] - 4.0 * coarse_step)
    fine_high = min(high, best[1] + 4.0 * coarse_step)
    for focal_px in np.linspace(fine_low, fine_high, 241):
        test_focal(float(focal_px))

    assert best is not None
    (
        score,
        focal_px,
        camera_matrix,
        rotation,
        translation,
        position,
    ) = best
    rms = score if score < 1000.0 else score - 1000.0

    if float(position[2, 0]) * expected_z_sign <= 0:
        raise RuntimeError(
            "Estimated camera is on the wrong Z side. Check left/right "
            "camera indices and marker coordinates."
        )

    return CameraPose(
        camera_matrix=camera_matrix,
        distortion=distortion,
        rotation_world_to_camera=rotation,
        translation_world_to_camera=translation,
        position_robot_mm=position,
        focal_length_px=focal_px,
        reprojection_rms_px=rms,
    )


def pose_to_json(pose: CameraPose) -> dict[str, Any]:
    return {
        "camera_matrix": pose.camera_matrix.tolist(),
        "distortion_coefficients": (
            pose.distortion.reshape(-1).tolist()
        ),
        "rotation_world_to_camera": (
            pose.rotation_world_to_camera.tolist()
        ),
        "translation_world_to_camera_mm": (
            pose.translation_world_to_camera.reshape(3).tolist()
        ),
        "camera_position_robot_mm": (
            pose.position_robot_mm.reshape(3).tolist()
        ),
        "focal_length_px": pose.focal_length_px,
        "reprojection_rms_px": pose.reprojection_rms_px,
    }


def projection_matrix(pose: CameraPose) -> np.ndarray:
    return pose.camera_matrix @ np.hstack(
        [
            pose.rotation_world_to_camera,
            pose.translation_world_to_camera,
        ]
    )


def detect_objects(
    model: YOLO,
    image: np.ndarray,
    class_names: set[str],
    confidence: float,
    image_size: int,
    maximum_detections: int,
) -> list[dict[str, Any]]:
    prediction = model.predict(
        source=image,
        imgsz=image_size,
        conf=confidence,
        iou=0.45,
        max_det=maximum_detections,
        verbose=False,
    )[0]

    detections = []
    if prediction.boxes is None:
        return detections

    wanted = {name.casefold() for name in class_names}
    for box, class_id, score in zip(
        prediction.boxes.xyxy.cpu().numpy(),
        prediction.boxes.cls.cpu().numpy(),
        prediction.boxes.conf.cpu().numpy(),
    ):
        name = str(model.names[int(class_id)])
        if wanted and name.casefold() not in wanted:
            continue

        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": name,
                "confidence": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "center_px": [
                    (x1 + x2) / 2.0,
                    (y1 + y2) / 2.0,
                ],
            }
        )

    detections.sort(
        key=lambda detection: (
            detection["class_name"],
            detection["center_px"][1],
            detection["center_px"][0],
        )
    )
    return detections


def undistort_pixel(
    pixel: list[float],
    pose: CameraPose,
) -> np.ndarray:
    corrected = cv2.undistortPoints(
        np.asarray(pixel, dtype=np.float64).reshape(1, 1, 2),
        pose.camera_matrix,
        pose.distortion,
        P=pose.camera_matrix,
    )
    return corrected.reshape(2)


def skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector.reshape(3)
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )


def fundamental_matrix(
    left_pose: CameraPose,
    right_pose: CameraPose,
) -> np.ndarray:
    relative_rotation = (
        right_pose.rotation_world_to_camera
        @ left_pose.rotation_world_to_camera.T
    )
    relative_translation = (
        right_pose.translation_world_to_camera
        - relative_rotation
        @ left_pose.translation_world_to_camera
    )
    essential = skew(relative_translation) @ relative_rotation
    return (
        np.linalg.inv(right_pose.camera_matrix).T
        @ essential
        @ np.linalg.inv(left_pose.camera_matrix)
    )


def epipolar_error(
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    fundamental: np.ndarray,
) -> float:
    left_h = np.array([left_pixel[0], left_pixel[1], 1.0])
    right_h = np.array([right_pixel[0], right_pixel[1], 1.0])

    line_right = fundamental @ left_h
    line_left = fundamental.T @ right_h
    numerator = abs(float(right_h @ fundamental @ left_h))

    right_denominator = math.hypot(
        float(line_right[0]),
        float(line_right[1]),
    )
    left_denominator = math.hypot(
        float(line_left[0]),
        float(line_left[1]),
    )
    if right_denominator < 1e-12 or left_denominator < 1e-12:
        return float("inf")

    return 0.5 * (
        numerator / right_denominator
        + numerator / left_denominator
    )


def triangulate_world_point(
    left_pixel: np.ndarray,
    right_pixel: np.ndarray,
    left_pose: CameraPose,
    right_pose: CameraPose,
) -> np.ndarray:
    homogeneous = cv2.triangulatePoints(
        projection_matrix(left_pose),
        projection_matrix(right_pose),
        left_pixel.reshape(2, 1),
        right_pixel.reshape(2, 1),
    )
    scale = float(homogeneous[3, 0])
    if abs(scale) < 1e-12:
        raise RuntimeError("Triangulation produced a point at infinity")
    return (homogeneous[:3, 0] / scale).reshape(3)


def project_world_point(
    point: np.ndarray,
    pose: CameraPose,
) -> np.ndarray:
    camera_point = (
        pose.rotation_world_to_camera @ point.reshape(3, 1)
        + pose.translation_world_to_camera
    ).reshape(3)
    projected = pose.camera_matrix @ camera_point
    return projected[:2] / projected[2]


def point_is_in_front(
    point: np.ndarray,
    pose: CameraPose,
) -> bool:
    camera_point = (
        pose.rotation_world_to_camera @ point.reshape(3, 1)
        + pose.translation_world_to_camera
    )
    return float(camera_point[2, 0]) > 0


def point_is_in_workspace(
    point: np.ndarray,
    workspace: dict[str, Any],
) -> bool:
    x, y, z = point
    return (
        float(workspace["x_min_mm"])
        <= x
        <= float(workspace["x_max_mm"])
        and float(workspace["y_min_mm"])
        <= y
        <= float(workspace["y_max_mm"])
        and float(workspace["z_min_mm"])
        <= z
        <= float(workspace["z_max_mm"])
    )


def match_and_triangulate(
    left_detections: list[dict[str, Any]],
    right_detections: list[dict[str, Any]],
    left_pose: CameraPose,
    right_pose: CameraPose,
    maximum_epipolar_error_px: float,
    maximum_reprojection_error_px: float,
    workspace: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    set[int],
    set[int],
]:
    fundamental = fundamental_matrix(left_pose, right_pose)
    candidates = []

    for left_index, left in enumerate(left_detections):
        left_pixel = undistort_pixel(
            left["center_px"],
            left_pose,
        )

        for right_index, right in enumerate(right_detections):
            if (
                left["class_name"].casefold()
                != right["class_name"].casefold()
            ):
                continue

            right_pixel = undistort_pixel(
                right["center_px"],
                right_pose,
            )
            epi_error = epipolar_error(
                left_pixel,
                right_pixel,
                fundamental,
            )
            if epi_error > maximum_epipolar_error_px:
                continue

            point = triangulate_world_point(
                left_pixel,
                right_pixel,
                left_pose,
                right_pose,
            )
            if not point_is_in_front(point, left_pose):
                continue
            if not point_is_in_front(point, right_pose):
                continue
            if not point_is_in_workspace(point, workspace):
                continue

            projected_left = project_world_point(point, left_pose)
            projected_right = project_world_point(point, right_pose)
            left_error = float(
                np.linalg.norm(projected_left - left_pixel)
            )
            right_error = float(
                np.linalg.norm(projected_right - right_pixel)
            )
            reprojection_error = 0.5 * (
                left_error + right_error
            )
            if (
                reprojection_error
                > maximum_reprojection_error_px
            ):
                continue

            candidates.append(
                {
                    "left_index": left_index,
                    "right_index": right_index,
                    "class_name": left["class_name"],
                    "confidence": min(
                        float(left["confidence"]),
                        float(right["confidence"]),
                    ),
                    "coordinates_robot_mm": point.tolist(),
                    "epipolar_error_px": epi_error,
                    "reprojection_error_px": reprojection_error,
                    "matching_cost": (
                        epi_error + reprojection_error
                    ),
                }
            )

    candidates.sort(
        key=lambda candidate: (
            candidate["matching_cost"],
            -candidate["confidence"],
        )
    )

    matched_left: set[int] = set()
    matched_right: set[int] = set()
    matches = []

    for candidate in candidates:
        left_index = int(candidate["left_index"])
        right_index = int(candidate["right_index"])
        if left_index in matched_left or right_index in matched_right:
            continue

        matched_left.add(left_index)
        matched_right.add(right_index)
        candidate["match_id"] = len(matches) + 1
        candidate.pop("matching_cost", None)
        matches.append(candidate)

    matches.sort(
        key=lambda match: (
            match["coordinates_robot_mm"][1],
            match["coordinates_robot_mm"][2],
        )
    )
    for match_id, match in enumerate(matches, start=1):
        match["match_id"] = match_id

    return matches, matched_left, matched_right


def annotate_detections(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    side: str,
    target_match_id: int | None,
) -> np.ndarray:
    annotated = image.copy()
    index_key = f"{side}_index"
    matches_by_index = {
        int(match[index_key]): match
        for match in matches
    }

    for index, detection in enumerate(detections):
        x1, y1, x2, y2 = (
            int(round(value))
            for value in detection["bbox_xyxy"]
        )
        match = matches_by_index.get(index)

        if match is None:
            color = (0, 200, 255)
            label = f"UNMATCHED {detection['class_name']}"
            thickness = 2
        else:
            match_id = int(match["match_id"])
            is_target = match_id == target_match_id
            color = (0, 0, 255) if is_target else (0, 200, 0)
            thickness = 4 if is_target else 2
            x_mm, y_mm, z_mm = match["coordinates_robot_mm"]
            label = (
                f"{'#' + str(match_id)} "
                f"X={x_mm:.1f} Y={y_mm:.1f} Z={z_mm:.1f} mm"
            )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(28, y1 - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def main() -> None:
    args = parse_args()
    repository_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(
        repository_root,
        str(args.config),
    )
    configuration = load_toml(config_path)

    paths = configuration["paths"]
    capture_config = configuration["capture"]
    detection_config = configuration["detection"]
    cameras_config = configuration["cameras"]
    validation_config = configuration["validation"]
    workspace = configuration["workspace"]

    marker_layout_path = resolve_path(
        repository_root,
        paths["marker_layout"],
    )
    model_path = resolve_path(
        repository_root,
        paths["model"],
    )
    capture_directory = resolve_path(
        repository_root,
        paths["capture_directory"],
    )
    calibration_output = resolve_path(
        repository_root,
        paths["calibration_output"],
    )
    coordinates_output = resolve_path(
        repository_root,
        paths["coordinates_output"],
    )

    capture_directory.mkdir(parents=True, exist_ok=True)
    raw_left_path = capture_directory / "stereo_left_raw.jpg"
    raw_right_path = capture_directory / "stereo_right_raw.jpg"
    oriented_left_path = capture_directory / "stereo_left.jpg"
    oriented_right_path = capture_directory / "stereo_right.jpg"

    capture_metadata: dict[str, Any] = {
        "capture_skipped": args.skip_capture
    }

    if args.skip_capture:
        left_raw = cv2.imread(str(raw_left_path))
        right_raw = cv2.imread(str(raw_right_path))
        if left_raw is None or right_raw is None:
            raise FileNotFoundError(
                "With --skip-capture, both stereo_left_raw.jpg and "
                "stereo_right_raw.jpg must exist in the capture directory."
            )
    else:
        print("Capturing camera 0 and camera 1...")
        left_raw, right_raw, camera_metadata = capture_two_cameras(
            int(cameras_config["left"]["index"]),
            int(cameras_config["right"]["index"]),
            int(capture_config["width"]),
            int(capture_config["height"]),
            float(capture_config["focus_wait_seconds"]),
        )
        capture_metadata.update(camera_metadata)
        save_image(
            raw_left_path,
            left_raw,
            int(capture_config["jpeg_quality"]),
        )
        save_image(
            raw_right_path,
            right_raw,
            int(capture_config["jpeg_quality"]),
        )

    print("Orienting images from shelf marker IDs...")
    left_image, left_rotation_degrees = orient_from_marker_layout(
        left_raw
    )
    right_image, right_rotation_degrees = orient_from_marker_layout(
        right_raw
    )
    save_image(
        oriented_left_path,
        left_image,
        int(capture_config["jpeg_quality"]),
    )
    save_image(
        oriented_right_path,
        right_image,
        int(capture_config["jpeg_quality"]),
    )

    if left_image.shape[:2] != right_image.shape[:2]:
        raise RuntimeError(
            "Left and right oriented images have different resolutions"
        )

    marker_layout = load_toml(marker_layout_path)

    print("Estimating left camera pose...")
    left_pose = estimate_camera_pose(
        left_image,
        marker_layout,
        int(cameras_config["left"]["expected_z_sign"]),
    )
    print("Estimating right camera pose...")
    right_pose = estimate_camera_pose(
        right_image,
        marker_layout,
        int(cameras_config["right"]["expected_z_sign"]),
    )

    baseline_mm = float(
        np.linalg.norm(
            right_pose.position_robot_mm
            - left_pose.position_robot_mm
        )
    )
    symmetry_error = {
        "x_difference_mm": abs(
            float(
                left_pose.position_robot_mm[0, 0]
                - right_pose.position_robot_mm[0, 0]
            )
        ),
        "y_difference_mm": abs(
            float(
                left_pose.position_robot_mm[1, 0]
                - right_pose.position_robot_mm[1, 0]
            )
        ),
        "z_sum_mm": abs(
            float(
                left_pose.position_robot_mm[2, 0]
                + right_pose.position_robot_mm[2, 0]
            )
        ),
    }

    maximum_marker_rms = float(
        validation_config[
            "maximum_marker_reprojection_rms_px"
        ]
    )
    if (
        left_pose.reprojection_rms_px > maximum_marker_rms
        or right_pose.reprojection_rms_px > maximum_marker_rms
    ):
        raise RuntimeError(
            "Shelf-marker reprojection error is too high: "
            f"left={left_pose.reprojection_rms_px:.3f}px, "
            f"right={right_pose.reprojection_rms_px:.3f}px, "
            f"maximum={maximum_marker_rms:.3f}px"
        )

    minimum_baseline = float(
        validation_config["minimum_baseline_mm"]
    )
    maximum_baseline = float(
        validation_config["maximum_baseline_mm"]
    )
    if not minimum_baseline <= baseline_mm <= maximum_baseline:
        raise RuntimeError(
            f"Estimated baseline {baseline_mm:.3f}mm is outside "
            f"{minimum_baseline:.3f}-{maximum_baseline:.3f}mm"
        )

    maximum_symmetry_error = float(
        validation_config[
            "maximum_mirror_symmetry_error_mm"
        ]
    )
    failed_symmetry_values = {
        name: value
        for name, value in symmetry_error.items()
        if value > maximum_symmetry_error
    }
    if failed_symmetry_values:
        raise RuntimeError(
            "Camera mirror-symmetry check failed: "
            f"{failed_symmetry_values}; maximum allowed per value is "
            f"{maximum_symmetry_error:.3f}mm"
        )

    calibration_data = {
        "status": "automatic_fixed_marker_stereo_calibration",
        "warning": (
            "Camera poses are calibrated from one fixed planar shelf-marker "
            "view. Lens distortion remains zero until a multi-view intrinsic "
            "calibration is supplied."
        ),
        "image_size": {
            "width": int(left_image.shape[1]),
            "height": int(left_image.shape[0]),
        },
        "left_image_rotation_degrees": left_rotation_degrees,
        "right_image_rotation_degrees": right_rotation_degrees,
        "capture_metadata": capture_metadata,
        "left_camera": pose_to_json(left_pose),
        "right_camera": pose_to_json(right_pose),
        "baseline_mm": baseline_mm,
        "mirror_symmetry_error": symmetry_error,
    }
    save_json(calibration_output, calibration_data)

    print("Running YOLO on both images...")
    model = YOLO(str(model_path))
    class_names = {
        str(name)
        for name in detection_config["class_names"]
    }
    left_detections = detect_objects(
        model,
        left_image,
        class_names,
        float(detection_config["confidence"]),
        int(detection_config["image_size"]),
        int(detection_config["maximum_detections"]),
    )
    right_detections = detect_objects(
        model,
        right_image,
        class_names,
        float(detection_config["confidence"]),
        int(detection_config["image_size"]),
        int(detection_config["maximum_detections"]),
    )

    print("Matching detections and triangulating...")
    matches, matched_left, matched_right = match_and_triangulate(
        left_detections,
        right_detections,
        left_pose,
        right_pose,
        float(
            detection_config[
                "maximum_epipolar_error_px"
            ]
        ),
        float(
            detection_config[
                "maximum_reprojection_error_px"
            ]
        ),
        workspace,
    )

    target_match_id = (
        int(matches[0]["match_id"])
        if matches
        else None
    )
    for match in matches:
        match["is_target"] = (
            int(match["match_id"]) == target_match_id
        )

    coordinates_data = {
        "mode": "real_stereo",
        "units": "mm",
        "coordinate_system": marker_layout["coordinate_system"],
        "baseline_mm": baseline_mm,
        "target_selection": (
            "smallest robot Y coordinate among valid matches"
        ),
        "target": matches[0] if matches else None,
        "matches": matches,
        "unmatched_left_detection_indices": sorted(
            set(range(len(left_detections))) - matched_left
        ),
        "unmatched_right_detection_indices": sorted(
            set(range(len(right_detections))) - matched_right
        ),
    }
    save_json(coordinates_output, coordinates_data)

    annotated_left = annotate_detections(
        left_image,
        left_detections,
        matches,
        "left",
        target_match_id,
    )
    annotated_right = annotate_detections(
        right_image,
        right_detections,
        matches,
        "right",
        target_match_id,
    )
    annotated_left_path = (
        capture_directory / "stereo_left_coordinates.jpg"
    )
    annotated_right_path = (
        capture_directory / "stereo_right_coordinates.jpg"
    )
    save_image(annotated_left_path, annotated_left)
    save_image(annotated_right_path, annotated_right)

    print()
    print("Stereo setup complete")
    print(
        "Left camera position [X, Y, Z] mm:",
        np.round(
            left_pose.position_robot_mm.reshape(3),
            3,
        ).tolist(),
    )
    print(
        "Right camera position [X, Y, Z] mm:",
        np.round(
            right_pose.position_robot_mm.reshape(3),
            3,
        ).tolist(),
    )
    print(f"Baseline: {baseline_mm:.3f} mm")
    print(
        "Marker reprojection RMS left/right:",
        f"{left_pose.reprojection_rms_px:.3f} / "
        f"{right_pose.reprojection_rms_px:.3f} px",
    )
    print(
        "Detections left/right:",
        f"{len(left_detections)} / {len(right_detections)}",
    )
    print(f"Valid stereo matches: {len(matches)}")

    for match in matches:
        x_mm, y_mm, z_mm = match["coordinates_robot_mm"]
        target_text = " TARGET" if match["is_target"] else ""
        print(
            f"#{match['match_id']}{target_text} "
            f"{match['class_name']} "
            f"XYZ=[{x_mm:.3f}, {y_mm:.3f}, {z_mm:.3f}] mm "
            f"reprojection={match['reprojection_error_px']:.3f}px"
        )

    print(f"Calibration JSON: {calibration_output}")
    print(f"Coordinates JSON: {coordinates_output}")
    print(f"Annotated left image: {annotated_left_path}")
    print(f"Annotated right image: {annotated_right_path}")

    if not matches:
        raise SystemExit(
            "No valid stereo matches. Inspect the annotated images and "
            "increase matching thresholds only if the detections are correct."
        )


if __name__ == "__main__":
    main()
