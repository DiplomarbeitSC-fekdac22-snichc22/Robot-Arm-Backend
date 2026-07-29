from __future__ import annotations

import argparse
import json
import math
import tomllib
from pathlib import Path

import cv2
import numpy as np


EXPECTED_IDS = {20, 21, 22, 23}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate camera position from shelf markers "
            "20, 21, 22 and 23."
        )
    )

    parser.add_argument(
        "image",
        type=Path,
    )

    parser.add_argument(
        "marker_layout",
        type=Path,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "calibration/camera_pose_bootstrap.json"
        ),
    )

    parser.add_argument(
        "--annotated-output",
        type=Path,
        default=Path(
            "calibration/captures/shelf_markers_pose.jpg"
        ),
    )

    return parser.parse_args()


def marker_world_corners(
    center_mm,
    marker_size_mm,
):
    x, y, z = (
        float(value)
        for value in center_mm
    )

    half = marker_size_mm / 2.0

    return np.array(
        [
            [
                x,
                y - half,
                z - half,
            ],
            [
                x,
                y - half,
                z + half,
            ],
            [
                x,
                y + half,
                z + half,
            ],
            [
                x,
                y + half,
                z - half,
            ],
        ],
        dtype=np.float64,
    )


def make_camera_matrix(
    focal_px,
    width,
    height,
):
    return np.array(
        [
            [
                focal_px,
                0.0,
                (width - 1) / 2.0,
            ],
            [
                0.0,
                focal_px,
                (height - 1) / 2.0,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=np.float64,
    )


def fit_pose(
    object_points,
    image_points,
    camera_matrix,
    distortion,
):
    success, rotation_vector, translation = (
        cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    )

    if not success:
        return None

    rotation_vector, translation = (
        cv2.solvePnPRefineLM(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            rotation_vector,
            translation,
        )
    )

    rotation, _ = cv2.Rodrigues(
        rotation_vector
    )

    camera_points = (
        rotation @ object_points.T
        + translation
    ).T

    if np.min(camera_points[:, 2]) <= 0:
        return None

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )

    residual = (
        projected.reshape(-1, 2)
        - image_points
    )

    rms = math.sqrt(
        float(
            np.mean(
                np.sum(
                    residual * residual,
                    axis=1,
                )
            )
        )
    )

    return (
        rms,
        rotation_vector,
        translation,
    )


def search_focal_length(
    object_points,
    image_points,
    width,
    height,
    distortion,
):
    minimum_focal = width * 0.45
    maximum_focal = width * 2.0

    best = None

    def try_focal(focal_px):
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

        (
            rms,
            rotation_vector,
            translation,
        ) = result

        candidate = (
            rms,
            focal_px,
            camera_matrix,
            rotation_vector,
            translation,
        )

        if (
            best is None
            or rms < best[0]
        ):
            best = candidate

    coarse_values = np.linspace(
        minimum_focal,
        maximum_focal,
        241,
    )

    for focal_px in coarse_values:
        try_focal(float(focal_px))

    if best is None:
        raise RuntimeError(
            "Could not estimate camera pose"
        )

    coarse_step = float(
        coarse_values[1]
        - coarse_values[0]
    )

    fine_minimum = max(
        minimum_focal,
        best[1] - 4.0 * coarse_step,
    )

    fine_maximum = min(
        maximum_focal,
        best[1] + 4.0 * coarse_step,
    )

    for focal_px in np.linspace(
        fine_minimum,
        fine_maximum,
        321,
    ):
        try_focal(float(focal_px))

    return best


def main():
    args = parse_args()

    image = cv2.imread(
        str(args.image)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: "
            f"{args.image}"
        )

    with args.marker_layout.open(
        "rb"
    ) as stream:
        layout = tomllib.load(stream)

    marker_size_mm = float(
        layout["aruco"][
            "shelf_marker_size_mm"
        ]
    )

    marker_configuration = (
        layout["markers"]
    )

    dictionary = (
        cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
    )

    parameters = (
        cv2.aruco.DetectorParameters()
    )

    parameters.cornerRefinementMethod = (
        cv2.aruco.CORNER_REFINE_SUBPIX
    )

    detector = cv2.aruco.ArucoDetector(
        dictionary,
        parameters,
    )

    corners, ids, _ = (
        detector.detectMarkers(image)
    )

    if ids is None:
        raise RuntimeError(
            "No ArUco markers detected"
        )

    detected = {
        int(marker_id):
        marker_corners.reshape(4, 2)

        for marker_id, marker_corners
        in zip(
            ids.flatten(),
            corners,
        )
    }

    missing = (
        EXPECTED_IDS
        - set(detected)
    )

    if missing:
        raise RuntimeError(
            f"Missing shelf markers: "
            f"{sorted(missing)}"
        )

    object_point_groups = []
    image_point_groups = []

    for marker_id in sorted(
        EXPECTED_IDS
    ):
        configuration = (
            marker_configuration[
                str(marker_id)
            ]
        )

        object_point_groups.append(
            marker_world_corners(
                configuration["center_mm"],
                marker_size_mm,
            )
        )

        image_point_groups.append(
            detected[marker_id]
        )

    object_points = np.concatenate(
        object_point_groups
    ).astype(np.float64)

    image_points = np.concatenate(
        image_point_groups
    ).astype(np.float64)

    height, width = image.shape[:2]

    distortion = np.zeros(
        (5, 1),
        dtype=np.float64,
    )

    result = search_focal_length(
        object_points,
        image_points,
        width,
        height,
        distortion,
    )

    (
        rms,
        focal_px,
        camera_matrix,
        rotation_vector,
        translation,
    ) = result

    rotation, _ = cv2.Rodrigues(
        rotation_vector
    )

    camera_position = (
        -rotation.T
        @ translation
    ).reshape(3)

    output = {
        "status": (
            "provisional_bootstrap_calibration"
        ),

        "warning": (
            "Camera pose calculated from one "
            "fixed planar marker view."
        ),

        "image_size": {
            "width": width,
            "height": height,
        },

        "camera_matrix": (
            camera_matrix.tolist()
        ),

        "distortion_coefficients": (
            distortion.reshape(-1).tolist()
        ),

        "focal_length_px": (
            float(focal_px)
        ),

        "reprojection_rms_px": (
            float(rms)
        ),

        "rotation_world_to_camera": (
            rotation.tolist()
        ),

        "translation_world_to_camera_mm": (
            translation.reshape(3).tolist()
        ),

        "camera_position_robot_mm": (
            camera_position.tolist()
        ),

        "marker_ids": sorted(
            EXPECTED_IDS
        ),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            output,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    annotated = image.copy()

    cv2.aruco.drawDetectedMarkers(
        annotated,
        corners,
        ids,
    )

    projected, _ = cv2.projectPoints(
        object_points,
        rotation_vector,
        translation,
        camera_matrix,
        distortion,
    )

    for point in projected.reshape(
        -1,
        2,
    ):
        cv2.circle(
            annotated,
            tuple(
                np.rint(point).astype(int)
            ),
            4,
            (0, 0, 255),
            -1,
        )

    args.annotated_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not cv2.imwrite(
        str(args.annotated_output),
        annotated,
    ):
        raise RuntimeError(
            f"Could not save annotated image: "
            f"{args.annotated_output}"
        )

    print(
        f"Focal length: "
        f"{focal_px:.3f} px"
    )

    print(
        f"Reprojection RMS: "
        f"{rms:.3f} px"
    )

    print(
        "Camera position [X, Y, Z] mm: "
        f"[{camera_position[0]:.3f}, "
        f"{camera_position[1]:.3f}, "
        f"{camera_position[2]:.3f}]"
    )

    print(
        f"Saved calibration: "
        f"{args.output}"
    )

    print(
        f"Saved annotated image: "
        f"{args.annotated_output}"
    )


if __name__ == "__main__":
    main()