"""Calibrate one fixed-focus camera from varied views of a measured ChArUco board."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def calibrate(images, board, maximum_rms=1.5):
    if not np.isfinite(maximum_rms) or maximum_rms <= 0:
        raise ValueError("Maximum RMS must be finite and positive")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("OpenCV with cv2.aruco.CharucoDetector is required")
    detector = cv2.aruco.CharucoDetector(board)
    object_points, image_points = [], []
    size = None
    for image in images:
        if image is None:
            raise ValueError("Could not read a calibration image")
        current_size = image.shape[1::-1]
        if size is not None and size != current_size:
            raise ValueError("All calibration images must use the same resolution and sensor crop")
        size = current_size
        corners, ids, _, _ = detector.detectBoard(image)
        if ids is None or len(ids) < 12:
            continue
        objects = board.getChessboardCorners()[ids.ravel()].astype(np.float32)
        if np.linalg.matrix_rank(objects - objects.mean(axis=0)) < 2:
            continue
        object_points.append(objects)
        image_points.append(np.asarray(corners, np.float32).reshape(-1, 1, 2))
    if len(object_points) < 12:
        raise ValueError("Need at least 12 sharp board views with at least 12 ChArUco corners each")
    rms, matrix, distortion, rotations, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    normals = [cv2.Rodrigues(r)[0][:, 2] for r in rotations]
    angles = [np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))) for a in normals for b in normals]
    if max(angles) < 20:
        raise ValueError("Board views need more tilt variation (at least 20 degrees)")
    if not np.isfinite(rms) or rms > maximum_rms:
        raise ValueError(f"Intrinsic RMS {rms:.3f}px exceeds {maximum_rms:.3f}px")
    return {"image_size": {"width": size[0], "height": size[1]}, "camera_matrix": matrix.tolist(),
            "distortion_coefficients": distortion.ravel().tolist(), "reprojection_rms_px": rms,
            "intrinsics_calibrated": True, "views": len(object_points)}


def capture(args):
    from libcamera import controls
    from picamera2 import Picamera2

    if args.images.exists():
        raise FileExistsError("Use a new image directory to preserve previous calibration captures")
    args.images.mkdir(parents=True)
    camera = Picamera2(camera_num=args.camera_index)
    try:
        camera.configure(camera.create_video_configuration(
            main={"size": (args.width, args.height), "format": "RGB888"}))
        camera.set_controls({"AfMode": controls.AfModeEnum.Manual, "LensPosition": args.lens_position})
        camera.start()
        for index in range(args.capture):
            input(f"[{index + 1}/{args.capture}] Change board position/tilt, hold still, then Enter: ")
            image = camera.capture_array("main")
            if not cv2.imwrite(str(args.images / f"view_{index:03d}.png"), image):
                raise RuntimeError("Could not write calibration capture")
    finally:
        try:
            camera.stop()
        finally:
            camera.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board", type=Path, required=True, help="JSON matching the physically measured printed board")
    parser.add_argument("--images", type=Path, required=True, help="Directory of unrotated PNG/JPG board images")
    parser.add_argument("--output", type=Path, required=True, help="New per-camera intrinsic JSON (never overwritten)")
    parser.add_argument("--camera-index", type=int, required=True)
    parser.add_argument("--lens-position", type=float, required=True, help="Fixed Camera Module 3 lens position")
    parser.add_argument("--rotation-degrees", type=int, choices=(0, 90, 180, 270), default=0)
    parser.add_argument("--capture", type=int, default=0, help="Capture this many views on the Pi before fitting")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--maximum-rms-px", type=float, default=1.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output filename to preserve existing intrinsic calibration")
    if (args.camera_index < 0 or not np.isfinite(args.lens_position) or args.lens_position < 0
            or args.capture < 0 or min(args.width, args.height) <= 0
            or not np.isfinite(args.maximum_rms_px) or args.maximum_rms_px <= 0):
        raise ValueError("Use a valid camera index, finite locked focus, positive dimensions and finite positive RMS limit")
    specification = json.loads(args.board.read_text())
    square, marker = float(specification["square_mm"]), float(specification["marker_mm"])
    counts = specification["squares"]
    if (not np.isfinite([square, marker]).all() or not 0 < marker < square
            or len(counts) != 2 or any(type(count) is not int or count < 2 for count in counts)):
        raise ValueError("Board needs two integer square counts and finite lengths with 0 < marker_mm < square_mm")
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
        raise RuntimeError("OpenCV with cv2.aruco.CharucoDetector is required")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, specification["dictionary"]))
    arguments = (tuple(counts), square, marker, dictionary)
    ids = specification.get("marker_ids")
    if ids is not None and (len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < dictionary.bytesList.shape[0] for i in ids)):
        raise ValueError("Board marker IDs must be unique integer IDs within the dictionary")
    board = cv2.aruco.CharucoBoard(*arguments) if ids is None else cv2.aruco.CharucoBoard(
        *arguments, np.asarray(ids, np.int32))
    if set(board.getIds().ravel()) & {20, 21, 22, 23}:
        raise ValueError("Board marker IDs must not overlap fixed shelf markers 20..23")
    if args.capture:
        capture(args)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    images = [cv2.imread(str(path)) for path in paths]
    if args.rotation_degrees:
        rotation = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        images = [None if image is None else cv2.rotate(image, rotation[args.rotation_degrees]) for image in images]
    result = calibrate(images, board, args.maximum_rms_px)
    result.update(camera_index=args.camera_index, rotation_degrees=args.rotation_degrees,
                  lens_position=args.lens_position, board=specification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {args.output}; RMS={result['reprojection_rms_px']:.3f}px. This is not a robot-frame pose.")


if __name__ == "__main__":
    main()
