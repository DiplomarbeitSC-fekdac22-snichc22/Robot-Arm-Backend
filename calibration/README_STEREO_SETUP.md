# Automatic two-camera setup

## Required physical setup

- Raspberry Pi Camera Module 3 devices connected as camera `0` and camera `1`.
- Camera `0` is the physical left camera.
- Camera `1` is the physical right camera.
- Both cameras can see shelf marker IDs `20`, `21`, `22`, and `23`.
- The four shelf markers and cameras must remain rigid.
- Objects must remain still during the two captures.

## Files

Keep all files in the existing backend structure:

```text
Robot-Arm-Backend/
├── calibration/
│   ├── stereo_auto_setup.py
│   ├── stereo_setup.toml
│   ├── run_stereo_setup.sh
│   ├── shelf_markers.toml
│   └── captures/
└── models/
    └── best.pt
```

## Run

From the backend repository:

```bash
bash calibration/run_stereo_setup.sh
```

The wrapper temporarily stops `robot-arm-backend.service` if it is running,
runs the calibration, and restarts the service before exiting.

## Automatic operations

1. Verify that two cameras exist.
2. Start and autofocus both cameras.
3. Lock each current focus position.
4. Capture both camera images at `2304 x 1296`.
5. Rotate each image automatically using shelf marker positions.
6. Detect IDs `20`, `21`, `22`, and `23`.
7. Estimate both cameras in the robot coordinate system.
8. Run YOLO on both images.
9. Match equal object classes using epipolar geometry.
10. Triangulate real robot-frame XYZ coordinates.
11. Reject points behind a camera, outside the workspace, or with excessive
    epipolar/reprojection error.

## Outputs

```text
calibration/stereo_calibration.json
calibration/stereo_coordinates.json
calibration/captures/stereo_left_raw.jpg
calibration/captures/stereo_right_raw.jpg
calibration/captures/stereo_left.jpg
calibration/captures/stereo_right.jpg
calibration/captures/stereo_left_coordinates.jpg
calibration/captures/stereo_right_coordinates.jpg
```

The `target` field in `stereo_coordinates.json` is the valid matched object
with the smallest robot `Y` coordinate.

## Re-run using saved images

This avoids accessing the cameras:

```bash
bash calibration/run_stereo_setup.sh --skip-capture
```

The script then reuses:

```text
calibration/captures/stereo_left_raw.jpg
calibration/captures/stereo_right_raw.jpg
```

## Current accuracy limitation

This workflow obtains real stereo depth when both cameras are installed.
However, each lens pose is bootstrapped from one fixed planar marker view.
Lens distortion is therefore currently stored as zero. A later multi-view
intrinsic calibration can replace this part without changing the detection,
matching, triangulation, or JSON output stages.
