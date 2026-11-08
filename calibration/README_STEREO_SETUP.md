# Two fixed cameras: calibration and runtime coordinates

The backend never calibrates cameras on startup. Raw and YOLO streams work
without calibration; object XYZ remains `null` until a usable calibration is
loaded. `GET /calibration/status` explains missing files or validation.

Historical one-view bootstrap files remain untouched and are not sufficient
for stereo coordinates. The discarded October calibration measurements must
not be used as current calibration. No valid intrinsic or stereo JSON is
currently included in this repository.

## 1. Measure and capture each lens independently

Use a rigid, flat ChArUco board and the JSON specification belonging to the
actual printed board. Measure its square size and printed scale; do not assume
PDF dimensions survived printing. Board marker IDs must not overlap the fixed
shelf markers 20–23. The JSON requires `dictionary`, `squares` (columns, rows),
`square_mm`, `marker_mm`, and optionally `marker_ids`.

On the Pi, use the existing Python environment with Picamera2, NumPy and
an OpenCV build exposing `cv2.aruco.CharucoDetector`. Stop only the backend camera service before
capturing; do not run deployment scripts that also start the arm controller.
Identify the physical cameras using `Picamera2.global_camera_info()`; indices
0/1 in the examples are configuration defaults, not verified identities.

From the backend directory, after setting `FOCUS_LEFT` and `FOCUS_RIGHT` to
measured, sharp lens positions and placing your board spec at
`calibration/board.json`:

```bash
sudo systemctl stop robot-arm-backend.service
python calibration/calibrate_intrinsics.py --board calibration/board.json \
  --images calibration/captures/left-intrinsics --capture 20 --camera-index 0 \
  --lens-position "$FOCUS_LEFT" --rotation-degrees 0 \
  --output calibration/left_intrinsics.json
python calibration/calibrate_intrinsics.py --board calibration/board.json \
  --images calibration/captures/right-intrinsics --capture 20 --camera-index 1 \
  --lens-position "$FOCUS_RIGHT" --rotation-degrees 0 \
  --output calibration/right_intrinsics.json
```

Move and tilt the board between captures. Cover the image edges and vary
distance; hold it still for each capture. Fitting requires at least 12 usable
views and at least 20 degrees of tilt variation. An excessive image RMS is
rejected; a small RMS alone does not establish millimetre accuracy.

Use `--width` / `--height` to match runtime resolution (default 1280×720).
Use the same sensor mode, crop, focus and image orientation for calibration and
runtime. Set `--rotation-degrees 180` if the corresponding runtime camera is
rotated. Images saved by this script are unrotated; it applies the requested
rotation before fitting. To fit existing images, omit `--capture`.

Existing capture directories and intrinsic JSON outputs are never overwritten.
For recalibration, choose new paths and update `stereo_setup.toml` accordingly.

## 2. Establish the measured robot frame

Inspect `calibration/shelf_markers.toml` rather than using historical positions.
Its current measurements are explicitly **not physically validated**. Check
marker edge lengths, all four marker centres, orientation, and the origin.
The `robot_base` frame is the roof mounting plate centre: X toward objects,
Y downward, Z right, in millimetres. Merely having matching axis directions
in a shelf frame does not prove that its origin matches the robot's origin.
If calibration is in a separate shelf frame, keep `name = "shelf"`; no robot
transform is inferred and no controller target is generated.

Only after measuring the layout and independently verifying positions may the
owner mark `shelf_layout_physically_validated = true`. Do not set this flag
merely to make the status green.

Set the two camera indices and independent intrinsic JSON paths in
`calibration/stereo_setup.toml`. Its capture dimensions must match the fitted
intrinsics. Both cameras must see fixed marker IDs 20, 21, 22, 23. Then run:

```bash
bash calibration/run_stereo_setup.sh
```

The existing wrapper stops the backend if active and restarts it on exit. The
script uses each lens's measured intrinsics and locked focus, estimates
independent world-to-camera poses from the markers, and records camera identity,
orientation, resolution, focus, frame name and validation state. It requires
`models/best.pt` for the final diagnostic YOLO pass. It does not move the arm.
Existing captures and result files are preserved: configure fresh output paths
for another run. `--skip-capture` can use raw left/right images in a fresh
capture directory; their focus and sensor configuration must match the files.

A calibration produced while the layout flag is false is retained for
inspection but the runtime rejects it. After physical validation, generate a
new result using new output paths; do not just edit a result's validation flag.

## 3. Runtime and independent checks

Set `STEREO_CALIBRATION_PATH` if using a result other than
`calibration/stereo_calibration.json`. Set `CAMERA_LEFT_INDEX`,
`CAMERA_RIGHT_INDEX`, `CAMERA_LEFT_LENS_POSITION`, `CAMERA_RIGHT_LENS_POSITION`,
and the per-camera dimensions/rotation to match the calibration, then restart
the backend service. Autofocus must not change the calibrated lens positions.
The runtime deliberately rejects mismatched resolution, orientation, selected
camera, focus, stale frames, or excessive capture timestamp skew.

Stereo combines only mutually unambiguous same-class observations passing
epipolar, positive-depth, ray-angle, workspace and per-camera reprojection
checks. Coordinates are triangulated in the calibrated frame, never averaged.
Unmatched or ambiguous observations retain their originating camera and null
XYZ. Bounding-box centres can correspond to different visible surfaces: even
a valid stereo estimate is **not a verified object centre or grasp point**.
`grasp_valid` stays false and `robot_coordinates` stays null.

On real hardware:

1. Check `/health`, `/calibration/status`, and both raw/detection streams.
2. Disconnect either camera; the other stream must remain usable and XYZ must
   become unavailable. Reconnect and restart the backend to reopen the device.
3. Check stationary targets at measured near/far and left/right positions;
   compare XYZ and distances with independent millimetre measurements.
4. Place two same-class targets on one epipolar line; ambiguous matches must
   show unavailable coordinates. Test wrong camera identity, rotation and focus.
5. Verify timestamps and position stability while moving a target slowly;
software timestamp gating does not provide hardware synchronization. Capture-completion
timestamps cannot establish exposure timing, so XYZ requires sensor timestamps.
6. Verify mode selection, buttons, permission expiry and emergency stop with
   the controller owner before enabling any automatic pickup integration.

The saved diagnostic file never selects an automatic grasp target. Neither
synthetic tests nor successful camera calibration proves safe robot pickup.
