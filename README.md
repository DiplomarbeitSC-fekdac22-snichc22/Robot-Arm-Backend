# Robot Arm Backend

FastAPI, two independent Picamera2 captures and one shared YOLO model. Importing
modules does not open cameras or load the model; FastAPI starts and releases
resources with its lifespan. A missing camera/model leaves diagnostic routes
available. Use one backend worker (`python run_backend.py`, port 8000).

## API

The `/video`, `/detect`, `/detections` and `/snapshot` routes default
to the left camera. Use `?camera_id=right` to select the other camera.

| Route | Result |
| --- | --- |
| `/health` | Camera availability/errors, model and calibration status |
| `/video/{left,right}` | Raw MJPEG |
| `/detect/{left,right}` | YOLO MJPEG |
| `/detections?camera_id=left` | Timestamped per-camera detections |
| `/snapshot?camera_id=left` | Latest JPEG |
| `/objects` or `/detections/objects` | Combined observations, image URLs, validated stereo estimates |
| `/calibration/status` | Missing calibration, runtime mismatches and file workflow |
| `/calibration/video/{left,right}` | Raw calibration preview |
| `GET/POST /operation-mode` | Manual/automatic selection; POST `{"mode":"automatic"}` |

`/objects` keeps unmatched observations with `position: null`. Correspondences
must be mutually unique by class and calibrated geometry; successful pairs
include `camera_ids`, `observation_ids`, `position: {x_mm,y_mm,z_mm}`,
`coordinate_system`, `unit: "mm"`, and reprojection error. Sensor timestamps
must agree within the configured time limit. The cameras are timestamped, not
hardware synchronized; verify stationary objects first. No current calibration
is shipped. No historical measurements are silently reused.

A YOLO box centre is only an image-point estimate. Even valid stereo XYZ has
`grasp_valid: false` and `robot_coordinates: null`. There is no measured
shelf-to-robot transform inferred from axis names. The file must declare its
actual coordinate frame.

`GET/POST /operation-mode` reads and changes the running controller's mode
through its loopback mode/status API. Set `ROBOT_CONTROLLER_URL` if its port
differs from `http://127.0.0.1:8765`. Mode changes never grant permission or
command servos. When disconnected, the response marks the requested selection
as unapplied; it is not replayed when the controller reconnects. The controller
starts in manual, and switching modes revokes its existing permission.

Both `/kinematics/target` and `/kinematics/scan-pose` still return **409
MOVEMENT_NOT_PERMITTED**. The mode/status link does not provide an automatic
motion driver or verify the emergency-stop circuit, and box centres are not
validated grasp points. Even when manual movement permission appears in the
nested controller status, the backend's vision `movement_allowed` remains false.
The public Kinematics repository also lacks the bridge's expected
`kinematics_hardware_runner.py`; automatic pickup needs those dependencies and
full-plan reachability checks integrated and tested.

## Raspberry Pi configuration

Keep the existing Deployment services and Python environment. Picamera2 and
libcamera come from the Pi system installation; the environment must expose
those packages. Install backend requirements in that environment. Calibration
also needs NumPy and OpenCV with `cv2.aruco.CharucoDetector` (OpenCV 4.8+).
Do not install a second conflicting OpenCV package over the system version.

Existing `MODEL_PATH`, `YOLO_CONF`, `CAMERA_WIDTH`, `CAMERA_HEIGHT`, and
`ROTATE_CAMERA_180` environment variables still work. The model defaults to
existing `best.pt`, otherwise `models/best.pt`.

Configure in the existing backend `.env` (paths are relative to this repository):

```dotenv
CAMERA_LEFT_INDEX=0
CAMERA_RIGHT_INDEX=1
CAMERA_LEFT_ENABLED=true
CAMERA_RIGHT_ENABLED=true
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720
STEREO_CALIBRATION_PATH=calibration/stereo_calibration.json
MAX_FRAME_AGE_SECONDS=5
STEREO_MAX_TIME_SKEW_SECONDS=0.10
```

Verify physical camera indices with `Picamera2.global_camera_info()` before
using these defaults. Each side also supports `WIDTH`, `HEIGHT`, `ROTATE_180`,
and `LENS_POSITION` under its `CAMERA_LEFT_` / `CAMERA_RIGHT_` prefix. Set the
lens positions to the measured calibration focus values; omitting them enables
normal autofocus for streams but makes stereo calibration unavailable. Disable
the right camera for single-camera deployments. Changing a lens, mounting,
sensor crop, resolution, orientation or index requires matching calibration.

See [the calibration workflow](calibration/README_STEREO_SETUP.md) for independent
ChArUco intrinsics, fixed-marker extrinsics and exact real-hardware checks.
Restart the backend after calibration/configuration changes; no recalibration
runs on startup. A disconnected camera is reported unavailable; restart to
reopen it. Cached preview images use bounded filenames rather than accumulating
new files on every UI poll. Runtime images and calibration outputs are ignored
by Git; existing calibration files remain untouched.

## Tests and hardware validation

```bash
python -m pytest tests/unit tests/test_stereo_localization.py -q
```

Mock/synthetic tests cover camera selection/failure/shutdown, model locking,
calibration rejection, distorted projection and triangulation, timing,
ambiguous matches, API compatibility and movement denial. Plain `pytest` skips
Pi integration tests. To run those explicitly on the Pi, stop the backend,
then run `RUN_HARDWARE_TESTS=1 python -m pytest tests/integration -v` (or the
existing `python tests/run_all_tests.py`). These tests open cameras and YOLO;
they do not command arm movement. Real-camera accuracy and pickup remain
unverified until measured hardware tests pass.
