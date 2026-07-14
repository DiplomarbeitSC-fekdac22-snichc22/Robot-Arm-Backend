import sys
from unittest.mock import MagicMock, patch


# Fake picamera2 module
fake_picamera2_module = MagicMock()
fake_camera = MagicMock()
fake_picamera2_module.Picamera2.return_value = fake_camera

sys.modules["picamera2"] = fake_picamera2_module


# Fake libcamera module
fake_libcamera_module = MagicMock()
fake_libcamera_module.Transform = MagicMock()
fake_libcamera_module.controls.AfModeEnum.Continuous = "continuous"

sys.modules["libcamera"] = fake_libcamera_module


# Prevent the actual YOLO model from loading
_yolo_patcher = patch("ultralytics.YOLO")
mock_yolo_class = _yolo_patcher.start()

fake_model = MagicMock()
fake_model.names = {
    0: "Ball",
    1: "Car",
    2: "Manner",
}

mock_yolo_class.return_value = fake_model