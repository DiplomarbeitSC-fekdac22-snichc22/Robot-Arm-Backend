import code
from email import message


class AppError(Exception):
    def __init__(self, code, message, status_code=500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code

class CameraUnavailableError(AppError):
    def __init__(self):
        super().__init__(
            code="CAMERA_UNAVAILABLE",
            message="Camera unavailable",
            status_code=503,
        )

class DetectionUnavailableError(AppError):
    def __init__(self):
        super().__init__(
            code="DETECTION_UNAVAILABLE",
            message="Detection disabled",
            status_code=503,
        )

class BackendTimeOutError(AppError):
    def __init__(self, reason="Robot movement not started"):
        super().__init__(
            code="MOVEMENT_REJECTED",
            message=reason,
            status_code=409
        )

class ImageSaveError(AppError):
    def __init__(self):
        super().__init__(
            code="IMAGE_SAVE_FAILED",
            message="Could not save camera image",
            status_code=500
        )