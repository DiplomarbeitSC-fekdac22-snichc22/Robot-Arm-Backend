from app.cameras.base import CameraSource
from app.cameras.mock_camera import MockCameraSource
from app.cameras.picamera import PicameraSource
from app.core.config import Settings
from app.services.vision_service import VisionService
from app.vision.yolo_detector import YoloDetector


def create_camera(
    settings: Settings,
) -> CameraSource:
    if settings.use_mock_camera:
        return MockCameraSource(
            settings.mock_image_path,
        )

    return PicameraSource(
        camera_index=0,
        width=settings.camera_width,
        height=settings.camera_height,
        rotate_180=settings.rotate_camera_180,
    )


def create_vision_service(
    settings: Settings,
) -> VisionService:
    camera = create_camera(settings)

    detector = YoloDetector(
        model_path=settings.model_path,
        confidence=settings.yolo_confidence,
    )

    return VisionService(
        camera=camera,
        detector=detector,
    )