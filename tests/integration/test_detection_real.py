from app.camera import read_camera
from app.detection import model, run_detection


EXPECTED_CLASSES = {
    0: "Ball",
    1: "Car",
    2: "Manner",
}


def test_real_model_has_expected_classes():
    assert model.names == EXPECTED_CLASSES


def test_real_detection_runs_without_crashing():
    frame = read_camera()
    result = run_detection(frame)

    assert result is not None
    assert hasattr(result, "boxes")


def test_detection_result_contains_class_names():
    frame = read_camera()
    result = run_detection(frame)

    assert result.names == EXPECTED_CLASSES
