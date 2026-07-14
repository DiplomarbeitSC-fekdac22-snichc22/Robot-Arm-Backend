# tests/integration/test_detection_real.py

from app.main import model, read_camera, run_detection


def test_real_model_has_expected_classes():
    assert model.names == {
        0: "Ball",
        1: "Car",
        2: "Manner",
    }


def test_real_detection_runs_without_crashing():
    frame = read_camera()
    result = run_detection(frame)

    assert result is not None
    assert hasattr(result, "boxes")


def test_detection_result_contains_class_names():
    frame = read_camera()
    result = run_detection(frame)

    assert result.names == {
        0: "Ball",
        1: "Car",
        2: "Manner",
    }