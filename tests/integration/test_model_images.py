# tests/integration/test_model_images.py

from pathlib import Path

import cv2

from app.main import run_detection


FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def get_detected_classes(result) -> list[str]:
    detected_classes = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        class_name = result.names[class_id]
        detected_classes.append(class_name)

    return detected_classes


def test_back_of_car_is_detected_as_car():
    image_path = FIXTURES / "car_back.jpeg"
    frame = cv2.imread(str(image_path))

    assert frame is not None, f"Could not load {image_path}"

    result = run_detection(frame)
    detected_classes = get_detected_classes(result)

    assert "Car" in detected_classes


def test_back_of_car_is_not_detected_as_ball():
    image_path = FIXTURES / "car_back.jpeg"
    frame = cv2.imread(str(image_path))

    assert frame is not None

    result = run_detection(frame)
    detected_classes = get_detected_classes(result)

    assert "Ball" not in detected_classes

def test_manner_package_is_detected():
    image_path = FIXTURES / "manner.jpeg"
    frame = cv2.imread(str(image_path))

    assert frame is not None

    result = run_detection(frame)
    detected_classes = get_detected_classes(result)

    assert "Manner" in detected_classes

def test_single_manner_package_is_detected_once():
    image_path = FIXTURES / "manner.jpeg"
    frame = cv2.imread(str(image_path))

    assert frame is not None

    result = run_detection(frame)
    detected_classes = get_detected_classes(result)

    manner_count = detected_classes.count("Manner")

    assert manner_count == 1