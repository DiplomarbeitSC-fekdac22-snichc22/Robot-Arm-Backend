# tests/integration/test_camera_real.py

import numpy as np

from app.main import read_camera


def test_real_camera_returns_an_image():
    frame = read_camera()

    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.size > 0


def test_real_camera_returns_color_image():
    frame = read_camera()

    assert frame.ndim == 3
    assert frame.shape[2] == 3


def test_real_camera_image_is_not_completely_black():
    frame = read_camera()

    assert frame.mean() > 1