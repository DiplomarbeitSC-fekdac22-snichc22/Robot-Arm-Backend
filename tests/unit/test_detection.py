from unittest.mock import MagicMock, patch

import numpy as np

from app.main import run_detection


def test_run_detection_returns_first_result():
    fake_frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    fake_result = MagicMock()

    with patch("app.main.model") as mock_model:
        mock_model.predict.return_value = [fake_result]

        result = run_detection(fake_frame)

    assert result is fake_result
    mock_model.predict.assert_called_once()


def test_run_detection_uses_frame():
    fake_frame = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    with patch("app.main.model") as mock_model:
        mock_model.predict.return_value = [MagicMock()]

        run_detection(fake_frame)

    arguments = mock_model.predict.call_args.kwargs

    assert arguments["source"] is fake_frame
    assert arguments["verbose"] is False