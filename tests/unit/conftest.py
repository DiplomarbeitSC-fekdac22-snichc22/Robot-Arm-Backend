from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_model(monkeypatch):
    model = MagicMock()
    model.names = {0: "Ball", 1: "Car", 2: "Manner"}
    monkeypatch.setattr("app.detection.model", model)
    return model
