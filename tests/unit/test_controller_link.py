import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app import main


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setattr(main, "_mode", "manual")
    return TestClient(main.app)


def test_disconnected_selection_is_not_reported_as_applied(api, monkeypatch):
    monkeypatch.setattr(main, "_controller_request", Mock(side_effect=OSError("No controller")))
    response = api.post("/operation-mode", json={"mode": "automatic"})
    data = response.json()
    assert response.status_code == 200
    assert data["mode"] == "automatic" and data["requested_mode"] == "automatic"
    assert data["controller_mode"] is None and data["controller_connected"] is False
    assert data["controller"] is None and data["movement_allowed"] is False
    assert "not been applied" in data["reason"]


def test_live_status_does_not_inherit_manual_permission_for_vision(api, monkeypatch):
    status = {"mode": "manual", "armed": True, "permission": "setup", "movement_allowed": True}
    request = Mock(return_value=status)
    monkeypatch.setattr(main, "_controller_request", request)
    data = api.get("/operation-mode").json()
    request.assert_called_once_with(None)
    assert data["controller_connected"] is True and data["controller_mode"] == "manual"
    assert data["controller"]["movement_allowed"] is True
    assert data["movement_allowed"] is False


def test_post_forwards_only_the_mode(api, monkeypatch):
    request = Mock(return_value={"mode": "automatic", "armed": False, "automatic_ready": False})
    monkeypatch.setattr(main, "_controller_request", request)
    data = api.post("/operation-mode", json={"mode": "automatic"}).json()
    request.assert_called_once_with("automatic")
    assert data["mode"] == "automatic" and data["controller_connected"] is True
    assert data["movement_allowed"] is False


def test_unavailable_selection_is_not_replayed_on_reconnection(api, monkeypatch):
    request = Mock(side_effect=[OSError("Disconnected"), {"mode": "manual", "armed": False}])
    monkeypatch.setattr(main, "_controller_request", request)
    api.post("/operation-mode", json={"mode": "automatic"})
    data = api.get("/operation-mode").json()
    assert data["mode"] == "manual" and data["controller_mode"] == "manual"
    assert request.call_args_list[1].args == (None,)


@pytest.mark.parametrize("url", ["https://127.0.0.1:8765", "http://192.168.0.2:8765", "http://user@localhost:8765", "http://localhost:8765/target"])
def test_control_link_rejects_nonlocal_or_ambiguous_urls(monkeypatch, url):
    monkeypatch.setenv("ROBOT_CONTROLLER_URL", url)
    with pytest.raises(ValueError):
        main._controller_request()


def test_end_to_end_mode_forwarding_without_hardware(api, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "Robot-Controller" / "control_api.py"
    if not path.exists():
        pytest.skip("Controller checkout is needed for the cross-repository mode-link test")
    spec = importlib.util.spec_from_file_location("robot_control_api_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    status = {"mode": "manual", "armed": False, "automatic_ready": False, "movement_allowed": False}
    requests = []

    def change(mode):
        requests.append(mode)
        if mode == "manual":
            raise RuntimeError("Mode change interrupted")
        status["mode"] = mode
        return dict(status)

    server = module.start_control_api(lambda: dict(status), change, port=0)
    monkeypatch.setenv("ROBOT_CONTROLLER_URL", f"http://127.0.0.1:{server.server_port}")
    try:
        data = api.post("/operation-mode", json={"mode": "automatic"}).json()
        assert requests == ["automatic"]
        assert data["controller_connected"] is True and data["mode"] == "automatic"
        assert data["movement_allowed"] is False and data["controller"]["armed"] is False
        rejected = api.post("/operation-mode", json={"mode": "manual"}).json()
        assert rejected["mode"] == "automatic" and rejected["requested_mode"] == "manual"
        assert rejected["controller_connected"] is True and "rejected" in rejected["reason"]
        assert api.post("/kinematics/scan-pose").status_code == 409
    finally:
        server.shutdown()
        server.server_close()
