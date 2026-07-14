from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_returns_200():
    response = client.get("/health")

    assert response.status_code == 200


def test_health_returns_json():
    response = client.get("/health")

    assert isinstance(response.json(), dict)


def test_unknown_endpoint_returns_404():
    response = client.get("/does-not-exist")

    assert response.status_code == 404