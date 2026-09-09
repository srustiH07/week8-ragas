from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_positive_prediction():
    response = client.post(
        "/predict",
        json={"text": "This is an excellent product"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "positive"


def test_negative_prediction():
    response = client.post(
        "/predict",
        json={"text": "This is a terrible product"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "negative"


def test_neutral_prediction():
    response = client.post(
        "/predict",
        json={"text": "This is a product"},
    )

    assert response.status_code == 200
    assert response.json()["prediction"] == "neutral"