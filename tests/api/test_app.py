from fastapi.testclient import TestClient
import pytest

from src.api.app import app


@pytest.fixture
def client(monkeypatch):
    async def fake_run_memorability(image_path: str) -> float:
        return 0.42

    async def fake_run_feedback(image_path: str) -> str:
        return "Keep the framing tight around the subject."

    monkeypatch.setattr("src.api.app.run_memorability", fake_run_memorability)
    monkeypatch.setattr("src.api.app.run_feedback", fake_run_feedback)
    monkeypatch.setattr("src.api.app.get_memorability_model", lambda: None)
    monkeypatch.setattr("src.api.app.get_feedback_manager", lambda: None)
    monkeypatch.setattr(
        "src.api.app._save_request_artifacts", lambda *args, **kwargs: None
    )

    with TestClient(app) as test_client:
        yield test_client


def test_score_endpoint(client):
    response = client.post(
        "/score",
        files={"image": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == pytest.approx(0.42)
    assert "latency_ms" in payload


def test_score_feedback_endpoint(client):
    response = client.post(
        "/score-feedback",
        files={"image": ("sample.jpg", b"fake-image-bytes", "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["score"] == pytest.approx(0.42)
    assert payload["feedback"] == "Keep the framing tight around the subject."
    assert "feedback_latency_ms" in payload
