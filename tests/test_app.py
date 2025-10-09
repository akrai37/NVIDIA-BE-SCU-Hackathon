from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == app.title
    assert body["version"] == app.version


def test_health_degraded_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_analyze_rejects_empty_payload():
    files = {"file": ("empty.pdf", b"", "application/pdf")}
    response = client.post("/v1/documents/analyze", files=files)
    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."
