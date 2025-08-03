import pytest
from fastapi.testclient import TestClient
from evolvattention.server import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_barycenter_and_cosine():
    # create barycenter
    r = client.post("/api/v1/barycenter", json={"strings": ["a", "b"]})
    assert r.status_code == 200

    # query cosine similarities
    r = client.post("/api/v1/cosine-similarities", json={"strings": ["c"]})
    assert r.status_code == 200
    payload = r.json()
    assert payload["status"] == "success"
    assert payload["data"]["similarities"] == [0.8]
