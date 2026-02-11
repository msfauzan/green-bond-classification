"""Tests for API endpoints via FastAPI TestClient."""
import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webapp'))

from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Must import after path setup
    from webapp.api import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "timestamp" in data


class TestLabelsEndpoint:
    def test_labels_returns_200(self, client):
        response = client.get("/labels")
        assert response.status_code == 200
        data = response.json()
        assert "labels" in data
        assert len(data["labels"]) == 4


class TestClassifyEndpoint:
    def test_rejects_non_pdf(self, client):
        response = client.post(
            "/api/classify",
            files={"file": ("test.txt", b"not a pdf", "text/plain")}
        )
        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]

    def test_rejects_oversized_file(self, client):
        # Create a file just over 50MB
        large_content = b"%" + b"x" * (50 * 1024 * 1024 + 1)
        response = client.post(
            "/api/classify",
            files={"file": ("big.pdf", large_content, "application/pdf")}
        )
        assert response.status_code == 413


class TestFeedbackEndpoints:
    def test_feedback_stats_returns_200(self, client):
        response = client.get("/api/feedback/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total_feedback" in data

    def test_feedback_export_returns_200(self, client):
        response = client.get("/api/feedback/export")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "data" in data


class TestPdfEndpoint:
    def test_invalid_file_id_rejected(self, client):
        # Characters like spaces and special chars should be rejected by regex
        response = client.get("/api/pdf/bad file id!@#")
        assert response.status_code == 400

    def test_safe_file_id_format(self, client):
        # Valid format but file doesn't exist
        response = client.get("/api/pdf/20250101120000_abc123def456")
        assert response.status_code == 404

    def test_delete_requires_auth_when_configured(self, client, monkeypatch):
        monkeypatch.setattr("webapp.api.API_KEY", "test-secret")
        response = client.delete("/api/pdf/nonexistent")
        assert response.status_code == 401

    def test_delete_works_without_auth_in_dev(self, client, monkeypatch):
        monkeypatch.setattr("webapp.api.API_KEY", None)
        response = client.delete("/api/pdf/nonexistent")
        # Should succeed (file not found is still a 200 with message)
        assert response.status_code == 200


class TestCleanupEndpoint:
    def test_cleanup_requires_auth(self, client, monkeypatch):
        monkeypatch.setattr("webapp.api.API_KEY", "test-secret")
        response = client.post("/api/cleanup")
        assert response.status_code == 401

    def test_cleanup_works_without_auth_in_dev(self, client, monkeypatch):
        monkeypatch.setattr("webapp.api.API_KEY", None)
        response = client.post("/api/cleanup")
        assert response.status_code == 200
