"""Tests for /live and /health probe endpoints."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, _gateway_is_ready
from app.models import HealthResponse


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


class TestGatewayIsReady:
    def test_requires_vllm_and_dynamodb(self):
        settings = Settings(rag_enabled=False, opensearch_endpoint="")
        health = HealthResponse(
            status="degraded",
            gateway="healthy",
            vllm_backend="unhealthy",
            dynamodb="healthy",
        )
        assert _gateway_is_ready(health, settings) is False

    def test_requires_opensearch_when_rag_on(self):
        settings = Settings(rag_enabled=True, opensearch_endpoint="https://os.example.com")
        health = HealthResponse(
            status="degraded",
            gateway="healthy",
            vllm_backend="healthy",
            dynamodb="healthy",
            opensearch="unhealthy",
        )
        assert _gateway_is_ready(health, settings) is False

    def test_ready_when_core_deps_up_and_rag_off(self):
        settings = Settings(rag_enabled=False, opensearch_endpoint="")
        health = HealthResponse(
            status="healthy",
            gateway="healthy",
            vllm_backend="healthy",
            dynamodb="healthy",
        )
        assert _gateway_is_ready(health, settings) is True


class TestLiveEndpoint:
    def test_always_returns_200(self, client):
        response = client.get("/live")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestHealthEndpoint:
    def _mock_deps(self, *, vllm_ok=True, ddb_ok=True, os_ok=True, rag_enabled=False):
        mock_http = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200 if vllm_ok else 503
        mock_http.get = AsyncMock(return_value=mock_response)

        mock_table = MagicMock()
        type(mock_table).table_status = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("ddb down"))
            if not ddb_ok
            else lambda self: "ACTIVE"
        )

        mock_session_mgr = MagicMock()
        mock_session_mgr.table = mock_table

        mock_os = MagicMock()
        mock_os.is_available.return_value = os_ok

        settings = Settings(
            rag_enabled=rag_enabled,
            opensearch_endpoint="https://os.example.com" if rag_enabled else "",
        )

        return mock_http, mock_session_mgr, mock_os, settings

    def test_returns_200_when_ready(self, client):
        mock_http, mock_session_mgr, mock_os, settings = self._mock_deps()
        with (
            patch("app.main.http_client", mock_http),
            patch("app.main.get_session_manager", return_value=mock_session_mgr),
            patch("app.main.get_settings", return_value=settings),
        ):
            response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "healthy"
        assert body["vllm_backend"] == "healthy"
        assert body["dynamodb"] == "healthy"

    def test_returns_503_when_vllm_down(self, client):
        mock_http, mock_session_mgr, mock_os, settings = self._mock_deps(vllm_ok=False)
        with (
            patch("app.main.http_client", mock_http),
            patch("app.main.get_session_manager", return_value=mock_session_mgr),
            patch("app.main.get_settings", return_value=settings),
        ):
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["vllm_backend"] == "unhealthy"

    def test_returns_503_when_dynamodb_down(self, client):
        mock_http, mock_session_mgr, mock_os, settings = self._mock_deps(ddb_ok=False)
        with (
            patch("app.main.http_client", mock_http),
            patch("app.main.get_session_manager", return_value=mock_session_mgr),
            patch("app.main.get_settings", return_value=settings),
        ):
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["dynamodb"] == "unhealthy"

    def test_returns_503_when_opensearch_down_and_rag_on(self, client):
        mock_http, mock_session_mgr, mock_os, settings = self._mock_deps(
            rag_enabled=True, os_ok=False
        )
        with (
            patch("app.main.http_client", mock_http),
            patch("app.main.get_session_manager", return_value=mock_session_mgr),
            patch("app.main.get_opensearch_rag", return_value=mock_os),
            patch("app.main.get_settings", return_value=settings),
        ):
            response = client.get("/health")
        assert response.status_code == 503
        assert response.json()["opensearch"] == "unhealthy"
