"""Tests for generic error responses."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


def test_unhandled_exception_does_not_leak_details():
    settings = Settings(api_key="test-key")
    with patch("app.main.get_settings", return_value=settings):
        with TestClient(app, raise_server_exceptions=False) as client:
            with patch(
                "app.main.get_session_manager",
                side_effect=RuntimeError("secret internal table name"),
            ):
                response = client.post(
                    "/v1/sessions",
                    json={},
                    headers={"Authorization": "Bearer test-key"},
                )
    assert response.status_code == 500
    body = response.json()
    assert body == {"error": "Internal server error"}
    assert "secret internal" not in response.text
