import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bazi_api.core.config import Settings
from bazi_api.main import create_app

ACCESS_KEY = "test-application-access-key-32-characters"


@pytest.fixture
def client(tmp_path: Path):
    settings = Settings(
        _env_file=None,
        app_access_key=ACCESS_KEY,
        database_path=tmp_path / "app.db",
        embedding_cache_path=tmp_path / "embeddings.db",
        embedding_provider="hash",
        reranker_provider="lexical",
        vector_backend="memory",
        openai_api_key="",
    )
    with TestClient(create_app(settings)) as client:
        yield client


def test_private_endpoints_reject_anonymous_access_before_reading_or_writing(client) -> None:
    headers = {"X-Bazi-Access-Key": ACCESS_KEY}
    created = client.post(
        "/api/v1/tasks",
        headers=headers,
        json={"birth": {"date": "1990-01-01", "time": "12:00:00", "name": "private-name"}},
    )
    assert created.status_code == 201
    task_id = created.json()["id"]
    for prefix in ("/api/v1", "/api"):
        for method, path in (
            ("GET", "/tasks"),
            ("GET", f"/tasks/{task_id}"),
            ("GET", "/tasks/events"),
            ("GET", f"/conversations/{task_id}"),
            ("POST", "/tasks"),
            ("PATCH", f"/tasks/{task_id}"),
            ("POST", f"/tasks/{task_id}/archive"),
            ("POST", f"/tasks/{task_id}/restore"),
            ("POST", f"/tasks/{task_id}/messages"),
            ("POST", f"/tasks/{task_id}/jobs/job/cancel"),
            ("POST", f"/tasks/{task_id}/jobs/job/retry"),
            ("POST", "/chat"),
            ("POST", "/chat/stream"),
            ("POST", "/feedback"),
            ("GET", "/observability/recent"),
        ):
            response = client.request(method, prefix + path)
            assert response.status_code == 401, (method, path, response.text)
            assert "private-name" not in response.text
            assert response.headers["WWW-Authenticate"].startswith("Basic ")
            assert "no-store" in response.headers["Cache-Control"]

    detail = client.get(f"/api/v1/tasks/{task_id}", headers=headers)
    assert detail.status_code == 200
    assert detail.headers["Cache-Control"] == "private, no-store"
    assert detail.json()["archived"] is False
    assert detail.json()["messages"] == []
    assert client.get("/api/v1/tasks", auth=("bazi", ACCESS_KEY)).status_code == 200
    assert client.get("/api/v1/tasks", auth=("other", ACCESS_KEY)).status_code == 401
    assert client.get("/api/v1/tasks", headers={"X-Bazi-Access-Key": "wrong"}).status_code == 401
    assert (
        client.get(
            "/api/v1/tasks", auth=("bazi", ACCESS_KEY), headers={"X-Bazi-Access-Key": "wrong"}
        ).status_code
        == 401
    )


@pytest.mark.parametrize("path", ["/api/v1/tasks", "/api/tasks", "/docs", "/openapi.json"])
def test_unconfigured_application_fails_closed(path: str) -> None:
    client = TestClient(create_app(Settings(_env_file=None, app_access_key="")))
    response = client.get(path, auth=("bazi", ACCESS_KEY))
    assert response.status_code == 503
    assert response.json()["code"] == "access_unconfigured"


@pytest.mark.parametrize(
    "authorization",
    [
        "Basic not-base64!",
        "Basic /w==",
        "Bearer token",
        "Basic",
        "",
        "Basic " + base64.b64encode(b"bazi").decode(),
    ],
)
def test_malformed_credentials_are_rejected(authorization: str) -> None:
    client = TestClient(create_app(Settings(_env_file=None, app_access_key=ACCESS_KEY)))
    response = client.get("/api/v1/tasks", headers={"Authorization": authorization})
    assert response.status_code == 401


def test_duplicate_credentials_are_rejected() -> None:
    client = TestClient(create_app(Settings(_env_file=None, app_access_key=ACCESS_KEY)))
    response = client.get(
        "/api/v1/tasks",
        headers=[("X-Bazi-Access-Key", ACCESS_KEY), ("X-Bazi-Access-Key", "wrong")],
    )
    assert response.status_code == 401


def test_cors_preflight_does_not_require_or_grant_access() -> None:
    client = TestClient(create_app(Settings(_env_file=None, app_access_key=ACCESS_KEY)))
    headers = {
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "x-bazi-access-key",
    }
    assert client.options("/api/v1/tasks", headers=headers).status_code == 200
    assert client.get("/api/v1/tasks", headers={"Origin": headers["Origin"]}).status_code == 401


def test_access_key_must_be_strong_and_is_not_in_settings_repr() -> None:
    with pytest.raises(ValueError, match="至少"):
        Settings(_env_file=None, app_access_key="weak")
    assert ACCESS_KEY not in repr(Settings(_env_file=None, app_access_key=ACCESS_KEY))


@pytest.mark.parametrize(
    "headers",
    [
        {"Origin": "https://untrusted.example"},
        {"Origin": "null"},
        {"Sec-Fetch-Site": "cross-site"},
        {"Origin": "http://localhost:3000", "Sec-Fetch-Site": "cross-site"},
    ],
)
def test_authenticated_cross_site_writes_are_rejected(headers: dict[str, str]) -> None:
    client = TestClient(create_app(Settings(_env_file=None, app_access_key=ACCESS_KEY)))
    for path in ("/api/v1/tasks/task/archive", "/api/tasks/task/jobs/job/cancel"):
        response = client.post(path, headers=headers, auth=("bazi", ACCESS_KEY))
        assert response.status_code == 403
        assert response.json()["code"] == "cross_site_request"


def test_browser_same_origin_and_trusted_frontend_writes_are_allowed(client) -> None:
    for origin in ("http://testserver", "http://localhost:3000"):
        response = client.post(
            "/api/v1/chart",
            headers={"Origin": origin, "Sec-Fetch-Site": "same-origin"},
            auth=("bazi", ACCESS_KEY),
            json={"date": "1990-01-01", "time": "12:00:00"},
        )
        assert response.status_code == 200
