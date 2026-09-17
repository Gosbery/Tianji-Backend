"""访问边界测试。

访问控制中间件已按用户决定移除，本文件不再断言 401/503/403 之类的认证行为，
只固定移除之后仍然成立的边界：文档端点关闭、legacy 镜像默认关闭、
观测端点自带独立 key、v1 端点在无凭证下可达（这是有意为之）。
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bazi_api.core.config import Settings
from bazi_api.main import create_app


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    return Settings(
        _env_file=None,
        database_path=tmp_path / "app.db",
        openai_api_key="",
        **overrides,  # type: ignore[arg-type]
    )


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(create_app(_settings(tmp_path))) as client:
        yield client


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_endpoints_are_disabled(client, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/health"),
        ("GET", "/api/tasks"),
        ("POST", "/api/chart"),
        ("POST", "/api/feedback"),
    ],
)
def test_legacy_api_mirror_is_disabled_by_default(client, method: str, path: str) -> None:
    response = client.request(method, path, json={} if method == "POST" else None)
    assert response.status_code == 404, (method, path, response.text)
    assert "Deprecation" not in response.headers
    assert "Sunset" not in response.headers


def test_legacy_api_mirror_can_be_enabled_explicitly(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path, legacy_api_enabled=True))) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.headers["Deprecation"] == "true"
    assert response.headers["Link"] == '</api/v1/health>; rel="successor-version"'
    assert response.headers["Cache-Control"] == "private, no-store"


def test_observability_needs_its_own_key_and_is_hidden_without_one(tmp_path: Path) -> None:
    # 未配置 OBSERVABILITY_API_KEY 时端点整体不可见。
    with TestClient(create_app(_settings(tmp_path))) as client:
        assert client.get("/api/v1/observability/recent").status_code == 404

    configured = _settings(tmp_path, observability_api_key="test-observability-key")
    with TestClient(create_app(configured)) as client:
        assert client.get("/api/v1/observability/recent").status_code == 401
        authorized = client.get(
            "/api/v1/observability/recent",
            headers={"Authorization": "Bearer test-observability-key"},
        )
        assert authorized.status_code == 200
        assert authorized.json() == []


def test_v1_endpoints_are_reachable_without_credentials(client) -> None:
    """访问控制被有意移除：v1 端点不再要求凭证，边界靠回环绑定与网络隔离维持。"""
    created = client.post(
        "/api/v1/tasks",
        json={"birth": {"date": "1990-01-01", "time": "12:00:00", "name": "private-name"}},
    )
    assert created.status_code == 201
    task_id = created.json()["id"]

    for method, path in (
        ("GET", "/api/v1/health"),
        ("GET", "/api/v1/tasks"),
        ("GET", f"/api/v1/tasks/{task_id}"),
        ("POST", f"/api/v1/tasks/{task_id}/archive"),
        ("POST", "/api/v1/chart"),
    ):
        payload = {"date": "1990-01-01", "time": "12:00:00"} if method == "POST" else None
        response = client.request(method, path, json=payload)
        assert response.status_code in {200, 201, 204}, (method, path, response.text)
        assert "no-store" in response.headers["Cache-Control"]

    detail = client.get(f"/api/v1/tasks/{task_id}")
    assert detail.json()["archived"] is True
    assert detail.json()["messages"] == []


def test_application_settings_no_longer_expose_an_access_key() -> None:
    """APP_ACCESS_KEY 已随访问控制一起移除，配置项不应再存在。"""
    assert "app_access_key" not in Settings.model_fields
