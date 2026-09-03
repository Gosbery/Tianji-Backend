import sqlite3
import tomllib
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bazi_api.core.config import Settings
from bazi_api.main import create_app

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def test_health_and_chart_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
        vector_backend="memory",
        embedding_provider="hash",
        reranker_provider="lexical",
        embedding_cache_path=tmp_path / "embedding-cache.sqlite3",
        observability_api_key="test-observability-key",
        llm_provider="openai",
        openai_api_key="",
    )

    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        health = client.get("/api/v1/health", headers={"X-Request-ID": "api-route-test-request"})
        legacy_health = client.get("/api/health")
        allowed_preflight = client.options(
            "/api/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "authorization,content-type",
            },
        )
        rejected_preflight = client.options(
            "/api/v1/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "DELETE",
            },
        )
        overview = client.get("/api/v1/knowledge/overview")
        originals = client.get("/api/v1/knowledge/originals")
        annotations = client.get("/api/v1/knowledge/annotations")
        graph = client.get("/api/v1/knowledge/graph")
        chart = client.post(
            "/api/v1/chart",
            json={
                "date": "1990-01-01",
                "time": "12:00:00",
                "timezone": "Asia/Shanghai",
                "name": "接口测试",
            },
        )
        invalid_timezone = client.post(
            "/api/v1/chart",
            json={
                "date": "1990-01-01",
                "time": "12:00:00",
                "timezone": "Mars/Olympus",
                "name": "接口测试",
            },
        )
        reviewed_chat = client.post(
            "/api/v1/chat",
            json={
                "chart": chart.json(),
                "question": "《子平真诠》如何讨论月令用神？",
                "mode": "hybrid",
            },
        )
        preview_chat = client.post(
            "/api/v1/chat",
            json={
                "chart": chart.json(),
                "question": "《子平真诠》如何讨论月令用神？",
                "mode": "hybrid",
                "evidence_scope": "personal_preview",
            },
        )
        unknown_session_chat = client.post(
            "/api/v1/chat",
            json={
                "chart": chart.json(),
                "question": "如何理解日主？",
                "session_id": str(uuid.uuid4()),
                "mode": "hybrid",
            },
        )
        invalid_session_chat = client.post(
            "/api/v1/chat",
            json={
                "chart": chart.json(),
                "question": "如何理解日主？",
                "session_id": "client-chosen-session",
                "mode": "hybrid",
            },
        )
        traces_without_key = client.get("/api/v1/observability/recent")
        traces = client.get(
            "/api/v1/observability/recent",
            headers={"Authorization": "Bearer test-observability-key"},
        )
        client.app.state.container.settings.observability_api_key = ""
        disabled_traces = client.get(
            "/api/v1/observability/recent",
            headers={"Authorization": "Bearer test-observability-key"},
        )
        feedback = client.post(
            "/api/v1/feedback",
            json={
                "session_id": preview_chat.json()["session_id"],
                "message_id": preview_chat.json()["message_id"],
                "rating": 1,
            },
        )
        updated_feedback = client.post(
            "/api/v1/feedback",
            json={
                "session_id": preview_chat.json()["session_id"],
                "message_id": preview_chat.json()["message_id"],
                "rating": -1,
                "note": "更新反馈",
            },
        )
        wrong_message_feedback = client.post(
            "/api/v1/feedback",
            json={
                "session_id": preview_chat.json()["session_id"],
                "message_id": str(uuid.uuid4()),
                "rating": 1,
            },
        )

        def fail_chart(_: object) -> None:
            raise KeyError("sensitive-internal-detail")

        monkeypatch.setattr(client.app.state.container.charts, "calculate", fail_chart)
        internal_chart_error = client.post(
            "/api/v1/chart",
            headers={
                "Origin": "http://localhost:3000",
                "X-Request-ID": "internal-error-request",
            },
            json={
                "date": "1990-01-01",
                "time": "12:00:00",
                "timezone": "Asia/Shanghai",
                "name": "接口测试",
            },
        )

    assert health.status_code == 200
    assert health.json()["llm_configured"] is False
    with (BACKEND_ROOT / "pyproject.toml").open("rb") as project_file:
        assert client.app.version == tomllib.load(project_file)["project"]["version"]
    assert health.json()["cards"] > 0
    assert health.json()["knowledge_layers"]["canonical_passages"] > 0
    assert legacy_health.status_code == 200
    assert legacy_health.headers["Deprecation"] == "true"
    assert legacy_health.headers["Sunset"] == settings.legacy_api_sunset
    assert legacy_health.headers["Link"] == '</api/v1/health>; rel="successor-version"'
    assert health.headers["X-Request-ID"] == "api-route-test-request"
    assert allowed_preflight.status_code == 200
    assert rejected_preflight.status_code == 400
    assert overview.status_code == 200
    assert overview.json()["layers"]["modern_annotations"] >= 10
    ziping = next(
        work for work in originals.json()["works"] if work["id"] == "work-ziping-zhenquan"
    )
    assert ziping["status"] == "machine_verified"
    ziping_passage = next(
        item for item in originals.json()["passages"] if item["id"].startswith("ziping-zhenquan-")
    )
    assert ziping_passage["status"] == "machine_verified"
    assert annotations.status_code == 200
    assert len(graph.json()["edges"]) >= 3
    assert chart.status_code == 200
    assert invalid_timezone.status_code == 422
    assert chart.json()["day_master"] == "丙"
    assert reviewed_chat.status_code == 200
    uuid.UUID(reviewed_chat.json()["session_id"])
    uuid.UUID(reviewed_chat.json()["message_id"])
    assert not any(
        item["review_status"] == "machine_verified" for item in reviewed_chat.json()["evidence"]
    )
    assert preview_chat.status_code == 200
    assert any(
        item["review_status"] == "machine_verified" for item in preview_chat.json()["evidence"]
    )
    assert unknown_session_chat.status_code == 404
    assert unknown_session_chat.json()["code"] == "session_not_found"
    assert invalid_session_chat.status_code == 422
    assert traces_without_key.status_code == 401
    assert traces.status_code == 200
    assert disabled_traces.status_code == 404
    assert traces.json()[0]["evidence_scope"] == "personal_preview"
    assert traces.json()[0]["model_version"].startswith("hash-")
    assert traces.json()[0]["index_version"]
    assert "hits_json" not in traces.json()[0]
    assert feedback.status_code == 204
    assert updated_feedback.status_code == 204
    assert wrong_message_feedback.status_code == 422
    assert internal_chart_error.status_code == 500
    assert internal_chart_error.json()["detail"] == "服务内部错误"
    assert internal_chart_error.json()["request_id"] == "internal-error-request"
    assert internal_chart_error.headers["X-Request-ID"] == internal_chart_error.json()["request_id"]
    assert internal_chart_error.headers["Access-Control-Allow-Origin"] == ("http://localhost:3000")
    assert "sensitive-internal-detail" not in internal_chart_error.text

    with sqlite3.connect(settings.database_path) as connection:
        stored_feedback = connection.execute(
            "SELECT rating, note FROM feedback WHERE message_id = ?",
            (preview_chat.json()["message_id"],),
        ).fetchall()
    assert stored_feedback == [(-1, "更新反馈")]


def test_health_reports_anthropic_configuration(tmp_path: Path) -> None:
    settings = Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        database_path=tmp_path / "app.db",
        qdrant_path=tmp_path / "qdrant",
        vector_backend="memory",
        embedding_provider="hash",
        reranker_provider="lexical",
        embedding_cache_path=tmp_path / "embedding-cache.sqlite3",
        llm_provider="anthropic",
        anthropic_auth_token="configured-secret",
    )

    with TestClient(create_app(settings)) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["llm_configured"] is True
