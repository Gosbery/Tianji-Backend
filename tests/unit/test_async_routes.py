import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from bazi_api.core.config import Settings
from bazi_api.modules.feedback.router import create_feedback, resolve_feedback_client_key
from bazi_api.modules.feedback.schemas import FeedbackInput
from bazi_api.modules.observability.router import recent_traces


@pytest.mark.asyncio
async def test_database_routes_offload_blocking_repository_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def tracked_to_thread(function: Any, /, *args: object) -> Any:
        calls.append((function, args))
        return function(*args)

    monkeypatch.setattr("asyncio.to_thread", tracked_to_thread)

    feedback_calls: list[tuple[object, ...]] = []

    def add_feedback(*args: object) -> None:
        feedback_calls.append(args)

    def recent(limit: int) -> list[dict[str, object]]:
        return [
            {
                "id": "trace",
                "session_id": "session",
                "question": "question",
                "mode": "hybrid",
                "evidence_scope": "reviewed_only",
                "model_version": "model",
                "index_version": "index",
                "hits": [],
                "latency_ms": 1,
                "token_usage": None,
                "created_at": "2026-09-01T00:00:00+00:00",
            }
        ]

    container = SimpleNamespace(
        settings=Settings(),
        feedback=SimpleNamespace(add=add_feedback),
        traces=SimpleNamespace(recent=recent),
    )
    payload = FeedbackInput(
        session_id=uuid.uuid4(),
        message_id=uuid.uuid4(),
        rating=1,
        note="ok",
    )
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/v1/feedback",
            "headers": [],
            "client": ("203.0.113.9", 12345),
        }
    )

    response = await create_feedback(payload, request, container)
    traces = await recent_traces(5, container)

    assert response.status_code == 204
    assert feedback_calls == [
        (str(payload.session_id), str(payload.message_id), 1, "ok", "203.0.113.9")
    ]
    assert traces[0].id == "trace"
    assert [call[1] for call in calls] == [feedback_calls[0], (5,)]


def test_feedback_proxy_identity_requires_trusted_peer_secret_and_single_ip() -> None:
    settings = Settings(
        internal_proxy_secret="proxy-secret",
        trusted_proxy_cidrs=["127.0.0.1/32", "::1/128"],
    )

    def request(peer: str, secret: str, client_ip: str) -> Request:
        return Request(
            {
                "type": "http",
                "headers": [
                    (b"x-bazi-proxy-secret", secret.encode()),
                    (b"x-bazi-client-ip", client_ip.encode()),
                ],
                "client": (peer, 12345),
            }
        )

    assert resolve_feedback_client_key(
        request("127.0.0.1", "proxy-secret", "2001:db8::7"), settings
    ) == "2001:db8::7"
    assert resolve_feedback_client_key(
        request("203.0.113.8", "forged", "198.51.100.4"), settings
    ) == "203.0.113.8"
    with pytest.raises(HTTPException) as exc_info:
        resolve_feedback_client_key(
            request("127.0.0.1", "forged", "198.51.100.4, 198.51.100.5"), settings
        )
    assert exc_info.value.status_code == 403
