from __future__ import annotations

import contextvars
import json
import logging
from datetime import UTC, datetime
from typing import Any

request_id_context: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

_LOG_FIELDS = (
    "attempt",
    "archived_rows",
    "duration_ms",
    "error_code",
    "hits",
    "journal_mode",
    "max_tokens",
    "method",
    "mode",
    "path",
    "provider",
    "request_id",
    "status_code",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_context.get()
        if request_id:
            payload["request_id"] = request_id
        for field in _LOG_FIELDS:
            value = getattr(record, field, None)
            if value is not None:
                payload[field] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    logger = logging.getLogger("bazi_api")
    logger.setLevel(level)
    if not any(getattr(handler, "_bazi_json_handler", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._bazi_json_handler = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.propagate = False
