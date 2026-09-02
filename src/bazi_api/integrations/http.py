from __future__ import annotations

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
# Connection establishment retries live in AsyncHTTPTransport; excluding them here keeps
# the configured attempt count from multiplying across two retry layers.
RETRYABLE_EXCEPTIONS = (
    httpx.PoolTimeout,
    httpx.ReadError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
    httpx.WriteError,
    httpx.WriteTimeout,
)


async def post_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    headers: dict[str, str],
    payload: dict[str, object],
    timeout: float,
    max_retries: int,
    operation: str,
) -> httpx.Response:
    max_attempts = max_retries + 1
    for attempt in range(1, max_attempts + 1):
        try:
            response = await client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except RETRYABLE_EXCEPTIONS:
            if attempt >= max_attempts:
                raise
            logger.warning(
                "upstream_request_retry",
                extra={"attempt": attempt, "provider": operation},
            )
            await asyncio.sleep(_retry_delay(None, attempt))
            continue

        if response.status_code not in RETRYABLE_STATUS_CODES or attempt >= max_attempts:
            return response
        logger.warning(
            "upstream_response_retry",
            extra={
                "attempt": attempt,
                "provider": operation,
                "status_code": response.status_code,
            },
        )
        await asyncio.sleep(_retry_delay(response.headers.get("Retry-After"), attempt))

    raise RuntimeError("unreachable retry state")


def _retry_delay(retry_after: str | None, attempt: int) -> float:
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 5.0)
        except ValueError:
            pass
    return min(0.25 * 2 ** (attempt - 1), 5.0)
