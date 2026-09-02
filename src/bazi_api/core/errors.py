from __future__ import annotations


class BaziApiError(Exception):
    """Base class for errors that are safe to expose through the API."""

    status_code = 500
    code = "internal_error"
    default_message = "服务内部错误"

    def __init__(self, message: str | None = None) -> None:
        self.public_message = message or self.default_message
        super().__init__(self.public_message)


class SessionNotFoundError(BaziApiError):
    status_code = 404
    code = "session_not_found"
    default_message = "会话不存在或已失效，请重新开始对话"


class FeedbackTargetError(BaziApiError):
    status_code = 422
    code = "invalid_feedback_target"
    default_message = "反馈目标不属于该会话"


class FeedbackRateLimitError(BaziApiError):
    status_code = 429
    code = "feedback_rate_limited"
    default_message = "反馈提交过于频繁，请稍后再试"


class ServiceUnavailableError(BaziApiError):
    status_code = 503
    code = "service_unavailable"
    default_message = "服务暂未配置或不可用"


class UpstreamServiceError(BaziApiError):
    status_code = 502
    code = "upstream_service_error"
    default_message = "上游模型服务暂时不可用"


class InvalidUpstreamResponseError(UpstreamServiceError):
    code = "invalid_upstream_response"
    default_message = "上游模型返回了无法识别的响应"
