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


class SessionContextMismatchError(BaziApiError):
    status_code = 409
    code = "session_context_mismatch"
    default_message = "命盘、流派或证据范围已改变，请开始新的对话"


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


class ExpertNotFoundError(BaziApiError):
    status_code = 404
    code = "expert_not_found"
    default_message = "专家方法不存在或不在当前审核范围"


class TaskNotFoundError(BaziApiError):
    status_code = 404
    code = "task_not_found"
    default_message = "任务不存在"


class TaskBusyError(BaziApiError):
    status_code = 409
    code = "task_busy"
    default_message = "该任务正在排队或生成，请完成后再提问"


class InvalidJobStateError(BaziApiError):
    status_code = 409
    code = "invalid_job_state"
    default_message = "当前生成状态不支持此操作"
