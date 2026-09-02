import json
from datetime import date, time

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.core.errors import InvalidUpstreamResponseError, UpstreamServiceError
from bazi_api.integrations.llm import (
    AnswerGenerator,
    classify_question_policy,
    requires_refusal,
)
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit


def test_prediction_policy_is_disabled() -> None:
    for question in (
        "我何年暴富？",
        "我的婚事肯定黄吗？",
        "我啥时候归西？",
        "这场诉讼胜券在握不？",
        "金是否可以直接解释成财富？",
        "《子平真诠》怎样解释正官？",
    ):
        assert classify_question_policy(question) == "evidence_answer"
        assert not requires_refusal(question)


def chart():
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="隐私姓名")
    )


def hit() -> RetrievalHit:
    return RetrievalHit(
        document=RetrievalDocument(
            id="evidence",
            kind="knowledge_card",
            layer=3,
            title="证据",
            text="证据内容",
            source="test",
            school="基础共识",
            concepts=[],
        ),
        score=1.0,
        matched_by=["test"],
    )


@pytest.mark.asyncio
async def test_llm_retries_then_falls_back_without_response_format() -> None:
    payloads: list[dict[str, object]] = []
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        idempotency_keys.append(request.headers["Idempotency-Key"])
        if len(payloads) == 1:
            return httpx.Response(503, headers={"Retry-After": "0"})
        if len(payloads) == 2:
            return httpx.Response(422)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "有依据的回答 [1]",
                                    "uncertainties": ["仍有边界"],
                                    "followups": ["继续追问"],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    settings = Settings(
        llm_provider="openai", openai_api_key="test-key", http_request_retries=2
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    assert result.answer == "有依据的回答 [1]"
    assert result.citations_validated
    assert result.token_usage == 42
    assert len(payloads) == 3
    assert "response_format" in payloads[0]
    assert "response_format" in payloads[1]
    assert "response_format" not in payloads[2]
    assert idempotency_keys[0] == idempotency_keys[1]
    assert idempotency_keys[1] != idempotency_keys[2]
    assert "隐私姓名" not in str(payloads)


@pytest.mark.asyncio
async def test_anthropic_provider_uses_messages_protocol() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["headers"] = dict(request.headers)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "answer": "有依据的回答 [1]",
                                "uncertainties": ["仍有边界"],
                                "followups": ["继续追问"],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
                "usage": {"input_tokens": 30, "output_tokens": 12},
            },
        )

    settings = Settings(
        llm_provider="anthropic",
        anthropic_auth_token="test-anthropic-key",
        anthropic_base_url="https://anthropic.invalid",
        anthropic_chat_model="claude-test",
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    headers = captured["headers"]
    payload = captured["payload"]
    assert captured["url"] == "https://anthropic.invalid/v1/messages"
    assert isinstance(headers, dict) and headers["x-api-key"] == "test-anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert isinstance(payload, dict) and payload["model"] == "claude-test"
    assert "system" in payload and "response_format" not in payload
    assert result.answer == "有依据的回答 [1]"
    assert result.token_usage == 42


@pytest.mark.asyncio
async def test_openai_streams_answer_field_and_returns_terminal_metadata() -> None:
    captured: dict[str, object] = {}
    model_chunks = [
        '{"answer":"有依据',
        '的回答 [1]","uncertainties":["仍有边界"],',
        '"followups":["继续追问"]}',
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        events = [
            "data: "
            + json.dumps({"choices": [{"delta": {"content": chunk}}]}, ensure_ascii=False)
            + "\n\n"
            for chunk in model_chunks
        ]
        events.append(
            f"data: {json.dumps({'choices': [], 'usage': {'total_tokens': 42}})}\n\n"
        )
        events.append("data: [DONE]\n\n")
        return httpx.Response(
            200,
            text="".join(events),
            headers={"Content-Type": "text/event-stream"},
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    streamed: list[str] = []

    async def collect(chunk: str) -> None:
        streamed.append(chunk)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate_stream(
            "如何理解正官？", chart(), [hit()], collect
        )

    assert "".join(streamed) == "有依据的回答 [1]"
    assert result.answer == "有依据的回答 [1]"
    assert result.uncertainties == ["仍有边界"]
    assert result.followups == ["继续追问"]
    assert result.token_usage == 42
    assert isinstance(captured["payload"], dict)
    assert captured["payload"]["stream"] is True  # type: ignore[index]


@pytest.mark.asyncio
async def test_anthropic_retries_when_thinking_exhausts_output_tokens() -> None:
    payloads: list[dict[str, object]] = []
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        idempotency_keys.append(request.headers["Idempotency-Key"])
        if len(payloads) == 1:
            return httpx.Response(
                200,
                json={
                    "content": [{"type": "thinking", "thinking": "分析中"}],
                    "stop_reason": "max_tokens",
                    "usage": {"input_tokens": 30, "output_tokens": 4096},
                },
            )
        return httpx.Response(
            200,
            json={
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(
                            {
                                "answer": "扩容后得到回答 [1]",
                                "uncertainties": ["仍有边界"],
                                "followups": [],
                            },
                            ensure_ascii=False,
                        ),
                    }
                ],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 30, "output_tokens": 80},
            },
        )

    settings = Settings(
        llm_provider="anthropic",
        anthropic_auth_token="test-anthropic-key",
        anthropic_base_url="https://anthropic.invalid",
        anthropic_chat_model="claude-test",
        llm_max_tokens=4096,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    assert result.answer == "扩容后得到回答 [1]"
    assert [payload["max_tokens"] for payload in payloads] == [4096, 8192]
    assert idempotency_keys[0] != idempotency_keys[1]


@pytest.mark.asyncio
async def test_llm_retry_count_is_bounded() -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(503, headers={"Retry-After": "0"})

    settings = Settings(
        llm_provider="openai", openai_api_key="test-key", http_request_retries=2
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UpstreamServiceError):
            await AnswerGenerator(settings, client).generate(
                "如何理解正官？", chart(), [hit()]
            )

    assert requests == 3


@pytest.mark.asyncio
async def test_llm_rejects_malformed_success_envelope() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    settings = Settings(
        llm_provider="openai", openai_api_key="test-key", http_request_retries=0
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InvalidUpstreamResponseError):
            await AnswerGenerator(settings, client).generate(
                "如何理解正官？", chart(), [hit()]
            )


@pytest.mark.asyncio
async def test_llm_calls_model_without_evidence() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "明年可能发财 [1]",
                                    "uncertainties": [],
                                    "followups": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        generator = AnswerGenerator(settings, client)
        no_evidence = await generator.generate("如何理解正官？", chart(), [])
        high_risk = await generator.generate("我何年暴富？", chart(), [hit()])

    assert requests == 2
    assert no_evidence.policy_decision == "allow"
    assert no_evidence.question_policy == "evidence_answer"
    assert no_evidence.answer == "明年可能发财 [1]"
    assert high_risk.policy_decision == "allow"
    assert high_risk.answer == "明年可能发财 [1]"
    assert no_evidence.citations_validated
    assert high_risk.citations_validated


@pytest.mark.asyncio
async def test_llm_output_without_valid_citations_is_returned() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "没有引用的确定结论",
                                    "uncertainties": [],
                                    "followups": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    assert result.policy_decision == "allow"
    assert result.answer == "没有引用的确定结论"
    assert not result.citations_validated


@pytest.mark.asyncio
async def test_llm_deterministic_prediction_is_returned_with_valid_citation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "明年中大奖，已经可以确定 [1]",
                                    "uncertainties": ["没有不确定性"],
                                    "followups": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    assert result.policy_decision == "allow"
    assert result.answer == "明年中大奖，已经可以确定 [1]"


@pytest.mark.asyncio
async def test_llm_allows_bazi_terms_with_prediction_boundary() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": (
                                        "本盘可讨论正财格 [1]，但不能据此预测现实财富或婚姻结果。"
                                    ),
                                    "uncertainties": ["格局判断仍需更多规则"],
                                    "followups": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "我是什么命格？", chart(), [hit()]
        )

    assert result.policy_decision == "allow"
    assert "正财格" in result.answer
    assert result.citations_validated


@pytest.mark.asyncio
async def test_llm_returns_personal_assertion_with_bazi_terms() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "answer": "命主明年一定发财 [1]。",
                                    "uncertainties": [],
                                    "followups": [],
                                },
                                ensure_ascii=False,
                            )
                        }
                    }
                ]
            },
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解财格？", chart(), [hit()]
        )

    assert result.policy_decision == "allow"
    assert result.answer == "命主明年一定发财 [1]。"
