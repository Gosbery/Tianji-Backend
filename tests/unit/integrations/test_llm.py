import json
import re
from datetime import date, time

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.core.errors import InvalidUpstreamResponseError, UpstreamServiceError
from bazi_api.integrations.llm import (
    AnswerGenerator,
    GenerationResult,
    classify_question_policy,
)
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("我何年暴富？", "evidence_answer"),
        ("我的婚事肯定黄吗？", "evidence_answer"),
        ("我啥时候归西？", "evidence_answer"),
        ("这场诉讼胜券在握不？", "evidence_answer"),
        ("金是否可以直接解释成财富？", "evidence_answer"),
        ("《子平真诠》怎样解释正官？", "evidence_answer"),
    ],
)
def test_question_policy_allows_every_topic(
    question: str, expected: str
) -> None:
    assert classify_question_policy(question) == expected


def chart():
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="隐私姓名")
    )


def test_system_prompt_requires_readable_markdown_layout() -> None:
    prompt = AnswerGenerator._system_prompt("基础共识", "reviewed_only")

    assert "## 标题" in prompt
    assert "每段最多三句话" in prompt
    assert "每项单独成行" in prompt
    assert "禁止把“一、二、三”多个部分连续写在同一段" in prompt
    assert "以上分析步骤不得展示给用户" in prompt
    assert "候选依据与逐项检验" in prompt
    assert "资料审核层级只在证据面板展示" in prompt


def test_chart_context_hides_internal_luck_status_values() -> None:
    context = AnswerGenerator._chart_context(chart())

    assert '"status"' not in context
    assert '"current"' not in context
    assert '"future"' not in context
    assert '"pattern_candidates"' not in context
    assert "候选" not in context


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

    settings = Settings(llm_provider="openai", openai_api_key="test-key", http_request_retries=2)
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
    assert payload["thinking"] == {"type": "disabled"}
    assert "system" in payload and "response_format" not in payload
    assert result.answer == "有依据的回答 [1]"
    assert result.token_usage == 42


@pytest.mark.asyncio
async def test_streaming_buffers_until_answer_passes_validation() -> None:
    captured: dict[str, object] = {}
    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps({
                    "answer": "有依据的回答 [1]",
                    "uncertainties": ["仍有边界"],
                    "followups": ["继续追问"],
                }, ensure_ascii=False)}}],
                "usage": {"total_tokens": 42},
            },
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
    assert "stream" not in captured["payload"]  # type: ignore[operator]


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

    settings = Settings(llm_provider="openai", openai_api_key="test-key", http_request_retries=2)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(UpstreamServiceError):
            await AnswerGenerator(settings, client).generate("如何理解正官？", chart(), [hit()])

    assert requests == 3


@pytest.mark.asyncio
async def test_llm_rejects_malformed_success_envelope() -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    settings = Settings(llm_provider="openai", openai_api_key="test-key", http_request_retries=0)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(InvalidUpstreamResponseError):
            await AnswerGenerator(settings, client).generate("如何理解正官？", chart(), [hit()])


@pytest.mark.asyncio
async def test_no_evidence_skips_model_but_prediction_request_calls_it() -> None:
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
                                    "uncertainties": ["具体结果仍受现实选择影响"],
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

    assert requests == 1
    assert no_evidence.policy_decision == "refuse_no_evidence"
    assert no_evidence.question_policy == "evidence_answer"
    assert no_evidence.degradation_reason == "no_evidence"
    assert high_risk.policy_decision == "allow"
    assert high_risk.question_policy == "evidence_answer"
    assert high_risk.answer == "明年可能发财 [1]"
    assert no_evidence.citations_validated
    assert high_risk.citations_validated


@pytest.mark.asyncio
async def test_llm_output_without_valid_citations_falls_back_after_one_repair() -> None:
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
    assert requests == 2
    assert "[1] **证据**" in result.answer
    assert result.citations_validated
    assert result.degradation_reason == "model_output_failed_validation"


@pytest.mark.asyncio
async def test_llm_repairs_out_of_range_citation_once() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        answer = "越界引用 [99]" if requests == 1 else "已修复引用 [1]"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({
                "answer": answer,
                "uncertainties": ["仍需结合更多资料"],
                "followups": [],
            }, ensure_ascii=False)}}]},
            request=request,
        )

    settings = Settings(llm_provider="openai", openai_api_key="test-key")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "如何理解正官？", chart(), [hit()]
        )

    assert requests == 2
    assert result.answer == "已修复引用 [1]"
    assert result.citations_validated
    assert result.degradation_reason == ""


def test_citation_validation_normalizes_named_bracket_labels() -> None:
    result = AnswerGenerator._validate_model_generation(
        GenerationResult(
            answer="命盘事实 [pattern_candidates]，资料依据 [1]。",
            uncertainties=["仍需核对"],
            followups=[],
        ),
        evidence_count=1,
    )

    assert result.answer == "命盘事实 （格局参考），资料依据 [1]。"
    assert result.citations_validated


def test_generation_validation_removes_internal_luck_labels_from_user_text() -> None:
    result = AnswerGenerator._validate_model_generation(
        GenerationResult(
            answer=(
                "当前辛亥（2018—2027，状态current），下一步壬子"
                "（2028—2037，status=future） [1]。"
            ),
            uncertainties=["current_cycle 仍需结合现实"],
            followups=["继续分析 next_cycle"],
        ),
        evidence_count=1,
    )

    assert result.answer == "当前辛亥（2018—2027），下一步壬子（2028—2037） [1]。"
    assert result.uncertainties == ["当前大运 仍需结合现实"]
    assert result.followups == ["继续分析 下一步大运"]


def test_generation_validation_removes_review_and_candidate_language() -> None:
    result = AnswerGenerator._validate_model_generation(
        GenerationResult(
            answer=(
                "月令酉金偏印当令（machine_verified 候选，未定成格），感情推进偏慢 [1]。"
                "偏印格候选为机器校勘候选，未定成格。整体仍需结合大运。"
            ),
            uncertainties=["仍需结合现实"],
            followups=[],
        ),
        evidence_count=1,
    )

    assert "感情推进偏慢 [1]" in result.answer
    assert "整体仍需结合大运" in result.answer
    assert not re.search(r"machine_verified|候选|未定成格|机器校勘", result.answer)


@pytest.mark.asyncio
async def test_llm_allows_deterministic_prediction_with_valid_citation() -> None:
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
                                    "uncertainties": ["实际结果仍可能受到现实条件影响"],
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
    assert result.degradation_reason == ""


@pytest.mark.asyncio
async def test_llm_allows_bazi_prediction_answer() -> None:
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
                                        "本盘正财格显示财富积累较稳，婚姻中重视现实基础 [1]。"
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
async def test_llm_allows_personal_assertion_with_bazi_terms() -> None:
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
                                    "uncertainties": ["现实结果仍可能变化"],
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
    assert result.degradation_reason == ""
