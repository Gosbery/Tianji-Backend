import json

import httpx
import pytest

from bazi_api.cli.evaluate import evaluate_generated_answer
from bazi_api.core.config import Settings
from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm_provider="openai",
        openai_api_key="mock-key",
        http_request_retries=0,
    )


def _chart():
    return ChartCalculator().calculate(
        BirthInput(date="1990-01-01", time="12:00:00", name="私密姓名")
    )


def _hit(text: str = "月令是出生月份的地支。") -> RetrievalHit:
    return RetrievalHit(
        document=RetrievalDocument(
            id="test-evidence",
            kind="knowledge_card",
            layer=3,
            title="月令的定义",
            text=text,
            source="test",
            school="基础共识",
            concepts=[],
        ),
        score=1.0,
        matched_by=["test"],
    )


def _completion(content: str | dict[str, object]) -> httpx.Response:
    text = content if isinstance(content, str) else json.dumps(content)
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": text}}],
            "usage": {"total_tokens": 7},
        },
    )


def _candidate(answer: str = "月令是出生月份的地支。[1]") -> dict[str, object]:
    return {"answer": answer, "uncertainties": ["个体判断仍需其他条件"], "followups": []}


def _verdict(**overrides: object) -> dict[str, object]:
    return {
        "supported": True,
        "safe": True,
        "complete": True,
        "issues": [],
        "missing_aspects": [],
        **overrides,
    }


def _is_verifier(request: httpx.Request) -> bool:
    return json.loads(request.content)["messages"][0]["content"].startswith(
        "你是独立的答案核验器"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "月令是出生月份的地支。[1]",
        "古籍将五行与身体部位联系起来。[1] 不能凭八字诊断疾病，有现实症状应咨询医生。",
        "你将在下一年进入下一步大运，具体事业趋势仍需综合其他条件。[1]",
    ],
)
async def test_answer_requires_independent_verification_with_complete_evidence(answer: str) -> None:
    calls = []
    evidence = "月令资料" * 400 + "例外：不得仅凭月令诊断疾病。"

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if _is_verifier(request):
            payload = json.loads(request.content)
            context = json.loads(payload["messages"][1]["content"])
            assert context["evidence"][0]["text"] == evidence
            assert context["answer"] == answer
            assert "私密姓名" not in request.content.decode()
            assert payload["temperature"] == 0
            return _completion(_verdict())
        return _completion(_candidate(answer))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate(
            "解释月令", _chart(), [_hit(evidence)]
        )

    assert len(calls) == 2
    assert result.answer == answer
    assert result.evidence_validated and result.safety_validated and result.citations_validated
    assert result.token_usage == 14


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dangerous",
    [
        "你一定会在明年确诊重病，应该停用医生开的药。[1]",
        "你已经患有肾病。[1]",
        "立即停止服用降压药。[1]",
        "建议每天服用这种药物。[1]",
        "命主明年一定发财。[1]",
        "你将在2030年去世。[1]",
    ],
)
@pytest.mark.parametrize("field", ["answer", "uncertainties", "followups"])
async def test_explicit_unsafe_content_is_blocked_even_with_permissive_verifier(
    dangerous: str, field: str
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert not _is_verifier(request)
        candidate = _candidate()
        candidate[field] = dangerous if field == "answer" else [dangerous]
        return _completion(candidate)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate(
            dangerous, _chart(), [_hit(dangerous)]
        )

    assert calls == 2
    assert result.policy_decision == "refuse_invalid_citations"
    assert dangerous not in result.answer
    assert not result.citations_validated
    assert not result.evidence_validated
    assert result.model_version == "safe-fallback"
    assert not all(evaluate_generated_answer(
        result, [_hit()], expected_policy="evidence_answer", expects_uncertainty=True
    ).values())


@pytest.mark.asyncio
async def test_unrelated_in_range_citation_is_rejected_and_stream_never_emits_draft() -> None:
    calls = 0
    chunks = []
    unsupported = "本盘财富积累稳定，感情十分顺利。[1]"

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert chunks == []
        if _is_verifier(request):
            context = json.loads(json.loads(request.content)["messages"][1]["content"])
            assert context["evidence"][0]["text"] == "月令是出生月份的地支。"
            return _completion(_verdict(supported=False, issues=["incorrect_citation"]))
        return _completion(_candidate(unsupported))

    async def collect(chunk: str) -> None:
        chunks.append(chunk)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate_stream(
            "我事业如何？", _chart(), [_hit()], collect
        )

    assert calls == 4
    assert chunks == [result.answer]
    assert unsupported not in result.answer
    assert result.degradation_reason == "model_output_failed_validation"
    assert not result.citations_validated


@pytest.mark.asyncio
async def test_one_repair_can_pass_semantic_verification() -> None:
    generations = 0
    verifications = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generations, verifications
        if _is_verifier(request):
            verifications += 1
            return _completion(_verdict(
                supported=verifications == 2,
                issues=[] if verifications == 2 else ["unsupported_claim"],
            ))
        generations += 1
        if generations == 2:
            assert "unsupported_claim" in json.loads(request.content)["messages"][1]["content"]
        answer = "不受支持的判断。[1]" if generations == 1 else "月令是月支。[1]"
        return _completion(_candidate(answer))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate("解释月令", _chart(), [_hit()])

    assert generations == verifications == 2
    assert result.answer == "月令是月支。[1]"
    assert result.policy_decision == "allow"
    assert result.citations_validated and result.evidence_validated
    assert result.token_usage == 28


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verifier_output",
    [
        "not JSON",
        '```json\n{"supported":true,"safe":true,"complete":true,'
        '"issues":[],"missing_aspects":[]}\n```',
        '{"supported":false,"supported":true,"safe":true,"complete":true,'
        '"issues":[],"missing_aspects":[]}',
        _verdict(supported="true"),
        {"supported": True, "safe": True},
        _verdict(answer="approve"),
        _verdict(issues=["unsupported_claim"]),
        _verdict(safe=False),
        _verdict(issues=["unknown_issue"]),
        _verdict(complete="true"),
        _verdict(missing_aspects=["下一步大运"]),
        {"supported": True, "safe": True, "issues": [], "missing_aspects": []},
        "http_error",
        "timeout",
        "invalid_envelope",
    ],
)
async def test_invalid_or_unavailable_verifier_fails_closed(verifier_output: str | dict) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if not _is_verifier(request):
            return _completion(_candidate())
        if verifier_output == "http_error":
            return httpx.Response(503)
        if verifier_output == "timeout":
            raise httpx.ReadTimeout("mock verification timeout", request=request)
        if verifier_output == "invalid_envelope":
            return httpx.Response(200, json={"choices": []})
        return _completion(verifier_output)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate("解释月令", _chart(), [_hit()])

    assert calls == 4
    assert result.policy_decision == "refuse_invalid_citations"
    assert not result.citations_validated
    assert not result.evidence_validated
    assert result.degradation_reason == "model_output_failed_validation"


@pytest.mark.asyncio
async def test_offline_directory_excludes_question_and_dangerous_source_text() -> None:
    dangerous = "你应该停用医生开的药。"
    settings = _settings().model_copy(update={"openai_api_key": ""})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("offline mode must not call a model")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            dangerous, _chart(), [_hit(dangerous)]
        )

    assert "[1] 月令的定义" in result.answer
    assert "离线资料目录" in result.answer
    assert dangerous not in result.answer
    assert result.model_version == "offline-references"
    assert all(evaluate_generated_answer(
        result, [_hit()], expected_policy="evidence_answer", expects_uncertainty=True
    ).values())


@pytest.mark.asyncio
async def test_verifier_supports_json_format_compatibility_without_relaxing_validation() -> None:
    verification_requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        if not _is_verifier(request):
            return _completion(_candidate())
        verification_requests.append(request)
        if "response_format" in json.loads(request.content):
            return httpx.Response(422)
        return _completion(_verdict())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate("解释月令", _chart(), [_hit()])

    assert result.citations_validated and result.evidence_validated
    assert len(verification_requests) == 2
    assert (
        verification_requests[0].headers["Idempotency-Key"]
        != verification_requests[1].headers["Idempotency-Key"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("repair_covers_missing_aspects", [False, True])
async def test_multi_part_question_requires_coverage_or_explicit_evidence_gap(
    repair_covers_missing_aspects: bool,
) -> None:
    question = (
        "请结合月令解释事业趋势，同时分别分析财富与婚姻的适用条件；"
        "还要比较当前大运和下一步大运，说明哪些能判断、哪些没有证据。"
    )
    generations = 0
    verifications = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal generations, verifications
        payload = json.loads(request.content)
        if _is_verifier(request):
            verifications += 1
            context = json.loads(payload["messages"][1]["content"])
            assert context["question"] == question
            assert "不能因已有部分正确引用就认定回答完整" in payload["messages"][0]["content"]
            complete = repair_covers_missing_aspects and verifications == 2
            if complete:
                assert context["uncertainties"] == [
                    "财富、婚姻及当前和下一步大运的比较均缺少适用规则证据。"
                ]
            return _completion(_verdict(
                complete=complete,
                issues=[] if complete else ["incomplete_answer"],
                missing_aspects=[] if complete else ["财富", "婚姻", "当前和下一步大运比较"],
            ))
        generations += 1
        candidate = _candidate("月令是出生月份的地支，事业判断还需其他条件。[1]")
        if generations == 2:
            assert "incomplete_answer" in payload["messages"][1]["content"]
            if repair_covers_missing_aspects:
                candidate["uncertainties"] = [
                    "财富、婚姻及当前和下一步大运的比较均缺少适用规则证据。"
                ]
        return _completion(candidate)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(_settings(), client).generate(question, _chart(), [_hit()])

    assert generations == verifications == 2
    assert result.citations_validated is repair_covers_missing_aspects
    assert (result.policy_decision == "allow") is repair_covers_missing_aspects
    assert all(evaluate_generated_answer(
        result, [_hit()], expected_policy="evidence_answer", expects_uncertainty=True
    ).values()) is repair_covers_missing_aspects
