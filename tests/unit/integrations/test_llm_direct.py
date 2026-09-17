from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.core.errors import (
    InvalidUpstreamResponseError,
    ServiceUnavailableError,
    UpstreamServiceError,
)
from bazi_api.integrations.llm import (
    AnswerGenerator,
    GenerationResult,
    _explicit_safety_violation,
)
from bazi_api.modules.charts.schemas import BirthInput, TopicFactPack
from bazi_api.modules.charts.service import ChartCalculator


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-key",
        "openai_base_url": "https://llm.example/v1",
        "openai_chat_model": "test-model",
        "database_path": tmp_path / "app.sqlite3",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def llm_payload(answer: str = "## 结论\n\n测试回答") -> dict[str, object]:
    return {
        "choices": [
            {"message": {"content": json.dumps(
                {"answer": answer, "uncertainties": ["测试不确定性"], "followups": ["测试追问"]},
                ensure_ascii=False,
            )}}
        ],
        "usage": {"total_tokens": 33},
    }


def chart():
    return ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=time(12, 0), name="测试")
    )


def raw_payload(answer: str, uncertainties: list[str]) -> dict[str, object]:
    return {
        "choices": [
            {"message": {"content": json.dumps(
                {"answer": answer, "uncertainties": uncertainties, "followups": []},
                ensure_ascii=False,
            )}}
        ],
        "usage": {"total_tokens": 11},
    }


def recording_transport(
    replies: list[httpx.Response],
) -> tuple[httpx.MockTransport, list[dict[str, object]], list[str]]:
    payloads: list[dict[str, object]] = []
    idempotency_keys: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        idempotency_keys.append(request.headers["Idempotency-Key"])
        return replies[min(len(payloads) - 1, len(replies) - 1)]

    return httpx.MockTransport(handler), payloads, idempotency_keys


@pytest.mark.asyncio
async def test_generate_direct_sends_topic_pack_and_returns_result(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        # 实际捕获的是完整请求体
        captured["body"] = request.read().decode("utf-8")
        return httpx.Response(200, json=llm_payload())

    transport = httpx.MockTransport(handler)
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )
    pack = TopicFactPack(
        topic_id="topic:wealth", label="财富与财运", facts=["偏财（庚金）藏于年支巳"]
    )

    result = await generator.generate_direct("看下财运", chart(), pack, school="基础共识")

    assert result.answer.startswith("## 结论")
    assert result.uncertainties == ["测试不确定性"]
    assert result.followups == ["测试追问"]
    assert result.prompt_version == "direct-v1"
    assert result.model_version == "test-model"
    assert result.token_usage == 33
    body = str(captured["body"])
    assert "偏财（庚金）藏于年支巳" in body
    assert "话题事实" in body
    assert "财富与财运" in str(captured["body"])
    # 系统提示词断言：通过请求体片段检查关键政策边界仍在
    assert "不得诊断疾病" in body or "健康内容只能说明传统文献观点" in body


@pytest.mark.asyncio
async def test_generate_direct_requires_api_key(tmp_path: Path) -> None:
    generator = AnswerGenerator(
        make_settings(tmp_path, openai_api_key=""),
        httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200))),
    )
    with pytest.raises(ServiceUnavailableError):
        await generator.generate_direct("看下财运", chart(), None)


@pytest.mark.asyncio
async def test_generate_direct_retries_without_response_format(tmp_path: Path) -> None:
    transport, payloads, idempotency_keys = recording_transport(
        [httpx.Response(400), httpx.Response(200, json=llm_payload("## 结论\n\n兼容回退成功"))]
    )
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )

    result = await generator.generate_direct("看下财运", chart(), None)

    assert result.answer == "## 结论\n\n兼容回退成功"
    assert result.prompt_version == "direct-v1"
    assert len(payloads) == 2
    assert "response_format" in payloads[0]
    assert "response_format" not in payloads[1]
    assert idempotency_keys[0] != idempotency_keys[1]


@pytest.mark.asyncio
async def test_generate_direct_falls_back_after_two_failed_attempts(tmp_path: Path) -> None:
    transport, payloads, _ = recording_transport(
        [httpx.Response(200, json=raw_payload("## 结论\n\n缺不确定性的判断", []))]
    )
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )

    result = await generator.generate_direct("看下财运", chart(), None)

    assert result.degradation_reason == "direct_validation_failed"
    assert result.prompt_version == "direct-v1"
    assert result.model_version == "test-model"
    assert len(payloads) == 2
    assert "上一次输出未通过校验（missing_uncertainty）" in json.dumps(
        payloads[1], ensure_ascii=False
    )


@pytest.mark.asyncio
async def test_generate_direct_repair_round_carries_failure_reason(tmp_path: Path) -> None:
    transport, payloads, _ = recording_transport(
        [
            httpx.Response(200, json=raw_payload("## 结论\n\n第一次回答", [])),
            httpx.Response(200, json=llm_payload("## 结论\n\n第二次回答")),
        ]
    )
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )

    result = await generator.generate_direct("看下财运", chart(), None)

    assert result.answer == "## 结论\n\n第二次回答"
    assert result.degradation_reason == ""
    assert len(payloads) == 2
    assert "上一次输出未通过校验" not in json.dumps(payloads[0], ensure_ascii=False)
    assert "上一次输出未通过校验（missing_uncertainty）" in json.dumps(
        payloads[1], ensure_ascii=False
    )


@pytest.mark.asyncio
async def test_generate_direct_prompt_drops_evidence_channel_wording(tmp_path: Path) -> None:
    transport, payloads, _ = recording_transport([httpx.Response(200, json=llm_payload())])
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )
    pack = TopicFactPack(
        topic_id="topic:wealth", label="财富与财运", facts=["偏财（庚金）藏于年支巳"]
    )

    await generator.generate_direct("看下财运", chart(), pack)

    body = json.dumps(payloads[0], ensure_ascii=False)
    for wording in ("[n]", "资料以 JSON", "审查范围", "reviewed_only", "personal_preview"):
        assert wording not in body
    assert "偏财（庚金）藏于年支巳" in body
    assert "不得诊断疾病" in body


def anthropic_settings(tmp_path: Path) -> Settings:
    return make_settings(
        tmp_path,
        llm_provider="anthropic",
        anthropic_auth_token="test-anthropic-key",
        anthropic_base_url="https://anthropic.invalid",
        anthropic_chat_model="claude-test",
    )


@pytest.mark.asyncio
async def test_generate_direct_anthropic_uses_messages_protocol(tmp_path: Path) -> None:
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
                                "answer": "## 结论\n\n有依据的回答",
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

    generator = AnswerGenerator(
        anthropic_settings(tmp_path), httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )

    result = await generator.generate_direct("如何理解正官？", chart(), None)

    headers = captured["headers"]
    payload = captured["payload"]
    assert captured["url"] == "https://anthropic.invalid/v1/messages"
    assert isinstance(headers, dict)
    assert headers["x-api-key"] == "test-anthropic-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert isinstance(payload, dict)
    assert payload["model"] == "claude-test"
    assert payload["thinking"] == {"type": "disabled"}
    assert "system" in payload
    assert "response_format" not in payload
    assert result.answer == "## 结论\n\n有依据的回答"
    assert result.token_usage == 42
    assert result.model_version == "claude-test"


@pytest.mark.asyncio
async def test_generate_direct_retry_count_is_bounded_and_maps_to_upstream_error(
    tmp_path: Path,
) -> None:
    requests = 0

    def handler(_: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(503, headers={"Retry-After": "0"})

    generator = AnswerGenerator(
        make_settings(tmp_path, http_request_retries=2),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(UpstreamServiceError):
        await generator.generate_direct("如何理解正官？", chart(), None)

    assert requests == 3


@pytest.mark.asyncio
async def test_generate_direct_rejects_malformed_success_envelope(tmp_path: Path) -> None:
    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    generator = AnswerGenerator(
        make_settings(tmp_path, http_request_retries=0),
        httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(InvalidUpstreamResponseError):
        await generator.generate_direct("如何理解正官？", chart(), None)


def test_direct_output_gate_rejects_deterministic_and_medical_claims() -> None:
    deterministic = GenerationResult(
        answer="命主明年一定发财，已经可以确定",
        uncertainties=["现实结果仍可能变化"],
        followups=[],
    )
    medical = GenerationResult(
        answer="建议立即停药，改为每日服用本方剂",
        uncertainties=["本回答不构成医疗建议"],
        followups=[],
    )
    safe = GenerationResult(
        answer="若条件成熟仍可能受益",
        uncertainties=["仍需结合现实条件"],
        followups=[],
    )

    assert AnswerGenerator._direct_output_is_trusted(deterministic) is False
    assert AnswerGenerator._direct_output_is_trusted(medical) is False
    assert AnswerGenerator._direct_output_is_trusted(safe) is True


@pytest.mark.asyncio
async def test_generate_direct_falls_back_when_safety_gate_rejects_output(
    tmp_path: Path,
) -> None:
    transport, payloads, _ = recording_transport(
        [
            httpx.Response(
                200,
                json=raw_payload("命主明年一定发财，已经可以确定", ["现实结果仍可能变化"]),
            )
        ]
    )
    generator = AnswerGenerator(
        make_settings(tmp_path), httpx.AsyncClient(transport=transport)
    )

    result = await generator.generate_direct("如何理解财格？", chart(), None)

    assert result.degradation_reason == "direct_validation_failed"
    assert "一定发财" not in result.answer
    assert result.uncertainties == ["答案未通过安全与完整性校验。"]
    assert len(payloads) == 2
    assert "unsafe_output" in json.dumps(payloads[1], ensure_ascii=False)


def _violation(answer: str, uncertainties: list[str] | None = None) -> bool:
    return _explicit_safety_violation(
        GenerationResult(
            answer=answer,
            uncertainties=uncertainties or ["仍需结合实际条件"],
            followups=[],
        )
    )


@pytest.mark.parametrize(
    "answer",
    [
        # 年份 + 断言词 + 结果词
        "2028年你会结婚",
        "你2035年会去世",
        "你2028年必然离婚",
        "命主2029年注定破财",
        "2027年你一定会升职",
        "二〇二八年你会结婚",
        # 疑问年份词直接接结果词
        "我何年暴富",
        "你哪一年能买房",
        # 寿元类
        "命主寿元止于63岁",
        "他寿命只有七十岁",
        "阳寿仅余十年",
        # 强化措辞（无年份也属承诺/死期断言）
        "发大财",
        "命主终将归西",
    ],
)
def test_safety_gate_rejects_year_assertions_and_death_claims(answer: str) -> None:
    assert _violation(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "发财是必然的",
        "结婚是早晚的事，命中注定如此",
        "升职已经可以确定",
    ],
)
def test_safety_gate_rejects_result_before_assertion_order(answer: str) -> None:
    assert _violation(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "《子平真诠》第3章有云：财旺生官",
        "按《滴天髓》第十二节所述",
        "《三命通会》第一卷记载此格",
        "《穷通宝鉴》第一条即言此理",
        "《渊海子平》第三篇论用神",
    ],
)
def test_safety_gate_rejects_fabricated_citations_in_direct_channel(answer: str) -> None:
    assert _violation(answer) is True


def test_safety_gate_catches_cross_clause_escape() -> None:
    # 模态词与结果词被逗号分到不同子句，仍应被整段扫描拦下。
    assert _violation("命主明年一定，会结婚") is True
    assert _violation("2028年你必然，暴富") is True


def test_safety_gate_catches_assertion_split_across_answer_and_uncertainties() -> None:
    # 断言词在 answer、结果词落在 uncertainties：拼接扫描仍应拦下。
    assert _violation("命主明年一定", ["会结婚"]) is True
    # uncertainties 里出现“可能”不能把 answer 里的确定性断言洗成条件式。
    assert _violation("2028年你会结婚", ["现实结果仍可能变化"]) is True


@pytest.mark.parametrize(
    "answer",
    [
        "不应承诺2028年你会结婚，也不应断言必然发财",
        "不能断言你2035年会去世，命理不作死期判断",
        "不要承诺某人何年暴富",
        "不得给出寿元止于63岁的结论",
        "避免把发大财说成必然结果",
        "没有说你会归西",
        "未被证明2028年你会结婚",
    ],
)
def test_safety_gate_keeps_negation_whitelist(answer: str) -> None:
    assert _violation(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        "若条件成熟仍可能受益",
        "2028年前后若条件成熟，感情有推进空间，但不保证具体结果",
        "婚姻的时间窗口需结合大运与命局条件观察，不作确定性判断",
        "传统文献认为此格局利于文书往来，实际仍需个人努力",
    ],
)
def test_safety_gate_keeps_conditional_wording(answer: str) -> None:
    assert _violation(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        # 普通枚举（产品提示词要求用“1. ”/“- ”编号列表，这类措辞必然常见）
        "第一条建议是结合大运观察，第二条是留意流年",
        "我建议分三步：第一条，先看月令",
        "命主第一条优势是印星得力",
        "本报告分为三篇，第一篇讲格局",
        # 引文语境之外的分段表述
        "本命局可分为三层来看，第三篇式的罗列并无必要",
    ],
)
def test_safety_gate_allows_plain_enumerations(answer: str) -> None:
    assert _violation(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        # 真实编造引用：书名号或引文线索词出现在条文编号之前
        "据《子平真诠》第八章所论，财旺生官需身强",
        "《滴天髓》第十二节云：财多身弱，富屋贫人",
        "书中第一条规定此格取用之法",
    ],
)
def test_safety_gate_rejects_citations_in_quotation_context(answer: str) -> None:
    assert _violation(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        # spec 要求“未来只给条件式趋势”，带对冲标记的年份表述是应有输出
        "2026年你会有结婚的念头，是否成行仍看条件",
        "2029年你会结婚，但是否成行取决于大运配合",
        "2030年你会升职，若流年与大运配合则更稳",
        "2031年你会有买房的可能，但仍需看具体条件",
        "2032年你会结婚一事，仍需视情况而定",
        "2033年你会升职，需视大运而定",
    ],
)
def test_safety_gate_allows_hedged_year_statements(answer: str) -> None:
    assert _violation(answer) is False


@pytest.mark.parametrize(
    "answer",
    [
        # “视”只在对冲用法（视情况/视大运）下才算弱化标记；
        # “重视/忽视”里的“视”不得把真断言洗白。
        "2028年你会结婚，值得重视",
        "2028年你会结婚，切莫忽视",
    ],
)
def test_safety_gate_does_not_treat_vision_compounds_as_hedges(answer: str) -> None:
    assert _violation(answer) is True


@pytest.mark.parametrize(
    "answer",
    [
        "绝不预测你2028年是否会结婚，因为命理不作此类断言",
        "决不认为2028年你会结婚",
        "不必把命主寿元止于63岁当作结论",
        "不作2028年你会结婚的判断",
        "不予断言你2035年会去世",
    ],
)
def test_safety_gate_negation_whitelist_extended(answer: str) -> None:
    assert _violation(answer) is False


# 真实风格样例：正常命理解读、条件式趋势、编号列表、专家口吻、含否定/拒答的句子。
# 修前基线：9 条中 6 条被安全闸误判（整篇答案会被替换为兜底文案）。
_REALISTIC_STYLE_SAMPLES = [
    "第一条建议是结合大运观察，第二条是留意流年",
    "我建议分三步：第一条，先看月令",
    "命主第一条优势是印星得力",
    "本报告分为三篇，第一篇讲格局",
    "2026年你会有结婚的念头，是否成行仍看条件",
    "绝不预测你2028年是否会结婚，因为命理不作此类断言",
    "从命局看，月令得气，日主偏强，宜以财官为用，结合大运流向逐步观察",
    "2028年前后若流年与命局配合，感情有推进的空间，需看具体条件，不作确定性判断",
    "传统典籍多言印星主学业，但身弱逢印仍需辨别，实际仍看个人努力与客观条件",
]


@pytest.mark.parametrize("answer", _REALISTIC_STYLE_SAMPLES)
def test_safety_gate_keeps_realistic_style_samples(answer: str) -> None:
    assert _violation(answer) is False


def test_direct_system_prompt_requires_at_least_one_uncertainty() -> None:
    prompt = AnswerGenerator._direct_system_prompt("基础共识", None)

    assert "uncertainties 至少列出 1 条本答案的适用边界或不确定性" in prompt
