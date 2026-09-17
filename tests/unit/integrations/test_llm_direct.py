from __future__ import annotations

import json
from datetime import date, time
from pathlib import Path

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.core.errors import ServiceUnavailableError
from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.schemas import BirthInput, TopicFactPack
from bazi_api.modules.charts.service import ChartCalculator


def make_settings(tmp_path: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "openai_api_key": "test-key",
        "openai_base_url": "https://llm.example/v1",
        "openai_chat_model": "test-model",
        "database_path": tmp_path / "app.sqlite3",
        "embedding_cache_path": tmp_path / "embed.sqlite3",
        "qdrant_path": tmp_path / "qdrant",
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
