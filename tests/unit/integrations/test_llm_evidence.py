import json
from pathlib import Path

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.retrieval.schemas import (
    KnowledgeRuleContext,
    RetrievalDocument,
    RetrievalHit,
)


@pytest.fixture(scope="module")
def knowledge() -> KnowledgeRepository:
    repository = KnowledgeRepository(Path(__file__).resolve().parents[3] / "knowledge")
    repository.load()
    return repository


def _hit(document: RetrievalDocument) -> RetrievalHit:
    return RetrievalHit(document=document, score=1.0, matched_by=["test"])


def test_actual_long_cards_keep_every_rule_field_and_source(knowledge: KnowledgeRepository) -> None:
    cards = {card.id: card for card in knowledge.cards}
    documents = [
        document
        for document in knowledge.documents("personal_preview")
        if document.kind == "knowledge_card" and len(document.text) > 1400
    ]
    assert documents

    for document in documents:
        card = cards[document.id]
        expected = card.model_dump(include=set(KnowledgeRuleContext.model_fields))
        assert document.rule_context is not None
        assert document.rule_context.model_dump() == expected
        restored = RetrievalDocument.model_validate_json(document.model_dump_json())
        payload = AnswerGenerator._evidence_payload([_hit(restored)])[0]
        context = payload["rule_context"]
        assert context == {key: value for key, value in expected.items() if key != "content"}
        assert payload["source"] == document.source
        assert payload["trace_refs"] == document.trace_refs
        assert payload["number"] == 1
        if payload.get("text_repeated_in_rule"):
            assert card.content in context["rule"]
        else:
            assert payload["text"] == card.content


def test_unique_supplement_and_unstructured_tail_are_never_cut() -> None:
    content = "独有说明。" * 400 + "尾部补充：不能脱离位置和岁运讨论。"
    document = RetrievalDocument(
        id="long-card",
        kind="knowledge_card",
        layer=3,
        title="长规则",
        text="旧的拼接文本不作为字段来源。",
        source="测试原文 · 第四节",
        school="基础共识",
        concepts=[],
        rule_context=KnowledgeRuleContext(
            content=content,
            card_type="rule",
            rule="规则：先核对月令与日主，再讨论适用范围。",
            conditions=["前置条件。" * 400],
            exceptions=["例外条件必须保留。"],
            prohibited_uses=["不得由单一十神推断个人的现实事件。"],
        ),
    )
    structured = AnswerGenerator._evidence_payload([_hit(document)])[0]
    assert structured["text"] == content
    assert structured["rule_context"]["conditions"] == document.rule_context.conditions
    assert structured["rule_context"]["exceptions"] == document.rule_context.exceptions
    assert structured["rule_context"]["prohibited_uses"] == document.rule_context.prohibited_uses

    unstructured = document.model_copy(update={"rule_context": None, "text": content})
    assert AnswerGenerator._evidence_payload([_hit(unstructured)])[0]["text"] == content


@pytest.mark.asyncio
async def test_generator_and_verifier_share_actual_long_card_payload(
    knowledge: KnowledgeRepository,
) -> None:
    document = next(
        document
        for document in knowledge.documents("personal_preview")
        if document.id == "classic-rule-sanming-tonghui-ch134-p001"
    )
    generation_evidence = []
    verifier_evidence = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system = payload["messages"][0]["content"]
        user = payload["messages"][1]["content"]
        if system.startswith("你是独立的答案核验器"):
            verifier_evidence.extend(json.loads(user)["evidence"])
            result = {
                "supported": True,
                "safe": True,
                "complete": True,
                "issues": [],
                "missing_aspects": [],
            }
        else:
            generation_evidence.extend(json.loads(user.split("\n\n资料：\n", 1)[1]))
            result = {
                "answer": "原文指出，正财还需结合日主力量和适用条件理解。[1]",
                "uncertainties": ["个人具体情况仍需其他资料。"],
                "followups": [],
            }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(result)}}]}
        )

    settings = Settings(
        _env_file=None, llm_provider="openai", openai_api_key="mock-key", http_request_retries=0
    )
    chart = ChartCalculator().calculate(BirthInput(date="1990-01-01", time="12:00:00"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await AnswerGenerator(settings, client).generate(
            "解释原文中正财的条件和例外。", chart, [_hit(document)]
        )

    assert result.evidence_validated
    assert generation_evidence == verifier_evidence
    assert generation_evidence[0]["rule_context"]["exceptions"]
    assert generation_evidence[0]["rule_context"]["prohibited_uses"]
    assert generation_evidence[0]["text_repeated_in_rule"]
