from datetime import date, time
from pathlib import Path

import pytest

from bazi_api.core.config import Settings
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.retrieval.service import RetrievalService, normalize_retrieval_text

BACKEND_ROOT = Path(__file__).resolve().parents[3]


def build_settings(tmp_path: Path) -> Settings:
    return Settings(
        knowledge_path=BACKEND_ROOT / "knowledge",
        database_path=tmp_path / "app.db",
    )


@pytest.mark.asyncio
async def test_bm25_retrieval_returns_expected_card(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(settings, repository.documents())
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    hits = await service.search("偏财是不是一定代表意外之财？", chart, "基础共识", "bm25", 5)

    assert "god-piancai" in {hit.document.id for hit in hits}
    assert all(hit.matched_by for hit in hits)


@pytest.mark.asyncio
async def test_graph_expansion_is_a_retrieval_hint(tmp_path: Path) -> None:
    settings = build_settings(tmp_path)
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(
        settings,
        repository.documents(),
        repository.graph_nodes,
        repository.graph_edges,
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    hits = await service.search("阴阳是不是代表好坏？", chart, "基础共识", "bm25", 10)

    yinyang = next(hit for hit in hits if hit.document.id == "concept-yinyang")
    assert "knowledge_graph" in yinyang.matched_by
    assert yinyang.document.layer == 3
    linked_annotations = {
        ref for ref in yinyang.document.trace_refs if ref.startswith("annotation:")
    }
    assert linked_annotations.intersection(hit.document.id for hit in hits)


@pytest.mark.asyncio
async def test_evidence_scopes_are_isolated_and_preview_keeps_canonical_chain(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(
        settings,
        repository.documents("personal_preview"),
        repository.graph_nodes,
        repository.graph_edges,
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    reviewed = await service.search(
        "子平真诠怎样从月令讨论用神？",
        chart,
        "基础共识",
        "bm25",
        6,
        "reviewed_only",
    )
    preview = await service.search(
        "子平真诠怎样从月令讨论用神？",
        chart,
        "子平格局法",
        "bm25",
        6,
        "personal_preview",
    )

    assert reviewed and all(hit.document.review_status == "reviewed" for hit in reviewed)
    assert preview and any(hit.document.review_status == "machine_verified" for hit in preview)
    assert not any(hit.document.review_status in {"draft", "retired"} for hit in preview)
    assert any(hit.document.kind == "canonical_passage" for hit in preview)
    assert all(
        hit.document.warning for hit in preview if hit.document.review_status == "machine_verified"
    )

    for selected_school in ("基础共识", "子平格局法", "其他用神体系"):
        isolated = await service.search(
            "月令用神与格局",
            chart,
            selected_school,
            "bm25",
            10,
            "personal_preview",
        )
        assert all(
            hit.document.school in {"基础共识", selected_school} for hit in isolated
        )


@pytest.mark.asyncio
async def test_generic_chart_question_retrieves_its_pattern_candidate_rules(
    tmp_path: Path,
) -> None:
    settings = build_settings(tmp_path)
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    service = await RetrievalService.create(
        settings,
        repository.documents("personal_preview"),
        repository.graph_nodes,
        repository.graph_edges,
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12, 0)))

    terms = service._chart_query_terms("这个命格如何？", chart)
    # 纯 bm25 通道不再有向量召回兜底：第31章（论正官）的规则仍会被检索到，
    # 但排在召回窗口的十四位左右，因此按召回窗口而不是前六条断言。
    hits = await service.search(
        "这个命格如何？",
        chart,
        "子平格局法",
        "bm25",
        20,
        "personal_preview",
    )

    assert {"正官", "正官格候选", "子午冲"}.issubset(terms)
    assert any("ch31" in hit.document.id for hit in hits)
    assert any(hit.document.kind == "canonical_passage" for hit in hits)


def test_traditional_and_variant_terms_are_normalized() -> None:
    assert normalize_retrieval_text("陰陽生剋與財運") == "阴阳生克与财运"
    assert "月令" in normalize_retrieval_text("提纲")
    assert normalize_retrieval_text("七煞") == "七杀"
