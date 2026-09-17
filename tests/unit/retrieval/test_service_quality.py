from pathlib import Path
from typing import Any

import pytest

from bazi_api.core.config import Settings
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit
from bazi_api.modules.retrieval.service import RetrievalService


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {"database_path": tmp_path / "app.sqlite3"}
    values.update(overrides)
    return Settings(_env_file=None, **values)


def document(document_id: str, **overrides: Any) -> RetrievalDocument:
    values: dict[str, Any] = {
        "id": document_id,
        "kind": "knowledge_card",
        "layer": 3,
        "title": document_id,
        "text": f"Target evidence for {document_id}.",
        "source": "test source",
        "school": "test-school",
        "concepts": [],
    }
    values.update(overrides)
    return RetrievalDocument(**values)


def hit(doc: RetrievalDocument, score: float = 1.0) -> RetrievalHit:
    return RetrievalHit(document=doc, score=score, matched_by=["test"])


def test_diversification_limits_each_book_chapter_independently() -> None:
    docs = [
        document(f"{book}-{number}", trace_refs=[f"{book}-ch001-p{number:03}"])
        for book in ("book-a", "book-b")
        for number in (1, 2, 3)
    ]

    selected = RetrievalService._diversify_hits([hit(doc) for doc in docs], 4)

    assert [item.document.id for item in selected] == [
        "book-a-1",
        "book-a-2",
        "book-b-1",
        "book-b-2",
    ]


def test_diversification_still_deduplicates_shared_source_families() -> None:
    source = "book-a-ch001-p001"
    docs = [
        document("card", trace_refs=[source]),
        document("annotation", kind="modern_annotation", layer=2, trace_refs=[source]),
        document("independent", trace_refs=["book-b-ch001-p001"]),
    ]

    selected = RetrievalService._diversify_hits([hit(doc) for doc in docs], 3)

    assert [item.document.id for item in selected] == ["card", "independent"]
    assert RetrievalService._diversify_hits([hit(docs[0])], 0) == []


@pytest.mark.asyncio
async def test_evidence_chain_covers_two_parents_without_consuming_the_whole_ranking(
    tmp_path: Path,
) -> None:
    parents = [document(f"parent-{number}", trace_refs=[f"source-{number}"]) for number in range(3)]
    originals = [
        document(f"source-{number}", kind="canonical_passage", layer=1) for number in range(3)
    ]
    fillers = [document(f"filler-{number}") for number in range(3)]
    service = await RetrievalService.create(
        settings(tmp_path), [*parents, *originals, *fillers]
    )

    ranked = [hit(doc, 1 - index / 10) for index, doc in enumerate([*parents, fillers[0]])]
    selected = service._expand_evidence_chain(ranked, set(service.documents), 6)
    assert [item.document.id for item in selected] == [
        "parent-0",
        "source-0",
        "parent-1",
        "parent-2",
        "filler-0",
        "source-1",
    ]
    assert sum("evidence_chain" in item.matched_by for item in selected) == 2
    assert selected[1].score == pytest.approx(
        ranked[0].score * service.settings.evidence_chain_score_ratio
    )
    assert service._expand_evidence_chain(ranked, set(service.documents), 1) == ranked[:1]
    assert service._expand_evidence_chain(ranked, set(service.documents), 0) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [4, 6, 7, 10])
async def test_second_source_never_displaces_full_ranked_evidence(
    tmp_path: Path, limit: int
) -> None:
    parents = [document(f"parent-{number}", trace_refs=[f"source-{number}"]) for number in range(2)]
    sources = [
        document(f"source-{number}", kind="canonical_passage", layer=1) for number in range(2)
    ]
    fillers = [document(f"filler-{number}") for number in range(8)]
    service = await RetrievalService.create(
        settings(tmp_path), [*parents, *sources, *fillers]
    )

    ranked = [hit(doc, 1 - index / 20) for index, doc in enumerate([*parents, *fillers])]
    selected = service._expand_evidence_chain(ranked, set(service.documents), limit)
    single_chain_order = ["parent-0", "source-0", "parent-1", *[doc.id for doc in fillers]]
    assert [item.document.id for item in selected] == single_chain_order[:limit]
    assert all(item.document.id != "source-1" for item in selected)
    assert len(selected) == limit


@pytest.mark.asyncio
async def test_second_source_already_retrieved_keeps_its_rank_and_does_not_duplicate(
    tmp_path: Path,
) -> None:
    parents = [document(f"parent-{number}", trace_refs=[f"source-{number}"]) for number in range(2)]
    sources = [
        document(f"source-{number}", kind="canonical_passage", layer=1) for number in range(2)
    ]
    fillers = [document(f"filler-{number}") for number in range(3)]
    service = await RetrievalService.create(
        settings(tmp_path), [*parents, *sources, *fillers]
    )

    ranked = [hit(doc) for doc in [*parents, sources[1], *fillers]]
    selected = service._expand_evidence_chain(ranked, set(service.documents), 6)
    assert [item.document.id for item in selected] == [
        "parent-0",
        "source-0",
        "parent-1",
        "source-1",
        "filler-0",
        "filler-1",
    ]
    assert selected[3] is ranked[2]


@pytest.mark.asyncio
async def test_evidence_chain_deduplicates_and_skips_ineligible_or_cyclic_references(
    tmp_path: Path,
) -> None:
    original = document("original", kind="canonical_passage", layer=1)
    blocked = document("blocked", kind="canonical_passage", layer=1, review_status="draft")
    annotation = document(
        "annotation", kind="modern_annotation", layer=2, trace_refs=["parent", "original"]
    )
    parent = document("parent", trace_refs=["blocked", "missing", "annotation"])
    other = document("other", trace_refs=["original"])
    service = await RetrievalService.create(
        settings(tmp_path), [parent, original, blocked, annotation, other]
    )

    allowed = {"parent", "original", "annotation", "other"}
    ranked = [hit(parent), hit(other), hit(original, 0.5)]
    selected = service._expand_evidence_chain(ranked, allowed, 4)
    assert [item.document.id for item in selected] == ["parent", "original", "other"]
    assert selected[1] is ranked[2]
    assert selected[1].matched_by == ["test"]
