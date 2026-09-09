from datetime import date, time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

from bazi_api.core.config import Settings
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit
from bazi_api.modules.retrieval.service import RetrievalService


class CountingEmbeddingProvider(HashEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__(dimension=8)
        self.model_version = "fake-semantic-v1"
        self.calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return await super().embed(texts)


def settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "embedding_provider": "hash",
        "vector_backend": "memory",
        "reranker_provider": "lexical",
        "database_path": tmp_path / "app.sqlite3",
        "embedding_cache_path": tmp_path / "embeddings.sqlite3",
        "qdrant_path": tmp_path / "qdrant",
    }
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


def chart():
    return ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12)))


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
@pytest.mark.parametrize("dense_available", [True, False])
async def test_dense_diversification_fills_from_the_recalled_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dense_available: bool
) -> None:
    docs = [
        document("top", trace_refs=["book-a-ch001-p001"]),
        document("top-duplicate", trace_refs=["book-a-ch001-p001"]),
        document("same-chapter", trace_refs=["book-a-ch001-p002"]),
        document("chapter-overflow", trace_refs=["book-a-ch001-p003"]),
        document("tail-b", trace_refs=["book-b-ch001-p001"]),
        document("tail-c", trace_refs=["book-c-ch002-p001"]),
    ]
    service = await RetrievalService.create(settings(tmp_path), docs, HashEmbeddingProvider(8))
    ranking = [(doc.id, 1 - index / 10) for index, doc in enumerate(docs)]
    monkeypatch.setattr(
        service, "_dense_search", AsyncMock(return_value=ranking if dense_available else [])
    )
    monkeypatch.setattr(service.bm25, "search", lambda *_: ranking)

    try:
        selected = await service.search("target", chart(), "test-school", "dense", 4)
    finally:
        service.close()

    assert [item.document.id for item in selected] == ["top", "same-chapter", "tail-b", "tail-c"]
    component = "dense" if dense_available else "bm25-fallback"
    assert all(component in item.matched_by for item in selected)


@pytest.mark.asyncio
async def test_cache_only_startup_and_dense_fallback_never_encode_a_cold_index(
    tmp_path: Path,
) -> None:
    provider = CountingEmbeddingProvider()
    docs = [document("target-first"), document("target-second")]
    service = await RetrievalService.create(settings(tmp_path), docs, provider, embed_missing=False)

    try:
        assert provider.calls == []
        assert service.embedding_provider is provider
        assert service.model_version == "fake-semantic-v1"
        assert service.vectors.documents == {}
        assert set(service.documents) == {doc.id for doc in docs}
        assert service.cache_stats == {"hits": 0, "misses": 2}

        selected = await service.search("target", chart(), "test-school", "dense", 2)
        assert {item.document.id for item in selected} == {doc.id for doc in docs}
        assert all("bm25-fallback" in item.matched_by for item in selected)
        assert provider.calls == []
    finally:
        service.close()


@pytest.mark.asyncio
async def test_partial_cache_retains_unindexed_results_in_dense_top_k(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = CountingEmbeddingProvider()
    docs = [document("cached-first"), document("cached-second"), document("uncached-target")]
    config = settings(tmp_path)
    warmed = await RetrievalService.create(config, docs[:2], provider)
    partial_collection = warmed.vectors.collection
    warmed.close()
    provider.calls.clear()
    service = await RetrievalService.create(config, docs, provider, embed_missing=False)
    searched: list[set[str]] = []
    vector_search = service.vectors.search

    def tracked_search(vector: list[float], allowed: set[str], limit: int):
        searched.append(allowed)
        return vector_search(vector, allowed, limit)

    monkeypatch.setattr(service.vectors, "search", tracked_search)
    try:
        assert provider.calls == []
        assert service.cache_stats == {"hits": 2, "misses": 1}
        assert set(service.vectors.documents) == {"cached-first", "cached-second"}
        assert len(service.bm25.documents) == 3
        assert service.vectors.collection == partial_collection

        selected = await service.search("uncached-target", chart(), "test-school", "dense", 2)
        assert len(selected) == 2
        uncached = next(item for item in selected if item.document.id == "uncached-target")
        assert "bm25-unindexed" in uncached.matched_by
        assert searched == [{"cached-first", "cached-second"}]
        assert len(provider.calls) == 1 and len(provider.calls[0]) == 1
    finally:
        service.close()

    completed = await RetrievalService.create(config, docs, provider)
    try:
        assert completed.vectors.collection != partial_collection
        assert completed.index_version != service.index_version
    finally:
        completed.close()


@pytest.mark.asyncio
async def test_no_query_encoding_when_cached_documents_are_outside_the_selected_school(
    tmp_path: Path,
) -> None:
    provider = CountingEmbeddingProvider()
    cached = document("cached", school="other-school")
    unindexed = document("target-unindexed")
    config = settings(tmp_path)
    warmed = await RetrievalService.create(config, [cached], provider)
    warmed.close()
    provider.calls.clear()
    service = await RetrievalService.create(
        config, [cached, unindexed], provider, embed_missing=False
    )
    try:
        selected = await service.search("target", chart(), "test-school", "dense", 2)
        assert [item.document.id for item in selected] == ["target-unindexed"]
        assert provider.calls == []
    finally:
        service.close()


@pytest.mark.asyncio
async def test_explicit_prewarm_uses_the_configured_persistent_embedding_batch(
    tmp_path: Path,
) -> None:
    provider = CountingEmbeddingProvider()
    service = await RetrievalService.create(
        settings(tmp_path, local_embedding_cache_batch_size=1),
        [document("first"), document("second"), document("third")],
        provider,
    )
    try:
        assert [len(batch) for batch in provider.calls] == [1, 1, 1]
        assert len(service.vectors.documents) == 3
    finally:
        service.close()


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
        settings(tmp_path), [*parents, *originals, *fillers], HashEmbeddingProvider(8)
    )
    try:
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
    finally:
        service.close()


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
        settings(tmp_path), [*parents, *sources, *fillers], HashEmbeddingProvider(8)
    )
    try:
        ranked = [hit(doc, 1 - index / 20) for index, doc in enumerate([*parents, *fillers])]
        selected = service._expand_evidence_chain(ranked, set(service.documents), limit)
        single_chain_order = ["parent-0", "source-0", "parent-1", *[doc.id for doc in fillers]]
        assert [item.document.id for item in selected] == single_chain_order[:limit]
        assert all(item.document.id != "source-1" for item in selected)
        assert len(selected) == limit
    finally:
        service.close()


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
        settings(tmp_path), [*parents, *sources, *fillers], HashEmbeddingProvider(8)
    )
    try:
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
    finally:
        service.close()


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
        settings(tmp_path), [parent, original, blocked, annotation, other], HashEmbeddingProvider(8)
    )
    try:
        allowed = {"parent", "original", "annotation", "other"}
        ranked = [hit(parent), hit(other), hit(original, 0.5)]
        selected = service._expand_evidence_chain(ranked, allowed, 4)
        assert [item.document.id for item in selected] == ["parent", "original", "other"]
        assert selected[1] is ranked[2]
        assert selected[1].matched_by == ["test"]
    finally:
        service.close()
