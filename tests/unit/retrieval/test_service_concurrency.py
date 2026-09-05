import asyncio
from datetime import date, time
from pathlib import Path
from typing import Any

import httpx
import pytest

from bazi_api.core.config import Settings
from bazi_api.core.errors import InvalidUpstreamResponseError
from bazi_api.integrations.embeddings import HashEmbeddingProvider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.schemas import GraphNode
from bazi_api.modules.retrieval.schemas import RetrievalDocument, RetrievalHit
from bazi_api.modules.retrieval.service import (
    KnowledgeGraphIndex,
    RetrievalService,
    VectorIndex,
    query_mentions_term,
)


def _settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "vector_backend": "memory",
        "embedding_provider": "hash",
        "reranker_provider": "lexical",
        "database_path": tmp_path / "app.db",
        "qdrant_path": tmp_path / "qdrant",
        "embedding_cache_path": tmp_path / "embeddings.sqlite3",
    }
    values.update(overrides)
    return Settings(**values)


def _documents() -> list[RetrievalDocument]:
    return [
        RetrievalDocument(
            id="fire",
            kind="knowledge_card",
            layer=3,
            title="火的基础象义",
            text="火是一种五行关系语言。",
            source="test",
            school="基础共识",
            concepts=["火", "五行"],
        ),
        RetrievalDocument(
            id="train",
            kind="knowledge_card",
            layer=3,
            title="出行说明",
            text="火车是一种交通工具。",
            source="test",
            school="基础共识",
            concepts=["出行"],
        ),
    ]


def test_single_han_concepts_require_boundaries() -> None:
    assert not query_mentions_term("我今天坐火车出行", "火")
    assert not query_mentions_term("甲方提出了方案", "甲")
    assert not query_mentions_term("不要冲动决定", "冲")
    assert query_mentions_term("请解释「火」", "火")
    assert query_mentions_term("火是不是只代表热情", "火")
    assert query_mentions_term("怎样理解五行", "五行")

    graph = KnowledgeGraphIndex(
        [
            GraphNode(
                id="concept:fire",
                type="concept",
                name="火",
                status="reviewed",
                verification_level="human_review",
                reviewed_by="test-reviewer",
                reviewed_at="2026-09-03",
                review_note="测试夹具人工确认",
            )
        ],
        [],
    )
    assert graph.expand("火车如何运行", "reviewed_only") == ([], set())
    assert graph.expand("解释「火」", "reviewed_only")[1] == {"concept:fire"}


@pytest.mark.asyncio
async def test_sync_indexes_and_reranker_run_in_worker_threads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service = await RetrievalService.create(
        _settings(tmp_path), _documents(), HashEmbeddingProvider()
    )
    chart = ChartCalculator().calculate(BirthInput(date=date(1990, 1, 1), time=time(12)))
    real_to_thread = asyncio.to_thread
    owners: list[object | None] = []

    async def tracked_to_thread(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        owners.append(getattr(function, "__self__", None))
        return await real_to_thread(function, *args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", tracked_to_thread)

    await service.search("请解释「火」", chart, "基础共识", "hybrid_rerank", 2)

    assert service.bm25 in owners
    assert service.vectors in owners
    assert service.reranker in owners


@pytest.mark.asyncio
async def test_tuning_changes_ranking_and_index_version(tmp_path: Path) -> None:
    baseline = await RetrievalService.create(
        _settings(tmp_path, exact_title_boost=0.5),
        _documents(),
        HashEmbeddingProvider(),
    )
    tuned = await RetrievalService.create(
        _settings(tmp_path, exact_title_boost=0.8),
        _documents(),
        HashEmbeddingProvider(),
    )
    baseline_hits = [
        RetrievalHit(document=baseline.documents["fire"], score=0.0, matched_by=[]),
        RetrievalHit(document=baseline.documents["train"], score=0.6, matched_by=[]),
    ]
    tuned_hits = [
        RetrievalHit(document=tuned.documents["fire"], score=0.0, matched_by=[]),
        RetrievalHit(document=tuned.documents["train"], score=0.6, matched_by=[]),
    ]

    baseline._apply_title_boost("火的基础象义", baseline_hits)
    tuned._apply_title_boost("火的基础象义", tuned_hits)

    assert [item.document.id for item in baseline_hits] == ["train", "fire"]
    assert [item.document.id for item in tuned_hits] == ["fire", "train"]
    assert baseline.index_version != tuned.index_version


@pytest.mark.asyncio
async def test_lightrag_rejects_malformed_success_payload(tmp_path: Path) -> None:
    def malformed_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"status": "success", "data": {"chunks": [42]}},
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(malformed_response)
    ) as http_client:
        service = await RetrievalService.create(
            _settings(tmp_path, lightrag_base_url="https://lightrag.invalid"),
            _documents(),
            HashEmbeddingProvider(),
            http_client=http_client,
        )
        chart = ChartCalculator().calculate(
            BirthInput(date=date(1990, 1, 1), time=time(12))
        )

        with pytest.raises(InvalidUpstreamResponseError, match="chunk 格式无效"):
            await service.search("测试", chart, "基础共识", "lightrag", 2)


@pytest.mark.asyncio
async def test_lightrag_only_returns_locally_verified_eligible_documents(
    tmp_path: Path,
) -> None:
    preview = RetrievalDocument(
        id="preview:rule",
        kind="knowledge_card",
        layer=3,
        title="预览规则",
        text="本地机器校勘内容。",
        source="test",
        school="实验流派",
        concepts=[],
        review_status="machine_verified",
        verification_level="single_source_integrity",
        confidence=0.6,
    )

    def lightrag_response(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "status": "success",
                "data": {
                    "chunks": [
                        {
                            "chunk_id": "external-draft",
                            "file_path": "injected.md",
                            "content": "远端注入内容",
                        },
                        {
                            "chunk_id": "chunk-reviewed",
                            "file_path": "/export/fire.md",
                            "content": "被篡改的远端文本",
                        },
                        {
                            "chunk_id": "chunk-preview",
                            "file_path": "/export/preview-rule.md",
                            "content": "被篡改的预览文本",
                        },
                    ],
                    "entities": [],
                },
            },
            request=request,
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lightrag_response)
    ) as http_client:
        service = await RetrievalService.create(
            _settings(tmp_path, lightrag_base_url="https://lightrag.invalid"),
            [*_documents(), preview],
            HashEmbeddingProvider(),
            http_client=http_client,
        )
        chart = ChartCalculator().calculate(
            BirthInput(date=date(1990, 1, 1), time=time(12))
        )

        reviewed = await service.search(
            "测试", chart, "基础共识", "lightrag", 3, "reviewed_only"
        )
        personal_preview = await service.search(
            "测试", chart, "实验流派", "lightrag", 3, "personal_preview"
        )

    assert [hit.document.id for hit in reviewed] == ["fire"]
    assert reviewed[0].document.text == "火是一种五行关系语言。"
    assert [hit.document.id for hit in personal_preview] == ["fire", "preview:rule"]
    assert all(hit.document.id != "lightrag:external-draft" for hit in personal_preview)


def test_qdrant_filters_before_top_k_and_releases_local_lock(tmp_path: Path) -> None:
    qdrant_client = pytest.importorskip("qdrant_client")
    models = qdrant_client.models
    settings = _settings(tmp_path, vector_backend="qdrant")
    documents = _documents()
    index = VectorIndex(
        settings,
        documents,
        [[1.0, 0.0], [0.0, 1.0]],
        "test-index",
    )
    assert index.qdrant is not None
    client = index.qdrant
    stale_points = [
        models.PointStruct(
            id=index_number + 1000,
            vector=[1.0, 0.0],
            payload={"document_id": f"removed-{index_number}"},
        )
        for index_number in range(8)
    ]
    client.upsert(index.collection, stale_points, wait=True)

    hits = index.search([1.0, 0.0], {"train"}, 1)

    assert [document_id for document_id, _ in hits] == ["train"]
    local_client = client._client
    index.close()
    assert local_client._closed is True

    reopened = VectorIndex(
        settings,
        documents,
        [[1.0, 0.0], [0.0, 1.0]],
        "new-index",
    )
    assert reopened.qdrant is not None
    collections = {
        collection.name
        for collection in reopened.qdrant.get_collections().collections
        if collection.name.startswith("bazi_knowledge_")
    }
    assert collections == {"bazi_knowledge_new-index"}
    reopened.close()
