#!/usr/bin/env python3
"""Download local models when needed and prewarm the full preview index."""

from __future__ import annotations

import asyncio
import statistics
import time
from datetime import date
from datetime import time as clock_time

import httpx

from bazi_api.core.config import Settings, get_settings
from bazi_api.integrations.embeddings import create_embedding_provider
from bazi_api.modules.charts.schemas import BirthInput
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.retrieval.service import RetrievalService


async def main() -> None:
    settings = get_settings()
    settings.ensure_directories()
    repository = KnowledgeRepository(settings.knowledge_path)
    repository.load()
    async with httpx.AsyncClient(
        timeout=max(settings.llm_timeout_seconds, settings.embedding_timeout_seconds, 120.0),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
        transport=httpx.AsyncHTTPTransport(retries=settings.http_connect_retries),
    ) as http_client:
        await prewarm(settings, repository, http_client)


async def prewarm(
    settings: Settings,
    repository: KnowledgeRepository,
    http_client: httpx.AsyncClient,
) -> None:
    started = time.perf_counter()
    retrieval = await RetrievalService.create(
        settings,
        repository.documents("personal_preview"),
        create_embedding_provider(settings, http_client),
        repository.graph_nodes,
        repository.graph_edges,
        http_client,
    )
    index_ms = int((time.perf_counter() - started) * 1000)
    chart = ChartCalculator().calculate(
        BirthInput(date=date(1990, 1, 1), time=clock_time(12, 0), name="模型预热")
    )
    query_started = time.perf_counter()
    hits = await retrieval.search(
        "《子平真诠》怎样从月令讨论用神？",
        chart,
        "基础共识",
        "hybrid_rerank",
        6,
        "personal_preview",
    )
    query_ms = int((time.perf_counter() - query_started) * 1000)
    benchmark_queries = [
        "《子平真诠》如何论用神？",
        "相神在格局中有什么作用？",
        "四吉神为何也可能破格？",
        "四凶神为何也可能成格？",
        "墓库是否必须刑冲？",
        "行运为什么要结合原局？",
        "正官格如何取运？",
        "财格与印绶有什么关系？",
        "建禄月劫怎样取运？",
        "外格应当怎样取舍？",
    ] * 2
    latencies: list[float] = []
    for query in benchmark_queries:
        benchmark_started = time.perf_counter()
        await retrieval.search(
            query,
            chart,
            "基础共识",
            "hybrid_rerank",
            6,
            "personal_preview",
        )
        latencies.append((time.perf_counter() - benchmark_started) * 1000)
    p95_ms = int(statistics.quantiles(latencies, n=20)[18])
    print(
        {
            "documents": len(retrieval.documents),
            "model_version": retrieval.model_version,
            "index_version": retrieval.index_version,
            "cache": retrieval.cache_stats,
            "index_load_ms": index_ms,
            "warmup_query_ms": query_ms,
            "warm_query_p95_ms": p95_ms,
            "evidence": [hit.document.id for hit in hits],
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
