from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

import httpx

from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.integrations.embeddings import create_embedding_provider
from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.experts.repository import ExpertRepository
from bazi_api.modules.feedback.repository import FeedbackRepository
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.service import RetrievalService
from bazi_api.modules.tasks.repository import TaskRepository
from bazi_api.modules.tasks.service import TaskService

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    http_client: httpx.AsyncClient
    database: SQLiteDatabase
    charts: ChartCalculator
    knowledge: KnowledgeRepository
    retrieval: RetrievalService
    conversations: ConversationRepository
    feedback: FeedbackRepository
    traces: TraceRepository
    chat: ChatService
    experts: ExpertRepository
    task_repository: TaskRepository
    tasks: TaskService

    async def close(self) -> None:
        try:
            await self.tasks.close()
        finally:
            try:
                self.database.close()
            finally:
                try:
                    await asyncio.to_thread(self.retrieval.close)
                finally:
                    await self.http_client.aclose()


async def build_container(settings: Settings) -> ApplicationContainer:
    settings.ensure_directories()
    http_client = httpx.AsyncClient(
        timeout=max(settings.llm_timeout_seconds, settings.embedding_timeout_seconds, 120.0),
        limits=httpx.Limits(
            max_connections=settings.http_max_connections,
            max_keepalive_connections=settings.http_max_keepalive_connections,
        ),
        transport=httpx.AsyncHTTPTransport(retries=settings.http_connect_retries),
    )
    database: SQLiteDatabase | None = None
    retrieval: RetrievalService | None = None
    try:
        knowledge = KnowledgeRepository(settings.knowledge_path)
        knowledge.load()
        experts = ExpertRepository(settings.knowledge_path / "experts", knowledge)
        experts.load()
        logger.info(
            "knowledge_loaded",
            extra={"provider": "yaml", "hits": len(knowledge.documents("personal_preview"))},
        )
        embedding_provider = create_embedding_provider(settings, http_client)
        retrieval = await RetrievalService.create(
            settings=settings,
            documents=knowledge.documents("personal_preview"),
            embedding_provider=embedding_provider,
            graph_nodes=knowledge.graph_nodes,
            graph_edges=knowledge.graph_edges,
            http_client=http_client,
        )
        logger.info(
            "retrieval_index_ready",
            extra={"provider": retrieval.model_version, "hits": len(retrieval.documents)},
        )
        database = SQLiteDatabase(
            settings.database_path,
            busy_timeout_ms=settings.database_busy_timeout_ms,
        )
        conversations = ConversationRepository(database)
        feedback = FeedbackRepository(
            database,
            rate_limit_per_minute=settings.feedback_rate_limit_per_minute,
            client_rate_limit_per_minute=settings.feedback_client_rate_limit_per_minute,
        )
        traces = TraceRepository(database)
        charts = ChartCalculator()
        chat = ChatService(
            retrieval=retrieval,
            generator=AnswerGenerator(settings, http_client),
            conversations=conversations,
            traces=traces,
            charts=charts,
            topics=knowledge.topics,
            experts=experts,
        )
        task_repository = TaskRepository(database)
        tasks = TaskService(
            repository=task_repository,
            conversations=conversations,
            charts=charts,
            experts=experts,
            chat=chat,
            concurrency=3,
        )
        container = ApplicationContainer(
            settings=settings,
            http_client=http_client,
            database=database,
            charts=charts,
            knowledge=knowledge,
            retrieval=retrieval,
            conversations=conversations,
            feedback=feedback,
            traces=traces,
            chat=chat,
            experts=experts,
            task_repository=task_repository,
            tasks=tasks,
        )
        await tasks.start()
        return container
    except BaseException:
        if database is not None:
            database.close()
        if retrieval is not None:
            await asyncio.to_thread(retrieval.close)
        await http_client.aclose()
        raise
