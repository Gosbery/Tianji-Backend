from __future__ import annotations

from dataclasses import dataclass

from bazi_api.db.sqlite import SQLiteDatabase
from bazi_api.integrations.embeddings import create_embedding_provider
from bazi_api.integrations.llm import AnswerGenerator
from bazi_api.modules.charts.service import ChartCalculator
from bazi_api.modules.conversations.repository import ConversationRepository
from bazi_api.modules.conversations.service import ChatService
from bazi_api.modules.feedback.repository import FeedbackRepository
from bazi_api.modules.knowledge.repository import KnowledgeRepository
from bazi_api.modules.observability.repository import TraceRepository
from bazi_api.modules.retrieval.service import RetrievalService

from .config import Settings


@dataclass(slots=True)
class ApplicationContainer:
    settings: Settings
    database: SQLiteDatabase
    charts: ChartCalculator
    knowledge: KnowledgeRepository
    retrieval: RetrievalService
    conversations: ConversationRepository
    feedback: FeedbackRepository
    traces: TraceRepository
    chat: ChatService

    def close(self) -> None:
        self.database.close()


async def build_container(settings: Settings) -> ApplicationContainer:
    settings.ensure_directories()
    knowledge = KnowledgeRepository(settings.knowledge_path)
    knowledge.load()
    embedding_provider = create_embedding_provider(settings)
    retrieval = await RetrievalService.create(
        settings=settings,
        documents=knowledge.documents(),
        embedding_provider=embedding_provider,
    )
    database = SQLiteDatabase(settings.database_path)
    conversations = ConversationRepository(database)
    feedback = FeedbackRepository(database)
    traces = TraceRepository(database)
    chat = ChatService(
        retrieval=retrieval,
        generator=AnswerGenerator(settings),
        conversations=conversations,
        traces=traces,
    )
    return ApplicationContainer(
        settings=settings,
        database=database,
        charts=ChartCalculator(),
        knowledge=knowledge,
        retrieval=retrieval,
        conversations=conversations,
        feedback=feedback,
        traces=traces,
        chat=chat,
    )
