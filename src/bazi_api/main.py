from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from bazi_api.core.config import Settings, get_settings
from bazi_api.core.container import build_container
from bazi_api.modules.charts.router import router as charts_router
from bazi_api.modules.conversations.router import router as conversations_router
from bazi_api.modules.feedback.router import router as feedback_router
from bazi_api.modules.knowledge.router import router as knowledge_router
from bazi_api.modules.observability.router import router as observability_router
from bazi_api.modules.system.router import router as system_router

API_PREFIX = "/api/v1"
LEGACY_API_PREFIX = "/api"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = await build_container(app_settings)
        app.state.container = container
        try:
            yield
        finally:
            container.close()

    app = FastAPI(
        title=app_settings.app_name,
        version="0.2.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in (
        system_router,
        charts_router,
        knowledge_router,
        conversations_router,
        feedback_router,
        observability_router,
    ):
        app.include_router(router, prefix=API_PREFIX)
        app.include_router(router, prefix=LEGACY_API_PREFIX, include_in_schema=False)
    return app


app = create_app()
