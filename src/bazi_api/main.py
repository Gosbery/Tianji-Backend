import logging
import re
import time
import tomllib
import uuid
from contextlib import asynccontextmanager
from importlib.metadata import PackageNotFoundError, version

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bazi_api.core.config import BACKEND_ROOT, Settings, get_settings
from bazi_api.core.container import build_container
from bazi_api.core.errors import BaziApiError, UpstreamServiceError
from bazi_api.core.logging import configure_logging, request_id_context
from bazi_api.modules.charts.router import router as charts_router
from bazi_api.modules.conversations.router import router as conversations_router
from bazi_api.modules.experts.router import router as experts_router
from bazi_api.modules.feedback.router import router as feedback_router
from bazi_api.modules.knowledge.router import router as knowledge_router
from bazi_api.modules.observability.router import router as observability_router
from bazi_api.modules.system.router import router as system_router
from bazi_api.modules.tasks.router import router as tasks_router

API_PREFIX = "/api/v1"
LEGACY_API_PREFIX = "/api"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")

logger = logging.getLogger(__name__)


def _package_version() -> str:
    project_path = BACKEND_ROOT / "pyproject.toml"
    if project_path.is_file():
        with project_path.open("rb") as project_file:
            project = tomllib.load(project_file)
        return str(project["project"]["version"])
    try:
        return version("bazi-api")
    except PackageNotFoundError:
        return "0+unknown"


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        container = await build_container(app_settings)
        app.state.container = container
        try:
            yield
        finally:
            await container.close()

    app = FastAPI(
        title=app_settings.app_name,
        version=_package_version(),
        lifespan=lifespan,
    )

    @app.exception_handler(BaziApiError)
    async def handle_domain_error(request: Request, exc: BaziApiError) -> JSONResponse:
        request_id = _request_id(request)
        logger.warning(
            "request_domain_error",
            extra={
                "error_code": exc.code,
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id,
                "status_code": exc.status_code,
            },
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "detail": exc.public_message,
                "code": exc.code,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(httpx.HTTPError)
    async def handle_http_client_error(request: Request, exc: httpx.HTTPError) -> JSONResponse:
        upstream_error = UpstreamServiceError()
        request_id = _request_id(request)
        logger.warning(
            "unhandled_upstream_error",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "error_code": upstream_error.code,
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id,
                "status_code": upstream_error.status_code,
            },
        )
        return JSONResponse(
            status_code=upstream_error.status_code,
            content={
                "detail": upstream_error.public_message,
                "code": upstream_error.code,
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception(
            "unhandled_request_error",
            exc_info=(type(exc), exc, exc.__traceback__),
            extra={
                "error_code": "internal_error",
                "method": request.method,
                "path": request.url.path,
                "request_id": request_id,
                "status_code": 500,
            },
        )
        return JSONResponse(
            status_code=500,
            content={
                "detail": "服务内部错误",
                "code": "internal_error",
                "request_id": request_id,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        supplied_request_id = request.headers.get("X-Request-ID", "")
        request_id = (
            supplied_request_id
            if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
            else str(uuid.uuid4())
        )
        request.state.request_id = request_id
        context_token = request_id_context.set(request_id)
        started = time.perf_counter()
        try:
            try:
                response = await call_next(request)
            except Exception as exc:
                response = await handle_unexpected_error(request, exc)
            duration_ms = round((time.perf_counter() - started) * 1000, 2)
            response.headers["X-Request-ID"] = request_id
            if _is_legacy_path(request.url.path):
                response.headers["Deprecation"] = "true"
                if app_settings.legacy_api_sunset:
                    response.headers["Sunset"] = app_settings.legacy_api_sunset
                successor = f"{API_PREFIX}{request.url.path.removeprefix(LEGACY_API_PREFIX)}"
                response.headers["Link"] = f'<{successor}>; rel="successor-version"'
                logger.warning(
                    "legacy_api_called",
                    extra={"method": request.method, "path": request.url.path},
                )
            logger.info(
                "request_completed",
                extra={
                    "duration_ms": duration_ms,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                },
            )
            return response
        finally:
            request_id_context.reset(context_token)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins,
        allow_credentials=app_settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "PATCH"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["Deprecation", "Link", "Sunset", "X-Request-ID"],
    )
    for router in (
        system_router,
        charts_router,
        knowledge_router,
        experts_router,
        conversations_router,
        tasks_router,
        feedback_router,
        observability_router,
    ):
        app.include_router(router, prefix=API_PREFIX)
        if app_settings.legacy_api_enabled:
            app.include_router(router, prefix=LEGACY_API_PREFIX, include_in_schema=False)
    return app


def _request_id(request: Request) -> str:
    state_request_id = getattr(request.state, "request_id", "")
    if isinstance(state_request_id, str) and state_request_id:
        return state_request_id
    context_request_id = request_id_context.get()
    return context_request_id or str(uuid.uuid4())


def _is_legacy_path(path: str) -> bool:
    return path == LEGACY_API_PREFIX or (
        path.startswith(f"{LEGACY_API_PREFIX}/")
        and path != API_PREFIX
        and not path.startswith(f"{API_PREFIX}/")
    )


app = create_app()
