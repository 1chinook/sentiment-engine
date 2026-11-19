"""FastAPI application entrypoint and lifespan management."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.api.v1.router import router as v1_router
from app.core.config import Settings, WorkMode, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.session import AsyncSessionLocal, dispose_db, engine, init_db
from app.services.ingestion.mock_stream import MockStreamIngestor
from app.services.nlp.analyzer import SentimentAnalyzer
from app.services.orchestrator import PipelineOrchestrator
from app.workers.scheduler import BackgroundWorkerManager

logger = get_logger(__name__)


def create_app(
    settings_override: Settings | None = None,
    engine_override: AsyncEngine | None = None,
    session_factory_override: async_sessionmaker[AsyncSession] | None = None,
    run_initial_seed: bool = True,
    force_fallback_nlp: bool = False,
) -> FastAPI:
    """Create and configure the sentiment-engine FastAPI application."""
    active_settings = settings_override or get_settings()
    active_engine = engine_override or engine
    active_session_factory = session_factory_override or AsyncSessionLocal
    owns_engine = engine_override is None

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging(
            level=active_settings.LOG_LEVEL,
            work_mode=active_settings.WORK_MODE,
            log_dir=active_settings.DLQ_FILE_PATH.parent,
        )
        logger.info(
            "Starting sentiment-engine microservice",
            extra={
                "app_name": active_settings.APP_NAME,
                "version": active_settings.APP_VERSION,
                "work_mode": active_settings.WORK_MODE.value,
            },
        )

        await init_db(active_engine)

        analyzer = SentimentAnalyzer()
        await asyncio.to_thread(
            analyzer.load_model, force_fallback=force_fallback_nlp
        )

        ingestor = MockStreamIngestor(
            failure_rate=active_settings.MOCK_FAILURE_RATE
        )
        orchestrator = PipelineOrchestrator(
            ingestor=ingestor,
            nlp_service=analyzer,
            session_factory=active_session_factory,
            dlq_file_path=active_settings.DLQ_FILE_PATH,
            corrupted_dlq_file_path=active_settings.CORRUPTED_DLQ_FILE_PATH,
            backoff_base_delay_sec=active_settings.BACKOFF_BASE_DELAY_SEC,
            backoff_max_retries=active_settings.BACKOFF_MAX_RETRIES,
            default_batch_size=active_settings.INGESTION_BATCH_SIZE,
        )
        worker_manager = BackgroundWorkerManager(
            orchestrator=orchestrator,
            work_mode=active_settings.WORK_MODE,
            ingest_interval_sec=active_settings.SCHEDULER_INGEST_INTERVAL_SEC,
            dlq_interval_sec=active_settings.SCHEDULER_DLQ_INTERVAL_SEC,
        )

        application.state.settings = active_settings
        application.state.engine = active_engine
        application.state.session_factory = active_session_factory
        application.state.analyzer = analyzer
        application.state.ingestor = ingestor
        application.state.orchestrator = orchestrator
        application.state.worker_manager = worker_manager

        if (
            active_settings.WORK_MODE == WorkMode.TEST_SIMULATION
            and run_initial_seed
        ):
            await orchestrator.run_cycle()

        worker_manager.start()

        try:
            yield
        finally:
            worker_manager.shutdown()
            if owns_engine:
                await dispose_db(active_engine)
            logger.info("sentiment-engine microservice shutdown complete")

    application = FastAPI(
        title="sentiment-engine",
        description=(
            "Asynchronous bilingual (TR/EN) sentiment analysis and synthetic stream "
            "processing microservice with PipelineOrchestrator & DLQ recovery."
        ),
        version=active_settings.APP_VERSION,
        lifespan=lifespan,
    )
    application.include_router(v1_router)

    @application.get("/", tags=["meta"])
    async def root() -> dict[str, str]:
        return {
            "service": active_settings.APP_NAME,
            "version": active_settings.APP_VERSION,
            "work_mode": active_settings.WORK_MODE.value,
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
        }

    return application


app = create_app()
