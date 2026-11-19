"""Shared pytest fixtures for async DB, orchestrator, and FastAPI client."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from app.core.config import Settings, WorkMode
from app.db.models import Base
from app.db.session import build_async_engine
from app.main import create_app
from app.services.ingestion.mock_stream import MockStreamIngestor
from app.services.nlp.analyzer import SentimentAnalyzer
from app.services.orchestrator import PipelineOrchestrator


@pytest.fixture
def dlq_paths(tmp_path: Path) -> tuple[Path, Path]:
    """Return isolated temporary DLQ and corrupted DLQ file paths."""
    return (
        tmp_path / "logs" / "failed_batches.jsonl",
        tmp_path / "logs" / "corrupted_dlq.jsonl",
    )


@pytest_asyncio.fixture
async def test_db_engine(tmp_path: Path) -> AsyncIterator[AsyncEngine]:
    """Create a fresh file-backed async SQLite engine per test."""
    db_file = tmp_path / "test_sentiment.db"
    engine = build_async_engine(
        database_url=f"sqlite+aiosqlite:///{db_file.as_posix()}",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture
def session_factory(
    test_db_engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    """Async session factory bound to the test SQLite engine."""
    return async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
    )


@pytest.fixture
def nlp_analyzer() -> SentimentAnalyzer:
    """Return singleton SentimentAnalyzer loaded with deterministic fallback backend."""
    analyzer = SentimentAnalyzer()
    analyzer.load_model(force_fallback=True)
    return analyzer


@pytest.fixture
def orchestrator(
    nlp_analyzer: SentimentAnalyzer,
    session_factory: async_sessionmaker[AsyncSession],
    dlq_paths: tuple[Path, Path],
) -> PipelineOrchestrator:
    """Create an isolated PipelineOrchestrator for unit & DLQ tests."""
    dlq_path, corrupted_path = dlq_paths
    ingestor = MockStreamIngestor(failure_rate=0.0, seed=42)
    return PipelineOrchestrator(
        ingestor=ingestor,
        nlp_service=nlp_analyzer,
        session_factory=session_factory,
        dlq_file_path=dlq_path,
        corrupted_dlq_file_path=corrupted_path,
        backoff_base_delay_sec=1.0,
        backoff_max_retries=3,
        default_batch_size=10,
    )


@pytest_asyncio.fixture
async def async_client(
    test_db_engine: AsyncEngine,
    session_factory: async_sessionmaker[AsyncSession],
    dlq_paths: tuple[Path, Path],
) -> AsyncIterator[httpx.AsyncClient]:
    """Provide an HTTPX AsyncClient wired to FastAPI with full lifespan active."""
    dlq_path, corrupted_path = dlq_paths
    test_settings = Settings(
        APP_NAME="sentiment-engine-test",
        WORK_MODE=WorkMode.TEST_SIMULATION,
        DATABASE_URL=str(test_db_engine.url),
        DLQ_FILE_PATH=dlq_path,
        CORRUPTED_DLQ_FILE_PATH=corrupted_path,
        SCHEDULER_INGEST_INTERVAL_SEC=3600,
        SCHEDULER_DLQ_INTERVAL_SEC=3600,
        INGESTION_BATCH_SIZE=10,
    )
    app = create_app(
        settings_override=test_settings,
        engine_override=test_db_engine,
        session_factory_override=session_factory,
        run_initial_seed=False,
        force_fallback_nlp=True,
    )

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            yield client
