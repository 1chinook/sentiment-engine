"""Async SQLAlchemy engine, session factory, and health probes."""

import time
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.db.models import Base


def build_async_engine(
    database_url: str | None = None,
    echo: bool | None = None,
) -> AsyncEngine:
    """Create an async SQLAlchemy engine tuned for PostgreSQL or SQLite."""
    settings = get_settings()
    url = database_url or settings.DATABASE_URL
    sql_echo = settings.DB_ECHO if echo is None else echo

    engine_kwargs: dict[str, Any] = {
        "echo": sql_echo,
    }

    if not url.startswith("sqlite"):
        engine_kwargs.update(
            {
                "pool_size": settings.DB_POOL_SIZE,
                "max_overflow": settings.DB_MAX_OVERFLOW,
                "pool_pre_ping": True,
            }
        )

    return create_async_engine(url, **engine_kwargs)


engine: AsyncEngine = build_async_engine()

AsyncSessionLocal: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)


async def init_db(target_engine: AsyncEngine | None = None) -> None:
    """Create database tables idempotently on application startup."""
    active_engine = target_engine or engine
    async with active_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_db(target_engine: AsyncEngine | None = None) -> None:
    """Dispose the async database engine connection pool cleanly."""
    active_engine = target_engine or engine
    await active_engine.dispose()


async def check_db_health(
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> tuple[bool, float, str | None]:
    """Run a lightweight connectivity check and return (healthy, latency_ms, error)."""
    factory = session_factory or AsyncSessionLocal
    started = time.perf_counter()
    try:
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return True, latency_ms, None
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        return False, latency_ms, str(exc)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with AsyncSessionLocal() as session:
        yield session
