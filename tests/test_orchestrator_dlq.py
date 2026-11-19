"""Unit and resilience tests for PipelineOrchestrator, backoff, and DLQ recovery."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.db.models import SentimentRecord
from app.db.repository import SentimentRepository
from app.services.ingestion.base import BaseIngestor, RawIngestedMessage
from app.services.ingestion.mock_stream import MockStreamIngestor
from app.services.nlp.analyzer import SentimentAnalyzer
from app.services.orchestrator import PipelineOrchestrator


class _SlowIngestor(BaseIngestor):
    """Test double that pauses during ingestion to verify concurrency locking."""

    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        await asyncio.sleep(0.15)
        return [
            RawIngestedMessage(
                topic="stream_chat",
                source="slow_test",
                text="Harika yayın efsane oyun!",
            )
            for _ in range(batch_size)
        ]

    async def trigger_spike(
        self, topic: str, sentiment_bias: str, batch_size: int
    ) -> list[RawIngestedMessage]:
        return await self.fetch_batch(batch_size)


@pytest.mark.asyncio
async def test_orchestrator_concurrency_lock_skips_overlapping_cycle(
    nlp_analyzer: SentimentAnalyzer,
    session_factory: async_sessionmaker[AsyncSession],
    dlq_paths: tuple[Path, Path],
) -> None:
    """A second run_cycle() invoked while one is active must skip immediately."""
    dlq_path, corrupted_path = dlq_paths
    orch = PipelineOrchestrator(
        ingestor=_SlowIngestor(),
        nlp_service=nlp_analyzer,
        session_factory=session_factory,
        dlq_file_path=dlq_path,
        corrupted_dlq_file_path=corrupted_path,
    )

    first_task = asyncio.create_task(orch.run_cycle(batch_size=3))
    await asyncio.sleep(0.02)
    assert orch.is_running is True

    overlapping_result = await orch.run_cycle(batch_size=3)
    assert overlapping_result["status"] == "skipped"
    assert overlapping_result["reason"] == "already_running"

    first_result = await first_task
    assert first_result["status"] == "completed"
    assert first_result["ingested_count"] == 3
    assert orch.is_running is False


def _install_sleep_recorder(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Intercept asyncio.sleep and record non-zero backoff delays."""
    recorded_sleeps: list[float] = []
    real_sleep = asyncio.sleep

    async def _mock_sleep(delay: float) -> None:
        if delay > 0:
            recorded_sleeps.append(delay)
        await real_sleep(0)

    monkeypatch.setattr("app.services.orchestrator.asyncio.sleep", _mock_sleep)
    return recorded_sleeps


@pytest.mark.asyncio
async def test_orchestrator_exponential_backoff_succeeds_after_two_failures(
    orchestrator: PipelineOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Transient failures trigger 1s, 2s backoff delays before succeeding on attempt 3."""
    assert isinstance(orchestrator.ingestor, MockStreamIngestor)
    orchestrator.ingestor.inject_transient_failures(2)
    recorded_sleeps = _install_sleep_recorder(monkeypatch)

    result = await orchestrator.run_cycle(batch_size=4)
    assert result["status"] == "completed"
    assert result["ingested_count"] == 4
    assert recorded_sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_orchestrator_exponential_backoff_exhaustion(
    orchestrator: PipelineOrchestrator,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exceeding max retries records full 1s, 2s, 4s exponential backoff sequence."""
    assert isinstance(orchestrator.ingestor, MockStreamIngestor)
    orchestrator.ingestor.inject_transient_failures(5)
    recorded_sleeps = _install_sleep_recorder(monkeypatch)

    result = await orchestrator.run_cycle(batch_size=4)
    assert result["status"] == "failed"
    assert result["ingested_count"] == 0
    assert recorded_sleeps == [1.0, 2.0, 4.0]


@pytest.mark.asyncio
async def test_db_write_failure_rolls_back_and_routes_to_dlq(
    orchestrator: PipelineOrchestrator,
    session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When DB bulk insert fails, session rolls back and batch writes safely to DLQ."""
    rollback_called = False
    original_rollback = AsyncSession.rollback

    async def _spy_rollback(self: AsyncSession) -> None:
        nonlocal rollback_called
        rollback_called = True
        await original_rollback(self)

    async def _failing_bulk_insert(
        self: SentimentRepository, records: Any
    ) -> int:
        raise RuntimeError("Simulated PostgreSQL outage")

    monkeypatch.setattr(AsyncSession, "rollback", _spy_rollback)
    monkeypatch.setattr(SentimentRepository, "bulk_insert", _failing_bulk_insert)

    result = await orchestrator.run_cycle(batch_size=6)

    assert rollback_called is True
    assert result["status"] == "dlq_fallback"
    assert result["persisted_to_db"] is False
    assert result["routed_to_dlq"] is True
    assert result["ingested_count"] == 6

    pending_dlq, corrupted_dlq = await orchestrator.get_dlq_counts()
    assert pending_dlq == 6
    assert corrupted_dlq == 0

    assert orchestrator.dlq_file_path.exists()
    lines = [
        line
        for line in orchestrator.dlq_file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 6
    parsed_first = json.loads(lines[0])
    assert "record" in parsed_first
    assert parsed_first["error_reason"] == "Simulated PostgreSQL outage"

    async with session_factory() as session:
        db_count = await session.scalar(select(func.count(SentimentRecord.id)))
        assert db_count == 0


@pytest.mark.asyncio
async def test_recover_dlq_inserts_valid_records_and_isolates_corrupted_lines(
    orchestrator: PipelineOrchestrator,
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """recover_dlq imports valid JSONL entries to DB, clears DLQ, and isolates bad lines."""
    now_iso = datetime.now(timezone.utc).isoformat()
    valid_entries = [
        {
            "failed_at": now_iso,
            "error_reason": "temporary_outage",
            "record": {
                "id": str(uuid.uuid4()),
                "topic": "stream_chat",
                "source": "mock_twitch",
                "raw_text": "POGGERS efsane klip!!",
                "sentiment_label": "positive",
                "confidence_score": 0.91,
                "sentiment_score": 0.91,
                "processing_time_ms": 1.2,
                "metadata_json": {"synthetic": True},
                "created_at": now_iso,
            },
        },
        {
            "id": str(uuid.uuid4()),
            "topic": "movie_reviews",
            "source": "mock_imdb",
            "raw_text": "Senaryo çok kopuktu tam hayal kırıklığı",
            "sentiment_label": "negative",
            "confidence_score": 0.87,
            "sentiment_score": -0.87,
            "processing_time_ms": 0.9,
            "created_at": now_iso,
        },
        {
            "failed_at": now_iso,
            "error_reason": "temporary_outage",
            "record": {
                "id": str(uuid.uuid4()),
                "topic": "market_pulse",
                "source": "mock_reuters",
                "raw_text": "Piyasalar Fed kararı öncesi yatay seyrediyor",
                "sentiment_label": "neutral",
                "confidence_score": 0.78,
                "sentiment_score": 0.0,
                "processing_time_ms": 0.8,
                "created_at": now_iso,
            },
        },
    ]

    corrupted_lines = [
        "{not_valid_json",
        json.dumps({"topic": "stream_chat"}),  # missing required sentiment fields
    ]

    orchestrator.dlq_file_path.parent.mkdir(parents=True, exist_ok=True)
    with orchestrator.dlq_file_path.open("w", encoding="utf-8") as fp:
        for entry in valid_entries:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for bad in corrupted_lines:
            fp.write(bad + "\n")

    recovery = await orchestrator.recover_dlq()

    assert recovery["status"] == "recovered"
    assert recovery["recovered_count"] == 3
    assert recovery["corrupted_count"] == 2
    assert recovery["remaining_dlq_count"] == 0
    assert recovery["error"] is None

    pending_dlq, corrupted_count = await orchestrator.get_dlq_counts()
    assert pending_dlq == 0
    assert corrupted_count == 2

    assert orchestrator.corrupted_dlq_file_path.exists()
    isolated = [
        line
        for line in orchestrator.corrupted_dlq_file_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert isolated == corrupted_lines

    async with session_factory() as session:
        db_count = await session.scalar(select(func.count(SentimentRecord.id)))
        assert db_count == 3
