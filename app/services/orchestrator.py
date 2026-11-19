"""Central PipelineOrchestrator coordinating ingestion, NLP, DB writes, and DLQ."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiofiles
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import SentimentLabel
from app.db.repository import SentimentRepository
from app.services.ingestion.base import BaseIngestor, RawIngestedMessage
from app.services.nlp.analyzer import SentimentAnalyzer, SentimentPrediction

logger = get_logger(__name__)

_REQUIRED_DLQ_FIELDS = (
    "topic",
    "raw_text",
    "sentiment_label",
    "confidence_score",
    "sentiment_score",
)
_VALID_SENTIMENT_LABELS = frozenset(label.value for label in SentimentLabel)


def _extract_valid_dlq_record(parsed: Any) -> dict[str, Any]:
    """Validate and unwrap a parsed DLQ JSON object into a DB-ready record dict."""
    if not isinstance(parsed, dict):
        raise ValueError("DLQ line JSON must be an object")

    candidate = parsed.get("record") if "record" in parsed else parsed
    if not isinstance(candidate, dict):
        raise ValueError("DLQ 'record' payload must be an object")

    for field in _REQUIRED_DLQ_FIELDS:
        if field not in candidate or candidate[field] is None:
            raise ValueError(f"Missing required DLQ field: {field}")

    raw_text = str(candidate["raw_text"])
    topic = str(candidate["topic"])
    label = str(candidate["sentiment_label"])
    if not raw_text.strip() or not topic.strip():
        raise ValueError("DLQ 'topic' and 'raw_text' must be non-empty")
    if label not in _VALID_SENTIMENT_LABELS:
        raise ValueError(f"Invalid sentiment_label in DLQ: {label!r}")

    confidence = float(candidate["confidence_score"])
    sentiment_score = float(candidate["sentiment_score"])
    if not (0.0 <= confidence <= 1.0):
        raise ValueError("DLQ confidence_score out of [0.0, 1.0] range")

    record_id = candidate.get("id")
    if record_id is not None:
        uuid.UUID(str(record_id))
    else:
        record_id = str(uuid.uuid4())

    created_at = candidate.get("created_at")
    if created_at is not None:
        datetime.fromisoformat(str(created_at))
    else:
        created_at = datetime.now(timezone.utc).isoformat()

    return {
        "id": str(record_id),
        "topic": topic,
        "source": str(candidate.get("source", "dlq_recovered")),
        "raw_text": raw_text,
        "sentiment_label": label,
        "confidence_score": confidence,
        "sentiment_score": sentiment_score,
        "processing_time_ms": float(candidate.get("processing_time_ms", 0.0)),
        "metadata_json": candidate.get("metadata_json"),
        "created_at": str(created_at),
    }


class PipelineOrchestrator:
    """Central runtime coordinator with locking, exponential backoff, and DLQ."""

    def __init__(
        self,
        ingestor: BaseIngestor,
        nlp_service: SentimentAnalyzer,
        session_factory: async_sessionmaker[AsyncSession],
        dlq_file_path: Path | None = None,
        corrupted_dlq_file_path: Path | None = None,
        backoff_base_delay_sec: float | None = None,
        backoff_max_retries: int | None = None,
        default_batch_size: int | None = None,
    ) -> None:
        settings = get_settings()
        self.ingestor = ingestor
        self.nlp_service = nlp_service
        self.session_factory = session_factory

        self.dlq_file_path: Path = Path(
            dlq_file_path or settings.DLQ_FILE_PATH
        )
        self.corrupted_dlq_file_path: Path = Path(
            corrupted_dlq_file_path or settings.CORRUPTED_DLQ_FILE_PATH
        )
        self.backoff_base_delay_sec: float = (
            settings.BACKOFF_BASE_DELAY_SEC
            if backoff_base_delay_sec is None
            else backoff_base_delay_sec
        )
        self.backoff_max_retries: int = (
            settings.BACKOFF_MAX_RETRIES
            if backoff_max_retries is None
            else backoff_max_retries
        )
        self.default_batch_size: int = (
            settings.INGESTION_BATCH_SIZE
            if default_batch_size is None
            else default_batch_size
        )

        self._is_running: bool = False
        self._cycle_lock = asyncio.Lock()
        self._dlq_lock = asyncio.Lock()

        self._last_cycle_at: datetime | None = None
        self._last_cycle_status: str = "idle"
        self._total_cycles_completed: int = 0
        self._total_records_persisted: int = 0

    @property
    def is_running(self) -> bool:
        """Whether an ingestion-analysis cycle is currently executing."""
        return self._is_running

    async def _fetch_with_backoff(
        self, batch_size: int
    ) -> list[RawIngestedMessage]:
        """Fetch raw messages with exponential backoff (1s, 2s, 4s)."""
        attempt = 0
        while True:
            try:
                return await self.ingestor.fetch_batch(batch_size=batch_size)
            except Exception as exc:
                if attempt >= self.backoff_max_retries:
                    logger.error(
                        "Ingestion failed after maximum backoff retries",
                        extra={
                            "attempts": attempt + 1,
                            "max_retries": self.backoff_max_retries,
                            "error": str(exc),
                        },
                    )
                    raise

                delay = self.backoff_base_delay_sec * (2**attempt)
                logger.warning(
                    "Transient ingestion error; retrying after exponential backoff",
                    extra={
                        "attempt": attempt + 1,
                        "delay_sec": delay,
                        "error": str(exc),
                    },
                )
                await asyncio.sleep(delay)
                attempt += 1

    @staticmethod
    def _build_enriched_records(
        messages: list[RawIngestedMessage],
        predictions: list[SentimentPrediction],
    ) -> list[dict[str, Any]]:
        """Combine raw ingested messages and NLP predictions into DB-ready dicts."""
        records: list[dict[str, Any]] = []
        for msg, pred in zip(messages, predictions, strict=True):
            records.append(
                {
                    "id": str(msg.id),
                    "topic": msg.topic,
                    "source": msg.source,
                    "raw_text": msg.text,
                    "sentiment_label": pred.label.value,
                    "confidence_score": pred.confidence,
                    "sentiment_score": pred.sentiment_score,
                    "processing_time_ms": pred.processing_time_ms,
                    "metadata_json": msg.metadata,
                    "created_at": msg.timestamp.astimezone(timezone.utc).isoformat(),
                }
            )
        return records

    async def _append_to_dlq(
        self, records: list[dict[str, Any]], error_reason: str
    ) -> None:
        """Persist failed records atomically to local JSONL Dead Letter Queue."""
        if not records:
            return

        failed_at = datetime.now(timezone.utc).isoformat()
        lines = [
            json.dumps(
                {
                    "failed_at": failed_at,
                    "error_reason": error_reason,
                    "record": record,
                },
                ensure_ascii=False,
            )
            + "\n"
            for record in records
        ]

        async with self._dlq_lock:
            self.dlq_file_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(
                self.dlq_file_path, mode="a", encoding="utf-8"
            ) as fp:
                await fp.writelines(lines)
                await fp.flush()

        logger.warning(
            "Batch routed to local Dead Letter Queue",
            extra={
                "dlq_path": str(self.dlq_file_path),
                "record_count": len(records),
                "error_reason": error_reason,
            },
        )

    async def _atomic_rewrite_file(
        self, target_path: Path, lines: list[str]
    ) -> None:
        """Atomically replace `target_path` contents using a temp file swap."""
        target_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target_path.with_name(f".{target_path.name}.{uuid.uuid4().hex}.tmp")
        async with aiofiles.open(tmp_path, mode="w", encoding="utf-8") as fp:
            if lines:
                await fp.writelines(lines)
            await fp.flush()
        await asyncio.to_thread(os.replace, tmp_path, target_path)

    async def _persist_with_dlq_fallback(
        self, records: list[dict[str, Any]]
    ) -> tuple[bool, bool, str | None]:
        """Attempt DB bulk insert; rollback and write to DLQ on failure."""
        if not records:
            return True, False, None

        async with self.session_factory() as session:
            repo = SentimentRepository(session)
            try:
                inserted = await repo.bulk_insert(records)
                self._total_records_persisted += inserted
                return True, False, None
            except Exception as exc:
                await session.rollback()
                error_msg = str(exc)
                logger.error(
                    "Database bulk insert failed; rolled back transaction and routing to DLQ",
                    extra={
                        "record_count": len(records),
                        "error": error_msg,
                    },
                )
                await self._append_to_dlq(records, error_reason=error_msg)
                return False, True, error_msg

    async def run_cycle(self, batch_size: int | None = None) -> dict[str, Any]:
        """Execute a full ingest -> NLP -> DB/DLQ cycle with concurrency guard."""
        if self._is_running or self._cycle_lock.locked():
            logger.warning("Cycle skipped: previous cycle still running")
            return {
                "status": "skipped",
                "reason": "already_running",
                "ingested_count": 0,
                "persisted_to_db": False,
                "routed_to_dlq": False,
                "processing_time_ms": 0.0,
            }

        async with self._cycle_lock:
            self._is_running = True
            started = time.perf_counter()
            resolved_batch_size = batch_size or self.default_batch_size
            try:
                messages = await self._fetch_with_backoff(resolved_batch_size)
                if not messages:
                    self._last_cycle_at = datetime.now(timezone.utc)
                    self._last_cycle_status = "empty"
                    return {
                        "status": "empty",
                        "ingested_count": 0,
                        "persisted_to_db": True,
                        "routed_to_dlq": False,
                        "processing_time_ms": 0.0,
                    }

                texts = [msg.text for msg in messages]
                # Isolate CPU-bound NLP inference off the asyncio event loop
                predictions, nlp_ms = await asyncio.to_thread(
                    self.nlp_service.analyze_batch, texts
                )

                records = self._build_enriched_records(messages, predictions)
                persisted, routed_to_dlq, db_error = (
                    await self._persist_with_dlq_fallback(records)
                )

                total_ms = round((time.perf_counter() - started) * 1000.0, 3)
                self._last_cycle_at = datetime.now(timezone.utc)
                self._total_cycles_completed += 1
                self._last_cycle_status = (
                    "completed" if persisted else "dlq_fallback"
                )

                logger.info(
                    "Pipeline cycle finished",
                    extra={
                        "status": self._last_cycle_status,
                        "ingested_count": len(records),
                        "nlp_ms": nlp_ms,
                        "total_ms": total_ms,
                        "routed_to_dlq": routed_to_dlq,
                    },
                )
                return {
                    "status": self._last_cycle_status,
                    "ingested_count": len(records),
                    "persisted_to_db": persisted,
                    "routed_to_dlq": routed_to_dlq,
                    "processing_time_ms": total_ms,
                    "error": db_error,
                }
            except Exception as exc:
                total_ms = round((time.perf_counter() - started) * 1000.0, 3)
                self._last_cycle_at = datetime.now(timezone.utc)
                self._last_cycle_status = "failed"
                logger.error(
                    "Pipeline cycle failed during ingestion or NLP phase",
                    extra={"error": str(exc), "total_ms": total_ms},
                )
                return {
                    "status": "failed",
                    "ingested_count": 0,
                    "persisted_to_db": False,
                    "routed_to_dlq": False,
                    "processing_time_ms": total_ms,
                    "error": str(exc),
                }
            finally:
                self._is_running = False

    async def trigger_simulation_spike(
        self,
        topic: str,
        sentiment_bias: str,
        batch_size: int = 25,
    ) -> dict[str, Any]:
        """Inject a targeted synthetic spike and process it end-to-end."""
        started = time.perf_counter()
        messages = await self.ingestor.trigger_spike(
            topic=topic,
            sentiment_bias=sentiment_bias,
            batch_size=batch_size,
        )
        texts = [msg.text for msg in messages]
        predictions, _ = await asyncio.to_thread(
            self.nlp_service.analyze_batch, texts
        )
        records = self._build_enriched_records(messages, predictions)
        persisted, routed_to_dlq, db_error = (
            await self._persist_with_dlq_fallback(records)
        )

        counts = Counter(pred.label.value for pred in predictions)
        distribution = {
            SentimentLabel.POSITIVE.value: counts.get(SentimentLabel.POSITIVE.value, 0),
            SentimentLabel.NEGATIVE.value: counts.get(SentimentLabel.NEGATIVE.value, 0),
            SentimentLabel.NEUTRAL.value: counts.get(SentimentLabel.NEUTRAL.value, 0),
        }
        total_ms = round((time.perf_counter() - started) * 1000.0, 3)

        return {
            "status": "completed" if persisted else "dlq_fallback",
            "topic": topic,
            "sentiment_bias": sentiment_bias,
            "ingested_count": len(records),
            "persisted_to_db": persisted,
            "routed_to_dlq": routed_to_dlq,
            "processing_time_ms": total_ms,
            "distribution": distribution,
            "error": db_error,
        }

    async def recover_dlq(self) -> dict[str, Any]:
        """Atomically replay failed_batches.jsonl into DB and isolate corrupted lines."""
        async with self._dlq_lock:
            if not self.dlq_file_path.exists():
                return {
                    "status": "empty",
                    "recovered_count": 0,
                    "corrupted_count": 0,
                    "remaining_dlq_count": 0,
                    "error": None,
                }

            async with aiofiles.open(
                self.dlq_file_path, mode="r", encoding="utf-8"
            ) as fp:
                raw_lines = await fp.readlines()

            non_empty_lines = [line for line in raw_lines if line.strip()]
            if not non_empty_lines:
                await self._atomic_rewrite_file(self.dlq_file_path, [])
                return {
                    "status": "empty",
                    "recovered_count": 0,
                    "corrupted_count": 0,
                    "remaining_dlq_count": 0,
                    "error": None,
                }

            valid_records: list[dict[str, Any]] = []
            valid_raw_lines: list[str] = []
            corrupted_lines: list[str] = []

            for line in non_empty_lines:
                normalized_line = line if line.endswith("\n") else f"{line}\n"
                try:
                    parsed = json.loads(line)
                    record = _extract_valid_dlq_record(parsed)
                    valid_records.append(record)
                    valid_raw_lines.append(normalized_line)
                except Exception:
                    corrupted_lines.append(normalized_line)

            if corrupted_lines:
                self.corrupted_dlq_file_path.parent.mkdir(parents=True, exist_ok=True)
                async with aiofiles.open(
                    self.corrupted_dlq_file_path, mode="a", encoding="utf-8"
                ) as bad_fp:
                    await bad_fp.writelines(corrupted_lines)
                    await bad_fp.flush()

                logger.warning(
                    "Isolated corrupted DLQ entries",
                    extra={
                        "corrupted_count": len(corrupted_lines),
                        "corrupted_dlq_path": str(self.corrupted_dlq_file_path),
                    },
                )

            if not valid_records:
                await self._atomic_rewrite_file(self.dlq_file_path, [])
                return {
                    "status": "completed_with_corrupted_only",
                    "recovered_count": 0,
                    "corrupted_count": len(corrupted_lines),
                    "remaining_dlq_count": 0,
                    "error": None,
                }

            async with self.session_factory() as session:
                repo = SentimentRepository(session)
                try:
                    inserted = await repo.bulk_insert(valid_records)
                    self._total_records_persisted += inserted
                except Exception as exc:
                    await session.rollback()
                    # Keep only valid lines in DLQ so corrupted lines aren't re-isolated
                    await self._atomic_rewrite_file(
                        self.dlq_file_path, valid_raw_lines
                    )
                    error_msg = str(exc)
                    logger.error(
                        "DLQ recovery DB insert failed; retained valid records in DLQ",
                        extra={
                            "remaining_dlq_count": len(valid_records),
                            "error": error_msg,
                        },
                    )
                    return {
                        "status": "failed",
                        "recovered_count": 0,
                        "corrupted_count": len(corrupted_lines),
                        "remaining_dlq_count": len(valid_records),
                        "error": error_msg,
                    }

            await self._atomic_rewrite_file(self.dlq_file_path, [])
            logger.info(
                "Successfully recovered DLQ batch into database",
                extra={
                    "recovered_count": inserted,
                    "corrupted_count": len(corrupted_lines),
                },
            )
            return {
                "status": "recovered",
                "recovered_count": inserted,
                "corrupted_count": len(corrupted_lines),
                "remaining_dlq_count": 0,
                "error": None,
            }

    async def get_dlq_counts(self) -> tuple[int, int]:
        """Return (pending_dlq_count, corrupted_dlq_count) asynchronously."""

        async def _count_lines(path: Path) -> int:
            if not path.exists():
                return 0
            async with aiofiles.open(path, mode="r", encoding="utf-8") as fp:
                lines = await fp.readlines()
            return sum(1 for line in lines if line.strip())

        async with self._dlq_lock:
            pending = await _count_lines(self.dlq_file_path)
            corrupted = await _count_lines(self.corrupted_dlq_file_path)
        return pending, corrupted

    def status_snapshot(self) -> dict[str, Any]:
        """Return runtime status snapshot for health diagnostics."""
        return {
            "is_running": self._is_running,
            "last_cycle_at": (
                self._last_cycle_at.isoformat() if self._last_cycle_at else None
            ),
            "last_cycle_status": self._last_cycle_status,
            "total_cycles_completed": self._total_cycles_completed,
            "total_records_persisted": self._total_records_persisted,
        }
