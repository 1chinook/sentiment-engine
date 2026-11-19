"""APScheduler-based background worker manager for periodic ingestion and DLQ."""

from __future__ import annotations

from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.core.config import WorkMode, get_settings
from app.core.logging import get_logger
from app.services.orchestrator import PipelineOrchestrator

logger = get_logger(__name__)


class BackgroundWorkerManager:
    """Manages APScheduler background jobs bound to the FastAPI event loop."""

    def __init__(
        self,
        orchestrator: PipelineOrchestrator,
        work_mode: WorkMode | None = None,
        ingest_interval_sec: int | None = None,
        dlq_interval_sec: int | None = None,
    ) -> None:
        settings = get_settings()
        self.orchestrator = orchestrator
        self.work_mode: WorkMode = work_mode or settings.WORK_MODE
        self.ingest_interval_sec: int = (
            ingest_interval_sec or settings.SCHEDULER_INGEST_INTERVAL_SEC
        )
        self.dlq_interval_sec: int = (
            dlq_interval_sec or settings.SCHEDULER_DLQ_INTERVAL_SEC
        )
        self._scheduler = AsyncIOScheduler(timezone="UTC")
        self._started: bool = False

    def start(self) -> None:
        """Register scheduled jobs and start the AsyncIOScheduler."""
        if self._started:
            return

        job_name = (
            "Synthetic Stream Ingestion & Sentiment Cycle"
            if self.work_mode == WorkMode.TEST_SIMULATION
            else "Production Stream Ingestion & Sentiment Cycle"
        )
        self._scheduler.add_job(
            self.orchestrator.run_cycle,
            trigger=IntervalTrigger(seconds=self.ingest_interval_sec),
            id="ingestion_cycle_job",
            name=job_name,
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self._scheduler.add_job(
            self.orchestrator.recover_dlq,
            trigger=IntervalTrigger(seconds=self.dlq_interval_sec),
            id="dlq_recovery_job",
            name="Dead Letter Queue Auto-Recovery",
            replace_existing=True,
            max_instances=1,
            coalesce=True,
        )

        self._scheduler.start()
        self._started = True
        logger.info(
            "Background worker scheduler started",
            extra={
                "work_mode": self.work_mode.value,
                "ingest_interval_sec": self.ingest_interval_sec,
                "dlq_interval_sec": self.dlq_interval_sec,
            },
        )

    def shutdown(self) -> None:
        """Gracefully shut down the scheduler without blocking."""
        if not self._started:
            return
        self._scheduler.shutdown(wait=False)
        self._started = False
        logger.info("Background worker scheduler stopped")

    def get_status(self) -> dict[str, Any]:
        """Return active worker state, scheduled jobs, and orchestrator snapshot."""
        jobs_info: list[dict[str, Any]] = []
        if self._started:
            for job in self._scheduler.get_jobs():
                jobs_info.append(
                    {
                        "id": job.id,
                        "name": job.name,
                        "next_run_time": (
                            job.next_run_time.isoformat()
                            if job.next_run_time
                            else None
                        ),
                    }
                )

        return {
            "active": self._started and self._scheduler.running,
            "work_mode": self.work_mode.value,
            "jobs": jobs_info,
            "orchestrator": self.orchestrator.status_snapshot(),
        }
