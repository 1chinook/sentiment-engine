"""FastAPI v1 endpoints for sentiment analysis, analytics, simulation, and DLQ."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1.schemas import (
    AnalyzedItemResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    DLQRecoveryResponse,
    HealthResponse,
    SimulationTriggerRequest,
    SimulationTriggerResponse,
    TopicSentimentResponse,
)
from app.core.config import Settings, get_settings
from app.db.repository import SentimentRepository
from app.db.session import AsyncSessionLocal, check_db_health
from app.services.nlp.analyzer import SentimentAnalyzer
from app.services.orchestrator import PipelineOrchestrator
from app.workers.scheduler import BackgroundWorkerManager

router = APIRouter(prefix="/api/v1", tags=["sentiment-engine"])


async def get_db_session(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield an async SQLAlchemy session bound to the app's session factory."""
    factory: async_sessionmaker[AsyncSession] = getattr(
        request.app.state, "session_factory", AsyncSessionLocal
    )
    async with factory() as session:
        yield session


def get_analyzer(request: Request) -> SentimentAnalyzer:
    """Resolve the singleton SentimentAnalyzer from app state."""
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        return SentimentAnalyzer()
    return analyzer


def get_orchestrator(request: Request) -> PipelineOrchestrator:
    """Resolve the central PipelineOrchestrator from app state."""
    orchestrator = getattr(request.app.state, "orchestrator", None)
    if orchestrator is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="PipelineOrchestrator is not initialized.",
        )
    return orchestrator


def get_worker_manager(request: Request) -> BackgroundWorkerManager:
    """Resolve the BackgroundWorkerManager from app state."""
    manager = getattr(request.app.state, "worker_manager", None)
    if manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="BackgroundWorkerManager is not initialized.",
        )
    return manager


def get_app_settings(request: Request) -> Settings:
    """Resolve runtime Settings from app state."""
    return getattr(request.app.state, "settings", get_settings())


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Analyze single or batch texts in real time",
)
async def analyze_texts(
    payload: AnalyzeRequest,
    analyzer: SentimentAnalyzer = Depends(get_analyzer),
    session: AsyncSession = Depends(get_db_session),
) -> AnalyzeResponse:
    """Classify one or many texts asynchronously without blocking the event loop."""
    texts = payload.resolved_texts()
    predictions, total_ms = await asyncio.to_thread(analyzer.analyze_batch, texts)

    persisted = False
    if payload.persist:
        now = datetime.now(timezone.utc)
        records = [
            {
                "id": uuid.uuid4(),
                "topic": payload.topic,
                "source": payload.source,
                "raw_text": pred.text,
                "sentiment_label": pred.label.value,
                "confidence_score": pred.confidence,
                "sentiment_score": pred.sentiment_score,
                "processing_time_ms": pred.processing_time_ms,
                "metadata_json": {"endpoint": "/api/v1/analyze"},
                "created_at": now,
            }
            for pred in predictions
        ]
        try:
            await SentimentRepository(session).bulk_insert(records)
            persisted = True
        except Exception as exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Failed to persist analyzed batch: {exc}",
            ) from exc

    return AnalyzeResponse(
        results=[
            AnalyzedItemResponse(
                text=pred.text,
                sentiment_label=pred.label,
                confidence_score=pred.confidence,
                sentiment_score=pred.sentiment_score,
                processing_time_ms=pred.processing_time_ms,
            )
            for pred in predictions
        ],
        batch_size=len(predictions),
        total_processing_time_ms=total_ms,
        persisted=persisted,
    )


@router.get(
    "/sentiment/{topic}",
    response_model=TopicSentimentResponse,
    summary="Get time-series sentiment metrics and moving averages for a topic",
)
async def get_topic_sentiment(
    topic: str,
    session: AsyncSession = Depends(get_db_session),
) -> TopicSentimentResponse:
    """Return 5-minute / 1-hour moving averages, volume, and time-series buckets."""
    analytics = await SentimentRepository(session).get_topic_analytics(topic=topic)
    return TopicSentimentResponse(**analytics)


@router.post(
    "/simulation/trigger",
    response_model=SimulationTriggerResponse,
    summary="Trigger an immediate biased synthetic sentiment spike",
)
async def trigger_simulation_spike(
    payload: SimulationTriggerRequest,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> SimulationTriggerResponse:
    """Inject a synthetic sentiment burst through the PipelineOrchestrator."""
    result = await orchestrator.trigger_simulation_spike(
        topic=payload.topic,
        sentiment_bias=payload.sentiment_bias,
        batch_size=payload.batch_size,
    )
    return SimulationTriggerResponse(**result)


@router.post(
    "/orchestrator/recover-dlq",
    response_model=DLQRecoveryResponse,
    summary="Manually trigger Dead Letter Queue recovery",
)
async def recover_dlq_endpoint(
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> DLQRecoveryResponse:
    """Replay failed JSONL batches back into PostgreSQL."""
    result = await orchestrator.recover_dlq()
    return DLQRecoveryResponse(**result)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Get system, DB, worker, DLQ, and NLP model health status",
)
async def health_check(
    request: Request,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
    worker_manager: BackgroundWorkerManager = Depends(get_worker_manager),
    analyzer: SentimentAnalyzer = Depends(get_analyzer),
    settings: Settings = Depends(get_app_settings),
) -> HealthResponse:
    """Inspect database connectivity, scheduler jobs, DLQ backlog, and NLP readiness."""
    session_factory: async_sessionmaker[AsyncSession] = getattr(
        request.app.state, "session_factory", AsyncSessionLocal
    )
    db_ok, db_latency_ms, db_error = await check_db_health(session_factory)
    pending_dlq, corrupted_dlq = await orchestrator.get_dlq_counts()
    worker_status = worker_manager.get_status()
    nlp_info = analyzer.health_info()

    overall_status = (
        "healthy" if db_ok and bool(nlp_info.get("loaded")) else "degraded"
    )

    return HealthResponse(
        status=overall_status,
        work_mode=settings.WORK_MODE,
        database={
            "connected": db_ok,
            "latency_ms": db_latency_ms,
            "error": db_error,
        },
        worker=worker_status,
        dlq={
            "pending_records": pending_dlq,
            "corrupted_records": corrupted_dlq,
        },
        nlp_model=nlp_info,
        timestamp=datetime.now(timezone.utc),
    )
