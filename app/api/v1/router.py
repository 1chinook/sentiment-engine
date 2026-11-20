"""FastAPI v1 endpoints for sentiment analysis, analytics, simulation, and DLQ."""

import asyncio
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.v1.schemas import (
    AlertsStatusResponse,
    AnalyzedItemResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    ChatClusterItem,
    DLQRecoveryResponse,
    HealthResponse,
    SimulationTriggerRequest,
    SimulationTriggerResponse,
    TopicSentimentResponse,
    TwitchChatMessageItem,
    TwitchLiveRequest,
    TwitchLiveResponse,
    WebhookRegistrationRequest,
)
from app.core.config import Settings, get_settings
from app.core.security import verify_api_key
from app.db.models import SentimentLabel
from app.db.repository import SentimentRepository
from app.db.session import AsyncSessionLocal, check_db_health
from app.services.ingestion.real_streams import (
    TwitchAnonymousIRCIngestor,
    normalize_twitch_channel,
)
from app.services.nlp.analyzer import SentimentAnalyzer
from app.services.nlp.hype_scorer import (
    analyze_chat_cohesion,
    compute_stream_hype_summary,
)
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
    dependencies=[Depends(verify_api_key)],
    summary="Analyze single or batch texts in real time",
)
async def analyze_texts(
    payload: AnalyzeRequest,
    request: Request,
    analyzer: SentimentAnalyzer = Depends(get_analyzer),
    session: AsyncSession = Depends(get_db_session),
) -> AnalyzeResponse:
    """Classify one or many texts asynchronously without blocking the event loop."""
    texts = payload.resolved_texts()
    predictions, total_ms = await asyncio.to_thread(analyzer.analyze_batch, texts)
    cohesion = analyze_chat_cohesion(texts)

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

        orchestrator: PipelineOrchestrator | None = getattr(
            request.app.state, "orchestrator", None
        )
        if orchestrator is not None:
            await orchestrator.alert_detector.evaluate_batch(records)

    return AnalyzeResponse(
        results=[
            AnalyzedItemResponse(
                text=pred.text,
                sentiment_label=pred.label,
                confidence_score=pred.confidence,
                sentiment_score=pred.sentiment_score,
                processing_time_ms=pred.processing_time_ms,
                hype_score=pred.hype_score,
                hype_tags=list(pred.hype_tags),
            )
            for pred in predictions
        ],
        batch_size=len(predictions),
        total_processing_time_ms=total_ms,
        persisted=persisted,
        collective_sync_score=round(cohesion.cohesion_score * 100.0, 1),
        wave_detected=cohesion.wave_detected,
        wave_type=cohesion.wave_type,
        chat_clusters=[
            ChatClusterItem(
                label=c.label,
                match_count=c.match_count,
                unique_users=c.unique_users,
                cohesion_type=c.cohesion_type,
                sample_texts=c.sample_texts,
            )
            for c in cohesion.clusters
        ],
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
    dependencies=[Depends(verify_api_key)],
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
    dependencies=[Depends(verify_api_key)],
    summary="Manually trigger Dead Letter Queue recovery",
)
async def recover_dlq_endpoint(
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> DLQRecoveryResponse:
    """Replay failed JSONL batches back into PostgreSQL."""
    result = await orchestrator.recover_dlq()
    return DLQRecoveryResponse(**result)


@router.get(
    "/alerts",
    response_model=AlertsStatusResponse,
    summary="Get recent sentiment spike alerts and registered webhook targets",
)
async def get_spike_alerts(
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> AlertsStatusResponse:
    """Return recent hype/backlash spike alerts for clip automation & monitoring."""
    detector = orchestrator.alert_detector
    return AlertsStatusResponse(
        threshold=detector.sentiment_threshold,
        min_batch_size=detector.min_batch_size,
        registered_webhooks=detector.webhook_urls,
        recent_alerts=detector.get_recent_alerts(limit=25),
    )


@router.post(
    "/alerts/webhooks",
    response_model=AlertsStatusResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Register a webhook URL to receive automatic sentiment spike callbacks",
)
async def register_spike_webhook(
    payload: WebhookRegistrationRequest,
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> AlertsStatusResponse:
    """Register an HTTP endpoint (e.g. clip automation service) for spike alerts."""
    detector = orchestrator.alert_detector
    detector.register_webhook(payload.url)
    return AlertsStatusResponse(
        threshold=detector.sentiment_threshold,
        min_batch_size=detector.min_batch_size,
        registered_webhooks=detector.webhook_urls,
        recent_alerts=detector.get_recent_alerts(limit=25),
    )


@router.post(
    "/live/twitch",
    response_model=TwitchLiveResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Connect to a live Twitch channel chat via IRC and analyze messages",
)
async def analyze_live_twitch_chat(
    payload: TwitchLiveRequest,
    analyzer: SentimentAnalyzer = Depends(get_analyzer),
    orchestrator: PipelineOrchestrator = Depends(get_orchestrator),
) -> TwitchLiveResponse:
    """Pull real-time chat messages from a live Twitch channel and run NLP inference."""
    channel = normalize_twitch_channel(payload.channel)
    if not channel:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid Twitch channel name.",
        )

    ingestor = TwitchAnonymousIRCIngestor(
        channel=channel,
        read_window_sec=payload.listen_seconds,
    )
    raw_messages = await ingestor.fetch_batch(batch_size=payload.max_messages)

    empty_dist = {
        SentimentLabel.POSITIVE.value: 0,
        SentimentLabel.NEGATIVE.value: 0,
        SentimentLabel.NEUTRAL.value: 0,
    }
    if not raw_messages:
        return TwitchLiveResponse(
            channel=channel,
            fetched_count=0,
            avg_sentiment_score=0.0,
            distribution=empty_dist,
            total_processing_time_ms=0.0,
            persisted=False,
            alerts_triggered=0,
            messages=[],
        )

    texts = [msg.text for msg in raw_messages]
    predictions, total_ms = await asyncio.to_thread(
        analyzer.analyze_batch, texts
    )

    records = orchestrator._build_enriched_records(raw_messages, predictions)
    persisted = False
    if payload.persist:
        persisted, _, _ = await orchestrator._persist_with_dlq_fallback(records)

    alerts = await orchestrator.alert_detector.evaluate_batch(records)

    dist = dict(empty_dist)
    total_score = 0.0
    items: list[TwitchChatMessageItem] = []

    for msg, pred in zip(raw_messages, predictions, strict=True):
        lbl = pred.label.value
        dist[lbl] = dist.get(lbl, 0) + 1
        total_score += pred.sentiment_score
        items.append(
            TwitchChatMessageItem(
                username=str(msg.metadata.get("username", "viewer")),
                text=pred.text,
                sentiment_label=pred.label,
                confidence_score=pred.confidence,
                sentiment_score=pred.sentiment_score,
                processing_time_ms=pred.processing_time_ms,
                hype_score=pred.hype_score,
                hype_tags=list(pred.hype_tags),
            )
        )

    usernames = [
        str(msg.metadata.get("username", f"viewer_{idx}"))
        for idx, msg in enumerate(raw_messages)
    ]
    cohesion = analyze_chat_cohesion(texts=texts, usernames=usernames)
    hype_summary = compute_stream_hype_summary(
        message_hype_scores=[pred.hype_score for pred in predictions],
        all_tags=[list(pred.hype_tags) for pred in predictions],
        listen_seconds=payload.listen_seconds,
        cohesion_result=cohesion,
    )
    avg_score = round(total_score / len(items), 4)
    return TwitchLiveResponse(
        channel=channel,
        fetched_count=len(items),
        avg_sentiment_score=avg_score,
        hype_score=hype_summary["hype_score"],
        hype_level=hype_summary["hype_level"],
        clip_recommended=hype_summary["clip_recommended"],
        messages_per_sec=hype_summary["messages_per_sec"],
        collective_sync_score=hype_summary["collective_sync_score"],
        wave_detected=hype_summary["wave_detected"],
        wave_type=hype_summary["wave_type"],
        chat_clusters=[
            ChatClusterItem(**cluster)
            for cluster in hype_summary["chat_clusters"]
        ],
        dominant_triggers=hype_summary["dominant_triggers"],
        distribution=dist,
        total_processing_time_ms=total_ms,
        persisted=persisted,
        alerts_triggered=len(alerts),
        messages=items,
    )


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
