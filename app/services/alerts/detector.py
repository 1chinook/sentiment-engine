"""Real-time sentiment spike detector and webhook dispatcher."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class SpikeAlertEvent(BaseModel):
    """Detected sentiment burst event emitted to subscribers/webhooks."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: str = "sentiment_spike_alert"
    topic: str
    direction: Literal["positive_hype", "negative_backlash"]
    avg_sentiment_score: float
    batch_size: int
    threshold: float
    dominant_label: str
    sample_texts: list[str]
    webhook_delivered: bool = False
    triggered_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class SpikeAlertDetector:
    """Evaluates analyzed batches for sentiment spikes and fires HTTP webhooks."""

    def __init__(
        self,
        sentiment_threshold: float | None = None,
        min_batch_size: int | None = None,
        webhook_url: str | None = None,
        webhook_timeout_sec: float | None = None,
        max_history: int = 50,
    ) -> None:
        settings = get_settings()
        self.sentiment_threshold: float = (
            settings.ALERT_SENTIMENT_THRESHOLD
            if sentiment_threshold is None
            else sentiment_threshold
        )
        self.min_batch_size: int = (
            settings.ALERT_MIN_BATCH_SIZE
            if min_batch_size is None
            else min_batch_size
        )
        self.webhook_timeout_sec: float = (
            settings.ALERT_WEBHOOK_TIMEOUT_SEC
            if webhook_timeout_sec is None
            else webhook_timeout_sec
        )

        self._webhook_urls: list[str] = []
        initial_url = (
            webhook_url
            if webhook_url is not None
            else settings.ALERT_WEBHOOK_URL
        )
        if initial_url and initial_url.strip():
            self._webhook_urls.append(initial_url.strip())

        self._history: deque[SpikeAlertEvent] = deque(maxlen=max_history)

    @property
    def webhook_urls(self) -> list[str]:
        """Return currently registered webhook target URLs."""
        return list(self._webhook_urls)

    def register_webhook(self, url: str) -> list[str]:
        """Add a webhook target URL idempotently."""
        cleaned = url.strip()
        if cleaned and cleaned not in self._webhook_urls:
            self._webhook_urls.append(cleaned)
            logger.info(
                "Registered spike alert webhook target",
                extra={"webhook_url": cleaned},
            )
        return self.webhook_urls

    def get_recent_alerts(self, limit: int = 20) -> list[SpikeAlertEvent]:
        """Return most recent spike alerts (newest first)."""
        items = list(self._history)
        items.reverse()
        return items[: max(1, limit)]

    async def _dispatch_webhooks(self, alert: SpikeAlertEvent) -> bool:
        """Send asynchronous HTTP POST notification to all registered webhooks."""
        if not self._webhook_urls:
            return False

        payload = alert.model_dump(mode="json")
        delivered_any = False

        async with httpx.AsyncClient(timeout=self.webhook_timeout_sec) as client:
            for target_url in self._webhook_urls:
                try:
                    response = await client.post(target_url, json=payload)
                    if 200 <= response.status_code < 300:
                        delivered_any = True
                        logger.info(
                            "Spike alert webhook delivered",
                            extra={
                                "webhook_url": target_url,
                                "alert_id": str(alert.id),
                                "topic": alert.topic,
                                "direction": alert.direction,
                                "status_code": response.status_code,
                            },
                        )
                    else:
                        logger.warning(
                            "Spike alert webhook returned non-2xx status",
                            extra={
                                "webhook_url": target_url,
                                "status_code": response.status_code,
                            },
                        )
                except Exception as exc:
                    logger.warning(
                        "Failed to deliver spike alert webhook",
                        extra={
                            "webhook_url": target_url,
                            "error": str(exc),
                        },
                    )

        return delivered_any

    async def evaluate_batch(
        self, records: list[dict[str, Any]]
    ) -> list[SpikeAlertEvent]:
        """Detect sentiment spikes per topic and notify registered webhooks."""
        if not records:
            return []

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            grouped[str(rec.get("topic", "general"))].append(rec)

        emitted: list[SpikeAlertEvent] = []
        for topic, group in grouped.items():
            if len(group) < self.min_batch_size:
                continue

            avg_score = round(
                sum(float(r.get("sentiment_score", 0.0)) for r in group)
                / len(group),
                4,
            )
            if abs(avg_score) < self.sentiment_threshold:
                continue

            direction: Literal["positive_hype", "negative_backlash"] = (
                "positive_hype" if avg_score > 0 else "negative_backlash"
            )
            label_counts = Counter(
                str(r.get("sentiment_label", "neutral")) for r in group
            )
            dominant_label = label_counts.most_common(1)[0][0]
            sample_texts = [str(r.get("raw_text", "")) for r in group[:3]]

            alert = SpikeAlertEvent(
                topic=topic,
                direction=direction,
                avg_sentiment_score=avg_score,
                batch_size=len(group),
                threshold=self.sentiment_threshold,
                dominant_label=dominant_label,
                sample_texts=sample_texts,
            )

            delivered = await self._dispatch_webhooks(alert)
            alert.webhook_delivered = delivered
            self._history.append(alert)
            emitted.append(alert)

            logger.info(
                "Sentiment spike alert triggered",
                extra={
                    "alert_id": str(alert.id),
                    "topic": topic,
                    "direction": direction,
                    "avg_sentiment_score": avg_score,
                    "batch_size": len(group),
                    "webhook_delivered": delivered,
                },
            )

        return emitted
