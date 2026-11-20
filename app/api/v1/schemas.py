"""Pydantic v2 request and response schemas for API v1."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import WorkMode
from app.db.models import SentimentLabel
from app.services.alerts.detector import SpikeAlertEvent


class AnalyzeRequest(BaseModel):
    """Request payload for real-time single or batch sentiment analysis."""

    model_config = ConfigDict(extra="forbid")

    text: str | None = None
    texts: list[str] | None = None
    topic: str = Field(default="general", min_length=1, max_length=128)
    source: str = Field(default="api", min_length=1, max_length=128)
    persist: bool = False

    @model_validator(mode="after")
    def validate_non_empty_inputs(self) -> "AnalyzeRequest":
        """Ensure at least one non-empty text string is supplied."""
        collected = self.resolved_texts()
        if not collected:
            raise ValueError(
                "At least one non-empty string must be provided via 'text' or 'texts'."
            )
        return self

    def resolved_texts(self) -> list[str]:
        """Return normalized list of texts combining `text` and `texts`."""
        items: list[str] = []
        if self.text is not None and self.text.strip():
            items.append(self.text.strip())
        if self.texts is not None:
            for item in self.texts:
                if item and item.strip():
                    items.append(item.strip())
        return items


class AnalyzedItemResponse(BaseModel):
    """Per-item sentiment and stream-hype inference result."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    sentiment_label: SentimentLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    hype_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hype_tags: list[str] = Field(default_factory=list)


class AnalyzeResponse(BaseModel):
    """Batch sentiment inference response."""

    model_config = ConfigDict(from_attributes=True)

    results: list[AnalyzedItemResponse]
    batch_size: int = Field(ge=1)
    total_processing_time_ms: float = Field(ge=0.0)
    persisted: bool


class WindowAggregationSchema(BaseModel):
    """Aggregated sentiment summary for a rolling time window."""

    model_config = ConfigDict(from_attributes=True)

    window_label: str
    message_volume: int = Field(ge=0)
    moving_avg_sentiment: float
    avg_confidence: float = Field(ge=0.0, le=1.0)
    sentiment_distribution: dict[str, int]
    sentiment_ratios: dict[str, float]


class TimeSeriesPointSchema(BaseModel):
    """Single time-bucket data point for topic sentiment trends."""

    model_config = ConfigDict(from_attributes=True)

    timestamp: datetime
    message_volume: int = Field(ge=0)
    avg_sentiment_score: float
    avg_confidence: float = Field(ge=0.0, le=1.0)
    dominant_label: str | None = None


class TopicSentimentResponse(BaseModel):
    """Topic-level temporal sentiment analytics response."""

    model_config = ConfigDict(from_attributes=True)

    topic: str
    total_volume: int = Field(ge=0)
    window_5m: WindowAggregationSchema
    window_1h: WindowAggregationSchema
    time_series: list[TimeSeriesPointSchema]
    generated_at: datetime


class SimulationTriggerRequest(BaseModel):
    """Control request to trigger a biased synthetic sentiment burst."""

    model_config = ConfigDict(extra="forbid")

    topic: Literal["stream_chat", "movie_reviews", "market_pulse"] = "stream_chat"
    sentiment_bias: Literal["positive", "negative", "neutral"] = "negative"
    batch_size: int = Field(default=25, ge=1, le=500)


class SimulationTriggerResponse(BaseModel):
    """Outcome of a triggered synthetic sentiment spike."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    topic: str
    sentiment_bias: str
    ingested_count: int = Field(ge=0)
    persisted_to_db: bool
    routed_to_dlq: bool
    processing_time_ms: float = Field(ge=0.0)
    distribution: dict[str, int]
    alerts_triggered: int = Field(default=0, ge=0)


class WebhookRegistrationRequest(BaseModel):
    """Request payload to register a target webhook URL for spike alerts."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(..., min_length=8, max_length=512)

    @model_validator(mode="after")
    def validate_http_scheme(self) -> "WebhookRegistrationRequest":
        cleaned = self.url.strip()
        if not (cleaned.startswith("http://") or cleaned.startswith("https://")):
            raise ValueError("Webhook URL must start with http:// or https://")
        self.url = cleaned
        return self


class AlertsStatusResponse(BaseModel):
    """Current spike detector configuration, webhooks, and recent alert history."""

    model_config = ConfigDict(from_attributes=True)

    threshold: float
    min_batch_size: int
    registered_webhooks: list[str]
    recent_alerts: list[SpikeAlertEvent]


class TwitchLiveRequest(BaseModel):
    """Request payload to sample and analyze a live Twitch streamer's chat."""

    model_config = ConfigDict(extra="forbid")

    channel: str = Field(..., min_length=1, max_length=128)
    listen_seconds: float = Field(default=4.0, ge=1.0, le=12.0)
    max_messages: int = Field(default=25, ge=1, le=100)
    persist: bool = True


class TwitchChatMessageItem(BaseModel):
    """Single analyzed live Twitch chat message with author username and hype score."""

    model_config = ConfigDict(from_attributes=True)

    username: str
    text: str
    sentiment_label: SentimentLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)
    hype_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hype_tags: list[str] = Field(default_factory=list)


class TwitchLiveResponse(BaseModel):
    """Batch sentiment & real-time Hype Meter summary for a live Twitch chat sample."""

    model_config = ConfigDict(from_attributes=True)

    channel: str
    fetched_count: int = Field(ge=0)
    avg_sentiment_score: float
    hype_score: float = Field(default=0.0, ge=0.0, le=100.0)
    hype_level: Literal["calm", "warming_up", "high_hype", "clip_moment"] = "calm"
    clip_recommended: bool = False
    messages_per_sec: float = Field(default=0.0, ge=0.0)
    dominant_triggers: list[str] = Field(default_factory=list)
    distribution: dict[str, int]
    total_processing_time_ms: float = Field(ge=0.0)
    persisted: bool
    alerts_triggered: int = Field(default=0, ge=0)
    messages: list[TwitchChatMessageItem]


class DLQRecoveryResponse(BaseModel):
    """Outcome of a manual or scheduled Dead Letter Queue recovery run."""

    model_config = ConfigDict(from_attributes=True)

    status: str
    recovered_count: int = Field(ge=0)
    corrupted_count: int = Field(ge=0)
    remaining_dlq_count: int = Field(ge=0)
    error: str | None = None


class HealthResponse(BaseModel):
    """Comprehensive service, database, worker, DLQ, and NLP model health."""

    model_config = ConfigDict(from_attributes=True)

    status: Literal["healthy", "degraded"]
    work_mode: WorkMode
    database: dict[str, Any]
    worker: dict[str, Any]
    dlq: dict[str, int]
    nlp_model: dict[str, Any]
    timestamp: datetime
