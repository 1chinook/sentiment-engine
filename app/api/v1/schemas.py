"""Pydantic v2 request and response schemas for API v1."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.core.config import WorkMode
from app.db.models import SentimentLabel


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
    """Per-item sentiment inference result."""

    model_config = ConfigDict(from_attributes=True)

    text: str
    sentiment_label: SentimentLabel
    confidence_score: float = Field(ge=0.0, le=1.0)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    processing_time_ms: float = Field(ge=0.0)


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
