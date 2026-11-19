"""SQLAlchemy ORM models and sentiment domain enums."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import DateTime, Float, Index, JSON, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


class SentimentLabel(str, Enum):
    """Standardized sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class SentimentRecord(Base):
    """Persisted sentiment analysis record with temporal indexing."""

    __tablename__ = "sentiment_records"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    topic: Mapped[str] = mapped_column(
        String(128),
        index=True,
        nullable=False,
    )
    source: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        default="unknown",
    )
    raw_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    sentiment_label: Mapped[str] = mapped_column(
        String(32),
        index=True,
        nullable=False,
    )
    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    sentiment_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    processing_time_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )
    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        index=True,
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_sentiment_records_topic_created_at", "topic", "created_at"),
    )

    def __repr__(self) -> str:
        return (
            f"<SentimentRecord id={self.id} topic={self.topic!r} "
            f"label={self.sentiment_label!r} score={self.sentiment_score:.3f}>"
        )
