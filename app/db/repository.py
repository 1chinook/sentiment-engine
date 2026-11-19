"""Repository for bulk sentiment persistence and time-series analytics."""

import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import SentimentLabel, SentimentRecord


def _ensure_utc(dt: datetime) -> datetime:
    """Normalize naive datetimes (e.g. from SQLite) to UTC-aware datetimes."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _parse_datetime(value: datetime | str | None) -> datetime:
    """Convert ISO strings or datetimes into a UTC-aware datetime."""
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        return _ensure_utc(value)
    parsed = datetime.fromisoformat(value)
    return _ensure_utc(parsed)


def _parse_uuid(value: uuid.UUID | str | None) -> uuid.UUID:
    """Convert string UUID representations to uuid.UUID objects."""
    if value is None:
        return uuid.uuid4()
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _coerce_record(item: dict[str, Any] | SentimentRecord) -> SentimentRecord:
    """Normalize a dictionary or ORM object into a SentimentRecord instance."""
    if isinstance(item, SentimentRecord):
        return item

    return SentimentRecord(
        id=_parse_uuid(item.get("id")),
        topic=str(item["topic"]),
        source=str(item.get("source", "unknown")),
        raw_text=str(item["raw_text"]),
        sentiment_label=str(item["sentiment_label"]),
        confidence_score=float(item["confidence_score"]),
        sentiment_score=float(item["sentiment_score"]),
        processing_time_ms=float(item.get("processing_time_ms", 0.0)),
        metadata_json=item.get("metadata_json"),
        created_at=_parse_datetime(item.get("created_at")),
    )


def _compute_window_summary(
    rows: list[tuple[str, float, float, datetime]],
    window_label: str,
) -> dict[str, Any]:
    """Aggregate sentiment metrics over a collection of rows."""
    distribution: dict[str, int] = {
        SentimentLabel.POSITIVE.value: 0,
        SentimentLabel.NEGATIVE.value: 0,
        SentimentLabel.NEUTRAL.value: 0,
    }

    count = len(rows)
    if count == 0:
        return {
            "window_label": window_label,
            "message_volume": 0,
            "moving_avg_sentiment": 0.0,
            "avg_confidence": 0.0,
            "sentiment_distribution": distribution,
            "sentiment_ratios": {k: 0.0 for k in distribution},
        }

    total_sentiment = 0.0
    total_confidence = 0.0

    for label, confidence, sentiment_score, _ in rows:
        total_sentiment += float(sentiment_score)
        total_confidence += float(confidence)
        if label in distribution:
            distribution[label] += 1
        else:
            distribution[label] = distribution.get(label, 0) + 1

    ratios = {
        label: round(val / count, 4) for label, val in distribution.items()
    }

    return {
        "window_label": window_label,
        "message_volume": count,
        "moving_avg_sentiment": round(total_sentiment / count, 4),
        "avg_confidence": round(total_confidence / count, 4),
        "sentiment_distribution": distribution,
        "sentiment_ratios": ratios,
    }


class SentimentRepository:
    """Async repository encapsulating writes and temporal aggregations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def bulk_insert(
        self, records: Sequence[dict[str, Any] | SentimentRecord]
    ) -> int:
        """Persist a batch of sentiment records atomically and return insert count."""
        if not records:
            return 0

        orm_objects = [_coerce_record(item) for item in records]
        self.session.add_all(orm_objects)
        await self.session.flush()
        await self.session.commit()
        return len(orm_objects)

    async def get_topic_analytics(
        self, topic: str, now: datetime | None = None
    ) -> dict[str, Any]:
        """Return 5m/1h moving averages, message volumes, and minute time-series."""
        reference_now = _ensure_utc(now or datetime.now(timezone.utc))
        cutoff_5m = reference_now - timedelta(minutes=5)
        cutoff_1h = reference_now - timedelta(hours=1)

        total_stmt = select(func.count(SentimentRecord.id)).where(
            SentimentRecord.topic == topic
        )
        total_result = await self.session.execute(total_stmt)
        total_volume = int(total_result.scalar_one() or 0)

        recent_stmt = (
            select(
                SentimentRecord.sentiment_label,
                SentimentRecord.confidence_score,
                SentimentRecord.sentiment_score,
                SentimentRecord.created_at,
            )
            .where(
                SentimentRecord.topic == topic,
                SentimentRecord.created_at >= cutoff_1h,
            )
            .order_by(SentimentRecord.created_at.asc())
        )
        recent_result = await self.session.execute(recent_stmt)
        rows_1h: list[tuple[str, float, float, datetime]] = [
            (
                str(label),
                float(confidence),
                float(score),
                _ensure_utc(created_at),
            )
            for label, confidence, score, created_at in recent_result.all()
        ]

        rows_5m = [row for row in rows_1h if row[3] >= cutoff_5m]

        window_5m = _compute_window_summary(rows_5m, window_label="5m")
        window_1h = _compute_window_summary(rows_1h, window_label="1h")

        buckets: dict[datetime, list[tuple[str, float, float]]] = defaultdict(list)
        for label, confidence, score, created_at in rows_1h:
            bucket_ts = created_at.replace(second=0, microsecond=0)
            buckets[bucket_ts].append((label, confidence, score))

        time_series: list[dict[str, Any]] = []
        for bucket_ts in sorted(buckets.keys()):
            items = buckets[bucket_ts]
            bucket_count = len(items)
            avg_sentiment = sum(item[2] for item in items) / bucket_count
            avg_confidence = sum(item[1] for item in items) / bucket_count

            label_counts: dict[str, int] = defaultdict(int)
            for item in items:
                label_counts[item[0]] += 1
            dominant_label = max(label_counts.items(), key=lambda kv: kv[1])[0]

            time_series.append(
                {
                    "timestamp": bucket_ts,
                    "message_volume": bucket_count,
                    "avg_sentiment_score": round(avg_sentiment, 4),
                    "avg_confidence": round(avg_confidence, 4),
                    "dominant_label": dominant_label,
                }
            )

        return {
            "topic": topic,
            "total_volume": total_volume,
            "window_5m": window_5m,
            "window_1h": window_1h,
            "time_series": time_series,
            "generated_at": reference_now,
        }
