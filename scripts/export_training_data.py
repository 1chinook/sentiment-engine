"""Export collected PostgreSQL sentiment records as JSONL dataset for Fine-Tuning."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from sqlalchemy import select

from app.db.models import SentimentRecord
from app.db.session import AsyncSessionLocal
from app.services.nlp.hype_scorer import score_message_hype


async def export_dataset(
    output_path: Path,
    topic: str | None = None,
    min_confidence: float = 0.65,
) -> int:
    """Query persisted records and write a Hugging Face compatible JSONL dataset."""
    stmt = select(SentimentRecord).where(
        SentimentRecord.confidence_score >= min_confidence
    )
    if topic:
        stmt = stmt.where(SentimentRecord.topic == topic)
    stmt = stmt.order_by(SentimentRecord.created_at.desc())

    async with AsyncSessionLocal() as session:
        result = await session.execute(stmt)
        rows = result.scalars().all()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    exported = 0

    with output_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            hype = score_message_hype(
                text=row.raw_text,
                base_label=row.sentiment_label,  # type: ignore[arg-type]
                base_confidence=row.confidence_score,
                base_sentiment_score=row.sentiment_score,
            )
            item = {
                "text": row.raw_text,
                "label": label_to_id.get(row.sentiment_label, 1),
                "label_name": row.sentiment_label,
                "confidence_score": row.confidence_score,
                "hype_score": hype.hype_score,
                "hype_tags": hype.hype_tags,
                "topic": row.topic,
                "source": row.source,
            }
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
            exported += 1

    print(f"Exported {exported} labeled records to {output_path}")
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export collected sentiment records to JSONL for Hugging Face fine-tuning."
    )
    parser.add_argument(
        "--output",
        default="logs/training_dataset.jsonl",
        help="Target JSONL file path",
    )
    parser.add_argument(
        "--topic",
        default=None,
        help="Optional topic filter (e.g. stream_chat)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.65,
        help="Minimum confidence threshold for pseudo-labeled training rows",
    )
    args = parser.parse_args()
    asyncio.run(
        export_dataset(
            output_path=Path(args.output),
            topic=args.topic,
            min_confidence=args.min_confidence,
        )
    )


if __name__ == "__main__":
    main()
