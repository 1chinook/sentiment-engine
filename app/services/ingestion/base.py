"""Abstract ingestion interface and canonical raw message schema."""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RawIngestedMessage(BaseModel):
    """Canonical representation of an ingested text event prior to NLP scoring."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    topic: str = Field(..., min_length=1, max_length=128)
    source: str = Field(..., min_length=1, max_length=128)
    text: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)


class BaseIngestor(ABC):
    """Abstract base class for asynchronous sentiment data ingestors."""

    @abstractmethod
    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        """Fetch the next batch of messages from the underlying data stream."""

    @abstractmethod
    async def trigger_spike(
        self,
        topic: str,
        sentiment_bias: str,
        batch_size: int,
    ) -> list[RawIngestedMessage]:
        """Generate or pull a targeted burst of messages biased toward a sentiment."""
