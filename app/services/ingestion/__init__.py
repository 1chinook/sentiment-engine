"""Data ingestion interfaces and synthetic stream generators."""

from app.services.ingestion.base import BaseIngestor, RawIngestedMessage
from app.services.ingestion.mock_stream import MockStreamIngestor

__all__ = ["BaseIngestor", "MockStreamIngestor", "RawIngestedMessage"]
