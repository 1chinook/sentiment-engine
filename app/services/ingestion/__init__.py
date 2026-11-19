"""Data ingestion interfaces, synthetic stream generators, and production feeds."""

from app.services.ingestion.base import BaseIngestor, RawIngestedMessage
from app.services.ingestion.mock_stream import MockStreamIngestor
from app.services.ingestion.real_streams import (
    CompositeProductionIngestor,
    RSSFeedIngestor,
    TwitchAnonymousIRCIngestor,
    build_ingestor,
    parse_rss_or_atom_xml,
)

__all__ = [
    "BaseIngestor",
    "CompositeProductionIngestor",
    "MockStreamIngestor",
    "RSSFeedIngestor",
    "RawIngestedMessage",
    "TwitchAnonymousIRCIngestor",
    "build_ingestor",
    "parse_rss_or_atom_xml",
]
