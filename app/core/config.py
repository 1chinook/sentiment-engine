"""Application settings and operating mode definitions."""

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class WorkMode(str, Enum):
    """Runtime operational mode for the sentiment engine."""

    TEST_SIMULATION = "TEST_SIMULATION"
    PRODUCTION = "PRODUCTION"


class Settings(BaseSettings):
    """Typed application settings loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "sentiment-engine"
    APP_VERSION: str = "1.0.0"
    WORK_MODE: WorkMode = WorkMode.TEST_SIMULATION
    LOG_LEVEL: str = "INFO"

    # Database (Async SQLAlchemy)
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@db:5432/sentiment_db"
    )
    DB_POOL_SIZE: int = Field(default=10, ge=1)
    DB_MAX_OVERFLOW: int = Field(default=20, ge=0)
    DB_ECHO: bool = False

    # NLP Model
    NLP_MODEL_NAME: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    NLP_USE_FALLBACK_IF_UNAVAILABLE: bool = True
    NLP_MAX_BATCH_SIZE: int = Field(default=64, ge=1, le=512)
    NLP_DEVICE: str = "cpu"

    # Ingestion & Pipeline Orchestrator
    INGESTION_BATCH_SIZE: int = Field(default=20, ge=1, le=500)
    BACKOFF_BASE_DELAY_SEC: float = Field(default=1.0, ge=0.0)
    BACKOFF_MAX_RETRIES: int = Field(default=3, ge=1, le=10)
    MOCK_FAILURE_RATE: float = Field(default=0.0, ge=0.0, le=1.0)

    # Dead Letter Queue (DLQ) Paths
    DLQ_FILE_PATH: Path = Path("logs/failed_batches.jsonl")
    CORRUPTED_DLQ_FILE_PATH: Path = Path("logs/corrupted_dlq.jsonl")

    # Background APScheduler Intervals
    SCHEDULER_INGEST_INTERVAL_SEC: int = Field(default=10, ge=1)
    SCHEDULER_DLQ_INTERVAL_SEC: int = Field(default=60, ge=1)

    # Security (Optional X-API-Key authentication; disabled when None/empty)
    API_KEY: str | None = None

    # Spike / Hype Alerting & Webhook Notification (Clip Automation Integration)
    ALERT_SENTIMENT_THRESHOLD: float = Field(default=0.45, ge=0.05, le=1.0)
    ALERT_MIN_BATCH_SIZE: int = Field(default=5, ge=1)
    ALERT_WEBHOOK_URL: str | None = None
    ALERT_WEBHOOK_TIMEOUT_SEC: float = Field(default=5.0, ge=0.5, le=30.0)

    # Real Data Ingestion Sources (Used in PRODUCTION mode)
    TWITCH_CHANNEL: str | None = None
    MARKET_RSS_URLS: str = "https://feeds.bbci.co.uk/news/business/rss.xml"
    MOVIE_RSS_URLS: str = "https://letterboxd.com/journal/rss/"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton Settings instance."""
    return Settings()
