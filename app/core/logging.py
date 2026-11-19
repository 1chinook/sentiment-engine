"""Structured JSON logging configuration for sentiment-engine."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import WorkMode, get_settings

_RESERVED_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line structured JSON objects."""

    def __init__(self, work_mode: WorkMode) -> None:
        super().__init__()
        self.work_mode = work_mode.value

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "work_mode": self.work_mode,
            "message": record.getMessage(),
        }

        extra_context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_LOG_ATTRS and not key.startswith("_")
        }
        if extra_context:
            payload["context"] = extra_context

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)


def configure_logging(
    level: str | None = None,
    work_mode: WorkMode | None = None,
    log_dir: Path | None = None,
) -> None:
    """Initialize root logger with structured JSON formatting."""
    settings = get_settings()
    resolved_level = (level or settings.LOG_LEVEL).upper()
    resolved_mode = work_mode or settings.WORK_MODE

    target_dir = log_dir or settings.DLQ_FILE_PATH.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, resolved_level, logging.INFO))

    # Avoid duplicate handlers on repeated lifespan setups (e.g. during tests)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(StructuredJsonFormatter(resolved_mode))
    root_logger.addHandler(stream_handler)

    # Reduce noisy third-party loggers while preserving warnings/errors
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("apscheduler").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
