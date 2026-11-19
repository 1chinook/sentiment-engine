"""Integration tests for FastAPI v1 endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
async def test_post_analyze_single_and_batch(
    async_client: httpx.AsyncClient,
) -> None:
    """POST /api/v1/analyze handles single text, batch texts, and validation errors."""
    single_resp = await async_client.post(
        "/api/v1/analyze",
        json={"text": "Harika clutch aldın helal olsun!"},
    )
    assert single_resp.status_code == 200
    single_data = single_resp.json()
    assert single_data["batch_size"] == 1
    assert single_data["persisted"] is False
    assert single_data["results"][0]["sentiment_label"] == "positive"
    assert 0.0 <= single_data["results"][0]["confidence_score"] <= 1.0

    batch_resp = await async_client.post(
        "/api/v1/analyze",
        json={
            "texts": [
                "Masterpiece! Directing and score were phenomenal",
                "Lag var yayında ses kayıyor rezalet",
                "Piyasalar yatay seyrediyor",
            ],
            "topic": "api_custom_topic",
            "source": "integration_test",
            "persist": True,
        },
    )
    assert batch_resp.status_code == 200
    batch_data = batch_resp.json()
    assert batch_data["batch_size"] == 3
    assert batch_data["persisted"] is True
    assert [item["sentiment_label"] for item in batch_data["results"]] == [
        "positive",
        "negative",
        "neutral",
    ]

    invalid_resp = await async_client.post(
        "/api/v1/analyze",
        json={"texts": ["   "]},
    )
    assert invalid_resp.status_code == 422


@pytest.mark.asyncio
async def test_simulation_trigger_and_topic_analytics(
    async_client: httpx.AsyncClient,
) -> None:
    """Triggering synthetic spikes populates DB analytics and rolling averages."""
    spike_resp = await async_client.post(
        "/api/v1/simulation/trigger",
        json={
            "topic": "movie_reviews",
            "sentiment_bias": "negative",
            "batch_size": 20,
        },
    )
    assert spike_resp.status_code == 200
    spike_data = spike_resp.json()
    assert spike_data["status"] == "completed"
    assert spike_data["topic"] == "movie_reviews"
    assert spike_data["ingested_count"] == 20
    assert spike_data["persisted_to_db"] is True
    assert spike_data["routed_to_dlq"] is False

    analytics_resp = await async_client.get("/api/v1/sentiment/movie_reviews")
    assert analytics_resp.status_code == 200
    analytics = analytics_resp.json()
    assert analytics["topic"] == "movie_reviews"
    assert analytics["total_volume"] == 20
    assert analytics["window_5m"]["message_volume"] == 20
    assert analytics["window_1h"]["message_volume"] == 20
    assert analytics["window_5m"]["moving_avg_sentiment"] < 0.0
    assert len(analytics["time_series"]) >= 1


@pytest.mark.asyncio
async def test_manual_dlq_recovery_endpoint(
    async_client: httpx.AsyncClient,
    dlq_paths: tuple[Path, Path],
) -> None:
    """POST /api/v1/orchestrator/recover-dlq replays pending DLQ items into DB."""
    dlq_path, _ = dlq_paths
    dlq_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat()
    sample_line = json.dumps(
        {
            "failed_at": now_iso,
            "error_reason": "simulated_db_restart",
            "record": {
                "id": str(uuid.uuid4()),
                "topic": "market_pulse",
                "source": "mock_bist_feed",
                "raw_text": "BIST ve teknoloji hisselerinde güçlü yükseliş, rekor hacim!",
                "sentiment_label": "positive",
                "confidence_score": 0.93,
                "sentiment_score": 0.93,
                "processing_time_ms": 1.1,
                "created_at": now_iso,
            },
        },
        ensure_ascii=False,
    )
    dlq_path.write_text(sample_line + "\n", encoding="utf-8")

    recover_resp = await async_client.post("/api/v1/orchestrator/recover-dlq")
    assert recover_resp.status_code == 200
    recover_data = recover_resp.json()
    assert recover_data["status"] == "recovered"
    assert recover_data["recovered_count"] == 1
    assert recover_data["remaining_dlq_count"] == 0

    market_resp = await async_client.get("/api/v1/sentiment/market_pulse")
    assert market_resp.status_code == 200
    assert market_resp.json()["total_volume"] == 1


@pytest.mark.asyncio
async def test_health_endpoint(async_client: httpx.AsyncClient) -> None:
    """GET /api/v1/health reports DB, APScheduler worker, DLQ, and NLP readiness."""
    resp = await async_client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["work_mode"] == "TEST_SIMULATION"
    assert data["database"]["connected"] is True
    assert data["worker"]["active"] is True
    assert len(data["worker"]["jobs"]) == 2
    assert data["dlq"]["pending_records"] == 0
    assert data["dlq"]["corrupted_records"] == 0
    assert data["nlp_model"]["loaded"] is True


@pytest.mark.asyncio
async def test_dashboard_endpoint(async_client: httpx.AsyncClient) -> None:
    """GET /dashboard serves the interactive live sentiment testing dashboard."""
    resp = await async_client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "sentiment-engine" in resp.text
