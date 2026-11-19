"""Integration tests for FastAPI v1 endpoints."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from app.core.config import Settings, WorkMode
from app.services.ingestion.base import RawIngestedMessage
from app.services.ingestion.mock_stream import MockStreamIngestor
from app.services.ingestion.real_streams import (
    CompositeProductionIngestor,
    TwitchAnonymousIRCIngestor,
    build_ingestor,
    normalize_twitch_channel,
    parse_rss_or_atom_xml,
)


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


@pytest.mark.asyncio
async def test_spike_alerts_and_webhook_registration(
    async_client: httpx.AsyncClient,
) -> None:
    """Spike bursts trigger SpikeAlertEvents and webhook registration persists."""
    reg_resp = await async_client.post(
        "/api/v1/alerts/webhooks",
        json={"url": "http://localhost:9999/webhooks/clip-trigger"},
    )
    assert reg_resp.status_code == 200
    reg_data = reg_resp.json()
    assert (
        "http://localhost:9999/webhooks/clip-trigger"
        in reg_data["registered_webhooks"]
    )

    spike_resp = await async_client.post(
        "/api/v1/simulation/trigger",
        json={
            "topic": "stream_chat",
            "sentiment_bias": "positive",
            "batch_size": 25,
        },
    )
    assert spike_resp.status_code == 200
    assert spike_resp.json()["alerts_triggered"] >= 1

    alerts_resp = await async_client.get("/api/v1/alerts")
    assert alerts_resp.status_code == 200
    alerts_data = alerts_resp.json()
    assert len(alerts_data["recent_alerts"]) >= 1
    latest = alerts_data["recent_alerts"][0]
    assert latest["topic"] == "stream_chat"
    assert latest["direction"] == "positive_hype"


@pytest.mark.asyncio
async def test_api_key_authentication_enforcement(
    async_client: httpx.AsyncClient,
) -> None:
    """When API_KEY is configured, protected endpoints enforce X-API-Key header."""
    transport = async_client._transport  # type: ignore[attr-defined]
    app = transport.app
    app.state.settings.API_KEY = "test-secret-key-123"
    try:
        unauth_resp = await async_client.post(
            "/api/v1/analyze",
            json={"text": "Harika yayın!"},
        )
        assert unauth_resp.status_code == 401

        auth_resp = await async_client.post(
            "/api/v1/analyze",
            headers={"X-API-Key": "test-secret-key-123"},
            json={"text": "Harika yayın!"},
        )
        assert auth_resp.status_code == 200
    finally:
        app.state.settings.API_KEY = None


def test_parse_rss_and_atom_xml_and_production_factory() -> None:
    """RSS 2.0 and Atom XML payloads parse into RawIngestedMessage items."""
    rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <item>
          <title>Tech stocks rally to record high</title>
          <description>&lt;p&gt;Strong institutional buying continues.&lt;/p&gt;</description>
          <link>https://example.com/market/1</link>
        </item>
      </channel>
    </rss>"""
    rss_msgs = parse_rss_or_atom_xml(
        rss_xml, topic="market_pulse", source_url="https://example.com/rss"
    )
    assert len(rss_msgs) == 1
    assert rss_msgs[0].topic == "market_pulse"
    assert "Tech stocks rally to record high" in rss_msgs[0].text
    assert "<p>" not in rss_msgs[0].text

    atom_xml = """<?xml version="1.0" encoding="utf-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <title>Masterpiece sci-fi film review</title>
        <summary>Phenomenal cinematography and score.</summary>
        <link href="https://example.com/movies/1"/>
      </entry>
    </feed>"""
    atom_msgs = parse_rss_or_atom_xml(
        atom_xml, topic="movie_reviews", source_url="https://example.com/atom"
    )
    assert len(atom_msgs) == 1
    assert atom_msgs[0].topic == "movie_reviews"
    assert "Masterpiece sci-fi film review" in atom_msgs[0].text

    sim_ingestor = build_ingestor(Settings(WORK_MODE=WorkMode.TEST_SIMULATION))
    prod_ingestor = build_ingestor(Settings(WORK_MODE=WorkMode.PRODUCTION))
    assert isinstance(sim_ingestor, MockStreamIngestor)
    assert isinstance(prod_ingestor, CompositeProductionIngestor)
    assert normalize_twitch_channel("https://www.twitch.tv/Elraenn/chat") == "elraenn"


@pytest.mark.asyncio
async def test_live_twitch_chat_endpoint(
    async_client: httpx.AsyncClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """POST /api/v1/live/twitch normalizes channel, analyzes IRC messages, and persists."""

    async def _fake_fetch_batch(
        self: TwitchAnonymousIRCIngestor, batch_size: int = 20
    ) -> list[RawIngestedMessage]:
        return [
            RawIngestedMessage(
                topic="stream_chat",
                source=f"twitch:{self.channel}",
                text="POGGERS efsane klip harika oynadın!! 🔥",
                metadata={"channel": self.channel, "username": "izleyici_1"},
            ),
            RawIngestedMessage(
                topic="stream_chat",
                source=f"twitch:{self.channel}",
                text="Lag var yayında ses kayıyor rezalet 😡",
                metadata={"channel": self.channel, "username": "izleyici_2"},
            ),
        ]

    monkeypatch.setattr(
        TwitchAnonymousIRCIngestor, "fetch_batch", _fake_fetch_batch
    )

    resp = await async_client.post(
        "/api/v1/live/twitch",
        json={
            "channel": "https://twitch.tv/Elraenn",
            "listen_seconds": 2.0,
            "max_messages": 10,
            "persist": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["channel"] == "elraenn"
    assert data["fetched_count"] == 2
    assert data["persisted"] is True
    assert len(data["messages"]) == 2
    assert data["messages"][0]["username"] == "izleyici_1"
    assert data["messages"][0]["sentiment_label"] == "positive"
    assert data["messages"][1]["username"] == "izleyici_2"
    assert data["messages"][1]["sentiment_label"] == "negative"
