"""Unit tests for the Singleton SentimentAnalyzer NLP service."""

from __future__ import annotations

from app.db.models import SentimentLabel
from app.services.nlp.analyzer import SentimentAnalyzer, _map_raw_label
from app.services.nlp.hype_scorer import (
    compute_stream_hype_summary,
    score_message_hype,
)


def test_sentiment_analyzer_is_singleton() -> None:
    """Multiple instantiations must return the exact same object identity."""
    first = SentimentAnalyzer()
    second = SentimentAnalyzer()
    assert first is second


def test_bilingual_batch_inference(nlp_analyzer: SentimentAnalyzer) -> None:
    """Mixed Turkish and English positive/negative/neutral texts classify accurately."""
    samples = [
        "POGGERS efsane klip!! 🔥",
        "Senaryo çok kopuktu, 2 saatim boşa gitti tam bir hayal kırıklığı",
        "Piyasalar Fed faiz kararı öncesi yatay seyrediyor",
        "Masterpiece! Directing and score were phenomenal ❤️",
        "Terrible pacing and wooden dialogue, worst movie of the year 👎",
    ]
    predictions, total_ms = nlp_analyzer.analyze_batch(samples)

    assert len(predictions) == len(samples)
    assert total_ms >= 0.0

    assert predictions[0].label == SentimentLabel.POSITIVE
    assert predictions[1].label == SentimentLabel.NEGATIVE
    assert predictions[2].label == SentimentLabel.NEUTRAL
    assert predictions[3].label == SentimentLabel.POSITIVE
    assert predictions[4].label == SentimentLabel.NEGATIVE

    for pred in predictions:
        assert 0.0 <= pred.confidence <= 1.0
        assert -1.0 <= pred.sentiment_score <= 1.0
        assert pred.processing_time_ms >= 0.0


def test_empty_batch_returns_empty_result(nlp_analyzer: SentimentAnalyzer) -> None:
    """Empty input batches return zero predictions with zero latency."""
    predictions, total_ms = nlp_analyzer.analyze_batch([])
    assert predictions == []
    assert total_ms == 0.0


def test_raw_label_normalization_mapping() -> None:
    """Hugging Face label variants map cleanly to canonical SentimentLabel values."""
    assert _map_raw_label("LABEL_0") == SentimentLabel.NEGATIVE
    assert _map_raw_label("LABEL_1") == SentimentLabel.NEUTRAL
    assert _map_raw_label("LABEL_2") == SentimentLabel.POSITIVE
    assert _map_raw_label("Negative") == SentimentLabel.NEGATIVE
    assert _map_raw_label("neutral") == SentimentLabel.NEUTRAL
    assert _map_raw_label("POSITIVE") == SentimentLabel.POSITIVE


def test_health_info_reports_loaded_state(nlp_analyzer: SentimentAnalyzer) -> None:
    """Health metadata reflects loaded state and CPU execution device."""
    info = nlp_analyzer.health_info()
    assert info["loaded"] is True
    assert info["device"] == "cpu"
    assert info["backend"] in ("huggingface_transformers", "bilingual_fallback")
    assert isinstance(info["total_analyzed"], int)


def test_hybrid_stream_hype_scoring_and_clip_detection(
    nlp_analyzer: SentimentAnalyzer,
) -> None:
    """Laughter, Twitch slang, elongation, and clip requests boost hype score and calibrate sentiment."""
    hype_chat = [
        "AHAHAHAHAHA asdasdasd KEKW",
        "OOOOOHA NE YAPTIN KLİP AL!! 🔥🚀",
        "WWWWWWWW CLUTCH GOD POGGERS",
    ]
    predictions, _ = nlp_analyzer.analyze_batch(hype_chat)
    for pred in predictions:
        assert pred.label == SentimentLabel.POSITIVE
        assert pred.hype_score >= 0.45
        assert len(pred.hype_tags) >= 1

    summary = compute_stream_hype_summary(
        message_hype_scores=[p.hype_score for p in predictions],
        all_tags=[list(p.hype_tags) for p in predictions],
        listen_seconds=1.0,
    )
    assert summary["hype_score"] >= 70.0
    assert summary["clip_recommended"] is True
    assert summary["hype_level"] in ("high_hype", "clip_moment")
    assert "Klip İsteği" in summary["dominant_triggers"]

    single = score_message_hype(
        text="AHAHAHAHA sjsjsjsj",
        base_label=SentimentLabel.NEUTRAL,
        base_confidence=0.60,
        base_sentiment_score=0.0,
    )
    assert single.calibrated_label == SentimentLabel.POSITIVE
    assert "Kahkaha/Random" in single.hype_tags
