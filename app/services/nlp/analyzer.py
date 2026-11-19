"""Thread-safe Singleton NLP Sentiment Analyzer supporting Turkish & English."""

from __future__ import annotations

import math
import re
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import SentimentLabel

logger = get_logger(__name__)

_TR_CHAR_MAP = str.maketrans(
    {
        "ç": "c",
        "ğ": "g",
        "ı": "i",
        "İ": "i",
        "ö": "o",
        "ş": "s",
        "ü": "u",
        "Ç": "c",
        "Ğ": "g",
        "Ö": "o",
        "Ş": "s",
        "Ü": "u",
    }
)

_POSITIVE_LEXICON: dict[str, float] = {
    # Turkish
    "efsane": 1.6,
    "harika": 1.5,
    "mukemmel": 1.7,
    "super": 1.4,
    "helal": 1.3,
    "basyapit": 1.8,
    "muhtesem": 1.6,
    "guzel": 1.1,
    "iyi": 0.9,
    "basarili": 1.3,
    "yukselis": 1.4,
    "rekor": 1.3,
    "guclu": 1.1,
    "artis": 1.0,
    "kazanc": 1.2,
    "sevdim": 1.3,
    "bayildim": 1.6,
    "begendim": 1.2,
    "sahane": 1.5,
    "kusursuz": 1.7,
    "etkileyici": 1.3,
    "keyifli": 1.2,
    "tebrikler": 1.2,
    "kralsin": 1.4,
    "fena": 0.0,  # handled with context; neutral standalone
    # English & Twitch/Kick/Market slang
    "poggers": 1.6,
    "pog": 1.5,
    "clutch": 1.4,
    "hype": 1.4,
    "gg": 1.1,
    "goated": 1.6,
    "insane": 1.3,
    "masterpiece": 1.8,
    "phenomenal": 1.6,
    "amazing": 1.5,
    "awesome": 1.4,
    "great": 1.3,
    "good": 1.0,
    "love": 1.4,
    "loved": 1.4,
    "best": 1.5,
    "fantastic": 1.5,
    "excellent": 1.5,
    "brilliant": 1.5,
    "bullish": 1.5,
    "breakout": 1.3,
    "rally": 1.3,
    "surge": 1.3,
    "profit": 1.2,
    "win": 1.2,
    "w": 0.9,
    "lets": 0.5,
    "gooo": 1.1,
}

_NEGATIVE_LEXICON: dict[str, float] = {
    # Turkish
    "rezalet": 1.7,
    "berbat": 1.7,
    "kotu": 1.3,
    "korkunc": 1.6,
    "hayal": 0.6,
    "kirikligi": 1.3,
    "bosa": 1.2,
    "kopuk": 1.1,
    "kopuktu": 1.2,
    "sikici": 1.3,
    "lag": 1.2,
    "donuyor": 1.2,
    "kayiyor": 1.0,
    "cop": 1.6,
    "cakildi": 1.6,
    "dusus": 1.3,
    "satis": 0.9,
    "endise": 1.1,
    "endiseyle": 1.2,
    "kriz": 1.4,
    "zarar": 1.3,
    "iflas": 1.6,
    "felaket": 1.7,
    "zaman": 0.3,
    "kayip": 1.2,
    "igrenc": 1.7,
    "sacma": 1.3,
    "yetersiz": 1.2,
    "vasat": 1.0,
    "fiyasko": 1.7,
    # English & Twitch/Kick/Market slang
    "terrible": 1.6,
    "worst": 1.7,
    "bad": 1.2,
    "boring": 1.3,
    "trash": 1.6,
    "garbage": 1.6,
    "awful": 1.6,
    "horrible": 1.6,
    "disappointing": 1.4,
    "disappointment": 1.5,
    "wooden": 1.1,
    "waste": 1.4,
    "bearish": 1.5,
    "crash": 1.6,
    "dump": 1.5,
    "plunge": 1.5,
    "collapse": 1.6,
    "loss": 1.3,
    "scam": 1.7,
    "unwatchable": 1.6,
    "cringe": 1.3,
    "sleep": 0.8,
    "rip": 1.0,
    "l": 0.8,
}

_NEGATION_TOKENS = frozenset(
    {
        "degil",
        "yok",
        "hic",
        "asla",
        "olmaz",
        "not",
        "never",
        "no",
        "dont",
        "don't",
        "didnt",
        "didn't",
        "isnt",
        "isn't",
        "cant",
        "can't",
    }
)

_INTENSIFIERS = frozenset(
    {
        "cok",
        "asiri",
        "tam",
        "gercekten",
        "fazla",
        "sert",
        "very",
        "so",
        "extremely",
        "totally",
    }
)

_POSITIVE_EMOJIS = ("🔥", "🚀", "❤️", "👏", "🎉", "💯", "😍", "🙌", "📈", "💚")
_NEGATIVE_EMOJIS = ("😡", "👎", "💔", "😭", "📉", "🤮", "💀", "😤")


@dataclass(frozen=True, slots=True)
class SentimentPrediction:
    """Single text sentiment inference result."""

    text: str
    label: SentimentLabel
    confidence: float
    sentiment_score: float
    processing_time_ms: float


def _normalize_token(token: str) -> str:
    return token.lower().translate(_TR_CHAR_MAP)


def _map_raw_label(raw_label: str) -> SentimentLabel:
    """Map Hugging Face model output labels to standardized SentimentLabel."""
    normalized = raw_label.strip().lower()
    if any(
        key in normalized
        for key in ("pos", "label_2", "5 star", "4 star")
    ) or normalized == "2":
        return SentimentLabel.POSITIVE
    if any(
        key in normalized
        for key in ("neg", "label_0", "1 star", "2 star")
    ) or normalized == "0":
        return SentimentLabel.NEGATIVE
    if any(
        key in normalized
        for key in ("neu", "label_1", "3 star")
    ) or normalized == "1":
        return SentimentLabel.NEUTRAL
    return SentimentLabel.NEUTRAL


def _signed_sentiment_score(label: SentimentLabel, confidence: float) -> float:
    """Convert label + confidence into a signed polarity score in [-1.0, 1.0]."""
    clamped = max(0.0, min(1.0, float(confidence)))
    if label == SentimentLabel.POSITIVE:
        return round(clamped, 4)
    if label == SentimentLabel.NEGATIVE:
        return round(-clamped, 4)
    return 0.0


class SentimentAnalyzer:
    """Thread-safe Singleton wrapper for Hugging Face NLP sentiment inference."""

    _instance: SentimentAnalyzer | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> SentimentAnalyzer:
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        settings = get_settings()
        self.model_name: str = settings.NLP_MODEL_NAME
        self.max_batch_size: int = settings.NLP_MAX_BATCH_SIZE
        self.use_fallback_if_unavailable: bool = (
            settings.NLP_USE_FALLBACK_IF_UNAVAILABLE
        )
        self.device: str = settings.NLP_DEVICE

        self._model_lock = threading.Lock()
        self._loaded: bool = False
        self._backend: str = "uninitialized"
        self._tokenizer: Any = None
        self._model: Any = None
        self._torch: Any = None
        self._total_analyzed: int = 0
        self._initialized = True

    def load_model(self, force_fallback: bool = False) -> None:
        """Load the Hugging Face model into memory once (idempotent)."""
        if self._loaded and not force_fallback:
            return

        with self._model_lock:
            if self._loaded and not force_fallback:
                return

            if force_fallback:
                self._backend = "bilingual_fallback"
                self._loaded = True
                logger.info(
                    "NLP analyzer initialized with bilingual fallback backend",
                    extra={"model_name": self.model_name, "backend": self._backend},
                )
                return

            try:
                import torch
                from transformers import (
                    AutoModelForSequenceClassification,
                    AutoTokenizer,
                )

                logger.info(
                    "Loading Hugging Face sentiment model into memory",
                    extra={"model_name": self.model_name, "device": self.device},
                )
                tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name
                )
                model.eval()
                model.to("cpu")

                self._torch = torch
                self._tokenizer = tokenizer
                self._model = model
                self._backend = "huggingface_transformers"
                self._loaded = True
                logger.info(
                    "Hugging Face sentiment model loaded successfully",
                    extra={
                        "model_name": self.model_name,
                        "backend": self._backend,
                    },
                )
            except Exception as exc:
                if not self.use_fallback_if_unavailable:
                    logger.exception(
                        "Failed to load Hugging Face model and fallback is disabled",
                        extra={"model_name": self.model_name, "error": str(exc)},
                    )
                    raise

                self._tokenizer = None
                self._model = None
                self._backend = "bilingual_fallback"
                self._loaded = True
                logger.warning(
                    "Hugging Face model unavailable; activated bilingual fallback engine",
                    extra={
                        "model_name": self.model_name,
                        "error": str(exc),
                        "backend": self._backend,
                    },
                )

    def analyze_batch(
        self, texts: list[str]
    ) -> tuple[list[SentimentPrediction], float]:
        """Run synchronous CPU-bound sentiment inference over a batch of texts.

        Intended to be invoked via ``asyncio.to_thread`` from async code so
        the main event loop is never blocked.
        """
        if not texts:
            return [], 0.0

        if not self._loaded:
            self.load_model()

        started = time.perf_counter()

        if self._backend == "huggingface_transformers":
            raw_results = self._infer_transformers(texts)
        else:
            raw_results = self._infer_bilingual_fallback(texts)

        total_ms = round((time.perf_counter() - started) * 1000.0, 3)
        per_item_ms = round(total_ms / max(len(texts), 1), 3)

        predictions: list[SentimentPrediction] = []
        for text, (label, confidence) in zip(texts, raw_results, strict=True):
            clamped_conf = round(max(0.0, min(1.0, float(confidence))), 4)
            predictions.append(
                SentimentPrediction(
                    text=text,
                    label=label,
                    confidence=clamped_conf,
                    sentiment_score=_signed_sentiment_score(label, clamped_conf),
                    processing_time_ms=per_item_ms,
                )
            )

        self._total_analyzed += len(predictions)
        return predictions, total_ms

    def _infer_transformers(
        self, texts: list[str]
    ) -> list[tuple[SentimentLabel, float]]:
        """Execute batched PyTorch/Transformers forward pass with torch.no_grad()."""
        assert self._torch is not None
        assert self._tokenizer is not None
        assert self._model is not None

        torch = self._torch
        id2label = getattr(self._model.config, "id2label", {}) or {}
        results: list[tuple[SentimentLabel, float]] = []

        for start_idx in range(0, len(texts), self.max_batch_size):
            chunk = [t if t.strip() else " " for t in texts[start_idx : start_idx + self.max_batch_size]]
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self._model(**encoded)
                probs = torch.softmax(outputs.logits, dim=-1)
                confidences, pred_ids = torch.max(probs, dim=-1)

            for idx, conf in zip(pred_ids.tolist(), confidences.tolist(), strict=True):
                raw_label = str(id2label.get(idx, id2label.get(str(idx), f"LABEL_{idx}")))
                mapped_label = _map_raw_label(raw_label)
                results.append((mapped_label, float(conf)))

        return results

    def _infer_bilingual_fallback(
        self, texts: list[str]
    ) -> list[tuple[SentimentLabel, float]]:
        """Deterministic Turkish/English lexicon + negation sentiment classifier."""
        results: list[tuple[SentimentLabel, float]] = []

        for text in texts:
            stripped = text.strip()
            if not stripped:
                results.append((SentimentLabel.NEUTRAL, 0.55))
                continue

            tokens = re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9']+", stripped)
            normalized = [_normalize_token(tok) for tok in tokens]

            score = 0.0
            for idx, tok in enumerate(normalized):
                prev_tok = normalized[idx - 1] if idx > 0 else ""
                next_tok = normalized[idx + 1] if idx + 1 < len(normalized) else ""
                is_negated = prev_tok in _NEGATION_TOKENS or next_tok in _NEGATION_TOKENS
                multiplier = 1.25 if prev_tok in _INTENSIFIERS else 1.0

                # Handle Turkish idiom "fena değil" ("not bad" -> positive)
                if tok == "fena":
                    if next_tok in _NEGATION_TOKENS:
                        score += 1.0
                    else:
                        score -= 0.9
                    continue

                if tok in _POSITIVE_LEXICON:
                    val = _POSITIVE_LEXICON[tok] * multiplier
                    score += -val * 0.8 if is_negated else val
                elif tok in _NEGATIVE_LEXICON:
                    val = _NEGATIVE_LEXICON[tok] * multiplier
                    score += val * 0.6 if is_negated else -val

            for emoji in _POSITIVE_EMOJIS:
                if emoji in stripped:
                    score += 0.8
            for emoji in _NEGATIVE_EMOJIS:
                if emoji in stripped:
                    score -= 0.8

            # Neutral phrase dampening
            if "yatay" in normalized or "ortalama" in normalized:
                score *= 0.35

            if score >= 0.45:
                confidence = min(0.98, 0.58 + 0.38 * math.tanh(abs(score) / 2.2))
                results.append((SentimentLabel.POSITIVE, confidence))
            elif score <= -0.45:
                confidence = min(0.98, 0.58 + 0.38 * math.tanh(abs(score) / 2.2))
                results.append((SentimentLabel.NEGATIVE, confidence))
            else:
                confidence = max(0.52, min(0.88, 0.78 - abs(score) * 0.4))
                results.append((SentimentLabel.NEUTRAL, confidence))

        return results

    def health_info(self) -> dict[str, Any]:
        """Return model readiness and inference metrics."""
        return {
            "loaded": self._loaded,
            "model_name": self.model_name,
            "backend": self._backend,
            "device": "cpu",
            "total_analyzed": self._total_analyzed,
        }
