"""Synthetic bilingual stream ingestor for Twitch/Kick chat, movies, and markets."""

from __future__ import annotations

import asyncio
import math
import random
import time
from typing import Final

from app.core.config import get_settings
from app.core.logging import get_logger
from app.db.models import SentimentLabel
from app.services.ingestion.base import BaseIngestor, RawIngestedMessage

logger = get_logger(__name__)

_TOPIC_SOURCES: Final[dict[str, tuple[str, ...]]] = {
    "stream_chat": ("mock_twitch", "mock_kick"),
    "movie_reviews": ("mock_imdb", "mock_letterboxd"),
    "market_pulse": ("mock_reuters", "mock_bist_feed"),
}

_SYNTHETIC_CORPUS: Final[dict[str, dict[str, tuple[str, ...]]]] = {
    "stream_chat": {
        SentimentLabel.POSITIVE.value: (
            "POGGERS efsane klip!! 🔥",
            "Harika clutch aldın helal olsun!",
            "LETS GOOO insane play, chat going crazy 🚀",
            "Bu yayın çok sarıyor, muhteşem enerji 👏",
            "GG WP kralsın, kusursuz refleks!",
            "What an awesome comeback, absolute cinema!",
            "Süper hamleydi, bayıldım bu maça ❤️",
            "Goated streamer, best stream all week!",
        ),
        SentimentLabel.NEGATIVE.value: (
            "Lag var yayında ses kayıyor rezalet 😡",
            "Boring stream going to sleep...",
            "Yayın donuyor izlenmiyor, berbat kalite 👎",
            "Tam bir hayal kırıklığı, çok sıkıcı bugün",
            "Trash gameplay, complete waste of time",
            "Mikrofon cızırdıyor korkunç ses var",
            "Cringe moments, worst match ever 💀",
            "Çöp internet yine gitti, rezalet yayın",
        ),
        SentimentLabel.NEUTRAL.value: (
            "Chat sakin olun modlar nerede",
            "Yayın saat kaçta bitecek bilen var mı?",
            "Next map starts in five minutes.",
            "Bugünkü programda hangi oyun var?",
            "Moderatörler linki sabitleyebilir mi?",
            "Checking stream resolution settings now.",
        ),
    },
    "movie_reviews": {
        SentimentLabel.POSITIVE.value: (
            "Sinematografi ve oyunculuk başyapıt seviyesinde, kesinlikle izleyin",
            "Masterpiece! Directing and score were phenomenal ❤️",
            "Müzikler ve atmosfer muhteşem, yılın en iyi filmi!",
            "Brilliant storytelling and unforgettable performances 👏",
            "Kurgu çok başarılı, sonuna kadar etkileyici bir yapım",
            "Loved every single minute, fantastic cinematography!",
        ),
        SentimentLabel.NEGATIVE.value: (
            "Senaryo çok kopuktu, 2 saatim boşa gitti tam bir hayal kırıklığı",
            "Terrible pacing and wooden dialogue, worst movie of the year 👎",
            "Tam bir fiyasko, berbat oyunculuk ve sıkıcı kurgu",
            "Awful script, completely unwatchable and disappointing 💔",
            "Görsel efektler korkunç, zaman kaybı bir film",
            "Horrible ending, total garbage.",
        ),
        SentimentLabel.NEUTRAL.value: (
            "Görsel efektler fena değil ama kurgu ortalama kalmış",
            "Decent visuals, standard two-hour runtime.",
            "Film cuma günü tüm sinemalarda vizyona giriyor.",
            "A typical mid-budget sci-fi release with familiar tropes.",
            "Yönetmenin önceki filmleriyle benzer tempoda ilerliyor.",
            "Cast list includes several returning actors from the sequel.",
        ),
    },
    "market_pulse": {
        SentimentLabel.POSITIVE.value: (
            "BIST ve teknoloji hisselerinde güçlü yükseliş, rekor hacim! 📈",
            "Bullish breakout above resistance with strong institutional buying 🚀",
            "İhracat verileri beklentileri aştı, hisselerde sert artış",
            "Market rally accelerates on positive earnings surprise!",
            "Bankacılık endeksinde güçlü kazanç ve rekor kapanış",
            "Strong profit margins drive surge across semiconductor stocks.",
        ),
        SentimentLabel.NEGATIVE.value: (
            "Enflasyon endişesiyle sert satış dalgası, piyasalar çakıldı 📉",
            "Flash crash warning as liquidity dries up across major exchanges",
            "Enerji maliyetlerindeki artışla hisselerde sert düşüş ve zarar",
            "Bearish plunge deepens amid recession fears and heavy dump",
            "Kriz sinyalleri sonrası borsada büyük kayıp yaşandı",
            "Market collapse fears trigger worst sell-off of the quarter.",
        ),
        SentimentLabel.NEUTRAL.value: (
            "Piyasalar Fed faiz kararı öncesi yatay seyrediyor",
            "Central bank minutes scheduled for release at 14:00 UTC.",
            "Döviz kurları sabah seansında önceki kapanışa yakın işlem görüyor",
            "Trading volume remains steady ahead of quarterly GDP report.",
            "Şirket bilançoları hafta boyunca açıklanmaya devam edecek.",
            "Index futures hover near flat ahead of market open.",
        ),
    },
}


class TransientIngestionError(RuntimeError):
    """Raised when an upstream data source experiences a temporary failure."""


class MockStreamIngestor(BaseIngestor):
    """Generates realistic Turkish/English sentiment waves and targeted spikes."""

    def __init__(
        self,
        failure_rate: float | None = None,
        seed: int | None = None,
    ) -> None:
        settings = get_settings()
        self.failure_rate: float = (
            settings.MOCK_FAILURE_RATE if failure_rate is None else failure_rate
        )
        self._rng = random.Random(seed)
        self._failures_remaining: int = 0
        self._wave_offset: float = self._rng.uniform(0.0, math.pi * 2.0)

    def inject_transient_failures(self, count: int) -> None:
        """Force the next `count` fetch_batch calls to fail deterministically."""
        self._failures_remaining = max(0, int(count))

    def _compute_wave_weights(self) -> list[float]:
        """Return dynamic [positive, negative, neutral] sampling weights."""
        phase = (time.monotonic() / 30.0) + self._wave_offset
        wave = math.sin(phase)  # oscillates smoothly in [-1.0, 1.0]
        pos_weight = max(0.15, 0.40 + 0.25 * wave)
        neg_weight = max(0.15, 0.35 - 0.22 * wave)
        neu_weight = 0.25
        return [pos_weight, neg_weight, neu_weight]

    def _build_message(
        self,
        topic: str,
        sentiment_bucket: str,
        spike: bool = False,
    ) -> RawIngestedMessage:
        corpus_for_topic = _SYNTHETIC_CORPUS[topic]
        candidates = corpus_for_topic[sentiment_bucket]
        text = self._rng.choice(candidates)
        source = self._rng.choice(_TOPIC_SOURCES[topic])
        if spike:
            source = f"simulation_spike:{source}"

        return RawIngestedMessage(
            topic=topic,
            source=source,
            text=text,
            metadata={
                "synthetic": True,
                "intended_sentiment": sentiment_bucket,
                "spike": spike,
            },
        )

    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        """Asynchronously produce a synthetic multi-topic wave batch."""
        await asyncio.sleep(0)

        if self._failures_remaining > 0:
            self._failures_remaining -= 1
            raise TransientIngestionError(
                "Simulated upstream stream timeout (injected transient fault)"
            )

        if self.failure_rate > 0.0 and self._rng.random() < self.failure_rate:
            raise TransientIngestionError(
                "Simulated probabilistic upstream connection reset"
            )

        topics = tuple(_SYNTHETIC_CORPUS.keys())
        labels = [
            SentimentLabel.POSITIVE.value,
            SentimentLabel.NEGATIVE.value,
            SentimentLabel.NEUTRAL.value,
        ]
        weights = self._compute_wave_weights()

        messages: list[RawIngestedMessage] = []
        for _ in range(max(1, batch_size)):
            topic = self._rng.choice(topics)
            bucket = self._rng.choices(labels, weights=weights, k=1)[0]
            messages.append(self._build_message(topic=topic, sentiment_bucket=bucket))

        return messages

    async def trigger_spike(
        self,
        topic: str,
        sentiment_bias: str,
        batch_size: int = 25,
    ) -> list[RawIngestedMessage]:
        """Generate an immediate burst heavily weighted toward `sentiment_bias`."""
        await asyncio.sleep(0)

        if topic not in _SYNTHETIC_CORPUS:
            raise ValueError(
                f"Unsupported topic '{topic}'. Expected one of: {list(_SYNTHETIC_CORPUS.keys())}"
            )
        if sentiment_bias not in (
            SentimentLabel.POSITIVE.value,
            SentimentLabel.NEGATIVE.value,
            SentimentLabel.NEUTRAL.value,
        ):
            raise ValueError(
                f"Unsupported sentiment_bias '{sentiment_bias}'."
            )

        all_labels = [
            SentimentLabel.POSITIVE.value,
            SentimentLabel.NEGATIVE.value,
            SentimentLabel.NEUTRAL.value,
        ]
        # 85% heavily biased toward target sentiment, 15% natural background noise
        weights = [0.85 if lbl == sentiment_bias else 0.075 for lbl in all_labels]

        messages: list[RawIngestedMessage] = []
        for _ in range(max(1, batch_size)):
            bucket = self._rng.choices(all_labels, weights=weights, k=1)[0]
            messages.append(
                self._build_message(
                    topic=topic,
                    sentiment_bucket=bucket,
                    spike=True,
                )
            )

        logger.info(
            "Generated synthetic sentiment spike batch",
            extra={
                "topic": topic,
                "sentiment_bias": sentiment_bias,
                "batch_size": len(messages),
            },
        )
        return messages
