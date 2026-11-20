"""Hybrid Live-Stream Hype & Excitement Scorer for Twitch/Kick chat."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from app.db.models import SentimentLabel

# Turkish & international laughter / keyboard-smash patterns in live chat
_LAUGHTER_RE = re.compile(
    r"(?:a*ha+h[ah]*|e*he+h[eh]*|asdasd[asd]*|sjsjsj[sj]*|ksksks[ks]*|"
    r"jsjsjs[js]*|qwqw[qw]*|lmao+|lmfao+|rofl+|kekw+|omegalul+|lulw*)",
    re.IGNORECASE,
)

# Character elongation (e.g. "OOOOOHA", "GOOOOOL", "WWWWWW", "AAAAA")
_ELONGATION_RE = re.compile(r"([a-zA-ZçğıöşüÇĞİÖŞÜ])\1{2,}")

# Explicit clip request keywords in Turkish & English
_CLIP_REQUEST_RE = re.compile(
    r"\b(?:klip\s*al\w*|kliple\w*|clip\s*it|clip\s*that|clip\s*this|klip)\b",
    re.IGNORECASE,
)

_POSITIVE_HYPE_TOKENS = frozenset(
    {
        "pog",
        "poggers",
        "pogchamp",
        "pogu",
        "hypers",
        "pepega",
        "kekw",
        "omegalul",
        "lul",
        "lulw",
        "clutch",
        "goat",
        "goated",
        "ez",
        "gg",
        "ggwp",
        "wp",
        "w",
        "ww",
        "www",
        "wwww",
        "letsgo",
        "letsgooo",
        "insane",
        "oha",
        "yuh",
        "helal",
        "efsane",
        "kralsin",
        "kral",
        "baba",
        "oyy",
        "oyyy",
        "sheesh",
        "holy",
        "noway",
        "boom",
        "banger",
        "cinema",
        "gol",
        "gool",
        "goool",
        "777",
        "10/10",
        "nt",
    }
)

_DRAMATIC_FAIL_HYPE_TOKENS = frozenset(
    {
        "f",
        "ff",
        "fff",
        "rip",
        "l",
        "ll",
        "lll",
        "bruh",
        "fail",
        "throw",
        "choke",
        "abo",
        "abooo",
        "eyvah",
        "wtf",
    }
)

_HYPE_EMOJIS = (
    "🔥",
    "🚀",
    "🤯",
    "😱",
    "👑",
    "💀",
    "😂",
    "🤣",
    "👏",
    "⚡",
    "💥",
    "❤️",
    "💯",
    "🎉",
)


@dataclass(frozen=True, slots=True)
class MessageHypeAnalysis:
    """Hype-calibrated message sentiment and excitement metadata."""

    calibrated_label: SentimentLabel
    calibrated_confidence: float
    calibrated_sentiment_score: float
    hype_score: float  # 0.0 to 1.0
    hype_tags: list[str]


def score_message_hype(
    text: str,
    base_label: SentimentLabel,
    base_confidence: float,
    base_sentiment_score: float,
) -> MessageHypeAnalysis:
    """Enhance standard NLP sentiment prediction with live-stream hype signals."""
    stripped = text.strip()
    if not stripped:
        return MessageHypeAnalysis(
            calibrated_label=base_label,
            calibrated_confidence=base_confidence,
            calibrated_sentiment_score=base_sentiment_score,
            hype_score=0.0,
            hype_tags=[],
        )

    hype_points = 0.0
    pos_hype_signal = 0.0
    neg_hype_signal = 0.0
    tags: list[str] = []

    tokens = [
        t.lower()
        for t in re.findall(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9/]+", stripped)
    ]

    # 1. Explicit clip requests ("klip al", "clip it")
    if _CLIP_REQUEST_RE.search(stripped):
        hype_points += 0.55
        pos_hype_signal += 0.50
        tags.append("Klip İsteği")

    # 2. Laughter / Turkish random ("ahahaha", "asdasd", "kekw", "sjsjsj")
    laughter_hits = _LAUGHTER_RE.findall(stripped)
    if laughter_hits:
        laugh_boost = min(0.65, 0.45 + 0.10 * (len(laughter_hits) - 1))
        hype_points += laugh_boost
        pos_hype_signal += laugh_boost
        tags.append("Kahkaha/Random")

    # 3. Twitch/Kick slang & emotes
    matched_pos_slang = sum(1 for t in tokens if t in _POSITIVE_HYPE_TOKENS)
    matched_fail_slang = sum(
        1 for t in tokens if t in _DRAMATIC_FAIL_HYPE_TOKENS
    )
    if matched_pos_slang > 0:
        boost = min(0.55, 0.32 + 0.12 * (matched_pos_slang - 1))
        hype_points += boost
        pos_hype_signal += boost
        tags.append("Twitch/Yayın Jargonu")
    if matched_fail_slang > 0:
        boost = min(0.45, 0.28 + 0.10 * (matched_fail_slang - 1))
        hype_points += boost
        neg_hype_signal += boost
        if "Twitch/Yayın Jargonu" not in tags:
            tags.append("Fail/Tepki")

    # 4. Character elongation ("OOOOOHA", "GOOOOOL", "WWWWWW")
    if _ELONGATION_RE.search(stripped):
        hype_points += 0.28
        pos_hype_signal += 0.22
        tags.append("Harf Uzatma")

    # 5. ALL CAPS burst
    alpha_chars = [c for c in stripped if c.isalpha()]
    if len(alpha_chars) >= 4:
        upper_ratio = sum(1 for c in alpha_chars if c.isupper()) / len(
            alpha_chars
        )
        if upper_ratio >= 0.65:
            hype_points += 0.24
            tags.append("BÜYÜK HARF")

    # 6. Exclamation / Question burst & Hype Emojis
    excl_count = stripped.count("!") + stripped.count("?")
    if excl_count >= 2:
        hype_points += min(0.24, 0.10 * excl_count)
        tags.append("Ünlem Coşkusu")

    emoji_hits = sum(stripped.count(em) for em in _HYPE_EMOJIS)
    if emoji_hits > 0:
        hype_points += min(0.32, 0.16 * emoji_hits)
        pos_hype_signal += min(0.25, 0.12 * emoji_hits)
        tags.append("Hype Emoji")

    # Factor in strong base sentiment intensity
    if base_label != SentimentLabel.NEUTRAL and base_confidence >= 0.72:
        hype_points += 0.22 * base_confidence

    hype_score = round(min(1.0, math.tanh(hype_points * 1.15)), 4)

    # Calibrate sentiment label if standard RoBERTa missed stream slang/laughter
    calibrated_label = base_label
    calibrated_conf = base_confidence
    calibrated_polarity = base_sentiment_score

    if base_label == SentimentLabel.NEUTRAL:
        if pos_hype_signal >= 0.30 and pos_hype_signal > neg_hype_signal:
            calibrated_label = SentimentLabel.POSITIVE
            calibrated_conf = round(
                min(0.96, max(base_confidence, 0.64 + pos_hype_signal * 0.45)),
                4,
            )
            calibrated_polarity = calibrated_conf
        elif neg_hype_signal >= 0.30 and neg_hype_signal > pos_hype_signal:
            calibrated_label = SentimentLabel.NEGATIVE
            calibrated_conf = round(
                min(0.94, max(base_confidence, 0.62 + neg_hype_signal * 0.45)),
                4,
            )
            calibrated_polarity = -calibrated_conf

    return MessageHypeAnalysis(
        calibrated_label=calibrated_label,
        calibrated_confidence=calibrated_conf,
        calibrated_sentiment_score=calibrated_polarity,
        hype_score=hype_score,
        hype_tags=tags,
    )


def compute_stream_hype_summary(
    message_hype_scores: list[float],
    all_tags: list[list[str]],
    listen_seconds: float,
) -> dict[str, Any]:
    """Compute 0-100 real-time channel Hype Meter and clip recommendation."""
    count = len(message_hype_scores)
    if count == 0:
        return {
            "hype_score": 0.0,
            "hype_level": "calm",
            "clip_recommended": False,
            "messages_per_sec": 0.0,
            "hype_message_ratio": 0.0,
            "dominant_triggers": [],
        }

    safe_window = max(0.5, float(listen_seconds))
    mps = round(count / safe_window, 2)

    avg_hype = sum(message_hype_scores) / count
    sorted_scores = sorted(message_hype_scores, reverse=True)
    top_k = max(1, count // 3)
    top_avg_hype = sum(sorted_scores[:top_k]) / top_k

    energetic_count = sum(1 for s in message_hype_scores if s >= 0.35)
    hype_ratio = round(energetic_count / count, 4)

    # Velocity contribution: 4+ msg/s in Twitch chat indicates active burst
    velocity_score = min(1.0, mps / 5.5)

    tag_counter: Counter[str] = Counter()
    for t_list in all_tags:
        tag_counter.update(t_list)

    clip_tag_bonus = 12.0 if tag_counter.get("Klip İsteği", 0) > 0 else 0.0

    raw_0_100 = (
        (avg_hype * 38.0)
        + (top_avg_hype * 32.0)
        + (hype_ratio * 15.0)
        + (velocity_score * 15.0)
        + clip_tag_bonus
    )
    hype_0_100 = round(max(0.0, min(100.0, raw_0_100)), 1)

    hype_level: Literal["calm", "warming_up", "high_hype", "clip_moment"]
    if hype_0_100 >= 72.0:
        hype_level = "clip_moment"
    elif hype_0_100 >= 50.0:
        hype_level = "high_hype"
    elif hype_0_100 >= 25.0:
        hype_level = "warming_up"
    else:
        hype_level = "calm"

    clip_recommended = (
        hype_0_100 >= 70.0 or tag_counter.get("Klip İsteği", 0) >= 2
    )
    dominant_triggers = [tag for tag, _ in tag_counter.most_common(6)]
    if "Klip İsteği" in tag_counter and "Klip İsteği" not in dominant_triggers:
        dominant_triggers.insert(0, "Klip İsteği")
    if mps >= 3.5 and "Yüksek Sohbet Hızı" not in dominant_triggers:
        dominant_triggers.append("Yüksek Sohbet Hızı")

    return {
        "hype_score": hype_0_100,
        "hype_level": hype_level,
        "clip_recommended": clip_recommended,
        "messages_per_sec": mps,
        "hype_message_ratio": hype_ratio,
        "dominant_triggers": dominant_triggers,
    }
