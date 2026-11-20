"""Hybrid Live-Stream Hype & Excitement Scorer for Twitch/Kick chat."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Literal

from app.db.models import SentimentLabel

_TR_FOLD_MAP = str.maketrans(
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

_CHAT_STOPWORDS = frozenset(
    {
        "ve",
        "veya",
        "ama",
        "fakat",
        "lakin",
        "bu",
        "su",
        "bir",
        "iki",
        "de",
        "da",
        "ki",
        "mi",
        "mu",
        "ne",
        "nasil",
        "neden",
        "niye",
        "kim",
        "cok",
        "az",
        "daha",
        "en",
        "ile",
        "icin",
        "gibi",
        "kadar",
        "olan",
        "var",
        "yok",
        "ben",
        "sen",
        "biz",
        "siz",
        "onlar",
        "bana",
        "sana",
        "bunu",
        "sunu",
        "onu",
        "oyle",
        "boyle",
        "simdi",
        "artik",
        "hala",
        "bile",
        "sadece",
        "zaten",
        "hani",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "to",
        "of",
        "and",
        "in",
        "that",
        "it",
        "for",
        "on",
        "with",
        "as",
        "at",
        "this",
        "from",
        "by",
        "or",
        "an",
        "you",
        "me",
        "my",
        "so",
        "just",
        "not",
        "no",
        "yes",
        "what",
        "who",
        "why",
        "how",
        "ya",
        "la",
        "lan",
        "abi",
        "kanka",
        "knk",
        "beyler",
        "arkadaslar",
    }
)

# Semantic concept bridges: connects different words referring to the same stream moment
_SEMANTIC_CONCEPT_GROUPS: dict[str, tuple[str, ...]] = {
    "Müzik / Şarkı": ("sark", "muzi", "song", "musi", "trac", "beat", "melod", "parc"),
    "Ses / Mikrofon Sorunu": ("mikr", "sesi", "ses ", "audi", "duyu", "cizi", "yank", "mute"),
    "Yayın Kalitesi / Lag": ("lag", "donu", "kasi", "ping", "fps", "drop", "pikse", "yayin"),
    "Hile / Şüpheli Oyun": ("hile", "hack", "aimb", "wall", "chea", "macro"),
    "Gol / Maç Anı": ("gol", "pena", "ofsa", "hake", "dire", "mac", "skor"),
    "Klip Anı": ("klip", "clip"),
}

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


@dataclass(frozen=True, slots=True)
class ChatCluster:
    """Group of repeated or thematically connected chat messages."""

    label: str
    match_count: int
    unique_users: int
    cohesion_type: Literal["Toplu Aynı Tepki", "Bağlantılı Konu Kümesi"]
    sample_texts: list[str]


@dataclass(frozen=True, slots=True)
class ChatCohesionResult:
    """Batch-level synchronization and related-topic analysis across chat messages."""

    cohesion_score: float  # 0.0 to 1.0
    wave_detected: bool
    wave_type: Literal["none", "collective_echo", "related_topic_cluster", "combined_wave"]
    clusters: list[ChatCluster]


def _normalize_for_echo(text: str) -> str:
    """Fold Turkish characters, lowercase, and collapse elongated letter runs."""
    folded = text.strip().lower().translate(_TR_FOLD_MAP)
    collapsed = re.sub(r"([a-z])\1+", r"\1", folded)
    tokens = re.findall(r"[a-z0-9]+", collapsed)
    return " ".join(tokens)


def _extract_semantic_keys(text: str) -> set[str]:
    """Extract normalized Turkish/English stems and semantic concept bridges."""
    folded = text.strip().lower().translate(_TR_FOLD_MAP)
    collapsed = re.sub(r"([a-z])\1+", r"\1", folded)
    raw_tokens = re.findall(r"[a-z0-9]+", collapsed)

    keys: set[str] = set()
    for tok in raw_tokens:
        if tok in _CHAT_STOPWORDS or len(tok) < 3:
            continue
        stem = tok[:4] if len(tok) >= 4 else tok
        matched_concept = False
        for concept_name, prefixes in _SEMANTIC_CONCEPT_GROUPS.items():
            if any(stem.startswith(p.strip()) or tok.startswith(p.strip()) for p in prefixes):
                keys.add(f"concept:{concept_name}")
                matched_concept = True
                break
        if not matched_concept:
            keys.add(f"stem:{stem}")
    return keys


def analyze_chat_cohesion(
    texts: list[str],
    usernames: list[str] | None = None,
) -> ChatCohesionResult:
    """Detect repeated identical reactions and thematically connected chat waves."""
    total = len(texts)
    if total < 2:
        return ChatCohesionResult(
            cohesion_score=0.0,
            wave_detected=False,
            wave_type="none",
            clusters=[],
        )

    resolved_users = (
        usernames
        if usernames is not None and len(usernames) == total
        else [f"msg_{i}" for i in range(total)]
    )

    # 1. Exact/normalized echo detection (same phrase, emote, or number spam)
    echo_buckets: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(texts):
        norm = _normalize_for_echo(text)
        if norm:
            echo_buckets[norm].append(idx)

    echo_clusters: list[ChatCluster] = []
    echo_covered_indices: set[int] = set()
    min_echo_count = 2 if total <= 8 else 3

    for norm_phrase, indices in sorted(
        echo_buckets.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        unique_u = len({resolved_users[i] for i in indices})
        if len(indices) >= min_echo_count and (
            unique_u >= 2 or usernames is None
        ):
            echo_covered_indices.update(indices)
            samples: list[str] = []
            for i in indices:
                if texts[i] not in samples:
                    samples.append(texts[i])
                if len(samples) >= 3:
                    break
            echo_clusters.append(
                ChatCluster(
                    label=norm_phrase[:40],
                    match_count=len(indices),
                    unique_users=unique_u,
                    cohesion_type="Toplu Aynı Tepki",
                    sample_texts=samples,
                )
            )

    # 2. Related / connected topic clustering via Turkish stems & concept bridges
    key_buckets: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(texts):
        for key in _extract_semantic_keys(text):
            key_buckets[key].append(idx)

    related_clusters: list[ChatCluster] = []
    related_covered_indices: set[int] = set()
    min_related_count = 2 if total <= 6 else 3

    for key, indices in sorted(
        key_buckets.items(), key=lambda kv: len(kv[1]), reverse=True
    ):
        # Deduplicate indices per message
        unique_indices = list(dict.fromkeys(indices))
        if len(unique_indices) < min_related_count:
            continue
        # Skip if this exact group is already fully explained by a single identical echo
        if set(unique_indices).issubset(echo_covered_indices) and len(echo_clusters) > 0:
            # Only skip if all texts in this bucket are practically identical
            if len({_normalize_for_echo(texts[i]) for i in unique_indices}) <= 1:
                continue

        unique_u = len({resolved_users[i] for i in unique_indices})
        if unique_u < 2 and usernames is not None:
            continue

        # Avoid overlapping clusters covering the exact same subset of messages
        overlap = len(set(unique_indices) & related_covered_indices)
        if overlap == len(unique_indices):
            continue

        related_covered_indices.update(unique_indices)
        display_label = (
            key.split(":", 1)[1]
            if key.startswith("concept:")
            else f"{key.split(':', 1)[1]}*"
        )
        samples = []
        for i in unique_indices:
            if texts[i] not in samples:
                samples.append(texts[i])
            if len(samples) >= 3:
                break

        related_clusters.append(
            ChatCluster(
                label=display_label,
                match_count=len(unique_indices),
                unique_users=unique_u,
                cohesion_type="Bağlantılı Konu Kümesi",
                sample_texts=samples,
            )
        )

    all_clusters = (echo_clusters[:3] + related_clusters[:3])[:5]
    all_clusters.sort(key=lambda c: c.match_count, reverse=True)

    covered_total = len(echo_covered_indices | related_covered_indices)
    max_cluster_size = max((c.match_count for c in all_clusters), default=0)

    # Cohesion score balances covered ratio and dominant cluster concentration
    coverage_ratio = covered_total / total
    dominant_ratio = max_cluster_size / total
    cohesion_score = round(
        min(1.0, (coverage_ratio * 0.55) + (dominant_ratio * 0.65)), 4
    )

    has_echo = len(echo_clusters) > 0
    has_related = len(related_clusters) > 0

    wave_detected = bool(
        all_clusters
        and (
            cohesion_score >= 0.35
            or max_cluster_size >= 3
            or (total <= 6 and max_cluster_size >= 2)
        )
    )

    wave_type: Literal[
        "none", "collective_echo", "related_topic_cluster", "combined_wave"
    ]
    if not wave_detected:
        wave_type = "none"
    elif has_echo and has_related:
        wave_type = "combined_wave"
    elif has_echo:
        wave_type = "collective_echo"
    else:
        wave_type = "related_topic_cluster"

    return ChatCohesionResult(
        cohesion_score=cohesion_score,
        wave_detected=wave_detected,
        wave_type=wave_type,
        clusters=all_clusters,
    )


def compute_stream_hype_summary(
    message_hype_scores: list[float],
    all_tags: list[list[str]],
    listen_seconds: float,
    cohesion_result: ChatCohesionResult | None = None,
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
            "collective_sync_score": 0.0,
            "wave_detected": False,
            "wave_type": "none",
            "chat_clusters": [],
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

    cohesion_score = cohesion_result.cohesion_score if cohesion_result else 0.0
    wave_detected = cohesion_result.wave_detected if cohesion_result else False
    wave_type = cohesion_result.wave_type if cohesion_result else "none"
    cohesion_bonus = cohesion_score * 26.0 if wave_detected else cohesion_score * 10.0

    raw_0_100 = (
        (avg_hype * 34.0)
        + (top_avg_hype * 28.0)
        + (hype_ratio * 14.0)
        + (velocity_score * 14.0)
        + clip_tag_bonus
        + cohesion_bonus
    )
    hype_0_100 = round(max(0.0, min(100.0, raw_0_100)), 1)

    hype_level: Literal["calm", "warming_up", "high_hype", "clip_moment"]
    if hype_0_100 >= 72.0:
        hype_level = "clip_moment"
    elif hype_0_100 >= 50.0:
        hype_level = "high_hype"
    elif hype_0_100 >= 25.0 or wave_detected:
        hype_level = "warming_up"
    else:
        hype_level = "calm"

    clip_recommended = (
        hype_0_100 >= 70.0
        or tag_counter.get("Klip İsteği", 0) >= 2
        or (wave_detected and cohesion_score >= 0.65 and hype_0_100 >= 55.0)
    )
    dominant_triggers = [tag for tag, _ in tag_counter.most_common(6)]
    if "Klip İsteği" in tag_counter and "Klip İsteği" not in dominant_triggers:
        dominant_triggers.insert(0, "Klip İsteği")
    if wave_detected:
        if wave_type in ("collective_echo", "combined_wave"):
            dominant_triggers.insert(0, "Toplu Aynı Tepki Dalgası")
        if wave_type in ("related_topic_cluster", "combined_wave"):
            dominant_triggers.insert(0, "Bağlantılı Konu Kümesi")
    if mps >= 3.5 and "Yüksek Sohbet Hızı" not in dominant_triggers:
        dominant_triggers.append("Yüksek Sohbet Hızı")

    return {
        "hype_score": hype_0_100,
        "hype_level": hype_level,
        "clip_recommended": clip_recommended,
        "messages_per_sec": mps,
        "hype_message_ratio": hype_ratio,
        "collective_sync_score": round(cohesion_score * 100.0, 1),
        "wave_detected": wave_detected,
        "wave_type": wave_type,
        "chat_clusters": [
            {
                "label": c.label,
                "match_count": c.match_count,
                "unique_users": c.unique_users,
                "cohesion_type": c.cohesion_type,
                "sample_texts": c.sample_texts,
            }
            for c in (cohesion_result.clusters if cohesion_result else [])
        ],
        "dominant_triggers": dominant_triggers,
    }
