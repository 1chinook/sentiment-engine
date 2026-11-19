"""Production data ingestors for real RSS/Atom feeds and anonymous Twitch IRC."""

from __future__ import annotations

import asyncio
import html
import random
import re
import xml.etree.ElementTree as ET
from collections import deque

import httpx

from app.core.config import Settings, WorkMode, get_settings
from app.core.logging import get_logger
from app.services.ingestion.base import BaseIngestor, RawIngestedMessage
from app.services.ingestion.mock_stream import MockStreamIngestor

logger = get_logger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PRIVMSG_RE = re.compile(
    r"^:(?P<user>[^!]+)![^ ]+ PRIVMSG #(?P<channel>[^ ]+) :(?P<text>.+)$"
)


def _clean_html_text(raw: str) -> str:
    """Strip HTML tags and unescape entities from RSS/Atom text."""
    unescaped = html.unescape(raw)
    cleaned = _HTML_TAG_RE.sub(" ", unescaped)
    return " ".join(cleaned.split()).strip()


def parse_rss_or_atom_xml(
    xml_text: str,
    topic: str,
    source_url: str,
    max_items: int = 15,
) -> list[RawIngestedMessage]:
    """Parse RSS 2.0 or Atom XML string into canonical RawIngestedMessage objects."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        logger.warning(
            "Malformed RSS/Atom XML payload",
            extra={"source_url": source_url, "error": str(exc)},
        )
        return []

    messages: list[RawIngestedMessage] = []

    # RSS 2.0 <item> elements
    for item in root.findall(".//item"):
        title = _clean_html_text(item.findtext("title") or "")
        desc = _clean_html_text(item.findtext("description") or "")
        link = (item.findtext("link") or item.findtext("guid") or "").strip()
        combined = f"{title}. {desc}".strip(". ") if desc else title
        if not combined:
            continue
        messages.append(
            RawIngestedMessage(
                topic=topic,
                source=f"rss:{source_url}",
                text=combined[:500],
                metadata={"link": link, "feed_url": source_url, "synthetic": False},
            )
        )
        if len(messages) >= max_items:
            return messages

    # Atom <entry> elements (namespace-agnostic tag suffix check)
    for elem in root.iter():
        if not elem.tag.endswith("entry"):
            continue
        title = ""
        summary = ""
        link = ""
        for child in elem:
            if child.tag.endswith("title") and child.text:
                title = _clean_html_text(child.text)
            elif (
                child.tag.endswith("summary") or child.tag.endswith("content")
            ) and child.text:
                summary = _clean_html_text(child.text)
            elif child.tag.endswith("link"):
                link = child.attrib.get("href", "").strip()
        combined = f"{title}. {summary}".strip(". ") if summary else title
        if not combined:
            continue
        messages.append(
            RawIngestedMessage(
                topic=topic,
                source=f"atom:{source_url}",
                text=combined[:500],
                metadata={"link": link, "feed_url": source_url, "synthetic": False},
            )
        )
        if len(messages) >= max_items:
            break

    return messages


class RSSFeedIngestor(BaseIngestor):
    """Fetches real market and movie review headlines from RSS/Atom feeds."""

    def __init__(
        self,
        feed_map: dict[str, list[str]] | None = None,
        timeout_sec: float = 8.0,
    ) -> None:
        settings = get_settings()
        if feed_map is None:
            market_urls = [
                u.strip()
                for u in settings.MARKET_RSS_URLS.split(",")
                if u.strip()
            ]
            movie_urls = [
                u.strip()
                for u in settings.MOVIE_RSS_URLS.split(",")
                if u.strip()
            ]
            self.feed_map = {
                "market_pulse": market_urls,
                "movie_reviews": movie_urls,
            }
        else:
            self.feed_map = feed_map

        self.timeout_sec = timeout_sec
        self._seen_keys: deque[str] = deque(maxlen=500)
        self._mock_fallback = MockStreamIngestor(failure_rate=0.0)

    async def _fetch_single_feed(
        self, client: httpx.AsyncClient, topic: str, url: str, max_items: int
    ) -> list[RawIngestedMessage]:
        response = await client.get(url)
        response.raise_for_status()
        parsed = parse_rss_or_atom_xml(
            xml_text=response.text,
            topic=topic,
            source_url=url,
            max_items=max_items,
        )
        fresh: list[RawIngestedMessage] = []
        for msg in parsed:
            dedup_key = f"{msg.topic}:{msg.text[:120]}"
            if dedup_key not in self._seen_keys:
                self._seen_keys.append(dedup_key)
                fresh.append(msg)
        return fresh

    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        """Pull fresh items concurrently across all configured RSS/Atom feeds."""
        tasks = []
        headers = {"User-Agent": "sentiment-engine/1.0 (+https://github.com/1chinook/sentiment-engine)"}
        async with httpx.AsyncClient(
            timeout=self.timeout_sec,
            follow_redirects=True,
            headers=headers,
        ) as client:
            for topic, urls in self.feed_map.items():
                for url in urls:
                    tasks.append(
                        self._fetch_single_feed(
                            client=client,
                            topic=topic,
                            url=url,
                            max_items=max(5, batch_size // 2),
                        )
                    )
            if not tasks:
                return []

            results = await asyncio.gather(*tasks, return_exceptions=True)

        collected: list[RawIngestedMessage] = []
        for res in results:
            if isinstance(res, Exception):
                logger.warning(
                    "RSS feed poll failed",
                    extra={"error": str(res)},
                )
                continue
            collected.extend(res)

        return collected[:batch_size]

    async def trigger_spike(
        self, topic: str, sentiment_bias: str, batch_size: int
    ) -> list[RawIngestedMessage]:
        """Delegate manual simulation spike triggers to the synthetic generator."""
        return await self._mock_fallback.trigger_spike(
            topic=topic,
            sentiment_bias=sentiment_bias,
            batch_size=batch_size,
        )


def normalize_twitch_channel(raw_channel: str) -> str:
    """Extract clean lowercase Twitch channel login from name or twitch.tv URL."""
    cleaned = raw_channel.strip()
    if "twitch.tv/" in cleaned.lower():
        cleaned = re.split(r"twitch\.tv/", cleaned, flags=re.IGNORECASE)[-1]
    cleaned = cleaned.split("/")[0].split("?")[0]
    cleaned = cleaned.lstrip("#@").strip().lower()
    return re.sub(r"[^a-z0-9_]", "", cleaned)


class TwitchAnonymousIRCIngestor(BaseIngestor):
    """Reads live Twitch chat using anonymous read-only IRC (`justinfan`)."""

    def __init__(
        self,
        channel: str,
        irc_host: str = "irc.chat.twitch.tv",
        irc_port: int = 6667,
        read_window_sec: float = 4.0,
    ) -> None:
        self.channel = normalize_twitch_channel(channel)
        self.irc_host = irc_host
        self.irc_port = irc_port
        self.read_window_sec = read_window_sec
        self._mock_fallback = MockStreamIngestor(failure_rate=0.0)

    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        """Connect briefly to Twitch anonymous IRC and drain up to `batch_size` messages."""
        if not self.channel:
            return []

        nick = f"justinfan{random.randint(10000, 99999)}"
        messages: list[RawIngestedMessage] = []

        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self.irc_host, self.irc_port),
                timeout=4.0,
            )
        except Exception as exc:
            logger.warning(
                "Twitch IRC connection failed",
                extra={"channel": self.channel, "error": str(exc)},
            )
            return []

        try:
            writer.write(f"PASS SCHMOOPIIE\r\nNICK {nick}\r\nJOIN #{self.channel}\r\n".encode("utf-8"))
            await writer.drain()

            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.read_window_sec
            while len(messages) < batch_size and loop.time() < deadline:
                remaining = max(0.1, deadline - loop.time())
                try:
                    raw_line = await asyncio.wait_for(
                        reader.readline(), timeout=remaining
                    )
                except asyncio.TimeoutError:
                    break
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if line.startswith("PING"):
                    writer.write(b"PONG :tmi.twitch.tv\r\n")
                    await writer.drain()
                    continue
                match = _PRIVMSG_RE.match(line)
                if match:
                    text = match.group("text").strip()
                    user = match.group("user").strip()
                    if text:
                        messages.append(
                            RawIngestedMessage(
                                topic="stream_chat",
                                source=f"twitch:{self.channel}",
                                text=text,
                                metadata={
                                    "channel": self.channel,
                                    "username": user,
                                    "synthetic": False,
                                },
                            )
                        )
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

        return messages

    async def trigger_spike(
        self, topic: str, sentiment_bias: str, batch_size: int
    ) -> list[RawIngestedMessage]:
        return await self._mock_fallback.trigger_spike(
            topic=topic,
            sentiment_bias=sentiment_bias,
            batch_size=batch_size,
        )


class CompositeProductionIngestor(BaseIngestor):
    """Aggregates real RSS/Atom feeds and optional Twitch chat with graceful fallback."""

    def __init__(
        self,
        rss_ingestor: RSSFeedIngestor | None = None,
        twitch_ingestor: TwitchAnonymousIRCIngestor | None = None,
        fallback_ingestor: MockStreamIngestor | None = None,
    ) -> None:
        settings = get_settings()
        self.rss_ingestor = rss_ingestor or RSSFeedIngestor()
        self.twitch_ingestor = twitch_ingestor or (
            TwitchAnonymousIRCIngestor(settings.TWITCH_CHANNEL)
            if settings.TWITCH_CHANNEL
            else None
        )
        self.fallback_ingestor = fallback_ingestor or MockStreamIngestor(
            failure_rate=0.0
        )

    async def fetch_batch(self, batch_size: int = 20) -> list[RawIngestedMessage]:
        tasks = [self.rss_ingestor.fetch_batch(batch_size=batch_size)]
        if self.twitch_ingestor is not None:
            tasks.append(
                self.twitch_ingestor.fetch_batch(batch_size=batch_size // 2 or 5)
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)
        combined: list[RawIngestedMessage] = []
        for res in results:
            if isinstance(res, list):
                combined.extend(res)

        if combined:
            return combined[:batch_size]

        # If all external RSS feeds had no unseen entries in this 10s tick,
        # return a small heartbeat batch from fallback so time-series stays alive.
        return await self.fallback_ingestor.fetch_batch(batch_size=min(batch_size, 5))

    async def trigger_spike(
        self, topic: str, sentiment_bias: str, batch_size: int
    ) -> list[RawIngestedMessage]:
        return await self.fallback_ingestor.trigger_spike(
            topic=topic,
            sentiment_bias=sentiment_bias,
            batch_size=batch_size,
        )


def build_ingestor(settings: Settings | None = None) -> BaseIngestor:
    """Factory selecting MockStreamIngestor or CompositeProductionIngestor by WORK_MODE."""
    active = settings or get_settings()
    if active.WORK_MODE == WorkMode.PRODUCTION:
        logger.info("Initializing CompositeProductionIngestor for PRODUCTION mode")
        return CompositeProductionIngestor()
    return MockStreamIngestor(failure_rate=active.MOCK_FAILURE_RATE)
