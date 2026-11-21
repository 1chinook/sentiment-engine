"""Independent high-volume async load, latency, accuracy, and consistency test."""

from __future__ import annotations

import argparse
import asyncio
import math
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any

import httpx

BENCHMARK_SAMPLES: list[tuple[str, str]] = [
    ("POGGERS efsane klip!! 🔥", "positive"),
    ("Harika clutch aldın helal olsun!", "positive"),
    ("Masterpiece! Directing and score were phenomenal ❤️", "positive"),
    ("BIST ve teknoloji hisselerinde güçlü yükseliş, rekor hacim! 📈", "positive"),
    ("Bullish breakout above resistance with strong institutional buying 🚀", "positive"),
    ("Lag var yayında ses kayıyor rezalet 😡", "negative"),
    ("Senaryo çok kopuktu, 2 saatim boşa gitti tam bir hayal kırıklığı", "negative"),
    ("Terrible pacing and wooden dialogue, worst movie of the year 👎", "negative"),
    ("Enflasyon endişesiyle sert satış dalgası, piyasalar çakıldı 📉", "negative"),
    ("Boring stream going to sleep...", "negative"),
    ("Piyasalar Fed faiz kararı öncesi yatay seyrediyor", "neutral"),
    ("Chat sakin olun modlar nerede", "neutral"),
    ("Central bank minutes scheduled for release at 14:00 UTC.", "neutral"),
]


@dataclass
class BatchCallMetrics:
    ok: bool
    latency_ms: float
    texts_count: int
    correct_predictions: int
    status_code: int
    error: str | None = None


def _percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[lower]
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


async def _send_analyze_batch(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    batch_size: int,
    persist: bool,
) -> BatchCallMetrics:
    samples = [
        BENCHMARK_SAMPLES[i % len(BENCHMARK_SAMPLES)]
        for i in range(batch_size)
    ]
    texts = [text for text, _ in samples]
    expected_labels = [label for _, label in samples]

    async with semaphore:
        started = time.perf_counter()
        try:
            response = await client.post(
                "/api/v1/analyze",
                json={
                    "texts": texts,
                    "topic": "load_test_benchmark",
                    "source": "simulate_load",
                    "persist": persist,
                },
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if response.status_code != 200:
                return BatchCallMetrics(
                    ok=False,
                    latency_ms=elapsed_ms,
                    texts_count=len(texts),
                    correct_predictions=0,
                    status_code=response.status_code,
                    error=response.text[:200],
                )

            payload = response.json()
            results = payload.get("results", [])
            correct = sum(
                1
                for item, expected in zip(results, expected_labels, strict=False)
                if item.get("sentiment_label") == expected
            )
            return BatchCallMetrics(
                ok=True,
                latency_ms=elapsed_ms,
                texts_count=len(texts),
                correct_predictions=correct,
                status_code=response.status_code,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return BatchCallMetrics(
                ok=False,
                latency_ms=elapsed_ms,
                texts_count=len(texts),
                correct_predictions=0,
                status_code=0,
                error=str(exc),
            )


async def _get_topic_volume(
    client: httpx.AsyncClient, topic: str
) -> dict[str, Any]:
    resp = await client.get(f"/api/v1/sentiment/{topic}")
    resp.raise_for_status()
    return resp.json()


async def run_load_simulation(
    base_url: str,
    concurrency: int,
    total_requests: int,
    batch_size: int,
    spike_size: int,
    persist_analyze: bool,
) -> int:
    """Execute concurrent API stress tests and verify DB consistency."""
    print("=" * 72)
    print(f"sentiment-engine Load & Consistency Simulation -> {base_url}")
    print(
        f"Requests: {total_requests} | Concurrency: {concurrency} | "
        f"Batch Size: {batch_size} | Spike Size: {spike_size}"
    )
    print("=" * 72)

    limits = httpx.Limits(
        max_connections=max(concurrency * 2, 20),
        max_keepalive_connections=max(concurrency, 10),
    )
    timeout = httpx.Timeout(30.0)

    async with httpx.AsyncClient(
        base_url=base_url,
        limits=limits,
        timeout=timeout,
    ) as client:
        # 1. Health Check
        health_resp = await client.get("/api/v1/health")
        health_resp.raise_for_status()
        health_data = health_resp.json()
        print(
            f"[Health] status={health_data['status']} "
            f"mode={health_data['work_mode']} "
            f"nlp_backend={health_data['nlp_model'].get('backend')} "
            f"db_connected={health_data['database'].get('connected')}"
        )

        # 2. Record baseline topic volumes before spikes
        spike_plan = [
            ("stream_chat", "positive"),
            ("movie_reviews", "negative"),
            ("market_pulse", "positive"),
        ]
        baseline_volumes: dict[str, int] = {}
        for topic, _ in spike_plan:
            before = await _get_topic_volume(client, topic)
            baseline_volumes[topic] = int(before["total_volume"])

        # 3. Concurrent Batch Analysis Stress Test
        semaphore = asyncio.Semaphore(concurrency)
        wall_started = time.perf_counter()
        tasks = [
            _send_analyze_batch(client, semaphore, batch_size, persist_analyze)
            for _ in range(total_requests)
        ]
        metrics_list = await asyncio.gather(*tasks)
        wall_sec = time.perf_counter() - wall_started

        successes = [m for m in metrics_list if m.ok]
        failures = [m for m in metrics_list if not m.ok]
        latencies = sorted(m.latency_ms for m in successes)

        total_texts = sum(m.texts_count for m in successes)
        total_correct = sum(m.correct_predictions for m in successes)
        accuracy = (total_correct / total_texts * 100.0) if total_texts else 0.0

        print("\n--- Analyze Batch Stress Results ---")
        print(f"Succeeded / Total Requests : {len(successes)} / {len(metrics_list)}")
        print(f"Total Texts Analyzed       : {total_texts}")
        print(f"Wall Time                  : {wall_sec:.3f} s")
        print(f"Request Throughput         : {len(successes) / max(wall_sec, 1e-6):.2f} req/s")
        print(f"Text Throughput            : {total_texts / max(wall_sec, 1e-6):.2f} texts/s")
        if latencies:
            print(
                f"Latency (ms)               : "
                f"min={latencies[0]:.2f} | "
                f"avg={statistics.mean(latencies):.2f} | "
                f"p50={_percentile(latencies, 50):.2f} | "
                f"p95={_percentile(latencies, 95):.2f} | "
                f"p99={_percentile(latencies, 99):.2f} | "
                f"max={latencies[-1]:.2f}"
            )
        print(f"Benchmark Accuracy         : {accuracy:.2f}% ({total_correct}/{total_texts})")
        if failures:
            print(f"Failed Requests ({len(failures)}):")
            for err in failures[:5]:
                print(f"  - status={err.status_code} err={err.error}")

        # 4. Trigger Simulation Spikes Concurrently & Verify DB Consistency
        print("\n--- DB Consistency & Spike Verification ---")
        spike_responses = await asyncio.gather(
            *[
                client.post(
                    "/api/v1/simulation/trigger",
                    json={
                        "topic": topic,
                        "sentiment_bias": bias,
                        "batch_size": spike_size,
                    },
                )
                for topic, bias in spike_plan
            ]
        )

        consistency_ok = True
        for (topic, bias), resp in zip(spike_plan, spike_responses, strict=True):
            resp.raise_for_status()
            spike_payload = resp.json()
            after = await _get_topic_volume(client, topic)
            delta = int(after["total_volume"]) - baseline_volumes[topic]
            expected_min = spike_size if spike_payload["persisted_to_db"] else 0

            passed = delta >= expected_min
            consistency_ok = consistency_ok and passed
            print(
                f"Topic={topic:<14} bias={bias:<8} "
                f"before={baseline_volumes[topic]:<5} after={after['total_volume']:<5} "
                f"delta={delta:<4} 5m_avg={after['window_5m']['moving_avg_sentiment']:+.4f} "
                f"consistent={passed}"
            )

        print("=" * 72)
        if failures or not consistency_ok:
            print("RESULT: FAILED")
            return 1
        print("RESULT: PASSED")
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run async load, latency, accuracy, and DB consistency tests."
    )
    parser.add_argument("--base-url", default="http://localhost:8001")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=13)
    parser.add_argument("--spike-size", type=int, default=30)
    parser.add_argument("--persist-analyze", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exit_code = asyncio.run(
        run_load_simulation(
            base_url=args.base_url,
            concurrency=args.concurrency,
            total_requests=args.requests,
            batch_size=args.batch_size,
            spike_size=args.spike_size,
            persist_analyze=args.persist_analyze,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
