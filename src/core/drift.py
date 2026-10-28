import hashlib
import math
import os
from collections import Counter
from typing import Any, Iterable, Sequence

from dotenv import load_dotenv
from openai import OpenAI

from src.core.otel_setup import get_tracer

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

EMBEDDING_MODEL = os.getenv("DRIFT_EMBEDDING_MODEL", "text-embedding-3-small")
BASELINE_WINDOW_SIZE = int(os.getenv("DRIFT_BASELINE_WINDOW_SIZE", "5"))
ROLLING_WINDOW_SIZE = int(os.getenv("DRIFT_ROLLING_WINDOW_SIZE", "5"))
COSINE_DRIFT_THRESHOLD = float(os.getenv("DRIFT_COSINE_THRESHOLD", "0.35"))
TOOL_TVD_THRESHOLD = float(os.getenv("DRIFT_TOOL_TVD_THRESHOLD", "0.25"))

tracer = get_tracer("src.core.drift")

_observations: list[dict[str, Any]] = []


def _get_client() -> OpenAI | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def _fallback_embedding(text: str) -> list[float]:
    digest = hashlib.sha256((text or "").encode("utf-8")).digest()
    vec = [0.0] * 1536
    for i in range(len(vec)):
        byte = digest[i % len(digest)]
        vec[i] = ((byte + i * 17) % 251) / 250.0 - 0.5
    return vec


def _embed_text(text: str) -> list[float]:
    client = _get_client()
    if client is None:
        return _fallback_embedding(text)

    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text or "",
        )
        embedding = response.data[0].embedding
        return [float(v) for v in embedding]
    except Exception:
        return _fallback_embedding(text)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def _mean_vector(vectors: Iterable[Sequence[float]]) -> list[float]:
    vectors = list(vectors)
    if not vectors:
        return []
    dim = len(vectors[0])
    totals = [0.0] * dim
    for vector in vectors:
        if len(vector) != dim:
            continue
        for i, value in enumerate(vector):
            totals[i] += float(value)
    count = len(vectors)
    return [total / count for total in totals]


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 1.0
    dot = _dot(a, b)
    denom = _norm(a) * _norm(b)
    if denom == 0:
        return 1.0
    similarity = dot / denom
    return 1.0 - max(-1.0, min(1.0, similarity))


def _normalize_counter(counter: Counter[str]) -> dict[str, float]:
    total = sum(counter.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def _total_variation_distance(counter_a: Counter[str], counter_b: Counter[str]) -> float:
    if not counter_a and not counter_b:
        return 0.0
    keys = set(counter_a) | set(counter_b)
    if not keys:
        return 0.0
    pa = _normalize_counter(counter_a)
    pb = _normalize_counter(counter_b)
    return 0.5 * sum(abs(pa.get(key, 0.0) - pb.get(key, 0.0)) for key in keys)


def _merge_tool_counts(observations: Sequence[dict[str, Any]]) -> Counter[str]:
    combined: Counter[str] = Counter()
    for obs in observations:
        combined.update(obs.get("tool_counts", Counter()))
    return combined


def observe(narrative: str, tool_sequence: Sequence[str] | None) -> dict[str, Any]:
    """Record a new SAR narrative and its tool-usage signature.

    Returns the latest drift snapshot for that observation.
    """
    with tracer.start_as_current_span("drift.observe") as span:
        span.set_attribute("narrative_length", len(narrative or ""))
        span.set_attribute("tool_count", len(tool_sequence or []))

        embedding = _embed_text(narrative or "")
        tool_counts = Counter(tool_sequence or [])

        observation = {
            "narrative": narrative or "",
            "embedding": embedding,
            "tool_counts": tool_counts,
            "tool_sequence": list(tool_sequence or []),
        }
        _observations.append(observation)

        baseline_size = min(BASELINE_WINDOW_SIZE, len(_observations))
        baseline_vectors = [obs["embedding"] for obs in _observations[:baseline_size]]
        baseline_centroid = _mean_vector(baseline_vectors) if baseline_vectors else []

        recent_observations = _observations[-ROLLING_WINDOW_SIZE:]
        recent_tool_counts = _merge_tool_counts(recent_observations)
        baseline_tool_counts = _merge_tool_counts(_observations[:baseline_size])
        tool_tvd = _total_variation_distance(baseline_tool_counts, recent_tool_counts)

        current_distance = 0.0
        if baseline_centroid and len(_observations) > baseline_size:
            current_distance = _cosine_distance(embedding, baseline_centroid)

        status = {
            "observations_seen": len(_observations),
            "baseline_window_size": baseline_size,
            "baseline_ready": bool(baseline_centroid),
            "rolling_window_size": len(recent_observations),
            "current_cosine_distance": current_distance,
            "baseline_centroid": baseline_centroid,
            "baseline_tool_counts": dict(baseline_tool_counts),
            "rolling_tool_counts": dict(recent_tool_counts),
            "tool_total_variation_distance": tool_tvd,
            "alert": current_distance > COSINE_DRIFT_THRESHOLD or tool_tvd > TOOL_TVD_THRESHOLD,
        }
        span.set_attribute("cosine_distance", current_distance)
        span.set_attribute("tool_tvd", tool_tvd)
        return status


def status() -> dict[str, Any]:
    """Return the current drift snapshot for the in-memory observation history."""
    if not _observations:
        return {
            "observations_seen": 0,
            "baseline_window_size": BASELINE_WINDOW_SIZE,
            "baseline_ready": False,
            "current_cosine_distance": 0.0,
            "rolling_window_size": 0,
            "baseline_tool_counts": {},
            "rolling_tool_counts": {},
            "tool_total_variation_distance": 0.0,
            "alert": False,
        }

    baseline_size = min(BASELINE_WINDOW_SIZE, len(_observations))
    baseline_vectors = [obs["embedding"] for obs in _observations[:baseline_size]]
    baseline_centroid = _mean_vector(baseline_vectors) if baseline_vectors else []
    recent_observations = _observations[-ROLLING_WINDOW_SIZE:]
    recent_tool_counts = _merge_tool_counts(recent_observations)
    baseline_tool_counts = _merge_tool_counts(_observations[:baseline_size])
    tool_tvd = _total_variation_distance(baseline_tool_counts, recent_tool_counts)

    current_distance = 0.0
    if baseline_centroid and len(_observations) > baseline_size:
        latest_embedding = _observations[-1]["embedding"]
        current_distance = _cosine_distance(latest_embedding, baseline_centroid)

    return {
        "observations_seen": len(_observations),
        "baseline_window_size": baseline_size,
        "baseline_ready": bool(baseline_centroid),
        "current_cosine_distance": current_distance,
        "rolling_window_size": len(recent_observations),
        "baseline_tool_counts": dict(baseline_tool_counts),
        "rolling_tool_counts": dict(recent_tool_counts),
        "tool_total_variation_distance": tool_tvd,
        "alert": current_distance > COSINE_DRIFT_THRESHOLD or tool_tvd > TOOL_TVD_THRESHOLD,
    }
