from __future__ import annotations

from typing import Any

from src.core.online_eval import recent_eval_history

_feedback_by_investigation: dict[str, dict[str, Any]] = {}


def record_feedback(investigation_id: str, rating: str | int, note: str | None = None) -> dict[str, Any]:
    """Record a human feedback item for an investigation.

    A thumbs-down or low score should later be surfaced through the annotation queue.
    """
    normalized_rating = str(rating).strip().lower()
    payload = {
        "investigation_id": str(investigation_id),
        "rating": normalized_rating,
        "note": note or "",
        "source": "human",
    }
    _feedback_by_investigation[str(investigation_id)] = payload
    return payload


def annotation_queue() -> list[dict[str, Any]]:
    """Return merged human and sampled low-scoring annotations, deduplicated by investigation_id."""
    merged: dict[str, dict[str, Any]] = {}

    for item in list(_feedback_by_investigation.values()):
        investigation_id = str(item["investigation_id"])
        merged[investigation_id] = {
            "investigation_id": investigation_id,
            "rating": item.get("rating"),
            "note": item.get("note") or "",
            "source": "human",
        }

    for item in recent_eval_history():
        if not item.get("sampled"):
            continue

        score = item.get("score")
        low_agreement = bool(item.get("low_agreement"))
        reason = str(item.get("reason") or "")
        if score is None or not low_agreement:
            continue

        investigation_id = str(item.get("investigation_id") or item.get("id") or "unknown")
        if investigation_id not in merged:
            merged[investigation_id] = {
                "investigation_id": investigation_id,
                "rating": "fail",
                "note": reason,
                "source": "online_eval",
            }
        else:
            existing = merged[investigation_id]
            existing["rating"] = existing.get("rating") or "fail"
            existing["note"] = existing.get("note") or reason
            existing["source"] = "merged"

    return list(merged.values())
