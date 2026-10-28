import json
import os
import random
import re
from typing import Any

from src.core.llm_provider import chat_raw, DEFAULT_MODEL

ONLINE_EVAL_SAMPLE_RATE = float(os.getenv("ONLINE_EVAL_SAMPLE_RATE", "0.10"))
ONLINE_EVAL_HISTORY_LIMIT = int(os.getenv("ONLINE_EVAL_HISTORY_LIMIT", "100"))

_recent_history: list[dict[str, Any]] = []


def _parse_json_response(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    return {}


def _score_from_payload(payload: dict[str, Any]) -> tuple[int, str, bool]:
    score = payload.get("score")
    try:
        score_int = int(score)
    except (TypeError, ValueError):
        score_int = 0

    reason = str(payload.get("reason") or payload.get("comment") or "No reason supplied.")
    low_agreement = bool(payload.get("low_agreement"))

    if not 1 <= score_int <= 5:
        score_int = 3 if not low_agreement else 1

    if score_int <= 2:
        low_agreement = True

    return score_int, reason, low_agreement


def maybe_grade(gathered_evidence: dict, sar_narrative: str, *, sample_rate: float | None = None, model: str = DEFAULT_MODEL) -> dict[str, Any]:
    """Sample a fraction of investigations and judge the SAR narrative against the evidence."""
    rate = ONLINE_EVAL_SAMPLE_RATE if sample_rate is None else sample_rate
    if rate <= 0:
        return {"sampled": False, "reason": "sampling disabled"}

    if random.random() > rate:
        return {"sampled": False, "reason": f"not sampled (rate={rate})"}

    prompt = f"""
You are a strict compliance judge. Compare the SAR narrative against the evidence dict specifically.
Use only the evidence dict; do not invent findings or give credit for unsupported claims.
Score agreement from 1 to 5:
1 = severe mismatch or fabricated claims
2 = weak agreement with important omissions or hallucinations
3 = partial agreement, some gaps or uncertainty
4 = strong agreement with minor details missing
5 = exact agreement with the evidence and no unsupported claims

Return JSON only with this schema:
{{"score": 1-5, "reason": "one-line reason", "low_agreement": true|false}}

Evidence dict:
{json.dumps(gathered_evidence, default=str, sort_keys=True)}

SAR narrative:
{sar_narrative}
"""

    result = chat_raw(
        messages=[
            {"role": "system", "content": "You are a careful evaluator of SAR quality; compare the SAR narrative against the evidence dict specifically and return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0,
    )

    payload = _parse_json_response(result.content or "")
    score, reason, low_agreement = _score_from_payload(payload)

    eval_record = {
        "sampled": True,
        "score": score,
        "reason": reason,
        "low_agreement": low_agreement,
        "requested_model": result.requested_model,
        "served_model": result.served_model,
        "cost_usd": result.cost_usd,
        "duration_ms": result.duration_ms,
    }

    _recent_history.append(eval_record)
    if len(_recent_history) > ONLINE_EVAL_HISTORY_LIMIT:
        _recent_history.pop(0)

    return eval_record


def rolling_average_score() -> float:
    if not _recent_history:
        return 0.0
    return sum(item["score"] for item in _recent_history) / len(_recent_history)


def recent_eval_history() -> list[dict[str, Any]]:
    return [dict(item) for item in _recent_history]
