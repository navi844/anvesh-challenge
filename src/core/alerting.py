import os
from collections import deque
from datetime import datetime, timezone
from typing import Any, Sequence

from src.core.drift import observe as observe_drift
from src.core.online_eval import rolling_average_score

ALERT_HISTORY_LIMIT = int(os.getenv("ALERT_HISTORY_LIMIT", "50"))
COST_SPIKE_FACTOR = float(os.getenv("ALERT_COST_SPIKE_FACTOR", "2.5"))
MIN_ROLLING_EVAL_SCORE = float(os.getenv("ALERT_MIN_ROLLING_EVAL_SCORE", "3.0"))
COST_BASELINE_WINDOW = int(os.getenv("ALERT_COST_BASELINE_WINDOW", "10"))

_alerts: list[dict[str, Any]] = []
_cost_history: deque[float] = deque(maxlen=COST_BASELINE_WINDOW)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _append_alert(alert: dict[str, Any]) -> None:
    global _alerts
    _alerts.append(alert)
    if len(_alerts) > ALERT_HISTORY_LIMIT:
        _alerts = _alerts[-ALERT_HISTORY_LIMIT:]


def list_alerts() -> list[dict[str, Any]]:
    return list(_alerts)


def observe_and_alert(
    requested_model: str,
    served_model: str,
    cost_usd: float,
    sar_narrative: str,
    tool_sequence: Sequence[str] | None,
) -> list[dict[str, Any]]:
    """Evaluate a completed investigation and append any firing alerts."""
    fired: list[dict[str, Any]] = []

    if str(requested_model) != str(served_model):
        alert = {
            "rule": "model_mismatch",
            "timestamp": _now_iso(),
            "requested_model": str(requested_model),
            "served_model": str(served_model),
            "cost_usd": float(cost_usd or 0.0),
            "details": "Requested model does not match served model.",
        }
        _append_alert(alert)
        fired.append(alert)

    _cost_history.append(float(cost_usd or 0.0))
    if len(_cost_history) >= 3:
        baseline = _mean(list(_cost_history)[:-1])
        if baseline > 0 and float(cost_usd or 0.0) > baseline * COST_SPIKE_FACTOR:
            alert = {
                "rule": "cost_spike",
                "timestamp": _now_iso(),
                "cost_usd": float(cost_usd or 0.0),
                "rolling_baseline_mean": round(baseline, 6),
                "threshold_multiplier": COST_SPIKE_FACTOR,
                "details": "Cost jumped above the rolling baseline mean by the configured threshold.",
            }
            _append_alert(alert)
            fired.append(alert)

    avg_score = rolling_average_score()
    if avg_score > 0 and avg_score < MIN_ROLLING_EVAL_SCORE:
        alert = {
            "rule": "low_online_eval_score",
            "timestamp": _now_iso(),
            "rolling_average_score": round(avg_score, 4),
            "threshold_score": MIN_ROLLING_EVAL_SCORE,
            "details": "Rolling online-eval score falls below the minimum acceptable threshold.",
        }
        _append_alert(alert)
        fired.append(alert)

    drift_snapshot = observe_drift(sar_narrative or "", list(tool_sequence or []))
    if bool(drift_snapshot.get("alert")):
        alert = {
            "rule": "drift_alert",
            "timestamp": _now_iso(),
            "drift_snapshot": drift_snapshot,
            "details": "Drift detector flagged this observation against the baseline.",
        }
        _append_alert(alert)
        fired.append(alert)

    return fired
