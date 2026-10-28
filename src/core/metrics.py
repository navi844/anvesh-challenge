from opentelemetry.metrics import Observation

from src.core.otel_setup import get_meter

meter = get_meter("src.core.metrics")

request_counter = meter.create_counter(
    name="investigation_requests_total",
    description="Number of investigation requests processed.",
    unit="1",
)

duration_histogram = meter.create_histogram(
    name="investigation_duration",
    description="Duration of investigation processing.",
    unit="1",
)

cost_histogram = meter.create_histogram(
    name="investigation_cost",
    description="Total cost of each investigation in USD.",
    unit="1",
)

tool_failure_counter = meter.create_counter(
    name="tool_failures_total",
    description="Number of tool failures encountered during investigations.",
    unit="1",
)


def record_investigation_turn(
    *,
    customer_id: str,
    model: str,
    turn_count: int,
    finish_reason: str,
    duration_ms: float,
    cost_usd: float,
    tool_failures: int = 0,
    tool_name: str | None = None,
) -> None:
    request_counter.add(1, {"customer.id": customer_id, "model": model, "finish_reason": finish_reason})
    duration_histogram.record(duration_ms, {"customer.id": customer_id, "model": model, "turn_count": turn_count})
    cost_histogram.record(cost_usd, {"customer.id": customer_id, "model": model})

    if tool_failures > 0:
        tool_failure_counter.add(tool_failures, {
            "customer.id": customer_id,
            "model": model,
            "tool_name": tool_name or "unknown",
        })
