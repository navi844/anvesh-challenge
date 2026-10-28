import os

from opentelemetry import metrics, trace

try:
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ModuleNotFoundError:  # pragma: no cover - optional dependency in starter setup
    OTLPMetricExporter = None
    OTLPSpanExporter = None

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider

try:
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
except ModuleNotFoundError:  # pragma: no cover - optional dependency in older SDK installs
    PeriodicExportingMetricReader = None

try:
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
except ModuleNotFoundError:  # pragma: no cover - optional dependency in older SDK installs
    BatchSpanProcessor = None

OTEL_COLLECTOR_BASE_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
OTEL_TRACES_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", f"{OTEL_COLLECTOR_BASE_ENDPOINT}/v1/traces")
OTEL_METRICS_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", f"{OTEL_COLLECTOR_BASE_ENDPOINT}/v1/metrics")

trace_provider = trace.get_tracer_provider()
if not isinstance(trace_provider, TracerProvider):
    trace_provider = TracerProvider(
        resource=Resource.create({
            "service.name": "anvesh-agent",
            "service.version": "1.0.0",
        })
    )
    if OTLPSpanExporter is not None and BatchSpanProcessor is not None:
        trace_exporter = OTLPSpanExporter(endpoint=OTEL_TRACES_ENDPOINT)
        trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(trace_provider)

metric_provider = metrics.get_meter_provider()
if not isinstance(metric_provider, MeterProvider):
    if OTLPMetricExporter is not None and PeriodicExportingMetricReader is not None:
        metric_reader = PeriodicExportingMetricReader(
            exporter=OTLPMetricExporter(endpoint=OTEL_METRICS_ENDPOINT),
            export_interval_millis=5000,
        )
        metric_provider = MeterProvider(
            resource=Resource.create({
                "service.name": "anvesh-agent",
                "service.version": "1.0.0",
            }),
            metric_readers=[metric_reader],
        )
    else:
        metric_provider = MeterProvider(
            resource=Resource.create({
                "service.name": "anvesh-agent",
                "service.version": "1.0.0",
            })
        )
    metrics.set_meter_provider(metric_provider)


def get_tracer(name: str):
    return trace.get_tracer(name)


def get_meter(name: str):
    return metrics.get_meter(name)
