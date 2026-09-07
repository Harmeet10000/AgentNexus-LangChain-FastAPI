from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from app.shared.otel.configuration import exporter_enabled


def _setup_tracer_provider(
    resource: Resource,
    sample_rate: float = 1.0,
    *,
    exporter: str = "otlp",
    endpoint: str | None = None,
) -> TracerProvider | None:
    if not exporter_enabled(exporter):
        return None

    provider = TracerProvider(
        resource=resource,
        sampler=ParentBased(root=TraceIdRatioBased(sample_rate)) if sample_rate < 1.0 else None,
    )
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint=endpoint),
        max_queue_size=2048,
        max_export_batch_size=512,
        schedule_delay_millis=5000,
    )
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    return provider
