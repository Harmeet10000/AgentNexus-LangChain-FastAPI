from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

_otel_prometheus_reader: PrometheusMetricReader | None = None


def _setup_meter_provider(
    resource: Resource,
    export_interval_ms: int = 15000,
) -> MeterProvider:
    global _otel_prometheus_reader  # noqa: PLW0603 — intentional module-level state for OTEL metrics

    readers: list = [
        PeriodicExportingMetricReader(
            OTLPMetricExporter(),
            export_interval_millis=export_interval_ms,
        ),
    ]

    try:
        prometheus_reader = PrometheusMetricReader()
        readers.append(prometheus_reader)
        _otel_prometheus_reader = prometheus_reader
    except Exception:  # noqa: BLE001 — metrics setup must not crash app
        _otel_prometheus_reader = None

    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    return provider


def get_prometheus_reader() -> PrometheusMetricReader | None:
    return _otel_prometheus_reader
