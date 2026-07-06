from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.shared.otel.metrics import get_prometheus_reader


def get_prometheus_metrics() -> tuple[bytes, str]:
    reader = get_prometheus_reader()
    if reader is None:
        return b"", CONTENT_TYPE_LATEST
    return generate_latest(reader._collector.registry), CONTENT_TYPE_LATEST  # type: ignore
