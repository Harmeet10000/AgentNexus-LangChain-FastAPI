"""OTel-specific hooks and middleware for FastAPI."""

import re
from typing import Any

from opentelemetry import trace

UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE
)
DIGIT_PATTERN = re.compile(r"^\d+$")


def _normalize_path_otel(path: str) -> str:
    segments = path.strip("/").split("/")
    normalized = []
    for seg in segments:
        if UUID_PATTERN.match(seg) or DIGIT_PATTERN.match(seg):
            normalized.append("{id}")
        else:
            normalized.append(seg)
    return "/" + "/".join(normalized)


def default_span_details(scope: dict[str, Any]) -> tuple[str, trace.SpanKind]:
    path = scope.get("path", "/")
    route = _normalize_path_otel(path)
    return route, trace.SpanKind.SERVER
