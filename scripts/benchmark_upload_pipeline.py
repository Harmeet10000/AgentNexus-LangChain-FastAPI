"""Benchmark bounded upload validation and the local extraction/ingestion stages.

The benchmark deliberately avoids databases, object storage, embeddings, and network calls. It
measures the expensive deterministic boundary that decides whether an upload can reach those
systems: bounded multipart reading, Docling extraction, and classification. A token-count proxy
is recorded; model-backed chunking is deliberately excluded so the benchmark stays offline.
Run it in the production image (rather than a developer laptop) before changing the documented
limit. The oversized case is expected to fail before extraction.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from returns.result import Failure

from app.config import get_settings
from app.features.documents.classification import classify_document
from app.features.documents.parser import parse_document
from app.shared.upload_validation import read_bounded_upload

if TYPE_CHECKING:
    from typing import Any

    from app.config.settings import Settings


class _MemoryUpload:
    def __init__(self, data: bytes) -> None:
        self.filename = "benchmark.txt"
        self.content_type = "text/plain"
        self.data = data

    async def read(self, size: int = -1) -> bytes:
        return self.data if size < 0 else self.data[:size]


def _peak_rss_bytes() -> int:
    # Linux reports KiB; macOS reports bytes. The deployment image is Linux, but
    # the branch keeps the artifact interpretable when run by a developer.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if value > 10_000_000 else value * 1024


async def _measure_upload(size: int, *, limit: int) -> dict[str, Any]:
    payload = b"x" * size
    started = time.perf_counter()
    result = await read_bounded_upload(
        _MemoryUpload(payload),
        max_size=limit,
        allowed_extensions=["txt"],
    )
    elapsed = time.perf_counter() - started
    if isinstance(result, Failure):
        return {
            "size_bytes": size,
            "wall_clock_seconds": elapsed,
            "accepted": False,
            "failure_code": result.failure().code.value,
            "output_bytes": 0,
        }
    return {
        "size_bytes": size,
        "wall_clock_seconds": elapsed,
        "accepted": True,
        "failure_code": None,
        "output_bytes": len(result.unwrap()),
    }


def _read_fixture(fixture: Path) -> bytes:
    return fixture.read_bytes()


async def _measure_extraction(fixture: Path, settings: Settings) -> dict[str, Any]:
    raw = _read_fixture(fixture)
    started = time.perf_counter()
    rss_before = _peak_rss_bytes()
    try:
        parsed = await parse_document(
            raw_bytes=raw,
            filename=fixture.name,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            extraction_timeout_seconds=settings.DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
            max_pages=settings.DOCUMENT_EXTRACTION_MAX_PAGES,
            max_output_bytes=settings.DOCUMENT_EXTRACTION_MAX_OUTPUT_BYTES,
        )
        classified = classify_document(markdown=parsed.markdown, filename=fixture.name)
        # Chunk preparation uses a Hugging Face tokenizer and is intentionally
        # not part of the default benchmark: downloading a model would make a
        # local budget measurement neither deterministic nor offline-safe.
        token_count = len(parsed.markdown.split())
        return {
            "fixture": str(fixture),
            "input_bytes": len(raw),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "rss_delta_bytes": max(0, _peak_rss_bytes() - rss_before),
            "page_count": parsed.page_count,
            "output_bytes": len(parsed.markdown.encode("utf-8")),
            "token_count_estimate": token_count,
            "classification": classified.document_kind,
            "failure_mode": None,
        }
    except Exception as exc:  # noqa: BLE001 — benchmark records parser failure modes
        return {
            "fixture": str(fixture),
            "input_bytes": len(raw),
            "wall_clock_seconds": time.perf_counter() - started,
            "peak_rss_bytes": _peak_rss_bytes(),
            "rss_delta_bytes": max(0, _peak_rss_bytes() - rss_before),
            "failure_mode": type(exc).__name__,
            "failure_message": str(exc)[:240],
        }


async def _run(*, fixture: Path) -> dict[str, Any]:
    settings = get_settings()
    limit = int(settings.MAX_UPLOAD_SIZE)
    upload_sizes = sorted({1_024, 1_048_576, limit // 2, limit, limit + 1})
    return {
        "benchmark": "upload-extraction-ingestion-boundary",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "production_limits": {
            "max_upload_size_bytes": limit,
            "extraction_timeout_seconds": settings.DOCUMENT_EXTRACTION_TIMEOUT_SECONDS,
            "extraction_max_pages": settings.DOCUMENT_EXTRACTION_MAX_PAGES,
            "extraction_max_output_bytes": settings.DOCUMENT_EXTRACTION_MAX_OUTPUT_BYTES,
        },
        "upload_validation": [
            await _measure_upload(size, limit=limit) for size in upload_sizes
        ],
        "extraction_and_classification": await _measure_extraction(fixture, settings),
        "scope_note": "Measures the bounded upload and parser/classification stages at ingestion; DB/object-store/network/embedding/chunk-model work is excluded for deterministic offline benchmarking.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=Path("tools/lynk/testdata/contracts/minimal-valid.docx"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evals/benchmarks/upload_pipeline_latest.json"),
    )
    args = parser.parse_args()
    fixture = args.fixture.resolve()
    result = asyncio.run(_run(fixture=fixture))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
