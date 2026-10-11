"""Wire-contract tripwires: round-trip models and the single dispatch path.

Every registered payload model must survive ``model_dump(mode="json")`` back
through ``model_validate`` unchanged — the broker carries the dumped form, so
anything lost there is lost for every consumer. And every production dispatch
must go through the validating helper in ``app.connections.celery_producer``:
a direct ``send_task`` / ``.delay()`` / ``apply_async`` anywhere else is a
second dispatch path that skips validation and routing.
"""

import re
from pathlib import Path

from app.shared.jobs.payloads import (
    AgentMemoryConsolidationPayload,
    CrawlerJobPayload,
    DocumentIngestPayload,
    NoKwargsPayload,
    PasswordResetEmailPayload,
    VerificationEmailPayload,
)
from app.shared.jobs.registry import CeleryTaskRegistry

_VALID_SAMPLES = {
    NoKwargsPayload: {},
    AgentMemoryConsolidationPayload: {"tenant_ids": ["tenant-1"], "session_ids": ["s-1"]},
    VerificationEmailPayload: {
        "user_id": "user-1",
        "email": "user@example.com",
        "token": "token-1",
        "idempotency_key": "key-1",
    },
    PasswordResetEmailPayload: {
        "user_id": "user-1",
        "email": "user@example.com",
        "token": "token-1",
        "idempotency_key": "key-1",
    },
    DocumentIngestPayload: {
        "document_id": "doc-1",
        "user_id": "user-1",
        "filename": "contract.pdf",
        "content_type": "application/pdf",
        "object_uri": "s3://documents/contract.pdf",
    },
    CrawlerJobPayload: {
        "crawl_id": "crawl-1",
        "request_payload": {"url": "https://example.com"},
    },
}

_DIRECT_DISPATCH = re.compile(r"\.send_task\(|\.delay\(|apply_async\(")

# Legacy validated sender (covered by test_typed_dispatch.py) and the single
# dispatch path itself. Everything else must route through them.
_DIRECT_DISPATCH_ALLOWLIST = frozenset(
    {
        "src/app/connections/celery_producer.py",
        "src/app/shared/jobs/registry.py",
    }
)


def test_every_registered_payload_model_has_a_round_trip_sample() -> None:
    """A registered model without a sample here is a contract nobody exercises."""
    missing = sorted(
        model.__name__
        for model in set(CeleryTaskRegistry._registry.values())
        if model not in _VALID_SAMPLES
    )

    assert missing == []


def test_every_registered_payload_round_trips_through_json_mode() -> None:
    """The dumped form the broker carries must validate back to the same model."""
    for model, sample in _VALID_SAMPLES.items():
        original = model.model_validate(sample)
        revived = model.model_validate(original.model_dump(mode="json"))

        assert revived == original


def test_no_production_dispatch_bypasses_the_validating_helper() -> None:
    """Direct broker sends outside the helper are a second, unvalidated path."""
    repo_root = Path(__file__).resolve().parents[3]
    offenders = sorted(
        str(path.relative_to(repo_root))
        for path in (repo_root / "src" / "app").rglob("*.py")
        if str(path.relative_to(repo_root)) not in _DIRECT_DISPATCH_ALLOWLIST
        and _DIRECT_DISPATCH.search(path.read_text(encoding="utf-8"))
    )

    assert offenders == []
