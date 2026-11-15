"""F5 regression: registry refusal exceptions must survive the pool process boundary.

Root cause of the ingestion worker's ~2-minute death, captured live with a
shutdown-stack hook on a temporary systemd unit: a pool child rejected a
poisoned ``tasks.documents_ingest`` delivery (payload predating the current
catalog model) by raising ``TaskPayloadValidationError``. Celery pickles the
raised exception into the pool result pipe; the parent's event loop
(``asynpool._recv_message → load(bufv)``) unpickles it. Both registry refusals
called ``super().__init__(formatted_message)``, so ``BaseException.__reduce__``
reconstructed them with the wrong signature and the parent died on
``TypeError: TaskPayloadValidationError.__init__() missing 1 required
positional argument: 'validation_error'`` — an "Unrecoverable error" that
Celery answers with a full graceful worker shutdown (exit ~0, no traceback in
the journals). Every boot redelivered the poisoned message, so the worker died
~40 s after its consumer started, ~2 min after boot, deterministically.

The test simulates that trigger without waiting two minutes: raise the refusal
exactly as the delivery path does, pickle it the way billiard does
(``HIGHEST_PROTOCOL``), and load it the way the pool parent does.
"""

from __future__ import annotations

import pickle  # noqa: S403 — round-tripping the exception IS the mechanism under test
from pickle import HIGHEST_PROTOCOL  # noqa: S403

import pytest
from pydantic import ValidationError

from app.shared.jobs.names import DOCUMENTS_INGEST
from app.shared.jobs.registry import (
    CeleryTaskRegistry,
    TaskPayloadValidationError,
    UnregisteredTaskError,
)

pytestmark = pytest.mark.unit

_INVALID_INGEST_PAYLOAD: dict[str, object] = {
    "document_id": "doc-poisoned",
    "user_id": "user-1",
    # missing every other required field of the current catalog model
}


def _raise_payload_validation_error() -> TaskPayloadValidationError:
    with pytest.raises(TaskPayloadValidationError) as excinfo:
        CeleryTaskRegistry.validate(DOCUMENTS_INGEST, _INVALID_INGEST_PAYLOAD)
    return excinfo.value


def _raise_unregistered_error() -> UnregisteredTaskError:
    with pytest.raises(UnregisteredTaskError) as excinfo:
        CeleryTaskRegistry.validate("auth.refresh_token_reuse_detected", {})
    return excinfo.value


@pytest.mark.parametrize(
    ("raise_error", "label"),
    [
        (_raise_payload_validation_error, "TaskPayloadValidationError"),
        (_raise_unregistered_error, "UnregisteredTaskError"),
    ],
    ids=["payload-validation", "unregistered-task"],
)
def test_refusal_survives_pool_process_boundary(raise_error, label: str) -> None:
    """Child raises → billiard pickles → parent loads: no TypeError kills the pool."""
    raised = raise_error()

    # The child side: billiard serializes the failure with the highest protocol.
    wire = pickle.dumps(raised, protocol=HIGHEST_PROTOCOL)
    # The parent side: asynpool._recv_message loads the result message.
    rebuilt = pickle.loads(wire)  # noqa: S301 — the exact mechanism under test

    assert type(rebuilt) is type(raised)
    assert rebuilt.task_name == raised.task_name
    assert str(rebuilt) == str(raised)


def test_rebuilt_payload_validation_error_keeps_the_pydantic_cause() -> None:
    raised = _raise_payload_validation_error()

    rebuilt = pickle.loads(pickle.dumps(raised, protocol=HIGHEST_PROTOCOL))  # noqa: S301

    assert isinstance(rebuilt.validation_error, ValidationError)
