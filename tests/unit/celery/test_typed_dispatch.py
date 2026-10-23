"""Registry validation, proven by invoking the validate-only seam directly.

Nothing here records or relays a durable outbound event. The outbox tables do
not exist yet, and even once they do, routing this check through them would
prove the relay rather than the guard: the registry is the seam where a payload
is either refused or handed to a job, so the registry is what gets invoked.

No broker is contacted either. Publication methods were removed from the
registry on purpose — a second send path is a second failure mode — and every
refusal happens before publication could ever occur. The one success case
asserts the validated payload AND that no send was attempted.

The real task application arrives through the ``real_celery`` fixture. Validating
against the suite-wide mock would prove nothing: a mock accepts every payload and
records every send, which is indistinguishable from the defect being tested for.
"""

import celery.exceptions
import pytest

from app.shared.jobs.names import DOCUMENTS_INGEST
from app.worker.celery.registration import TASK_DECLARING_MODULES

_VALID_INGEST_PAYLOAD = {
    "document_id": "doc-1",
    "user_id": "user-1",
    "filename": "contract.pdf",
    "content_type": "application/pdf",
    "object_uri": "s3://bucket/contract.pdf",
}


@pytest.fixture
def sends(real_celery, monkeypatch):
    """Replace the send with a spy: a leaked registry send must be visible."""
    recorded = []

    def _record(task_name, **kwargs):
        recorded.append((task_name, kwargs))
        return object()

    monkeypatch.setattr(real_celery.producer, "send_task", _record)
    return recorded


def test_unregistered_name_is_reported_as_a_failure_naming_the_task(real_celery, sends):
    """The tightening over the harvested contract, which let this through with a warning.

    A name nobody registered is a name no consumer answers to. Sending it anyway
    produced a well-formed message addressed to nobody, which Celery discards in
    silence — so the dispatch looked successful and the work simply never happened.
    """
    unknown = "tasks.no_such_task_is_declared_anywhere"

    with pytest.raises(real_celery.registry.UnregisteredTaskError) as excinfo:
        real_celery.registry.CeleryTaskRegistry.validate(unknown, {"anything": 1})

    assert unknown in str(excinfo.value)
    assert excinfo.value.task_name == unknown
    assert sends == []


def test_registered_name_with_a_missing_field_is_refused_at_validation(real_celery, sends):
    """A payload the consumer cannot accept must not reach a queue.

    This is the defect the harvested contract was written for: a task gains a
    required parameter, existing producers keep sending the old shape, and the
    mismatch surfaces in the worker once per retry with a traceback naming the
    worker rather than the producer.
    """
    with pytest.raises(real_celery.registry.TaskPayloadValidationError) as excinfo:
        real_celery.registry.CeleryTaskRegistry.validate(DOCUMENTS_INGEST, {"document_id": "doc-1"})

    assert DOCUMENTS_INGEST in str(excinfo.value)
    assert excinfo.value.task_name == DOCUMENTS_INGEST
    assert sends == []


def test_registered_name_with_an_unexpected_field_is_refused_at_validation(real_celery, sends):
    """An extra key is a producer/consumer disagreement too, so it is refused as well."""
    payload = {**_VALID_INGEST_PAYLOAD, "not_a_declared_argument": True}

    with pytest.raises(real_celery.registry.TaskPayloadValidationError) as excinfo:
        real_celery.registry.CeleryTaskRegistry.validate(DOCUMENTS_INGEST, payload)

    assert DOCUMENTS_INGEST in str(excinfo.value)
    assert sends == []


def test_a_matching_payload_validates_to_a_payload_instance(real_celery, sends):
    """The guard has to let correct payloads through, or it is just an outage."""
    validated = real_celery.registry.CeleryTaskRegistry.validate(
        DOCUMENTS_INGEST, dict(_VALID_INGEST_PAYLOAD)
    )

    assert validated.model_dump(mode="json") == _VALID_INGEST_PAYLOAD
    assert sends == []


@pytest.mark.parametrize("error_name", ["UnregisteredTaskError", "TaskPayloadValidationError"])
def test_both_refusals_are_celery_errors(real_celery, error_name):
    """Not cosmetic: the relay's publish path catches this base class.

    A refusal that is not a Celery error escapes that narrow catch into the
    relay's outer blanket handler, which logs a warning and drops the event —
    putting the invisibility back one layer up. As Celery errors they are recorded
    as failed events and retried toward the dead-letter table instead.
    """
    assert issubclass(
        getattr(real_celery.registry, error_name),
        celery.exceptions.CeleryError,
    )


def test_the_original_validation_detail_is_preserved(real_celery, sends):
    """Naming the task must not cost the per-field detail Pydantic produced."""
    with pytest.raises(real_celery.registry.TaskPayloadValidationError) as excinfo:
        real_celery.registry.CeleryTaskRegistry.validate(DOCUMENTS_INGEST, {})

    assert [error["loc"] for error in excinfo.value.validation_error.errors()]
    assert excinfo.value.__cause__ is excinfo.value.validation_error
    assert sends == []


def test_validation_does_not_import_declaring_modules(real_celery, monkeypatch, sends):
    import importlib

    def refuse_import(name, *args, **kwargs):
        pytest.fail(f"Validation must not load consumer module {name}")

    monkeypatch.setattr(importlib, "import_module", refuse_import)
    validated = real_celery.registry.CeleryTaskRegistry.validate(
        DOCUMENTS_INGEST, dict(_VALID_INGEST_PAYLOAD)
    )
    assert validated.model_dump(mode="json") == _VALID_INGEST_PAYLOAD
    assert sends == []


def test_a_name_with_no_declaring_module_is_refused_rather_than_searched_for(real_celery, sends):
    """An unmapped name cannot be rescued by an import, so it fails without one."""
    unmapped = "tasks.unmapped_and_unregistered"

    assert unmapped not in TASK_DECLARING_MODULES

    with pytest.raises(real_celery.registry.UnregisteredTaskError):
        real_celery.registry.CeleryTaskRegistry.validate(unmapped, {})

    assert sends == []
