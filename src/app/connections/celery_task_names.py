"""Single definition site for every dispatchable Celery task name.

Each task name used to be written twice: once as a string literal in the
``@celery_app.task(name=...)`` declaration, and again as a string literal on the
dispatching side — an outbox ``event_type``, or a ``beat_schedule`` entry. Two
literals for one wire contract is a rename waiting to go wrong. Rename the
declaration and the producer keeps dispatching a name no consumer answers to;
and because an unregistered name used to be waved through rather than reported,
the only symptom was work that quietly never happened. Every name therefore
lives here exactly once and both sides import it, so a rename is a single edit
the type checker follows.

``TASK_DECLARING_MODULES`` records which module declares each name. It has two
jobs beyond documentation:

* the typed dispatch helper uses it to import the one module that declares the
  name it was handed. Without that, a process which never imported the task
  package — the API process, where the outbox relay lives — holds an empty
  payload registry, and every dispatch it made was validated against nothing at
  all. That was measured, not assumed: nothing under ``src/`` imports the task
  package, so the harvested typed-dispatch contract was inert in exactly the
  process that used it.
* it lets a unit test assert that every declaring module is named explicitly in
  the task application's ``include`` list, which is what stops registration from
  sliding back into an import side effect of the task package's initialiser.

The tempting shortcut is to drop the mapping and have the dispatch helper call
``celery_app.loader.import_default_modules()`` — Celery's own mechanism, and it
needs no bookkeeping. Measured, that imports *every* listed module, and the
ingestion modules pull the document-converter stack behind them: fourteen
seconds on a cold interpreter. Paying that inside the first dispatch of a live
API process is a worse trade than keeping this mapping honest, and the test
above is what keeps it honest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Final

# --- ingestion ---------------------------------------------------------------

DOCUMENTS_INGEST: Final = "tasks.documents_ingest"
SEARCH_INGEST: Final = "tasks.search_ingest"
PAGEINDEX_INGEST: Final = "tasks.pageindex_ingest"
LEGAL_BATCH_EXTRACTION: Final = "document_extraction.legal_batch"

#: The names whose work is measured in minutes, and which therefore consume the
#: dedicated ingestion queue rather than the default one. Membership lives here,
#: beside the names, so the routing table and the worker's queue can be derived
#: from one list instead of agreeing by hand.
#:
#: ``LEGAL_BATCH_EXTRACTION`` is deliberately absent even though it sits in this
#: section and also runs minutes of model work per message. Which names share a
#: queue is an operational decision with a cost — a third queue needs a third
#: consumer or it silently accumulates — and the answer that was given covered
#: the three ingest names. Moving it is a decision to be asked for, not inferred
#: from the fact that it looks similar.
#:
#: TODO(queues): give ``LEGAL_BATCH_EXTRACTION`` its own queue and consumer.
#: Deferred 2026-08-23 as a scope decision, not a technical blocker. It is
#: minutes-long model work (``src/tasks/document_extraction_tasks.py:31``) still
#: routed to ``default``, where one message head-of-line blocks every short task
#: behind it — ``worker_prefetch_multiplier=1`` stops a worker *hoarding*
#: messages but cannot stop the message already in its hands from occupying the
#: slot. Do NOT simply add the name to the frozenset above: that shares the
#: ingestion queue and reintroduces the same blocking between the two workloads.
#: The full change is six files —
#:   1. ``app/config/settings.py`` — new ``CELERY_EXTRACTION_QUEUE``, mirroring
#:      ``CELERY_INGESTION_QUEUE`` (same default-plus-env shape).
#:   2. ``app/connections/celery.py`` — a fourth ``Queue(...)`` with
#:      ``x-queue-type: quorum`` and the existing DLX, and ``_task_routes()``
#:      must stop being a two-way ternary (``ingestion_route if name in
#:      INGESTION_TASK_NAMES else default_route``). Three buckets need an
#:      explicit name→route mapping; extending the ternary is where this
#:      silently goes wrong.
#:   3. ``Makefile`` and 4. ``docker-compose.yml`` — a third ``worker -Q`` entry.
#:      A *declared* queue with no consumer accumulates forever and raises no
#:      error, and this is invisible locally because a worker started with no
#:      ``-Q`` consumes every declared queue including ``default.dlq``.
#:   5. ``tests/unit/celery/test_documented_worker_command.py`` — it asserts the
#:      documented command covers every declared queue, so it fails first and is
#:      the tripwire for forgetting 3/4.
#:   6. ``src/app/examples/CELERY.md`` — the queue table and the prose at :68
#:      that describes this frozenset as *the* minutes-long set.
#: Acceptance is behavioural, not structural: with a ``legal_batch`` message in
#: flight, a task dispatched to ``default`` must still start. Asserting the
#: fourth queue exists only proves it was declared.
INGESTION_TASK_NAMES: Final[frozenset[str]] = frozenset(
    {DOCUMENTS_INGEST, SEARCH_INGEST, PAGEINDEX_INGEST}
)

# --- transactional email -----------------------------------------------------

SEND_VERIFICATION_EMAIL: Final = "auth.send_verification_email"
SEND_PASSWORD_RESET_EMAIL: Final = "auth.send_password_reset_email"  # noqa: S105 — task name, not a password

# --- billing (scheduler-dispatched) ------------------------------------------

BILLING_RENEWAL: Final = "billing.renewal"
BILLING_DUNNING: Final = "billing.dunning"
BILLING_INVOICE_GENERATION: Final = "billing.invoice_generation"
BILLING_RECEIPT_GENERATION: Final = "billing.receipt_generation"
BILLING_PAUSE_RESUME: Final = "billing.pause_resume"
BILLING_RECONCILIATION: Final = "billing.reconciliation"

# --- credits (scheduler-dispatched) ------------------------------------------

CREDITS_EXPIRE: Final = "credits.expire"
CREDITS_RECONCILE: Final = "credits.reconcile"

# --- reference implementations -----------------------------------------------

EXAMPLE_ADD: Final = "tasks.add"
EXAMPLE_PROCESS_DOCUMENT: Final = "tasks.process_document"


# Module paths are private to this file: the authoritative copy for the worker is
# the task application's ``include`` list, and a unit test asserts the two agree
# rather than deriving one from the other. Deriving would make the ``include``
# list unreadable at the one place an operator looks for it.
_AUTH_EMAIL_TASKS: Final = "tasks.auth_email_tasks"
_BILLING_TASKS: Final = "tasks.billing_tasks"
_CREDIT_TASKS: Final = "tasks.credit_tasks"
_DOCUMENT_EXTRACTION_TASKS: Final = "tasks.document_extraction_tasks"
_DOCUMENT_TASKS: Final = "tasks.document_tasks"
_EXAMPLE_TASKS: Final = "tasks.example"
_PAGEINDEX_TASKS: Final = "tasks.pageindex_tasks"
_SEARCH_TASKS: Final = "tasks.search_tasks"


TASK_DECLARING_MODULES: Final[Mapping[str, str]] = {
    DOCUMENTS_INGEST: _DOCUMENT_TASKS,
    SEARCH_INGEST: _SEARCH_TASKS,
    PAGEINDEX_INGEST: _PAGEINDEX_TASKS,
    LEGAL_BATCH_EXTRACTION: _DOCUMENT_EXTRACTION_TASKS,
    SEND_VERIFICATION_EMAIL: _AUTH_EMAIL_TASKS,
    SEND_PASSWORD_RESET_EMAIL: _AUTH_EMAIL_TASKS,
    BILLING_RENEWAL: _BILLING_TASKS,
    BILLING_DUNNING: _BILLING_TASKS,
    BILLING_INVOICE_GENERATION: _BILLING_TASKS,
    BILLING_RECEIPT_GENERATION: _BILLING_TASKS,
    BILLING_PAUSE_RESUME: _BILLING_TASKS,
    BILLING_RECONCILIATION: _BILLING_TASKS,
    CREDITS_EXPIRE: _CREDIT_TASKS,
    CREDITS_RECONCILE: _CREDIT_TASKS,
    EXAMPLE_ADD: _EXAMPLE_TASKS,
    EXAMPLE_PROCESS_DOCUMENT: _EXAMPLE_TASKS,
}
