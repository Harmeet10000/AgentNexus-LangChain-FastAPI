"""G6 regression: the dead-letter topology must exist at the broker, not only in code.

The 2026-11-15 sweep found no ``tasks.dlx`` exchange, no ``default.dlq``
queue, and no binding on the live RabbitMQ — the declaration existed only as
``x-dead-letter-*`` arguments inside the producer's ``task_queues`` entry, and
nothing ever declared them.  A ``Reject(requeue=False)`` poison message was
therefore silently discarded by the broker: no retention, no forensics.

**No broker is opened here.**  The declaration is pure configuration driven
through kombu's channel contract, so a recording channel proves the payload,
and the real application object (via the ``real_celery`` fixture) proves the
declared binding matches what every work queue publishes into its
dead-letter route.
"""

from __future__ import annotations

import weakref

import pytest

from app.config import get_settings
from app.shared.jobs.topology import (
    DLQ_MAX_LENGTH,
    DLQ_MESSAGE_TTL_MS,
    dead_letter_queue,
    declare_dead_letter_topology,
)

pytestmark = pytest.mark.unit

_settings = get_settings()


class _RecordingChannel:
    """Duck-typed kombu channel capturing the declaration payload.

    kombu calls its channel contract with keyword arguments; the recorder
    keeps the merged call so assertions stay readable either way.
    """

    def __init__(self) -> None:
        self.exchange_declares: list[dict] = []
        self.queue_declares: list[dict] = []
        self.queue_binds: list[dict] = []

    def prepare_queue_arguments(self, arguments: dict, **_: object) -> dict:
        return arguments

    def exchange_declare(self, *args: object, **kwargs: object) -> None:
        self.exchange_declares.append({**dict(enumerate(args)), **kwargs})

    def queue_declare(self, *args: object, **kwargs: object) -> None:
        self.queue_declares.append({**dict(enumerate(args)), **kwargs})

    def queue_bind(self, *args: object, **kwargs: object) -> None:
        self.queue_binds.append({**dict(enumerate(args)), **kwargs})

    def __enter__(self) -> _RecordingChannel:
        return self

    def __exit__(self, *args: object) -> None:
        return None


class _RecordingConnection:
    def __init__(self, channel: _RecordingChannel) -> None:
        self._channel = channel
        self.opened = 0

    def channel(self) -> _RecordingChannel:
        self.opened += 1
        return self._channel

    def __enter__(self) -> _RecordingConnection:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def test_declaration_declares_exchange_queue_and_binding() -> None:
    """The payload must be exactly the route poisons travel: DLX -> binding -> DLQ."""
    channel = _RecordingChannel()
    connection = _RecordingConnection(channel)

    declare_dead_letter_topology(connection)  # type: ignore[arg-type]

    assert channel.exchange_declares  # the DLX must exist; kombu may declare it
    # both from the queue's binding exchange and explicitly.
    exchange_call = channel.exchange_declares[0]
    assert exchange_call.get("exchange") == _settings.CELERY_DEAD_LETTER_EXCHANGE
    assert exchange_call.get("type") == "direct"
    assert exchange_call.get("durable") is True
    assert all(
        call.get("exchange") == _settings.CELERY_DEAD_LETTER_EXCHANGE
        for call in channel.exchange_declares
    )

    assert len(channel.queue_declares) == 1
    queue_call = channel.queue_declares[0]
    assert queue_call.get("queue") == _settings.CELERY_DEAD_LETTER_QUEUE
    assert queue_call.get("durable") is True
    arguments = queue_call.get("arguments") or {}
    assert arguments["x-queue-type"] == "classic"
    assert arguments["x-max-length"] == DLQ_MAX_LENGTH
    assert arguments["x-overflow"] == "reject-publish"
    assert arguments["x-message-ttl"] == DLQ_MESSAGE_TTL_MS

    assert len(channel.queue_binds) == 1
    bind_call = channel.queue_binds[0]
    assert bind_call.get("queue") == _settings.CELERY_DEAD_LETTER_QUEUE
    assert bind_call.get("exchange") == _settings.CELERY_DEAD_LETTER_EXCHANGE
    assert bind_call.get("routing_key") == _settings.CELERY_DEAD_LETTER_ROUTING_KEY


def test_the_dlq_cannot_grow_unbounded() -> None:
    """Forensics parking is capped and expiring."""
    queue = dead_letter_queue()

    assert queue.durable
    assert queue.queue_arguments["x-max-length"] > 0
    assert queue.queue_arguments["x-message-ttl"] > 0
    assert queue.queue_arguments["x-overflow"] == "reject-publish"


def test_declared_binding_matches_what_the_producer_publishes(real_celery) -> None:
    """The terminal route must agree with the work queues' dead-letter args.

    Every work queue dead-letters into ``x-dead-letter-exchange`` with
    ``x-dead-letter-routing-key``; the DLQ's binding is the consumer side of
    that route.  The two sides have no mechanical link in code — this is the
    drift guard: a renamed exchange or routing key in either place drops
    poisons on the broker floor again.
    """
    from app.shared.jobs.topology import TASK_DLX_EXCHANGE

    queues = real_celery.app.amqp.queues
    dlq = queues[_settings.CELERY_DEAD_LETTER_QUEUE]
    work_queues = [
        queue for name, queue in queues.items() if name != _settings.CELERY_DEAD_LETTER_QUEUE
    ]

    assert work_queues, "the work queues must be declared for this comparison"
    assert dlq.exchange.name == TASK_DLX_EXCHANGE.name == _settings.CELERY_DEAD_LETTER_EXCHANGE
    assert dlq.routing_key == _settings.CELERY_DEAD_LETTER_ROUTING_KEY

    for work_queue in work_queues:
        dead_letter_args = work_queue.queue_arguments
        assert dead_letter_args["x-dead-letter-exchange"] == dlq.exchange.name
        assert dead_letter_args["x-dead-letter-routing-key"] == dlq.routing_key


def test_producer_declares_before_first_publish_and_only_once() -> None:
    """The declaration is per-process, idempotent, and failure-tolerant."""
    from app.connections.celery_producer import _dead_letter_state, ensure_dead_letter_declared

    channel = _RecordingChannel()
    connection = _RecordingConnection(channel)

    class _StubCelery:
        def broker_connection(self) -> _RecordingConnection:
            return connection  # type: ignore[return-value]

    previous = _dead_letter_state["declared"]
    _dead_letter_state["declared"] = False
    try:
        ensure_dead_letter_declared(_StubCelery())  # type: ignore[arg-type]
        ensure_dead_letter_declared(_StubCelery())  # type: ignore[arg-type]

        assert connection.opened == 1
        assert channel.queue_declares, "the DLQ must be declared before publishing"
    finally:
        _dead_letter_state["declared"] = previous


def test_worker_start_declares_the_topology() -> None:
    """Workers reject poisons too, so the worker-start signal must declare.

    ``celeryd_after_setup`` fires only when a worker actually starts — never
    at ``Worker`` construction (which unit tests perform) — so the
    declaration cannot dial the live broker from a test process.
    """
    from celery.signals import celeryd_after_setup

    from app.worker.celery.signals import declare_broker_topology

    receivers = [
        receiver() if isinstance(receiver, weakref.ref) else receiver
        for _, receiver in celeryd_after_setup.receivers
    ]
    assert declare_broker_topology in receivers


def test_a_failing_declaration_never_blocks_publishing() -> None:
    """The declaration is best-effort: work queues already exist on the broker.

    Any failure shape leaves the flag unset so the next publish retries, and
    the publish itself proceeds — the poison path is a safety net, not the
    delivery path.
    """
    from kombu.exceptions import OperationalError

    from app.connections.celery_producer import _dead_letter_state, ensure_dead_letter_declared

    broker_error = OperationalError("broker down")

    class _BrokenBroker:
        def broker_connection(self) -> object:
            raise broker_error

    previous = _dead_letter_state["declared"]
    _dead_letter_state["declared"] = False
    try:
        ensure_dead_letter_declared(_BrokenBroker())  # type: ignore[arg-type]

        assert _dead_letter_state["declared"] is False
    finally:
        _dead_letter_state["declared"] = previous
