"""C7: ingestion has its own queue, its own worker, and cannot starve the short work.

The decision this pins was an open question in the change's design, deliberately left unanswered
because it has an operational cost and no locked decision covered it: **does ingestion get its own
queue, or share the default one?** It was answered — a dedicated queue with its own concurrency and
its own worker service — and the reason is the thing these tests exist to keep true. Document
ingestion is minutes of model work per message. The default queue carries sub-second billing and
transactional-email tasks. One shared worker pool makes those wait behind ingestion whenever every
slot is busy, and `worker_prefetch_multiplier=1` does **not** prevent it: prefetch stops one worker
hoarding messages off the broker, and says nothing about head-of-line blocking once every slot is
already occupied. Disjoint queues with disjoint consumers is what removes the coupling, and it is a
property of two files agreeing — the routing table and the deployment — which is exactly the kind of
agreement that rots silently.

**No broker is opened here, and no worker is started.** Both facts are in-process: queue declaration
and route resolution are pure configuration, and the deployment's queue selection is text in
`docker-compose.yml`. That is not only cheaper, it is required — the configured broker is a live
managed instance carrying real billing, credit and password-reset work, so a consuming worker
started against it could execute it.

**Why the deployment is read from compose rather than from the `Makefile`.** Compose is what actually
runs. `test_documented_worker_command.py` separately proves compose, the `Makefile` and the README
hold the same strings, so reading compose here loses nothing and keeps this module's subject — what
the deployed processes consume — literal. Every parse in this file is guarded by a count assertion,
because a regex that silently stops matching turns every assertion below into a comparison of two
empty sets.
"""

import re
import shlex
from pathlib import Path

import pytest
from celery.exceptions import QueueNotFound

from app.config import get_settings
from app.connections.celery_task_names import (
    DOCUMENTS_INGEST,
    INGESTION_TASK_NAMES,
    SEND_PASSWORD_RESET_EMAIL,
    TASK_DECLARING_MODULES,
)

pytestmark = pytest.mark.unit

#: Queue names are read from the project's settings rather than from `celery_app.conf`. Both hold
#: them, but a missing key on Celery's own `Settings` raises from inside the library, and pytest
#: renders that frame's locals — which include the broker URL, credentials and all. A project
#: settings object cannot leak that: its secret fields are `SecretStr`.
_settings = get_settings()

_COMPOSE = Path(__file__).resolve().parents[3] / "docker-compose.yml"

_COMPOSE_CELERY_COMMAND = re.compile(
    r"^\s*command:\s*(?P<command>uv run celery .*?)\s*$", re.MULTILINE
)

#: One per worker, one for the scheduler. Asserted rather than assumed, so that a service deleted by
#: accident shows up as a failure here instead of as a smaller set that still satisfies everything.
_EXPECTED_CELERY_SERVICES = 3
_EXPECTED_WORKER_SERVICES = 2


def _compose_celery_commands() -> list[str]:
    commands = [m["command"] for m in _COMPOSE_CELERY_COMMAND.finditer(_COMPOSE.read_text("utf-8"))]

    assert len(commands) == _EXPECTED_CELERY_SERVICES, (
        f"found {len(commands)} celery commands in {_COMPOSE.name}, expected "
        f"{_EXPECTED_CELERY_SERVICES}; if the services were renamed or rewritten as a list, this "
        "module's parse needs updating rather than its assertions relaxing"
    )
    return commands


def _selected_queues(command: str) -> frozenset[str]:
    """The queues one command tells its worker to consume, or nothing for the scheduler."""
    argv = shlex.split(command)
    if "-Q" not in argv:
        return frozenset()
    return frozenset(argv[argv.index("-Q") + 1].split(","))


def _worker_queue_selections() -> list[frozenset[str]]:
    selections = [
        selected
        for selected in (_selected_queues(command) for command in _compose_celery_commands())
        if selected
    ]

    assert len(selections) == _EXPECTED_WORKER_SERVICES, (
        f"{len(selections)} of the deployed celery commands select queues, expected "
        f"{_EXPECTED_WORKER_SERVICES}"
    )
    return selections


@pytest.fixture(scope="module")
def routed_queue(real_celery):
    """Resolve a task name to the queue the application would publish it to.

    Goes through `amqp.router`, which is the same object the publishing path uses, rather than
    reading `task_routes` directly — the route table is an input to routing, not the outcome of it,
    and the difference is where a default fallthrough would hide.
    """

    def resolve(task_name: str) -> str:
        return real_celery.app.amqp.router.route({}, task_name)["queue"].name

    return resolve


# --------------------------------------------------------------------------------------
# The queue exists, because it cannot come into existence by accident
# --------------------------------------------------------------------------------------


def test_the_ingestion_queue_is_declared(real_celery, routed_queue) -> None:
    """`task_create_missing_queues=False` means an undeclared queue is a publish-time failure.

    That setting is why this is worth its own test: the queue set is closed, so routing a task to a
    name that is not in `task_queues` raises rather than quietly creating a queue nothing consumes.
    The check therefore runs through the router — if the declaration were missing, resolving the
    route would raise before any assertion here ran.
    """
    assert routed_queue(DOCUMENTS_INGEST) in real_celery.app.amqp.queues


def test_the_ingestion_queue_matches_the_shape_of_the_queue_it_sits_beside(
    real_celery, routed_queue
) -> None:
    """Quorum, durable, and dead-lettering — the same three properties as the default queue.

    Compared against the default queue's own arguments rather than against literals, so the two
    cannot drift apart: whichever way the project's queue conventions move, both queues move
    together or this fails. Dead-lettering is the one that matters most here. Without it a rejected
    or delivery-limited ingestion message is dropped instead of parked, which is indistinguishable
    from a document that was silently never processed — the exact failure shape this change exists
    to remove.
    """
    queues = real_celery.app.amqp.queues
    ingestion = queues[routed_queue(DOCUMENTS_INGEST)]
    default = queues[routed_queue(SEND_PASSWORD_RESET_EMAIL)]

    assert ingestion.durable
    assert ingestion.queue_arguments == default.queue_arguments
    assert ingestion.exchange.name == default.exchange.name
    assert ingestion.routing_key != default.routing_key


# --------------------------------------------------------------------------------------
# Routing is explicit for every dispatchable name
# --------------------------------------------------------------------------------------


def test_every_dispatchable_name_resolves_through_an_explicit_route(real_celery) -> None:
    """No name may reach its queue through the default fallthrough.

    `lookup_route` returning `None` is the discriminator, and it is the only one available: a name
    that matches no route still *arrives* on the default queue, via `task_default_queue`, so
    resolving the queue cannot tell the two apart. Before this change 11 of the 16 declared names
    took that path — every `auth.*`, `billing.*`, `credits.*` and `document_extraction.*` name —
    because the single route was a `tasks.*` glob that none of them match. Two mechanisms delivering
    to one queue read as one mechanism right up until a name needs a different queue.
    """
    router = real_celery.app.amqp.router

    unrouted = sorted(name for name in TASK_DECLARING_MODULES if router.lookup_route(name) is None)

    assert unrouted == []


@pytest.mark.parametrize("task_name", sorted(INGESTION_TASK_NAMES))
def test_the_ingest_names_route_to_the_ingestion_queue(task_name, routed_queue) -> None:
    """Anchored on the configured queue, not on the other ingest names.

    The first version of this compared each name to `routed_queue(DOCUMENTS_INGEST)` — the ingest
    names to one another — and a mutation caught it: routing *every* task to the default queue left
    the three still equal, so the test passed while the ingestion queue had no producers at all.
    Naming the configured queue removes that degree of freedom, and the inequality states the
    premise the whole split rests on, which the comparison silently assumed.
    """
    assert _settings.CELERY_INGESTION_QUEUE != _settings.CELERY_DEFAULT_QUEUE
    assert routed_queue(task_name) == _settings.CELERY_INGESTION_QUEUE


def test_the_latency_sensitive_names_route_to_the_default_queue(routed_queue) -> None:
    """The other half of the split, asserted positively so it cannot pass by everything moving.

    Without this, routing *every* task to the ingestion queue would satisfy the test above and the
    disjointness test below, and the starvation the split exists to prevent would be back with the
    queue names swapped.
    """
    misrouted = {
        name: routed_queue(name)
        for name in TASK_DECLARING_MODULES
        if name not in INGESTION_TASK_NAMES and routed_queue(name) != _settings.CELERY_DEFAULT_QUEUE
    }

    assert misrouted == {}


def test_routing_to_an_undeclared_queue_is_refused(real_celery) -> None:
    """The positive control for the closed queue set.

    Every assertion above is about a route resolving to the right queue; this is the one that shows
    resolution can fail at all. Without it, a router that returned a queue for anything would satisfy
    the whole module.
    """
    with pytest.raises(QueueNotFound):
        real_celery.app.amqp.router.expand_destination({"queue": "queue-that-is-not-declared"})


# --------------------------------------------------------------------------------------
# The deployment consumes exactly what the routing produces
# --------------------------------------------------------------------------------------


def test_the_deployed_workers_consume_exactly_the_queues_tasks_are_routed_to(routed_queue) -> None:
    """The drift guard between two files that have no mechanical link.

    The routing table takes its queue names from settings; the deployment writes them as literals in
    a `-Q` flag, because a compose command cannot read the application's settings. So the two agree
    by hand, and both failure directions are silent. A routed queue nobody consumes accepts messages
    forever and nothing runs them — which is the state this whole change began from. A consumed queue
    nothing routes to gives a worker that reports itself healthy, logs nothing, and processes
    nothing. Equality catches both, including a one-character typo in either place.
    """
    routed = {routed_queue(name) for name in TASK_DECLARING_MODULES}
    consumed = frozenset().union(*_worker_queue_selections())

    assert consumed == routed


def test_no_deployed_worker_consumes_the_dead_letter_queue(real_celery) -> None:
    """The reason every worker command carries `-Q` at all.

    Measured, not assumed: a worker started **without** `-Q` consumes every queue the application
    declares, and that includes the dead-letter queue — asserted directly below so this test cannot
    be read as defending against a hazard that does not exist. Such a worker re-runs precisely the
    messages that were parked for a human to look at, and it looks perfectly healthy while doing it.
    The dead-letter queue is a parking space, not a second inbox.
    """
    queues = real_celery.app.amqp.queues
    dead_letter_queue = _settings.CELERY_DEAD_LETTER_QUEUE

    # The hazard is real: with no selection, Celery consumes the whole declared set.
    assert set(queues.consume_from) == set(queues)
    assert dead_letter_queue in queues

    for selected in _worker_queue_selections():
        assert dead_letter_queue not in selected


def test_the_two_worker_pools_share_no_queue(routed_queue) -> None:
    """The starvation property itself, stated as the structural fact that guarantees it.

    A latency check — dispatch long work, then short work, and watch the short work start — needs a
    broker and two consuming workers, and the configured broker is a live managed instance carrying
    real billing and password-reset work. The property that check would observe is this one: two
    consumer sets over disjoint queues cannot head-of-line block each other, because neither pool's
    slots are ever occupied by the other pool's messages. There is no ordering, timing or load under
    which they can, so this holds strictly more firmly than one measurement of it would.
    """
    selections = _worker_queue_selections()
    ingestion_queue = routed_queue(DOCUMENTS_INGEST)
    latency_sensitive_queue = routed_queue(SEND_PASSWORD_RESET_EMAIL)

    assert ingestion_queue != latency_sensitive_queue

    first, second = selections
    assert not (first & second), f"the two worker pools both consume {sorted(first & second)}"

    consumes_ingestion = [selected for selected in selections if ingestion_queue in selected]
    assert len(consumes_ingestion) == 1
    assert latency_sensitive_queue not in consumes_ingestion[0]
