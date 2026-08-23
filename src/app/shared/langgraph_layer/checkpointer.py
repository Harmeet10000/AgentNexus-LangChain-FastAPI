"""Async PostgreSQL checkpointer for LangGraph persistence.

The saver is psycopg-based, so it wants a libpq DSN and takes it from
``get_database_url(flavour="plain")``. Neither of the two strings closer to hand will do:
``settings.POSTGRES_URL`` carries no credential until the accessor injects one, and the
relational engine's own URL carries that engine's dialect alias, which psycopg cannot
parse. One is unauthenticated, the other unparseable. The plain flavour exists for this
consumer; nothing is repaired here — a Proof for this task greps this file for the string
operations that repair would need, which is why the alias is named rather than written out.

**The lifespan wiring stays commented, by decision (D17).** Ingestion runs in the queue
worker process, which never executes the application lifespan, so "build the saver once"
is a per-process requirement and never involved shared application state at all. The
shutdown path does call ``teardown_langgraph_checkpointer``, but behind a
``hasattr(app.state, "langgraph_checkpointer")`` guard that nothing currently satisfies —
so teardown is reachable only from a test until that decision changes. Read nothing here
as an invitation to re-enable the block.
"""

from __future__ import annotations

import contextlib
import re
from enum import StrEnum
from typing import TYPE_CHECKING
from urllib.parse import quote

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.conninfo import conninfo_to_dict
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.connections.postgres import get_database_url
from app.utils.logger import logger

if TYPE_CHECKING:
    # Type-only under `from __future__ import annotations`: every annotation below is a
    # string at runtime, so importing these eagerly is what ruff's TC003 objects to.
    from typing import Any, Final

#: One saver per process, and a process that may never checkpoint should not hold four
#: idle server connections against a managed instance. `min_size` is also what an awaited
#: `open(wait=True)` blocks for, so raising it lengthens startup for no gain here.
_POOL_MIN_SIZE: Final[int] = 1
_POOL_MAX_SIZE: Final[int] = 4
_POOL_OPEN_TIMEOUT_SECONDS: Final[float] = 10.0
_POOL_CLOSE_TIMEOUT_SECONDS: Final[float] = 5.0

# The saver's own connection-string helper connects with exactly these three settings,
# and each is load-bearing rather than stylistic:
#   * `row_factory=dict_row` -- `setup()` reads `row["v"]` off its migration query, which
#     a tuple row cannot answer;
#   * `autocommit=True` -- the saver issues DDL and checkpoint writes without opening an
#     explicit transaction and never commits one;
#   * `prepare_threshold=0` -- prepare on first use rather than after five, which is what
#     makes the repeated checkpoint write cheap.
# Building the pool here instead of borrowing the library's context manager means
# inheriting the obligation to pass them. A pool built without them yields a saver that
# imports and constructs happily and then fails inside `setup()`.
_CONNECTION_KWARGS: Final[dict[str, Any]] = {
    "autocommit": True,
    "prepare_threshold": 0,
    "row_factory": dict_row,
}

_REDACTED: Final[str] = "***"

#: The ``user:secret@`` authority of a URL-shaped substring. Both halves go: a username is
#: a credential too, and this project has already had one leak out of a connection URL
#: through a fixed-width slice that reached into userinfo.
_URL_USERINFO: Final[re.Pattern[str]] = re.compile(r"://[^/\s@]*:[^/\s@]*@")


def _scrub(text: str, dsn: str | None = None) -> str:
    """``text`` with anything derived from a connection credential removed.

    This exists because the project's log redaction is **key-based and message-blind**:
    the patcher in ``utils/logger.py`` blanks ``extra`` entries whose *key name* contains
    "password"/"token"/"secret", and never inspects a value or the log message. So an
    exception's own text passes through it entirely untouched — and a psycopg connection
    error can quote the DSN it failed on. A redaction mechanism that looks like it covers
    this case and does not is worse than no mechanism, because it stops anyone looking.

    The pattern substitution is the defence that depends on parsing nothing and so runs
    first. When ``dsn`` is supplied, its secret is additionally removed in **both**
    encodings. psycopg's own DSN parser is used rather than a URL parser for two reasons:
    a DSN may legally be keyword-value form rather than a URL, where a URL parser finds no
    credential at all and silently scrubs nothing; and the parser returns the secret
    already percent-decoded, while the DSN itself carries the encoded form the accessor
    wrote, so an error echoing either one has to be matched.
    """
    scrubbed: str = _URL_USERINFO.sub(f"://{_REDACTED}@", text)
    if dsn is None:
        return scrubbed
    try:
        secret: object = conninfo_to_dict(dsn).get("password")
    except psycopg.ProgrammingError:
        # A DSN psycopg itself cannot parse cannot be matched substring-wise either. The
        # pattern above has already run, which is the point of doing it first.
        return scrubbed
    if not isinstance(secret, str) or not secret:
        return scrubbed
    for form in {secret, quote(secret, safe="")}:
        scrubbed = re.sub(re.escape(form), _REDACTED, scrubbed)
    return scrubbed


async def setup_langgraph_checkpointer(conn_string: str | None = None) -> AsyncPostgresSaver:
    """Open a connection pool and return a saver that can read and write checkpoints.

    The previous version called the saver's connection-string classmethod and then
    ``.setup()`` on what came back. That classmethod is decorated
    ``@classmethod @asynccontextmanager``, so the call returned an *unentered async context
    manager* and ``.setup()`` on it was an ``AttributeError`` — uncaught, because the
    handler named ``(ConnectionError, TimeoutError, OSError)`` and ``psycopg.Error``
    derives from none of the three. Reading its source shows the deeper problem: it opens
    the connection with ``async with`` and *yields* the saver, so the connection is closed
    when the block exits. There is no way to get a long-lived saver out of it at all. The
    pool has to be owned here.

    (Both that classmethod's name and the old suppression are described rather than quoted
    throughout this module: two of this task's Proofs are greps for those literals, and
    prose containing them would defeat the guard.)

    Ownership follows the constructing process, which for ingestion is the queue worker.

    Args:
        conn_string: A libpq DSN. Defaults to ``get_database_url(flavour="plain")``, the
            only value production should pass. The parameter survives so a test can own
            the construction outright rather than patching module state.

    Returns:
        A saver whose migrations have run. Never ``None``: the version this replaces
        returned ``None`` down two separate paths, each behind a type-checker suppression
        naming an invalid return type. A suppression sitting on a return statement is the
        tell that the signature is lying about what the caller gets.

    Raises:
        psycopg.Error: the pool could not open, or the migrations in ``setup()`` failed.
            The pool is closed before this propagates.
    """
    dsn: str = get_database_url(flavour="plain") if conn_string is None else conn_string

    logger.info("Initialising LangGraph async checkpointer")

    pool: AsyncConnectionPool[psycopg.AsyncConnection[dict[str, Any]]] = AsyncConnectionPool(
        conninfo=dsn,
        kwargs=_CONNECTION_KWARGS,
        min_size=_POOL_MIN_SIZE,
        max_size=_POOL_MAX_SIZE,
        # `open=False` followed by an awaited `open()`. Letting the constructor open the
        # pool is deprecated from async code precisely because a constructor cannot await:
        # it would spawn the first connections from a worker thread outside this loop.
        open=False,
    )
    pool_handed_off: bool = False
    try:
        await pool.open(wait=True, timeout=_POOL_OPEN_TIMEOUT_SECONDS)
        saver: AsyncPostgresSaver = AsyncPostgresSaver(conn=pool)
        # `setup()` is neither optional nor free: it creates `checkpoint_migrations` and
        # applies every migration past the recorded version. This is the DDL that makes
        # every Proof in this area a unit test or a local scratch database, never the
        # managed instance.
        await saver.setup()
    except psycopg.Error as e:
        # `PoolTimeout` needs no clause of its own -- it derives from
        # `psycopg.OperationalError`. Naming both would imply they were siblings and
        # invite someone to "complete" the tuple with more of the same family.
        e.add_note(f"operation=setup_langgraph_checkpointer, pool_min_size={_POOL_MIN_SIZE}")
        logger.error(
            "Failed to initialise LangGraph checkpointer",
            error=_scrub(str(e), dsn),
            error_type=type(e).__name__,
        )
        raise
    else:
        pool_handed_off = True
        logger.info("LangGraph async checkpointer initialised")
        return saver
    finally:
        if not pool_handed_off:
            # Covers what the `except` clause cannot: an unexpected exception type, and
            # cancellation inside `open()`. A shutdown racing startup would otherwise
            # leave a pool holding live server connections with nothing referencing it.
            with contextlib.suppress(Exception):
                await pool.close(timeout=_POOL_CLOSE_TIMEOUT_SECONDS)


class CheckpointerTeardown(StrEnum):
    """Which outcome ``teardown_langgraph_checkpointer`` reached.

    Returned rather than only logged, so a caller can act on it and a test can assert on
    it without scraping log records. The version this replaces returned ``None`` for every
    outcome and logged only one of them, which made "closed the pool", "there was nothing
    to close", and "I was handed something I do not know how to close" indistinguishable
    both at the call site and in the log.
    """

    POOL_CLOSED = "pool_closed"
    NOT_PROVISIONED = "not_provisioned"
    NO_POOL_TO_CLOSE = "no_pool_to_close"
    #: A fourth outcome beyond the three the task named. Reporting a failed close as
    #: `NO_POOL_TO_CLOSE` would be exactly the conflation this change exists to remove:
    #: one says there was nothing to do, the other says the thing was attempted and did
    #: not work, and only the second is worth investigating.
    CLOSE_FAILED = "close_failed"


async def teardown_langgraph_checkpointer(
    checkpointer: AsyncPostgresSaver | None,
) -> CheckpointerTeardown:
    """Close the connection pool ``checkpointer`` holds, and report what happened.

    The guard this replaces tested for a ``pool`` attribute on the saver, which is false
    for every saver the library can build: ``AsyncPostgresSaver.__init__`` sets ``conn``,
    ``pipe``, ``lock``, ``loop``, and ``supports_pipeline``, and no such attribute exists
    on it or on either of its bases. So the pool went unclosed on every shutdown, and
    *silently*, because a false ``if`` fell straight through to a successful return. (The
    old guard is described rather than quoted: this task's first Proof is a grep for it.)

    ``conn`` is the attribute that actually holds it, and it is a union: the saver accepts
    either a single ``AsyncConnection`` or an ``AsyncConnectionPool``. Only the pool is
    ours to close — a bare connection belongs to whoever opened it — which is why that
    case reports rather than closing.
    """
    if checkpointer is None:
        logger.debug("No LangGraph checkpointer was provisioned; nothing to tear down")
        return CheckpointerTeardown.NOT_PROVISIONED

    held: object = getattr(checkpointer, "conn", None)
    if not isinstance(held, AsyncConnectionPool):
        logger.warning(
            "LangGraph checkpointer holds no connection pool; nothing was closed",
            held_type=type(held).__name__,
        )
        return CheckpointerTeardown.NO_POOL_TO_CLOSE

    try:
        await held.close(timeout=_POOL_CLOSE_TIMEOUT_SECONDS)
    except (psycopg.Error, OSError) as e:
        e.add_note("operation=teardown_langgraph_checkpointer")
        logger.warning(
            "Error closing LangGraph checkpointer pool",
            error=_scrub(str(e)),
            error_type=type(e).__name__,
        )
        return CheckpointerTeardown.CLOSE_FAILED
    else:
        logger.info("LangGraph checkpointer connection pool closed")
        return CheckpointerTeardown.POOL_CLOSED
