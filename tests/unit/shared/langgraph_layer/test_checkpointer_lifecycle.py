"""C1–C4: setup must hand back a usable saver, and teardown must say what it closed.

Every test here is a unit test over a construction it owns outright. That is not squeamishness:
`AsyncPostgresSaver.setup()` issues DDL, so pointing any of this at the managed instance would
be a schema change disguised as a test. The pool class is replaced by a recording subclass and
the saver's migration step is neutralised, which leaves the module's own decisions — which DSN
it asks for, which connection settings it passes on, what it closes, and what it says about it
— as the only things under test.

Two of these tests exist because a *type suppression* was hiding the defect rather than a
missing branch. The version this replaces was annotated as returning a saver and returned
`None` down two paths, each with the type checker silenced on the line. So "does it ever return
`None`" is asserted here rather than delegated to `ty`, which had already been told not to look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import psycopg
import pytest
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.connections import postgres as postgres_module
from app.shared.langgraph_layer import checkpointer as checkpointer_module
from app.shared.langgraph_layer.checkpointer import (
    CheckpointerTeardown,
    setup_langgraph_checkpointer,
    teardown_langgraph_checkpointer,
)
from app.utils.logger import logger

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any

pytestmark = pytest.mark.unit

#: A source URL with both halves of the flavour contract present: a parameter the plain
#: driver keeps (`sslmode`) and one it must drop (`channel_binding`). The secret is chosen to
#: percent-encode, because the accessor writes the encoded form into the URL while psycopg's
#: DSN parser hands back the decoded one — a scrubber that knows only one of the two leaks.
_SOURCE_URL = "postgresql://svc:s3cr3t%2Fpw@db.invalid.test:5432/appdb?sslmode=require&channel_binding=require"
_SECRET_DECODED = "s3cr3t/pw"
_SECRET_ENCODED = "s3cr3t%2Fpw"
_DIALECT_ALIAS = "postgresql+asyncpg"


class _Spy:
    """What the recording pool saw, and what it should do about it."""

    def __init__(self) -> None:
        self.conninfo: str | None = None
        self.connect_kwargs: dict[str, Any] = {}
        self.pool: Any = None
        self.open_error: BaseException | None = None
        self.setup_error: BaseException | None = None
        self.setup_calls = 0


def _recording_pool_class(spy: _Spy) -> type[AsyncConnectionPool]:
    """A real `AsyncConnectionPool` subclass that connects to nothing.

    A subclass rather than an unrelated double, because teardown decides what it is allowed
    to close with an `isinstance` check against the real class. A double that merely looked
    like a pool would be reported as "nothing to close" and the test would pass for the wrong
    reason. `__init__` deliberately does not call `super().__init__` — the base would open
    background workers and try to reach a server.
    """

    class _RecordingPool(AsyncConnectionPool):  # type: ignore[misc]
        def __init__(self, *, conninfo: str = "", kwargs: Any = None, **_ignored: Any) -> None:
            spy.conninfo = conninfo
            spy.connect_kwargs = dict(kwargs or {})
            spy.pool = self
            self.close_calls = 0
            self.opened = False

        async def open(self, wait: bool = False, timeout: float = 30.0) -> None:
            del wait, timeout
            if spy.open_error is not None:
                raise spy.open_error
            self.opened = True

        async def close(self, timeout: float = 5.0) -> None:
            del timeout
            self.close_calls += 1

    return _RecordingPool


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> _Spy:
    """Intercept pool construction and the migration step, leaving the saver class real.

    `AsyncPostgresSaver` itself is never replaced: one of C2's claims is that setup returns an
    instance of the *real* class, and a test that patched the class could not tell the
    difference between that and returning a stand-in.
    """
    spy = _Spy()
    monkeypatch.setattr(checkpointer_module, "AsyncConnectionPool", _recording_pool_class(spy))

    async def _fake_setup(_self: AsyncPostgresSaver) -> None:
        spy.setup_calls += 1
        if spy.setup_error is not None:
            raise spy.setup_error

    monkeypatch.setattr(AsyncPostgresSaver, "setup", _fake_setup)
    return spy


@pytest.fixture
def _controlled_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the accessor at `_SOURCE_URL`.

    The whole settings object is replaced rather than one field: `Settings` is declared frozen
    and slotted, so assigning to an attribute raises. `postgres.py` binds its own module-level
    `settings`, so that binding is the one to swap — patching the factory would not reach it.
    """
    monkeypatch.setattr(
        postgres_module,
        "settings",
        postgres_module.settings.model_copy(update={"POSTGRES_URL": _SOURCE_URL}),
    )


@pytest.fixture
def log_lines() -> Iterator[list[str]]:
    """Every log record emitted during the test, message and bound fields together.

    `caplog` does not see loguru, and — more to the point — the project's redaction patcher
    only inspects field *names*. Capturing the rendered `extra` is what makes a credential
    that slipped through as a *value* visible to an assertion.
    """
    lines: list[str] = []
    sink_id = logger.add(lines.append, level="DEBUG", format="{message} | {extra}")
    try:
        yield lines
    finally:
        logger.remove(sink_id)


# --------------------------------------------------------------------------------------
# C2 — setup yields a usable saver, or raises
# --------------------------------------------------------------------------------------


async def test_setup_returns_an_instance_of_the_real_saver_class(spy: _Spy) -> None:
    """The mandatory C2 claim.

    What this replaces returned an unentered async context manager, because the classmethod it
    called is decorated as one. `isinstance` against the real class is the assertion that
    distinguishes the two — a context manager wrapping the saver is not the saver.
    """
    saver = await setup_langgraph_checkpointer(_SOURCE_URL)

    assert isinstance(saver, AsyncPostgresSaver)
    assert saver.conn is spy.pool, "the saver was not given the pool this module opened"
    assert spy.setup_calls == 1, "the migration step did not run"


async def test_the_saver_has_no_pool_attribute(spy: _Spy) -> None:
    """The fact the old teardown guard was betting on, pinned as a regression test.

    Nothing in the library sets a `pool` attribute — the pool is held under `conn`, which is a
    union of pool and single connection. This is asserted rather than assumed because the whole
    silent-shutdown defect was one wrong attribute name, and a future library version adding
    the attribute should show up here as a decision to revisit, not as behaviour that quietly
    changes.
    """
    saver = await setup_langgraph_checkpointer(_SOURCE_URL)

    assert not hasattr(saver, "pool")
    assert isinstance(saver.conn, AsyncConnectionPool)


async def test_a_pool_that_cannot_open_raises_and_leaves_no_pool_behind(spy: _Spy) -> None:
    """A construction failure must raise, and must not leak the pool it half-built."""
    spy.open_error = psycopg.OperationalError("connection to server failed")

    with pytest.raises(psycopg.OperationalError):
        await setup_langgraph_checkpointer(_SOURCE_URL)

    assert spy.pool.close_calls == 1, "the pool was left open after a failed startup"


async def test_a_failed_migration_also_closes_the_pool(spy: _Spy) -> None:
    """The second failure path: the pool opened, the DDL did not land.

    Distinct from the test above because the cleanup happens after a *successful* open, which
    is the case a naive `try` around only the constructor would miss.
    """
    spy.setup_error = psycopg.errors.InsufficientPrivilege("permission denied for schema public")

    with pytest.raises(psycopg.Error):
        await setup_langgraph_checkpointer(_SOURCE_URL)

    assert spy.pool.opened is True
    assert spy.pool.close_calls == 1


async def test_the_connection_settings_the_saver_depends_on_are_passed_on(spy: _Spy) -> None:
    """Owning the pool means owning the three settings the library's own helper applied.

    Not one of C2's stated Proofs, but the defect its fix is most likely to introduce: a pool
    built without these constructs a saver that imports and initialises fine and then fails
    inside its migration query, because a tuple row cannot answer a lookup by column name.
    """
    await setup_langgraph_checkpointer(_SOURCE_URL)

    assert spy.connect_kwargs["row_factory"] is dict_row
    assert spy.connect_kwargs["autocommit"] is True
    assert spy.connect_kwargs["prepare_threshold"] == 0


# --------------------------------------------------------------------------------------
# C3 — the DSN comes from the accessor, and never reaches a log
# --------------------------------------------------------------------------------------


@pytest.mark.usefixtures("_controlled_settings")
async def test_the_default_dsn_is_the_plain_accessor_flavour(spy: _Spy) -> None:
    """No argument, no repair: the module asks the accessor for its own flavour."""
    await setup_langgraph_checkpointer()

    assert spy.conninfo == postgres_module.get_database_url(flavour="plain")


@pytest.mark.usefixtures("_controlled_settings")
async def test_the_dsn_is_parseable_authenticated_and_still_transport_secure(spy: _Spy) -> None:
    """C3's three assertions in one place, because they fail as a set.

    The two strings closer to hand each satisfy exactly one of them: the engine's URL is
    authenticated but carries a dialect alias this driver cannot parse, and the configured
    value is parseable but carries no credential. Only the accessor's plain flavour satisfies
    all three, so asserting them together is what proves the right one was consumed.
    """
    await setup_langgraph_checkpointer()
    dsn = spy.conninfo or ""

    assert _DIALECT_ALIAS not in dsn, "the dialect alias reached a driver that cannot parse it"
    assert _SECRET_ENCODED in dsn, "the DSN carries no credential"
    assert "sslmode=require" in dsn, "transport security was dropped"
    assert "channel_binding" not in dsn, "a parameter this driver rejects was kept"


@pytest.mark.usefixtures("_controlled_settings")
async def test_a_successful_setup_logs_no_credential(spy: _Spy, log_lines: list[str]) -> None:
    await setup_langgraph_checkpointer()

    assert log_lines, "nothing was logged, so this proves nothing"
    _assert_no_credential(log_lines)


@pytest.mark.usefixtures("_controlled_settings")
async def test_a_failed_setup_logs_no_credential_even_when_the_driver_quotes_the_dsn(
    spy: _Spy, log_lines: list[str]
) -> None:
    """The realistic leak, and the reason a value-blind redactor is not enough.

    psycopg reports a connection failure by quoting the connection info it failed on. That text
    arrives as the *value* of a field whose name is innocuous, so the project's key-matching
    redaction patcher passes it through untouched. Both encodings of the secret are planted in
    the error, since the DSN carries one and the driver's own parser returns the other.
    """
    spy.open_error = psycopg.OperationalError(
        f'connection failed: dsn="{postgres_module.get_database_url(flavour="plain")}" '
        f"password={_SECRET_DECODED}"
    )

    with pytest.raises(psycopg.OperationalError):
        await setup_langgraph_checkpointer()

    assert any("Failed to initialise" in line for line in log_lines), "the failure was not logged"
    _assert_no_credential(log_lines)


def _assert_no_credential(lines: list[str]) -> None:
    """Assert without ever putting the secret in a failure message.

    A plain `assert secret not in line` prints the offending line on failure, which would write
    the credential into the test output — turning a leak into a leak that is also in CI logs.
    So the line index is reported and the content is not.
    """
    for index, line in enumerate(lines):
        assert _SECRET_DECODED not in line, f"log line {index} carries the decoded credential"
        assert _SECRET_ENCODED not in line, f"log line {index} carries the encoded credential"
        assert "svc:" not in line, f"log line {index} carries the connection authority"


# --------------------------------------------------------------------------------------
# C4 — teardown reports which outcome it reached
# --------------------------------------------------------------------------------------


async def test_teardown_of_nothing_reports_that_nothing_was_provisioned() -> None:
    assert await teardown_langgraph_checkpointer(None) is CheckpointerTeardown.NOT_PROVISIONED


async def test_teardown_closes_a_pool_and_reports_the_close(spy: _Spy) -> None:
    saver = await setup_langgraph_checkpointer(_SOURCE_URL)

    outcome = await teardown_langgraph_checkpointer(saver)

    assert outcome is CheckpointerTeardown.POOL_CLOSED
    assert spy.pool.close_calls == 1


async def test_a_saver_holding_no_pool_reports_that_it_closed_nothing() -> None:
    """The outcome the old code was indistinguishable from success.

    A saver may legally hold a single connection instead of a pool — the constructor accepts
    either — and that connection belongs to whoever opened it. Reporting rather than closing is
    the correct behaviour; reporting *silently*, as a bare early return, is what made a pool
    that genuinely needed closing look like a clean shutdown.
    """

    class _NotAPool:
        pass

    saver = AsyncPostgresSaver(conn=_NotAPool())  # type: ignore[arg-type]

    outcome = await teardown_langgraph_checkpointer(saver)

    assert outcome is CheckpointerTeardown.NO_POOL_TO_CLOSE


async def test_a_close_that_fails_is_reported_as_a_failure_not_as_nothing_to_close(
    spy: _Spy,
) -> None:
    """A fourth outcome beyond the three the task named, and the reason it exists.

    Folding a failed close into "there was nothing to close" would rebuild the exact conflation
    this task removes: one of those says no action was needed, the other says an action was
    attempted and did not work, and only the second is worth waking anyone for.
    """
    saver = await setup_langgraph_checkpointer(_SOURCE_URL)

    async def _failing_close(timeout: float = 5.0) -> None:
        del timeout
        msg = "server closed the connection unexpectedly"
        raise psycopg.OperationalError(msg)

    spy.pool.close = _failing_close

    outcome = await teardown_langgraph_checkpointer(saver)

    assert outcome is CheckpointerTeardown.CLOSE_FAILED


async def test_teardown_never_logs_a_credential_when_a_close_fails(
    spy: _Spy, log_lines: list[str]
) -> None:
    """Teardown holds no DSN, so its scrub has only the pattern pass to work with.

    Worth its own test because that is the weaker of the two defences and the one a refactor
    would be tempted to drop as redundant.
    """
    saver = await setup_langgraph_checkpointer(_SOURCE_URL)

    async def _failing_close(timeout: float = 5.0) -> None:
        del timeout
        msg = f'closing: dsn="{_SOURCE_URL}"'
        raise psycopg.OperationalError(msg)

    spy.pool.close = _failing_close

    assert await teardown_langgraph_checkpointer(saver) is CheckpointerTeardown.CLOSE_FAILED
    _assert_no_credential(log_lines)
