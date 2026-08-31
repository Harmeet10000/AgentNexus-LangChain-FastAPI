"""Neon Postgres database configuration with SQLAlchemy."""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Final, Literal
from urllib.parse import (
    SplitResult,
    parse_qsl,
    quote,
    unquote,
    urlencode,
    urlsplit,
    urlunsplit,
)

from fastapi.requests import HTTPConnection
from pydantic import BaseModel, ConfigDict, SecretStr
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.config.settings import Settings
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.engine.cursor import CursorResult
    from ty_extensions import Unknown

settings: Settings = get_settings()

type DatabaseUrlFlavour = Literal["async", "plain"]
"""Which driver the URL is built for.

``"async"`` serves the application's SQLAlchemy pool; ``"plain"`` serves low-level
drivers that speak libpq DSNs — `asyncpg.connect` and psycopg.
"""

_POSTGRES_SCHEMES: Final[frozenset[str]] = frozenset({"postgres", "postgresql"})
_ASYNC_SCHEME: Final[str] = "postgresql+asyncpg"
_PLAIN_SCHEME: Final[str] = "postgresql"

# Connection parameters each driver cannot accept as URL query arguments.
# SQLAlchemy's asyncpg dialect forwards leftover query arguments to `asyncpg.connect`
# as keyword arguments, so both must go. A plain DSN goes to asyncpg (or libpq)
# directly, and both parse `sslmode` natively -- so the plain flavour keeps transport
# security and drops only `channel_binding`, which neither understands and which
# asyncpg would otherwise forward as a server setting.
_ASYNC_DROPPED_PARAMS: Final[frozenset[str]] = frozenset({"sslmode", "channel_binding"})
_PLAIN_DROPPED_PARAMS: Final[frozenset[str]] = frozenset({"channel_binding"})


class DatabaseConnectionFields(BaseModel):
    """Discrete connection fields for components that accept no connection string.

    Derived from `get_database_url` rather than from the individual settings fields, so
    such a component cannot be pointed at a different database than the application's
    own pool while still holding a valid credential.
    """

    model_config = ConfigDict(frozen=True)

    host: str
    port: int
    username: str
    password: SecretStr
    database: str


def _unconfigured_password() -> str:
    """Return the placeholder `POSTGRES_PASSWORD` stands at when nothing configured it.

    Read off the field's own default so the sentinel is not duplicated as a literal in
    a second module, where a change to the default would silently stop matching.
    """
    default: object = Settings.model_fields["POSTGRES_PASSWORD"].default
    return default.get_secret_value() if isinstance(default, SecretStr) else ""


def _scheme_for(scheme: str, flavour: DatabaseUrlFlavour) -> str:
    """Rewrite a recognised Postgres scheme for `flavour`, leaving anything else alone."""
    if scheme.partition("+")[0] not in _POSTGRES_SCHEMES:
        return scheme
    return _ASYNC_SCHEME if flavour == "async" else _PLAIN_SCHEME


def _query_for(query: str, flavour: DatabaseUrlFlavour) -> str:
    """Drop the connection parameters `flavour`'s driver cannot accept."""
    dropped: frozenset[str] = _ASYNC_DROPPED_PARAMS if flavour == "async" else _PLAIN_DROPPED_PARAMS
    return urlencode(
        [
            (key, value)
            for key, value in parse_qsl(query, keep_blank_values=True)
            if key not in dropped
        ]
    )


def _netloc_with_credential(parsed: SplitResult) -> str:
    """Inject the configured password into `parsed`'s authority when it carries none.

    Only the credential is percent-encoded: the username and host come straight out of
    a URL and are already encoded, while the secret comes from settings and may hold
    reserved characters. SQLAlchemy and asyncpg both decode userinfo with
    `urllib.parse.unquote`, so `quote` is the matching encoder -- `quote_plus` would
    turn a space into a literal ``+`` and authenticate with the wrong secret.
    """
    if parsed.password or not parsed.username:
        return parsed.netloc
    secret: str = settings.POSTGRES_PASSWORD.get_secret_value()
    if secret == _unconfigured_password():
        return parsed.netloc
    # rpartition keeps host and port exactly as written, brackets and case included,
    # and cannot append a port the authority already carries.
    host_and_port: str = parsed.netloc.rpartition("@")[2]
    return f"{parsed.username}:{quote(secret, safe='')}@{host_and_port}"


def get_database_url(flavour: DatabaseUrlFlavour = "async") -> str:
    """Build the connection URL for `flavour` from the one configured Postgres URL.

    Both flavours normalise the same source value: the scheme is rewritten for the
    target driver, connection parameters that driver rejects are dropped, and the
    configured password is injected when the URL carries none.

    Args:
        flavour: `"async"` for the application's SQLAlchemy pool, which needs
            ``postgresql+asyncpg://`` and no libpq query arguments. `"plain"` for
            low-level drivers -- `asyncpg.connect`, psycopg -- which need
            ``postgresql://`` and do accept ``sslmode``.

    Returns:
        A connection URL the driver behind `flavour` can use as-is.
    """
    parsed: SplitResult = urlsplit(settings.POSTGRES_URL)
    return urlunsplit(
        (
            _scheme_for(parsed.scheme, flavour),
            _netloc_with_credential(parsed),
            parsed.path,
            _query_for(parsed.query, flavour),
            parsed.fragment,
        )
    )


def get_database_fields() -> DatabaseConnectionFields:
    """Decompose the plain connection URL into discrete, percent-decoded fields.

    Returns:
        The host, port, username, credential and database name the application's own
        pool connects to. Each falls back to its individual setting only when the URL
        omits that component, which is the only case where there is no application
        value to disagree with.
    """
    parsed: SplitResult = urlsplit(get_database_url(flavour="plain"))
    username: str | None = parsed.username
    password: str | None = parsed.password
    return DatabaseConnectionFields(
        host=parsed.hostname or settings.POSTGRES_HOST,
        port=parsed.port or settings.POSTGRES_PORT,
        username=unquote(username) if username else settings.POSTGRES_USERNAME,
        password=SecretStr(unquote(password)) if password else settings.POSTGRES_PASSWORD,
        database=unquote(parsed.path.removeprefix("/")) or settings.POSTGRES_DB_NAME,
    )


async def init_db() -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Initialize database engine and session factory.

    Returns:
        tuple: (engine, AsyncSessionLocal) for app.state injection
    """
    engine: AsyncEngine = create_async_engine(
        url=get_database_url(),
        echo=False,
        pool_size=settings.POSTGRES_POOL_SIZE,
        max_overflow=settings.POSTGRES_MAX_OVERFLOW,
        pool_pre_ping=True,
        pool_timeout=30,
        pool_recycle=3600,
        connect_args={
            # Timeouts are critical when managing your own pool directly
            "server_settings": {
                "statement_timeout": "10000",
                "idle_in_transaction_session_timeout": "10000",
            }
        },
        # if using Pg-Bouncer, set poolclass to NullPool to disable connection pooling at the SQLAlchemy level
        # Disable SQLAlchemy's pool, let PgBouncer handle concurrency.
        # Alternatively, keep poolclass default but set pool_size to something very small (e.g., 2 to 5).
        # poolclass=NullPool,
        # connect_args={
        #     # CRITICAL: Disables prepared statements to prevent PgBouncer transaction errors
        #     "prepared_statement_cache_size": 0,
        #     "statement_cache_size": 0,
        #     # Prevent bad queries from hanging connections forever (value in milliseconds)
        #     "server_settings": {
        #         "statement_timeout": "10000",
        #         "idle_in_transaction_session_timeout": "10000"
        #     }
        # }
    )

    session_local: async_sessionmaker[AsyncSession] = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )

    try:
        await _verify_postgres_connection(engine)
    except Exception as e:
        logger.error("PostgreSQL initialization failed: {}", e, exc_info=True)
        await engine.dispose()
        raise

    return engine, session_local


async def _verify_postgres_connection(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        result: CursorResult[Any] = await conn.execute(text("SELECT version()"))
        version: Unknown = result.scalar()
        fields: DatabaseConnectionFields = get_database_fields()
        logger.info(
            "PostgreSQL connected",
            host=fields.host,
            database=fields.database,
            version=version,
        )


@asynccontextmanager
async def independent_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Give one unit of work an independent transaction."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_postgres_db(connection: HTTPConnection) -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database sessions retrieved from app.state."""
    session_local: Any = connection.app.state.db_session_local
    async with session_local() as session:
        try:
            yield session  # noqa: ASYNC119
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
