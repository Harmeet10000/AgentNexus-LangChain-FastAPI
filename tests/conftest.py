import sys
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from fakeredis.aioredis import FakeRedis

# Break circular/broken imports before any app module loads.
# A MagicMock has no __path__ (dunders are not auto-created), so every
# `from <stubbed root>.<sub> import ...` in src/ needs its own entry — otherwise
# the import machinery reports "'<root>' is not a package". Regenerate with:
#   rg -o 'from (mcp_core|tasks|app\.shared\.langgraph_layer)(\.[\w.]+)? import' src/
sys.modules["app.connections.mcp"] = MagicMock()
sys.modules["app.connections.celery"] = MagicMock()
sys.modules["mcp_core"] = MagicMock()
sys.modules["mcp_core.client.auth"] = MagicMock()
sys.modules["mcp_core.client.manager"] = MagicMock()
sys.modules["mcp_core.client.settings"] = MagicMock()
sys.modules["mcp_core.common.errors"] = MagicMock()
sys.modules["mcp_core.common.models"] = MagicMock()
sys.modules["mcp_core.lifespan_mcp"] = MagicMock()
sys.modules["mcp_core.mcp"] = MagicMock()
sys.modules["mcp_core.server.factory"] = MagicMock()
sys.modules["mcp_core.server.http"] = MagicMock()
sys.modules["mcp_core.server.middleware"] = MagicMock()
sys.modules["mcp_core.server.tools"] = MagicMock()
sys.modules["tasks"] = MagicMock()
sys.modules["tasks.auth_email_tasks"] = MagicMock()

# The twenty-one `app.features.search.*` imports that used to sit below this block were **dead**:
# nothing in this file referenced a single one of them, and no test imports symbols from a conftest.
# They are why step 3 of `documents-unified-schema` relocated chunking, fusion and RAG behind
# re-export shims rather than moving them outright — the stated reason being that a missing symbol
# here is a collection error for every test in the repository. That reason was sound in form and
# void in fact: the imports bound names no fixture used. Worth remembering next time a shim is
# justified by an import list — whether the imports are *used* is a different question from whether
# they exist, and only the first one constrains a move.

# `app.shared.langgraph_layer` and four of its submodules were stubbed here too. They are
# not any more, and the entries must not come back:
#
#   * The cycles they worked around are gone, severed by 319c698 and 6525c6f. Removing all
#     five leaves the suite at exactly its prior counts, and faster.
#   * The stub made every Band C proof in `ingestion-pipeline-unification` unwritable. A
#     MagicMock has no `__path__`, so `app.shared.langgraph_layer.<anything>` raised
#     "is not a package" — and C2, C4, C5, and C6 each require a unit test over
#     `checkpointer.py` or `kb_retry.py`.
#   * It was hiding a live defect. Nothing had ever constructed `IngestionState`, so nobody
#     saw that it could not be constructed at all: `Annotated` sat in a `TYPE_CHECKING`
#     block while a Pydantic field annotation needed it at runtime. Removing the stub
#     surfaced it on the first attempt.
#
# A stub that makes a package unimportable does not isolate a test from that package; it
# removes the package from the suite's reach entirely, and takes its defects out of view
# with it.

_tal_mock = MagicMock()
_tal_mock.TokenAuditLog = MagicMock()
_tal_mock.TokenAuditLog.find_one.return_value = MagicMock()
_tal_mock.TokenAuditLog.find_one.return_value.update = AsyncMock(return_value=None)
_tal_mock.TokenAuditLog.find.return_value = MagicMock()
_tal_mock.TokenAuditLog.find.return_value.update = AsyncMock(return_value=None)
_tal_instance = MagicMock()
_tal_instance.insert = AsyncMock(return_value=None)
_tal_mock.TokenAuditLog.return_value = _tal_instance
sys.modules["app.features.auth.token_audit_log"] = _tal_mock

from app.features.auth.dto import UserResponse
from app.features.auth.repository import (
    RefreshTokenRepository,
    SessionData,
    UserRepository,
)
from app.features.auth.service import AuthService
from app.utils.exceptions import (
    ConflictException,
    NotFoundException,
    ServiceUnavailableException,
    UnauthorizedException,
)


@pytest.fixture
def redis() -> FakeRedis:
    fake = FakeRedis(decode_responses=True)
    return fake


@pytest.fixture
def refresh_token_repo(redis) -> RefreshTokenRepository:
    return RefreshTokenRepository(redis)


@pytest.fixture
def auth_service(refresh_token_repo) -> AuthService:
    mock_user_repo = MagicMock(spec=UserRepository)
    service = AuthService(mock_user_repo, refresh_token_repo)
    return service


def make_mock_user(**kwargs) -> MagicMock:
    user = MagicMock()
    user.id = kwargs.get("id", "507f1f77bcf86cd799439011")
    user.email = kwargs.get("email", "test@example.com")
    user.hashed_password = kwargs.get(
        "hashed_password", "$argon2id$v=19$m=65536,t=3,p=4$abc"
    )
    user.is_active = kwargs.get("is_active", True)
    user.is_verified = kwargs.get("is_verified", True)
    user.role = kwargs.get("role", "user")
    user.full_name = kwargs.get("full_name", None)
    user.oauth_accounts = []
    user.verification_token_hash = None
    user.reset_token_hash = None
    user.reset_token_expires_at = None
    user.created_at = datetime.now(UTC)
    user.updated_at = datetime.now(UTC)
    return user
