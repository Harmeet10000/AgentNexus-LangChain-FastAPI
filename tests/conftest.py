import sys
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import fakeredis.aioredis
import pytest
from fakeredis.aioredis import FakeRedis

# Break circular/broken imports before any app module loads
sys.modules["app.connections.mcp"] = MagicMock()
sys.modules["app.connections.celery"] = MagicMock()
sys.modules["mcp_core"] = MagicMock()
sys.modules["mcp_core.server.middleware"] = MagicMock()
sys.modules["tasks"] = MagicMock()
sys.modules["tasks.auth_email_tasks"] = MagicMock()
sys.modules["tasks.search_tasks"] = MagicMock()
sys.modules["app.shared.langgraph_layer"] = MagicMock()
sys.modules["app.shared.langgraph_layer.retrieval_kb"] = MagicMock()

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
from app.features.search.chunking import chunk_text, TextChunk
from app.features.search.constants import (
    DEFAULT_SEARCH_CACHE_TTL_SECONDS,
    HYBRID_CANDIDATE_LIMIT,
    INGEST_CHUNK_OVERLAP,
    INGEST_CHUNK_SIZE,
    RRF_K,
)
from app.features.search.dto import (
    HybridSearchRequest,
    RagSearchRequest,
    SearchIngestRequest,
    SearchIngestResponse,
    SearchResponse,
    SearchResultItem,
)
from app.features.search.fusion import (
    RankedChunk,
    RankedResultRow,
    reciprocal_rank_fusion,
)
from app.features.search.rag import (
    ContextSection,
    SearchChunkRecord,
    assemble_rag_context,
)
from app.features.search.repository import SearchRepository
from app.features.search.service import SearchService
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
