"""Dependency wiring for the site-search feature.

Authentication posture matches ``features/documents``: every endpoint resolves
``CurrentAuthorizationContext`` (bearer token + live SQL tenant membership), and
the resolved ``tenant_id`` — never anything client-supplied — scopes every
service call. The legacy feature had none of this; that was its biggest defect.
"""

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.connections import get_postgres_db, get_redis
from app.features.auth.api.http.dependencies import (
    AuthorizationContext,
    CurrentAuthorizationContext,
)
from app.features.search.adapters.outbound.embeddings import ProviderSearchEmbeddings
from app.features.search.adapters.outbound.jobs import DurableSearchJobQueue
from app.features.search.adapters.persistence.postgres.search_repository import (
    PostgresSearchRepository,
)
from app.features.search.application.service import SearchService

if TYPE_CHECKING:
    from app.shared.jobs import JobBus


async def get_search_repository(
    session: Annotated[AsyncSession, Depends(get_postgres_db)],
) -> PostgresSearchRepository:
    return PostgresSearchRepository(session)


async def get_search_service(
    repo: Annotated[PostgresSearchRepository, Depends(get_search_repository)],
    redis: Annotated[Redis, Depends(get_redis)],
    request: Request,
) -> SearchService:
    job_bus: JobBus | None = getattr(request.app.state, "job_bus", None)
    job_queue = DurableSearchJobQueue(job_bus) if job_bus is not None else None
    return SearchService(
        repo=repo,
        redis=redis,
        job_queue=job_queue,
        embeddings=ProviderSearchEmbeddings(),
    )


async def get_search_authorization(
    context: CurrentAuthorizationContext,
) -> AuthorizationContext:
    """Expose the authenticated caller's tenant to search handlers.

    No document allowlist is resolved here: site search filters by tenant in
    SQL on every branch, so forged ids cannot widen authority.
    """
    return context


SearchRepositoryDep = Annotated[PostgresSearchRepository, Depends(get_search_repository)]
SearchServiceDep = Annotated[SearchService, Depends(get_search_service)]
SearchAuthorizationDep = Annotated[AuthorizationContext, Depends(get_search_authorization)]
