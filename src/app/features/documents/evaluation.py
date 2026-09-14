"""Documents-service binding for the infrastructure-independent evaluation harness."""

from collections.abc import Awaitable, Callable

from returns.result import Failure

from app.shared.evaluation.runner import RetrievalEvaluation, run_retrieval_eval
from app.shared.evaluation.schema import GoldenQuery
from app.utils.exceptions import InfrastructureException

from .dto import SearchMetadataFilter, UnifiedSearchRequest
from .service import DocumentQueryService


class EmptyRetrievalException(InfrastructureException):
    """Live retrieval returned no identifiers and therefore proved no wiring."""

    def __init__(self) -> None:
        super().__init__(
            detail="Live retrieval returned no chunk identifiers for every golden query",
            retryable=False,
        )


def service_retriever(
    *, service: DocumentQueryService, user_id: str, limit: int = 10
) -> Callable[[GoldenQuery], Awaitable[list[str]]]:
    """Adapt the documents query service to the shared retriever protocol."""

    async def retrieve(query: GoldenQuery) -> list[str]:
        result = await service.search(
            user_id=user_id,
            payload=UnifiedSearchRequest(
                query=query.query,
                limit=limit,
                candidate_limit=max(limit, 50),
                metadata_filter=SearchMetadataFilter(
                    document_kind=query.document_kind,
                    jurisdiction=query.jurisdiction,
                ),
                bypass_cache=True,
            ),
        )
        if isinstance(result, Failure):
            error = result.failure()
            raise InfrastructureException(
                detail=f"Documents retrieval failed: {error.message}",
                retryable=error.retryable,
                data=error.details,
            )
        return [item.chunk_id for item in result.unwrap().items]

    return retrieve


async def run_live_retrieval_eval(
    *, service: DocumentQueryService, user_id: str, queries: list[GoldenQuery]
) -> RetrievalEvaluation:
    """Score the live service path and reject a result that proves no connection."""
    evaluation = await run_retrieval_eval(
        queries=queries,
        retrieve=service_retriever(service=service, user_id=user_id),
    )
    if not any(row.retrieved_chunk_ids for row in evaluation.rows):
        raise EmptyRetrievalException
    return evaluation
