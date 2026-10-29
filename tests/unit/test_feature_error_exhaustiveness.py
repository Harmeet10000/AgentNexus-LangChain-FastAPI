"""Every feature error union has an owner-local exhaustive classifier."""

from collections.abc import Callable

import pytest

from app.features.agent_saul.domain.errors import (
    AgentSaulInfrastructureError,
    AgentSaulValidationError,
    agent_saul_error_to_http_status,
)
from app.features.audit.exceptions import AuditInfrastructureError, audit_error_to_http_status
from app.features.auth.domain.errors import (
    AuthAuthenticationError,
    AuthAuthorizationError,
    AuthConflictError,
    AuthInfrastructureError,
    AuthNotFoundError,
    AuthValidationError,
    auth_error_to_http_status,
)
from app.features.billing.dunning.exceptions import (
    DunningExternalServiceError,
    DunningInfrastructureError,
    dunning_error_to_http_status,
)
from app.features.billing.plans.exceptions import (
    PlanConflictError,
    PlanInfrastructureError,
    PlanNotFoundError,
    PlanValidationError,
    plan_error_to_http_status,
)
from app.features.billing.subscriptions.exceptions import (
    SubscriptionDuplicateError,
    SubscriptionInfrastructureError,
    SubscriptionInvalidTransitionError,
    SubscriptionNotFoundError,
    SubscriptionPlanNotFoundError,
    SubscriptionTransientInfrastructureError,
    SubscriptionValidationError,
    SubscriptionVersionConflictError,
    subscription_error_to_http_status,
)
from app.features.crawler.exceptions import (
    CrawlerSearchError,
    CrawlerValidationError,
    crawler_error_to_http_status,
)
from app.features.documents.domain.errors import (
    DocumentChunkConflictError,
    DocumentConflictError,
    DocumentDatabaseError,
    DocumentEmbeddingWidthError,
    DocumentGraphWriteError,
    DocumentNotFoundError,
    DocumentStatusNotFoundError,
    DocumentStorageError,
    DocumentValidationError,
    document_error_to_http_status,
)
from app.features.documents.ingestion.legacy_errors import (
    IngestionGraphError,
    IngestionInternalError,
    IngestionPipelineError,
    IngestionValidationError,
    ingestion_error_to_http_status,
)
from app.features.profile.exceptions import (
    ProfileAuthenticationError,
    ProfileConflictError,
    ProfileInfrastructureError,
    ProfileStorageError,
    profile_error_to_http_status,
)
from app.features.users.exceptions import (
    UsersAuthorizationError,
    UsersConflictError,
    UsersInfrastructureError,
    UsersNotFoundError,
    UsersValidationError,
    users_error_to_http_status,
)
from app.kernel.errors import FeatureError

type Classifier = Callable[[object], int]
_USER_OPERATION_ERRORS = {UsersConflictError, UsersAuthorizationError, UsersInfrastructureError}
_PROFILE_OPERATION_ERRORS = {ProfileConflictError, ProfileInfrastructureError}


def _error(error_type: type[FeatureError], **fields: object) -> FeatureError:
    return error_type(message="test", **fields)


@pytest.mark.parametrize(
    ("classifier", "errors"),
    [
        (
            auth_error_to_http_status,
            [
                _error(error_type)
                for error_type in [
                    AuthValidationError,
                    AuthNotFoundError,
                    AuthConflictError,
                    AuthAuthenticationError,
                    AuthAuthorizationError,
                    AuthInfrastructureError,
                ]
            ],
        ),
        (
            document_error_to_http_status,
            [
                _error(error_type)
                for error_type in [
                    DocumentNotFoundError,
                    DocumentStatusNotFoundError,
                    DocumentConflictError,
                    DocumentChunkConflictError,
                    DocumentValidationError,
                    DocumentStorageError,
                    DocumentDatabaseError,
                    DocumentEmbeddingWidthError,
                    DocumentGraphWriteError,
                ]
            ],
        ),
        (
            users_error_to_http_status,
            [
                _error(error_type, user_id="u")
                if error_type is UsersNotFoundError
                else _error(error_type, operation="test")
                if error_type in _USER_OPERATION_ERRORS
                else _error(error_type)
                for error_type in (
                    UsersValidationError,
                    UsersNotFoundError,
                    UsersConflictError,
                    UsersAuthorizationError,
                    UsersInfrastructureError,
                )
            ],
        ),
        (
            profile_error_to_http_status,
            [
                _error(error_type, operation="test")
                if error_type in _PROFILE_OPERATION_ERRORS
                else _error(error_type)
                for error_type in (
                    ProfileConflictError,
                    ProfileAuthenticationError,
                    ProfileInfrastructureError,
                    ProfileStorageError,
                )
            ],
        ),
        (
            ingestion_error_to_http_status,
            [
                _error(error_type, doc_id="d")
                for error_type in (
                    IngestionGraphError,
                    IngestionPipelineError,
                    IngestionInternalError,
                    IngestionValidationError,
                )
            ],
        ),
        (
            plan_error_to_http_status,
            [
                _error(error_type, plan_id="p")
                if error_type is PlanNotFoundError
                else _error(error_type, operation="test")
                if error_type is PlanInfrastructureError
                else _error(error_type)
                for error_type in (
                    PlanNotFoundError,
                    PlanConflictError,
                    PlanValidationError,
                    PlanInfrastructureError,
                )
            ],
        ),
        (
            crawler_error_to_http_status,
            [_error(CrawlerValidationError), _error(CrawlerSearchError)],
        ),
        (audit_error_to_http_status, [_error(AuditInfrastructureError, operation="test")]),
        (
            dunning_error_to_http_status,
            [
                _error(error_type, operation="test")
                for error_type in (DunningInfrastructureError, DunningExternalServiceError)
            ],
        ),
        (
            agent_saul_error_to_http_status,
            [_error(AgentSaulValidationError), _error(AgentSaulInfrastructureError)],
        ),
        (
            subscription_error_to_http_status,
            [
                _error(error_type)
                for error_type in (
                    SubscriptionNotFoundError,
                    SubscriptionDuplicateError,
                    SubscriptionVersionConflictError,
                    SubscriptionInvalidTransitionError,
                    SubscriptionPlanNotFoundError,
                    SubscriptionInfrastructureError,
                    SubscriptionTransientInfrastructureError,
                    SubscriptionValidationError,
                )
            ],
        ),
    ],
)
def test_exhaustive_feature_classifier_covers_every_union_member(
    classifier: Classifier, errors: list[FeatureError]
) -> None:
    for error in errors:
        assert classifier(error) in {401, 403, 404, 409, 422, 500, 502, 503}
