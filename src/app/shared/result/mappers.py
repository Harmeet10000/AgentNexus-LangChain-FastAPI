"""Boundary mappers from internal Result errors to project exceptions."""

from app.utils import (
    APIException,
    ConflictException,
    DatabaseException,
    ExternalServiceException,
    NotFoundException,
    ServiceUnavailableException,
    ValidationException,
)

from .errors import (
    AppError,
    ConflictAppError,
    ExternalServiceAppError,
    InfrastructureAppError,
    NotFoundAppError,
    ValidationAppError,
)


def app_error_to_exception(error: AppError) -> APIException:
    """Translate expected internal failures to the existing exception boundary."""
    match error:
        case ValidationAppError():
            return ValidationException(
                detail=error.message,
                error_code=error.code,
                data=error.details,
            )
        case NotFoundAppError(resource=resource, identifier=identifier):
            return NotFoundException(
                resource=resource,
                identifier=identifier,
                error_code=error.code,
            )
        case ConflictAppError():
            return ConflictException(
                detail=error.message,
                error_code=error.code,
                data=error.details,
            )
        case ExternalServiceAppError(service=service):
            return ExternalServiceException(
                service=service,
                detail=error.message,
                error_code=error.code,
            )
        case InfrastructureAppError(retryable=True):
            return ServiceUnavailableException(
                detail=error.message,
                error_code=error.code,
                data=error.details,
            )
        case InfrastructureAppError():
            return DatabaseException(
                detail=error.message,
                error_code=error.code,
            )
        case AppError():
            return ValidationException(
                detail=error.message,
                error_code=error.code,
                data=error.details,
            )
