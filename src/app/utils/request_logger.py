"""Request-specific logging utilities."""

from fastapi import Request

from app.utils.logger import logger


def get_request_logger(request: Request):
    """
    Get a logger bound to the current request context.
    
    Usage in routes:
        @app.get("/users")
        async def get_users(request: Request):
            log = get_request_logger(request)
            log.info("Fetching users")
            # ... your code
            log.info("Users fetched", count=len(users))
    """
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    return logger.bind(
        correlation_id=correlation_id,
        path=request.url.path,
        method=request.method,
    )
