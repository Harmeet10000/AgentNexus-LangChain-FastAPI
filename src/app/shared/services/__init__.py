"""Shared services module."""

from app.shared.services.mailer import MailerConfig, send_template
from app.shared.services.rate_limiter import (
    RateLimitConfig,
    RateLimiter,
    RateLimitScope,
    get_rate_limiter,
)
from app.shared.services.tavily import (
    SearchResponse,
    SearchResult,
    get_context,
    get_tavily_client,
    search,
)

__all__ = [
    "MailerConfig",
    "RateLimitConfig",
    "RateLimitScope",
    "RateLimiter",
    "SearchResponse",
    "SearchResult",
    "get_context",
    "get_rate_limiter",
    "get_tavily_client",
    "search",
    "send_template",
]
