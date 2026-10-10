"""Shared services module."""

from app.connections.mailer import MailerConfig, send_template
from app.resilience.rate_limit import (
    RateLimitConfig,
    RateLimiter,
    RateLimitScope,
    build_rate_limiter,
)
from app.connections.tavily import (
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
    "build_rate_limiter",
    "get_context",
    "get_tavily_client",
    "search",
    "send_template",
]
