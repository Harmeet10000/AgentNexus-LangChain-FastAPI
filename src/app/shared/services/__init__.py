"""Shared service classifiers (error unions only).

Implementations moved to their owners: mail/Tavily/storage adapters live in
`app.connections`, rate limiting in `app.resilience`. The `errors.py` unions
stay here — the classifier rule permits shared error modules, and
`app.connections` adapters own no error files of their own.
"""

from .errors import (
    MailerError,
    StorageError,
    TavilyError,
)

__all__ = ["MailerError", "StorageError", "TavilyError"]
