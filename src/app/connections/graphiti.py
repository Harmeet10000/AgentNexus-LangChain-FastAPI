"""Knowledge-graph connection establishment.

Thin re-export: the implementation stays in ``app.shared.rag.graphiti``
(multiple importers reference it directly), so this module only moves the
import seam — every connection-shaped import in lifespan resolves to
``app.connections``.
"""

from __future__ import annotations

from app.shared.rag.graphiti import (
    close_graphiti,
    setup_graphiti,
    setup_graphiti_indices,
)

__all__ = [
    "close_graphiti",
    "setup_graphiti",
    "setup_graphiti_indices",
]
