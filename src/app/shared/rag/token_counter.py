"""Token counting for retrieval context budgets.

Neutral home for the counter callable: the tokenizer *choice* belongs to
`ingestion-chunking`, and every consumer here takes the counter as an injected
callable rather than importing a tokenizer library. The default below is the
currently configured counting function, not the choice — replacing it means
editing this module once, not every call site.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import tiktoken

type CountTokens = Callable[[str], int]
"""A token-counting callable, injected wherever a context budget is enforced."""


@lru_cache(maxsize=1)
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens with the configured encoding."""
    return len(_encoding().encode(text))
