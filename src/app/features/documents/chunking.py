"""Pure chunk-policy resolution for structure-aware ingestion."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ChunkPolicy(BaseModel):
    """Resolved chunking policy for one classified document kind.

    Identity (`name`) is recorded on every emitted chunk so a reader can tell
    which policy was in force without re-running classification. Overlap is
    zero for every structure-aware policy; only the flagged fallback path may
    overlap (task 5.3).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    max_tokens: int = Field(gt=0)
    overlap: int = Field(default=0, ge=0)


_POLICIES: dict[str, ChunkPolicy] = {
    "contract": ChunkPolicy(name="contract", max_tokens=512, overlap=0),
    "statute": ChunkPolicy(name="statute", max_tokens=384, overlap=0),
    "judgment": ChunkPolicy(name="judgment", max_tokens=640, overlap=0),
    "filing": ChunkPolicy(name="filing", max_tokens=448, overlap=0),
    "default": ChunkPolicy(name="default", max_tokens=512, overlap=0),
}

_KIND_ALIASES: dict[str, str] = {
    "contracts": "contract",
    "legal_contract": "contract",
    "contract": "contract",
    "statutes": "statute",
    "legal_statute": "statute",
    "statute": "statute",
    "judgments": "judgment",
    "legal_judgment": "judgment",
    "judgment": "judgment",
    "filings": "filing",
    "legal_filing": "filing",
    "filing": "filing",
    "legal_policy": "filing",
}


def resolve_chunk_policy(document_kind: str) -> ChunkPolicy:
    """Map a classified document kind to a chunk policy. Pure — no I/O.

    Unclassifiable kinds resolve the default policy rather than raising, so
    chunking proceeds.
    """
    key = _KIND_ALIASES.get((document_kind or "").strip().lower(), "default")
    return _POLICIES[key]
