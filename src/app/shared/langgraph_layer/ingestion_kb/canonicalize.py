"""Deterministic entity canonicalisation (ADR-2).

Every extracted entity resolves to a stable canonical identity **before**
anything is written to the knowledge graph. The rule is a pure function of the
surface form, so two processes canonicalising the same string produce the same
identity with no shared state. The raw surface form is retained on the result
as the audit trail — it is never discarded.

Where canonicalisation cannot be performed (the surface form carries no
identity at all — blank, punctuation-only, or a bare corporate suffix), the
entity is **refused**: callers record a terminal document failure and write
nothing for that document. There is no raw-text fallback identity, because a
fallback would reintroduce exactly the irreversible defect (three surface forms
of one company becoming three party nodes) in the situation least likely to be
noticed.

Ordering constraint (ADR-2): this lands **before** any graph write goes live.
Both graph-reaching stages (`embed_store`'s identity map and the
`graphiti_upsert` episode path) canonicalise first and refuse on the same
condition, so a missed site cannot poison the graph while another is guarded.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from .state import ExtractedEntity

# Trailing corporate designators stripped during normalisation. Stored longest
# first in a tuple — never iterated as a set: `str` hashing is salted per
# process, so set iteration order differs between processes and a rule that
# strips "co" before checking "company" in one process and the reverse in
# another is not deterministic. Periods are removed before matching, so the
# entries carry none.
_CORPORATE_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        (
            "incorporated",
            "corporation",
            "company",
            "limited",
            "services",
            "group",
            "holdings",
            "partners",
            "ventures",
            "inc",
            "corp",
            "llc",
            "ltd",
            "plc",
            "gmbh",
            "pty",
            "bhd",
            "sdn",
            "pte",
            "lp",
            "llp",
            "pllc",
            "ltd",
            "co",
            "ag",
            "sa",
            "pa",
        ),
        key=len,
        reverse=True,
    )
)

_WHITESPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9 ]")


class CanonicalEntity(BaseModel):
    """One extracted entity under its canonical identity, raw form retained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_ref: str
    entity_type: str
    canonical_id: str
    canonical_name: str
    raw_name: str


def canonical_identity_key(*, entity_type: str, name: str) -> str:
    """Return the deterministic graph identity for one surface form.

    Raises `ValueError` when the surface form carries no identity (blank,
    punctuation-only, or a bare corporate suffix) — the caller converts that
    into the refusal path rather than a fallback identity.
    """
    normalized = _normalize(name)
    if not normalized:
        msg = f"entity surface form carries no canonicalisable identity: {name!r}"
        raise ValueError(msg)
    return f"{entity_type.strip().upper()}:{normalized}"


def canonicalize_entities(
    entities: list[ExtractedEntity],
) -> tuple[dict[str, CanonicalEntity], list[ExtractedEntity]]:
    """Split extracted entities into canonicalised and refused.

    Returns the identity map (original entity id → canonical record) for the
    entities that resolved, and the entities that did not. A non-empty refused
    list means the caller must record a terminal document failure and write
    nothing — not fall back to raw text.
    """
    canonical: dict[str, CanonicalEntity] = {}
    refused: list[ExtractedEntity] = []
    for entity in entities:
        try:
            key = canonical_identity_key(entity_type=entity.type.value, name=entity.name)
        except ValueError:
            refused.append(entity)
            continue
        canonical[entity.id] = CanonicalEntity(
            entity_ref=entity.id,
            entity_type=entity.type.value,
            canonical_id=key,
            canonical_name=key.split(":", 1)[1],
            raw_name=entity.name,
        )
    return canonical, refused


def _normalize(name: str) -> str:
    """Fold one surface form to its canonical stem (pure function)."""
    folded = unicodedata.normalize("NFKC", name).lower()
    folded = folded.replace("&", " and ")
    # The typographic possessive apostrophe is spelled as an escape so the
    folded = _NON_ALNUM_RE.sub(
        " ", folded.replace(chr(39), chr(32)).replace(chr(8217), chr(32))
    )
    folded = _WHITESPACE_RE.sub(" ", folded).strip()
    # Possessive remnant: "acme s" (from "Acme's") is the same stem as "acme".
    words = folded.split(" ")
    if words and words[-1] == "s" and len(words) > 1:
        words = words[:-1]
    if words and words[0] == "the":
        words = words[1:]
    # Strip at most the designator run: "acme holdings group" → "acme", while a
    # stem that merely ends in a designator word keeps it when nothing precedes.
    while len(words) > 1 and words[-1] in _CORPORATE_SUFFIXES:
        words = words[:-1]
    # A conjunction left trailing by the strip ("acme and co" → "acme and") is
    # the same remnant class as the designators: no party is named "... and".
    if len(words) > 1 and words[-1] == "and":
        words = words[:-1]
    if len(words) == 1 and words[0] in _CORPORATE_SUFFIXES:
        return ""
    return " ".join(words)
