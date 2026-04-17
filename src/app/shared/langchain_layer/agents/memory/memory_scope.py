"""
Per-agent memory access contracts.

Each retrieval call accepts a `MemoryScope` so reasoning agents only see the
slice of memory they are allowed to use. This keeps retrieval focused and
prevents accidental cross-agent memory contamination.

Pre-defined scopes:
  `RISK_SCOPE`: clauses + obligations from graph/vector memory, recent only
  `COMPLIANCE_SCOPE`: clauses + contracts + orgs from graph/structured memory
  `PRECEDENT_SCOPE`: broad access across all supported entity types/sources
  `ORCHESTRATOR_SCOPE`: contract-level structured memory only
  `GROUNDING_SCOPE`: structured clause/obligation grounding context only
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from collections.abc import Mapping




class MemoryEntityType(StrEnum):
    CLAUSE = "CLAUSE"
    CONTRACT = "CONTRACT"
    OBLIGATION = "OBLIGATION"
    ORG = "ORG"
    PERSON = "PERSON"


class MemorySource(StrEnum):
    GRAPH = "graph"
    VECTOR = "vector"
    STRUCTURED = "structured"


class MemoryTimeFilter(StrEnum):
    RECENT = "recent"
    ALL = "all"


class MemoryScope(BaseModel):
    """Immutable memory retrieval policy for a single agent role."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    agent_id: str = Field(min_length=1)
    allowed_entity_types: frozenset[MemoryEntityType] = Field(min_length=1)
    allowed_sources: frozenset[MemorySource] = Field(min_length=1)
    graph_depth: int = Field(ge=0)
    time_filter: MemoryTimeFilter
    top_k: int = Field(gt=0)

    @field_validator("allowed_entity_types", "allowed_sources", mode="before")
    @classmethod
    def _normalize_frozenset(cls, value: object) -> object:
        if isinstance(value, frozenset):
            return value
        if isinstance(value, str):
            return frozenset({value})
        if isinstance(value, Iterable):
            return frozenset(value)
        return value

    def allows_source(self, source: str) -> bool:
        try:
            return MemorySource(source) in self.allowed_sources
        except ValueError:
            return False

    def allows_entity_type(self, entity_type: str) -> bool:
        try:
            return MemoryEntityType(entity_type) in self.allowed_entity_types
        except ValueError:
            return False

    def to_log_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "entity_types": sorted(entity_type.value for entity_type in self.allowed_entity_types),
            "sources": sorted(source.value for source in self.allowed_sources),
            "depth": self.graph_depth,
            "time_filter": self.time_filter.value,
            "top_k": self.top_k,
        }

def _coerce_entity_types(
    entity_types: Iterable[MemoryEntityType | str],
) -> frozenset[MemoryEntityType]:
    return frozenset(
        entity_type if isinstance(entity_type, MemoryEntityType) else MemoryEntityType(entity_type)
        for entity_type in entity_types
    )

def _coerce_sources(
    sources: Iterable[MemorySource | str],
) -> frozenset[MemorySource]:
    return frozenset(
        source if isinstance(source, MemorySource) else MemorySource(source) for source in sources
    )


def _coerce_time_filter(time_filter: MemoryTimeFilter | str) -> MemoryTimeFilter:
    if isinstance(time_filter, MemoryTimeFilter):
        return time_filter
    return MemoryTimeFilter(time_filter)


def _read_int_field(
    *,
    decision: Mapping[str, object],
    field_name: str,
    fallback: int,
) -> int:
    value = decision.get(field_name, fallback)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return fallback


def _build_scope(
    *,
    agent_id: str,
    entity_types: Iterable[MemoryEntityType | str],
    sources: Iterable[MemorySource | str],
    graph_depth: int,
    time_filter: MemoryTimeFilter | str,
    top_k: int,
) -> MemoryScope:
    return MemoryScope(
        agent_id=agent_id,
        allowed_entity_types=_coerce_entity_types(entity_types),
        allowed_sources=_coerce_sources(sources),
        graph_depth=graph_depth,
        time_filter=_coerce_time_filter(time_filter),
        top_k=top_k,
    )


RISK_SCOPE: MemoryScope = _build_scope(
    agent_id="risk_agent",
    entity_types={MemoryEntityType.CLAUSE, MemoryEntityType.OBLIGATION},
    sources={MemorySource.GRAPH, MemorySource.VECTOR},
    graph_depth=2,
    time_filter=MemoryTimeFilter.RECENT,
    top_k=8,
)

COMPLIANCE_SCOPE: MemoryScope = _build_scope(
    agent_id="compliance_agent",
    entity_types={
        MemoryEntityType.CLAUSE,
        MemoryEntityType.CONTRACT,
        MemoryEntityType.ORG,
    },
    sources={MemorySource.GRAPH, MemorySource.STRUCTURED},
    graph_depth=1,
    time_filter=MemoryTimeFilter.ALL,
    top_k=5,
)

PRECEDENT_SCOPE: MemoryScope = _build_scope(
    agent_id="precedent_agent",
    entity_types=set(MemoryEntityType),
    sources=set(MemorySource),
    graph_depth=3,
    time_filter=MemoryTimeFilter.ALL,
    top_k=5,
)

ORCHESTRATOR_SCOPE: MemoryScope = _build_scope(
    agent_id="orchestrator",
    entity_types={MemoryEntityType.CONTRACT},
    sources={MemorySource.STRUCTURED},
    graph_depth=1,
    time_filter=MemoryTimeFilter.RECENT,
    top_k=3,
)

GROUNDING_SCOPE: MemoryScope = _build_scope(
    agent_id="grounding_agent",
    entity_types={MemoryEntityType.CLAUSE, MemoryEntityType.OBLIGATION},
    sources={MemorySource.STRUCTURED},
    graph_depth=0,
    time_filter=MemoryTimeFilter.RECENT,
    top_k=5,
)


def scope_from_router_decision(decision: Mapping[str, object]) -> MemoryScope:
    """Build a validated scope from router-produced structured output."""

    return _build_scope(
        agent_id=str(decision.get("agent_id", "dynamic")),
        entity_types=_read_iterable_field(
            decision=decision,
            field_name="entity_types",
            fallback=(MemoryEntityType.CLAUSE, MemoryEntityType.OBLIGATION),
        ),
        sources=_read_iterable_field(
            decision=decision,
            field_name="sources",
            fallback=(MemorySource.GRAPH,),
        ),
        graph_depth=_read_int_field(decision=decision, field_name="graph_depth", fallback=2),
        time_filter=str(decision.get("time_filter", MemoryTimeFilter.RECENT.value)),
        top_k=_read_int_field(decision=decision, field_name="top_k", fallback=5),
    )


def _read_iterable_field(
    *,
    decision: Mapping[str, object],
    field_name: str,
    fallback: tuple[StrEnum, ...],
) -> tuple[str | StrEnum, ...]:
    value = decision.get(field_name)
    if value is None and field_name == "entity_types":
        value = decision.get("allowed_entity_types")
    if value is None and field_name == "sources":
        value = decision.get("allowed_sources")
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(item for item in value if isinstance(item, str | StrEnum))
    return fallback





