from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, ConfigDict, Field

FetchNode = Callable[["ReconciliationState"], Awaitable[dict[str, object]]]
ApplyNode = Callable[["ReconciliationState"], Awaitable[dict[str, object]]]
ReconciliationRunnable = Runnable[list[BaseMessage], Any]


class ReconciliationEntityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    entity_type: str
    name: str
    normalized_name: str
    confidence: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str | None = None


class MergeDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    keep_id: str
    discard_id: str
    reason: str


class UpdateDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    fields: dict[str, Any] = Field(default_factory=dict)
    reason: str


class IgnoreDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_id: str
    reason: str


class ReconciliationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    merge: list[MergeDecision] = Field(default_factory=list)
    update: list[UpdateDecision] = Field(default_factory=list)
    ignore: list[IgnoreDecision] = Field(default_factory=list)


class ReconciliationState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = ""
    run_id: str = ""
    lookback_hours: int = 24

    new_entities: list[ReconciliationEntityRecord] = Field(default_factory=list)
    existing_entities: list[ReconciliationEntityRecord] = Field(default_factory=list)
    fetch_error: str | None = None

    reconciliation_decision: ReconciliationDecision = Field(
        default_factory=ReconciliationDecision
    )
    reconcile_error: str | None = None

    merged_count: int = 0
    updated_count: int = 0
    versions_written: int = 0
    apply_error: str | None = None

    completed: bool = False
