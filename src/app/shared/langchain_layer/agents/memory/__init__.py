"""Public exports for the shared agent memory package."""

from .agent_memory_service import (
    AgentMemoryError,
    AgentMemoryService,
    ConsolidationPreconditionError,
    ConversationIdentityRequiredError,
    PartitionIdentityInvalidError,
    memory_partition,
)
from .cognee_client import setup_cognee
from .memory_scope import (
    COMPLIANCE_SCOPE,
    GROUNDING_SCOPE,
    ORCHESTRATOR_SCOPE,
    PRECEDENT_SCOPE,
    RISK_SCOPE,
    MemoryEntityType,
    MemoryScope,
    MemorySource,
    MemoryTimeFilter,
    scope_from_router_decision,
)

__all__ = [
    "COMPLIANCE_SCOPE",
    "GROUNDING_SCOPE",
    "ORCHESTRATOR_SCOPE",
    "PRECEDENT_SCOPE",
    "RISK_SCOPE",
    "AgentMemoryError",
    "AgentMemoryService",
    "ConsolidationPreconditionError",
    "ConversationIdentityRequiredError",
    "MemoryEntityType",
    "MemoryScope",
    "MemorySource",
    "MemoryTimeFilter",
    "PartitionIdentityInvalidError",
    "memory_partition",
    "scope_from_router_decision",
    "setup_cognee",
]
