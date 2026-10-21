"""Worker-independent task wire contracts and caller-first enqueue seam."""

from .bus import (
    AggregateRef,
    DirectEnqueueReceipt,
    DirectPublicationAdapter,
    DurableEnqueueReceipt,
    EnqueueOptions,
    JobBus,
    TransactionalJobBus,
    TransactionalStagingAdapter,
    TransactionScope,
    TypedJob,
    is_transaction_scope,
    prepare_job,
    require_direct_receipt,
)
from .in_memory import (
    InMemoryDirectAdapter,
    InMemoryJobBus,
    InMemoryTransaction,
    InMemoryTransactionalAdapter,
)

__all__ = [
    "AggregateRef",
    "DirectEnqueueReceipt",
    "DirectPublicationAdapter",
    "DurableEnqueueReceipt",
    "EnqueueOptions",
    "InMemoryDirectAdapter",
    "InMemoryJobBus",
    "InMemoryTransaction",
    "InMemoryTransactionalAdapter",
    "JobBus",
    "TransactionScope",
    "TransactionalJobBus",
    "TransactionalStagingAdapter",
    "TypedJob",
    "is_transaction_scope",
    "prepare_job",
    "require_direct_receipt",
]
