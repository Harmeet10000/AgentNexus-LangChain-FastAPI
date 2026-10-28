"""Outbox: model, writer, relay, and relay factory.

One events package: the model declares the tables, the writer stages events
inside business transactions, the relay delivers them, and the factory builds
a provisioned relay. Task ownership (spawning/cancelling the listener) stays
with the caller.
"""

from .model import DeadLetterEvent, OutboxEvent
from .relay import OutboxRelay
from .writer import terminal_event_id, with_outbox, with_terminal_outbox

__all__ = [
    "DeadLetterEvent",
    "OutboxEvent",
    "OutboxRelay",
    "terminal_event_id",
    "with_outbox",
    "with_terminal_outbox",
]
