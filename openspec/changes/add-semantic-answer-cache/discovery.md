# Discovery: Event Storm add-semantic-answer-cache

No new async behavior in this change.

The ask path gains no domain events, websocket topics, commands, presence, or telemetry
channels: the semantic lookup is a synchronous in-request read and the store is an in-request
write, both over the existing Redis client. The only expiry is the TTL discipline the
exact-match answer cache already carries, reused unchanged for semantic entries. Contracts
will record: no async interfaces in this change.
