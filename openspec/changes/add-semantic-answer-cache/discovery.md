# Discovery: Event Storm add-semantic-answer-cache

No new async behavior in this change.

The ask path gains no domain events, websocket topics, commands, presence, or telemetry
channels: the semantic lookup is a synchronous in-request read and the store is an in-request
write, both over the existing Redis client. Expiry is Redis key TTL with a non-sliding
maximum lifetime fixed before atomic publication (hits, observe checks, and replays cannot extend it) — not
the exact-match cache's 900s hygiene TTL. Contracts will record: no async interfaces in this
change.
