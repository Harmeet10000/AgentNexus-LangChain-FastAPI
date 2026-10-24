# Distributed WebSocket revocation

WebSocket sessions are revoked authoritatively in Redis by
`RefreshTokenRepository`. Each successful single-session, refresh-family, or
user-wide revocation publishes a versioned event to the bounded
`ws:revocation:events` Redis Stream. Every application worker consumes the
stream and closes only its own matching sockets.

## Why Redis Streams

Redis Pub/Sub has low latency but is lossy: a worker disconnected during a
publish cannot recover the event. Streams retain a bounded history, so a worker
can reconnect with its last stream ID and replay events. Workers deliberately
do not use consumer groups because every worker must see every event. The
stream is a propagation optimization, not the authorization boundary. A
periodic session sweep and outbound/inbound session checks remain enabled and
close sockets when an event was trimmed, delayed, or lost during a Redis
outage.

The stream uses approximate trimming at 10,000 entries. The listener reads at
most `WEBSOCKET_REVOCATION_BATCH_SIZE` entries per call and maintains a bounded
in-memory event-ID deduplication set. Duplicate and out-of-order events are
safe because closing a local socket is idempotent.

## Event contract and matching

Events contain `version`, `event_id`, `scope` (`session` or `user`),
`user_id`, an optional `session_id` or `except_session_id`, `reason`, and
`occurred_at`. Unknown fields, invalid versions, missing scope identity, and
malformed values are ignored. A tenant ID, when present, must match the local
connection context exactly. A session event closes one session; a user event
closes all sessions except the explicitly excluded session.

Publication is best-effort after the authoritative Redis mutation. A stream
write failure must not turn a successful revoke into an API failure, because
the pull-based checks provide the security guarantee.

## Operations and rollback

The listener starts alongside the existing periodic revocation sweep and is
cancelled and awaited during graceful shutdown. Redis connection errors cause
bounded exponential reconnect backoff. Inspect `ws:revocation:events` with
`XRANGE`/`XINFO STREAM` when diagnosing propagation; do not delete the stream
as an operational workaround. If propagation is noisy, set
`WEBSOCKET_REVOCATION_ENABLED=false` and rely on the existing pull checks while
investigating. Re-enable it after Redis health is restored. Disabling the
listener does not disable revocation enforcement.

Use the application metrics/logs for publication failures, malformed events,
listener reconnects, and local close counts. A Redis outage or an unexpectedly
old stream last-generated ID should be treated as a latency/degradation alert,
not as permission to allow a revoked session to continue indefinitely.
