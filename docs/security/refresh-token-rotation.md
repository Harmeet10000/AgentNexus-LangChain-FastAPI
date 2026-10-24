# Refresh-token rotation and reuse detection

Refresh tokens are now one-time credentials in an explicit token family. The
client-facing contract remains the existing `TokenResponse`, but a successful
refresh returns a new refresh token. Clients must replace the stored token
atomically after each successful response.

## Protocol

Each login creates a family record at `auth:refresh_family:{family_id}` and a
session record containing the family ID. The family stores only SHA-256
digests of the current and previous token identifier/secret, plus the family
owner, device/session binding, authorization revision, generation, expiry and
revocation state. Raw refresh material is never written to Redis, logs or
audit events. The signed JWT carries the opaque identifier and secret so the
server can verify it without persisting a replayable secret.

Redis Lua performs the compare-and-swap. The first request for the current
generation atomically installs generation `n+1`. A duplicate request for the
same parent is accepted only during the small configured grace window
(`REFRESH_TOKEN_ROTATION_GRACE_SECONDS`, default 10 seconds) and deterministically
reconstructs the same child. A later replay, secret mismatch, wrong device,
authorization revision mismatch or revoked family fails closed and a replay
revokes the entire family and its session.

The grace window is an idempotency allowance for a client retry or two
legitimate requests racing over the network; it is not a second-use allowance.
Clients should still serialize refreshes where possible and use an atomic
replace/rollback-safe write for their token store.

## Migration and rollback

Legacy refresh JWTs without family claims are accepted once and migrated into a
family derived from the existing session. They remain bounded by the existing
session expiry and configured refresh lifetime; no legacy token is extended.
The migration response is a rotated family token, so clients need no endpoint
change, only replacement-token handling.

To roll out clients, deploy server support first, then release clients that
replace their refresh token on every successful response and retry a failed
request only with the newest token. During a rollback, keep family validation
and revocation enabled and roll back only issuance after confirming all active
clients understand replacement tokens. Never re-enable reuse of an old token:
that would make replay detection ambiguous and would invalidate the security
guarantee for already-rotated families.

## Revocation and operations

Logout, explicit session revocation, revoke-all-sessions, account disablement
and authorization-revision changes revoke the relevant family. Operators can
revoke a family with `RefreshTokenRepository.revoke_refresh_family`; the
existing revoke-all session operation marks every associated family revoked.
Security events contain only family/session identifiers and reasons. Metrics
separate successful rotations, benign grace retries, infrastructure failures
and replay detections so a retry storm is not confused with theft.

## Testing

The deterministic unit suite covers digest-only storage, atomic concurrent
rotation, grace idempotency and post-grace replay revocation. The opt-in live
Redis suite (`LIVE_REDIS_URL`) exercises the same Lua transition with actual
Redis connections; CI workflows intentionally do not enable live infrastructure
tests in this repository.
