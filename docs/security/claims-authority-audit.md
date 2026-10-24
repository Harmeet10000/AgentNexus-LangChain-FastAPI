# JWT claims authority audit

Issue: [#78](https://github.com/Harmeet10000/AgentNexus-LangChain-FastAPI/issues/78)

## Rule

JWTs establish identity and carry a session identifier plus an authorization
revision hint. They do **not** establish current role or permissions. Every
privileged decision must use the live `AuthAccount` returned by the auth
dependency chain, or an explicitly versioned trusted cache. `sid=None` remains
reserved for short-lived stateless impersonation tokens; those tokens are
allowed only where the endpoint explicitly accepts the impersonation policy
and the audit trail records `impersonated_by`.

The canonical policy seams are:

- `get_token_claims`: decode the token and verify the Redis session for
  stateful access tokens.
- `get_current_active_user` / `get_current_verified_user`: hydrate live account
  state and enforce account status.
- `require_permission` / `require_role`: compare requested authority with the
  live account and return `AuthorizationContext`, never raw claim authority.

## Inventory

| Consumer | Fields used | Classification | Authority / remediation |
| --- | --- | --- | --- |
| `features/auth/dependencies.py:get_token_claims` | `sub`, `sid`, `authz_revision` | Identity/session gate | Redis session plus token subject and revision; role/permission fields are not consulted. |
| `features/auth/dependencies.py:require_permission` | `authz_revision` | Privileged policy | Live `AuthAccount.permissions` and `role_revision`; returns `AuthorizationContext`. |
| `features/auth/dependencies.py:require_role` | `authz_revision` | Privileged policy | Live `AuthAccount.role` and `role_revision`; returns `AuthorizationContext`. |
| `features/users/router.py` | requester identity | Privileged admin mutations | Uses `AuthorizationContext.user_id` after live permission/role guard. |
| `features/billing/plans/router.py` | requester identity | Privileged admin mutations | Uses `AuthorizationContext.user_id` after live admin-role guard. |
| `features/billing/credits/router.py:grant_credit` | none | Privileged admin mutation | Live admin-role dependency; no token role/permission read. |
| `features/billing/credits/router.py:balance/history` | live user role | Ownership/admin decision | `CurrentVerifiedUser` is live account state; cross-user access requires live admin role. |
| `features/billing/payments/router.py:refund_payment` | requester identity | Ownership mutation | Uses `CurrentVerifiedUser` and subscription ownership; no role/permission claim authority. |
| `features/auth/router.py:session endpoints` | `sub`, `sid` | Session self-management | Identity and session selector only; Redis-backed `CurrentClaims` gate runs first. |
| `features/profile/router.py:change_password` | `sid` | Session continuity | Live verified account plus current session identifier; no privilege fields. |
| `features/agent_saul/router.py` | `sub` | Resource ownership | Session-validated subject is passed to transcript/session ownership checks. |
| `features/agent_saul/dependencies.py` | `sub`, `sid` | WebSocket admission | Redis session owner/expiry is authoritative; no role/permission decision. |
| `features/auth/websocket_security.py` | `sub`, `sid` | WebSocket identity/revocation | Live Redis session checks on handshake and frames; no privilege fields. |
| `features/documents/dependencies.py` | `sub` | Identity scoping | User ID only; no authorization decision from role/permissions. |
| `features/ingestion/dependencies.py` | `sub` | Identity scoping | User ID only; no authorization decision from role/permissions. |
| `mcp/server/middleware.py` | `sub` | Rate-limit identity | Subject keys throttling only; MCP catalog is read-only and has no role/permission gate. |
| `security/tokens.py` | role/permissions serialization | Token compatibility | Emits and parses wire fields; never decides authorization. |
| `features/auth/service.py` / `users/service.py` | live account role/permissions | Token issuance | Copies current account state into newly issued tokens; revision binds state. |

## Static guard

`scripts/check_claims_authority.py` parses `src/app` and rejects direct reads
of `role` or `permissions` from claim-like variables, including dictionary
access and common aliases. The unit test runs this check on every test suite.
If a future endpoint needs a privileged decision, add or reuse a live policy
dependency and update this matrix; do not add an allowlist entry for stale JWT
authority.

## Rollout and exceptions

This change is behavior-preserving for identity and ownership endpoints. The
admin/user/billing mutation handlers now receive a live authorization context,
which makes the trust boundary explicit. No database or Redis read is added to
identity-only handlers beyond the existing stateful session gate. Stateless
impersonation remains an explicit exception bounded by its short JWT expiry and
`impersonated_by` audit context.
