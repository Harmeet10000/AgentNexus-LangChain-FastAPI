# Redis session-index repair runbook

`auth:user_sessions:<user_id>` is a revocation index, not the authority for a
session. The authoritative record is `auth:session:<session_id>`. A valid
index must remain discoverable for at least the longest remaining lifetime of
every live session it references. The normal session-store Lua script enforces
that invariant for new writes; this bounded repair handles indexes created by
older versions or manually shortened in Redis.

## Key-format inventory

The current namespace has three relevant key families: `auth:session:<id>`
(JSON `SessionData` with `expires_at`), `auth:user_sessions:<user_id>` (a Redis
set of session IDs), and `auth:user_authz_revision:<user_id>` (the monotonic
authorization epoch). There is no historical schema-version marker in these
keys, so mixed-version deployments are identified by the dry-run counters and
the index TTL observed by the command. The repair only matches the exact first
two families under the configured prefix; it never scans unrelated Redis keys.
Roll out the normal writer invariant before applying repairs, then run one
dry-run per namespace and retain its JSON report as the migration checkpoint.

## Safe operating procedure

The command uses Redis `SCAN`, never `KEYS`, and processes a finite number of
indexes per invocation. It is namespaced by the exact `key_prefix` supplied to
the command. It returns both cursors so an operator can resume without losing
progress:

```bash
uv run python scripts/repair_auth_session_indexes.py \
  --key-prefix 'production:' --batch-limit 100 --member-limit 1000 \
  --orphan-limit 1000
```

The default is dry-run. Review `ttl_violations`, `malformed_records`,
`memberships_removed`, `orphan_sessions`, `failures`, and the returned cursors.
Apply a reviewed batch explicitly:

```bash
uv run python scripts/repair_auth_session_indexes.py \
  --key-prefix 'production:' --apply --cursor 1234 \
  --batch-limit 100 --member-limit 1000
```

The apply path is idempotent and concurrency-safe. A Lua compare-and-remove
checks the current session value before removing a membership or restoring an
orphan. If a login replaces a malformed, expired, or missing observation while
repair is running, the replacement is not removed. A persistent existing index
keeps its persistent TTL; only a missing index or an expiring index is assigned
the verified session lifetime. TTL extension is never applied to a session key.

## Policies and failure handling

- An expired or missing member is removed from the index; its session key is
  never deleted by this tool.
- A malformed record is reported and its membership is removed only when the
  exact malformed value is still present. The malformed session key itself is
  left for separate investigation.
- An orphaned live session (a valid, unexpired session key with no membership)
  is reported and, in apply mode, reattached to its user index with the same
  exact-value guard and TTL rule. This restores the membership that the normal
  session-store operation would have created; it does not create a session or
  widen session authority. Malformed or logically expired records remain
  report-only.
- An index larger than `--member-limit` is deferred unchanged. Increase the
  limit and rerun that namespace rather than repairing an incomplete view.
- A non-set index, Redis timeout, or other per-index failure is reported and
  does not stop later scan keys. Stop applying if failures persist.

There is no destructive rollback for a TTL extension: shortening the TTL after
the fact could revoke live sessions unexpectedly. To recover a removed stale
membership, reissue the normal authenticated session-store operation only for a
verified, unexpired `SessionData` record. Do not restore expired or malformed
records. If the repair process must be stopped, stop invoking `--apply`; the
next dry-run reports the remaining work.

The structured `auth_session_index_repair_completed` log includes the same
counters as the JSON report. Alert on non-zero `failures`, deferred indexes,
orphan growth, and repeated TTL violations. Run the command in small batches
with an external scheduler when Redis latency is sensitive.
