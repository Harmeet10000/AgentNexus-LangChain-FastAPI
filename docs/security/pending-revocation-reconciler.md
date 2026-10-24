# Pending session-revocation reconciliation

Role changes advance `User.role_revision` and set
`session_revocation_pending=true` before asking Redis to revoke sessions. The
request-path epoch check remains authoritative while the Celery beat task
eventually removes the old Redis sessions and conditionally clears the marker.
No worker failure can grant a stale session access or clear a newer marker.

## Operation

`auth.pending_revocation_reconcile` runs once per minute and processes at most
`AUTH_PENDING_REVOCATION_BATCH_SIZE` oldest pending users. Each user has a
short Redis lease (`auth:session_revocation:lease:<id>`), so multiple workers
share the queue without unbounded duplicate work. Retry state is namespaced
under `auth:session_revocation:retry:<id>` and contains only a revision,
attempt count, next-attempt timestamp, error category, and poison flag; it
never stores credentials or tokens.

The task uses bounded exponential backoff with jitter. A permanent failure or
the maximum attempt count leaves the Mongo marker set and marks the record
poisoned for operator attention. A newer role revision always wins: the
conditional Mongo update can clear only the exact revision processed, and the
existing Redis Lua fencing keeps the authorization epoch monotonic.

The report/log fields are `attempted`, `succeeded`, `stale_revision`,
`retryable_failure`, `permanent_failure`, `poisoned`, queue depth, and oldest
pending age. `auth_pending_revocation_oldest_alert` is emitted when the oldest
marker exceeds `AUTH_PENDING_REVOCATION_OLDEST_ALERT_SECONDS`; the default
convergence target is five minutes under healthy Mongo/Redis.

## Incident recovery

Preview a bounded pass before changing Redis:

```bash
uv run python -m app.worker.operations.reconcile_pending_revocations --dry-run
```

Apply it after confirming the target set:

```bash
uv run python -m app.worker.operations.reconcile_pending_revocations --apply
uv run python -m app.worker.operations.reconcile_pending_revocations --apply --user-id <id> --force
```

`--force` is deliberately explicit: it bypasses a retry delay or poison flag,
but never bypasses the revision-conditional clear. If Redis is unhealthy,
pause the worker task with `AUTH_PENDING_REVOCATION_ENABLED=false`; this only
pauses cleanup and does not weaken request authorization. Restore Redis, run a
dry-run, then re-enable the task. Re-running is idempotent and safe after a
worker crash or expired lease.

## Deployment and rollback

Deploy the worker code before enabling the schedule. The retry keys are
additive and can be ignored by older workers; the Mongo marker and revision
schema are already backward-compatible. On rollback, leave pending markers in
place: request-time epoch checks preserve security, and the next compatible
worker can replay the operation. Do not manually set a marker false unless
the exact revision's Redis invalidation has been verified.
