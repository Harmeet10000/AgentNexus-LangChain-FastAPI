# Tenant migration rollout gate

Personal tenants are derived from the Mongo account id (`personal:<user_id>`).
PostgreSQL owns the tenant, membership, legal roles, role revision, and
`self_review_allowed`; Mongo fields are mirrors used for account hydration.

Run the Alembic upgrade before the application rollout:

```sh
uv run alembic upgrade head
```

Preview the Mongo-to-SQL backfill, then apply it in resumable batches (the
command can be safely rerun after a process or Mongo outage):

```sh
uv run python scripts/migrate_personal_tenants.py --dry-run
uv run python scripts/migrate_personal_tenants.py
```

The rollout gate must pass before enabling tenant-scoped writes. It scans every
active Mongo account, including accounts with no documents, and fails with exit
status 1 if the account has zero/multiple memberships, a missing personal
tenant, an invalid legal role, a pending SQL-committed Mongo mirror, or a
non-matching Mongo mirror:

```sh
uv run python scripts/migrate_personal_tenants.py --check
```

If a SQL transaction committed but the Mongo mirror write failed, the
membership is marked `mirror_sync_pending` with a bounded diagnostic. Retry the
pending rows and rerun the gate:

```sh
uv run python scripts/migrate_personal_tenants.py --reconcile --limit 250
uv run python scripts/migrate_personal_tenants.py --check
```

The reconciler never changes SQL roles or tenant policy. It clears a pending
marker only after the Mongo write succeeds and the SQL role revision is still
the revision that was read. A revision change leaves the row pending so the
new authority is retried instead of being overwritten by an old mirror.

`corpus_admin` is a tenant-scoped legal role. It is intentionally distinct from
the platform `admin` role; tenant membership administration requires the
requester to be a member of the exact route tenant and to hold `corpus_admin`,
unless the route explicitly opts into its platform-admin override. Platform
admin authority is never inferred from the legal-role alias.
