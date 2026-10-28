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
status 1 if the account has no SQL memberships, is missing its required
personal membership, references a missing tenant, has an invalid legal role, a
pending SQL-committed Mongo mirror, or a Mongo tenant mirror that is not one of
the account's SQL memberships. Multiple valid team memberships are expected
and are not a readiness failure; requests with more than one membership must
select their tenant explicitly:

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

## Graphiti canonical-group rollout

Graphiti reads now use canonical tenant/document/version/contract group IDs.
Existing episodes written under the historical document-only group are not
implicitly visible through the new scope, so complete the graph backfill before
enabling scoped graph evidence in production:

1. Pause graph evidence consumers or mark the affected documents unavailable.
2. Requeue each affected document version through the normal ingestion workflow
   so clause episodes and relationship episodes are rewritten into canonical
   groups. Do not copy legacy group IDs into the new scope.
3. Verify the re-ingestion reports `graphiti_verified` for every expected chunk
   and run the tenant-scoped graph smoke checks before resuming consumers.

Until that backfill is complete, an authorized document with no canonical
partition is intentionally reported as `scope_unavailable`; it must not fall
back to a broad or legacy group because that would cross the tenant/version
boundary.
