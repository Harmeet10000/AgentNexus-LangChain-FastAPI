# Live infrastructure tests

The live test workflow runs two isolated jobs against pinned disposable services:

- Redis `7.4.2-alpine` runs the session-store Lua scripts, Redis refusal and
  timeout classification, client reconnect, authorization revision, and
  revocation-index behavior.
- PostgreSQL `16.6` runs `AsyncPostgresSaver.setup()`, checkpoint write/read,
  parent-linked resume, thread isolation, and pool cleanup.

The jobs run for pull requests targeting `main`, nightly at 03:17 UTC, and via
the `Live Infrastructure Tests` workflow-dispatch button. They use only local
service credentials and never require provider API keys or production secrets.

To run the tests locally with disposable services, use the same pinned images
and clean them up when the test finishes:

```bash
docker run -d --rm --name agentnexus-live-redis -p 6379:6379 \
  redis:7.4.2-alpine
docker run -d --rm --name agentnexus-live-postgres -p 5432:5432 \
  -e POSTGRES_USER=ci_langgraph \
  -e POSTGRES_PASSWORD=ci_langgraph_password \
  -e POSTGRES_DB=langgraph_ci \
  postgres:16.6
```

Wait for both health checks, then run:

```bash
LIVE_REDIS_URL=redis://127.0.0.1:6379/15 \
LANGGRAPH_POSTGRES_TEST_DSN=postgresql://ci_langgraph:ci_langgraph_password@127.0.0.1:5432/langgraph_ci \
uv run pytest tests/integration/features/auth/test_live_redis_session_store.py \
  tests/integration/features/agent_saul/test_langgraph_checkpointer_roundtrip.py \
  -m integration -v
```

Stop the disposable containers after the run:

```bash
docker stop agentnexus-live-redis agentnexus-live-postgres
```

Without those environment variables, the integration tests skip with an
explicit reason; ordinary local unit-test runs remain service-free. CI always
sets both variables and the service health checks fail before the test step if
either dependency is unavailable.

There is no production migration. The saver creates its own checkpoint tables
in the disposable CI database through `AsyncPostgresSaver.setup()`. To upgrade
a service image, change the pinned tag and run the workflow manually before
merging; to roll back, restore the previous tag. A known incompatibility must
be quarantined with a tracked issue and a narrowly scoped workflow change, not
by disabling the required job.
