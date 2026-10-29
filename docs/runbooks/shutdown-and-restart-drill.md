# Runbook: shutdown, restart, and deploy drill

**Whoever owns this runbook owns the layers no framework configures.** FastAPI,
uvicorn, and Celery each manage their slice; everything that decides whether
their slice survives a stop — container kill patience, restart policy,
healthcheck, the SIGTERM→SIGKILL order — lives in deploy configs and this
document. If this file has never been executed end-to-end, treat the shutdown
chain as untested-prod code.

---

## 1. The shutdown budget chain (read before touching any timeout)

Every layer must cut off strictly AFTER the layer beneath it finishes draining:

| Layer | Source of truth | Budget |
|---|---|---|
| App drain (lifespan cleanup) | `settings.APP_SHUTDOWN_TIMEOUT` | 25s |
| Uvicorn connection cap | `settings.UVICORN_GRACEFUL_SHUTDOWN_TIMEOUT` | 30s |
| Container runtime | `docker-compose.prod.yml` `stop_grace_period` | 45s |
| (If k8s ever) `terminationGracePeriodSeconds` | manifest | 60s |

Invariant: each row's budget must be greater than the row above it. When you
change one, walk the whole table — the failure mode recurs at every scale,
which is how `timeout_graceful_shutdown` stayed unset for months: each layer
looked fine on its own; the chain had a hole.

---

## 2. The restart drill (run this to prove it)

This drill distinguishes "beautiful code" from "exercised behavior". Run it
after any change touching: lifespan shutdown paths, uvicorn settings, compose
files, uvicorn workers/fork mode, Celery init/capability code.

### Procedure

1. **Up and healthy**
   ```bash
   IMAGE_TAG=<pinned-tag> docker compose -f docker-compose.prod.yml up -d
   curl -fsS http://localhost:5000/health/ready  # record the pre-drill state
   ```

2. **Load** — generate steady traffic against `/health/ready` and one real
   route (an agent or document route), long enough to cross the stop window.

3. **Send TERM, never KILL, watch it exit itself**
   ```bash
   docker compose -f docker-compose.prod.yml stop --timeout 45 ai-service-1
   ```
   Expected: the container exits on uvicorn's graceful path inside ~30s —
   NOT SIGKILLed at 45s. (Locally, `docker kill --signal=TERM <id>` also
   works; `docker compose stop --timeout` varies the stop window.)

4. **Assertions — write the measured numbers into the PR**:
   - [ ] `"ExitCode": 0` in `docker inspect <id>` state
   - [ ] no `Shutdown drain timed out` line in logs
   - [ ] no `Application was running, but the shutdown never completed` lines
   - [ ] outbox drained: `Outbox relay drained` present
   - [ ] closes finished: `MCP upstream client sessions closed`,
         `WebSocket revocation listener stopped`,
         `Application shutdown complete` present
   - [ ] zero 5xx attributable to the restart window (compare with baseline)
   - [ ] worker containers: no rabbitmq reconnection storms post-restart,
         no in-flight task loss unless `task_acks_late` is confirmed OFF
         (check `src/app/worker/celery/app.py` before assuming either way)

5. **Healthcheck-rescue test** — proves `restart: unless-stopped` works alone:
   wedge the app loop (e.g. `py-spy` pause via a tiny script or an
   intentionally hung route under a feature flag), and verify the healthcheck
   flips `unhealthy` after ~90s (`interval 30s × retries 3`) and restarts the
   container. If a wedge mode exists that the liveness probe cannot see,
   extend `features/health` — do not accept the wedge.

### Failure interpretation matrix

| Observed | Meaning | Action |
|---|---|---|
| Exit 143 without finishing, then SIGKILL | drain exceeds budget | find the hung closer in `_shutdown_resources`; stack-dump before killing |
| `Shutdown drain timed out` present, exit within cap | forced-exit discipline working as built | inventory what leaked; OTEL must still flush (outer `finally`) |
| Exit 0, but NO OTEL flush line in logs | flush missing, not drain-hang | verify `shutdown_otel()` still in the lifespan outer `finally` |
| Workers lose tasks on restart | early acks + killed in-flight | confirm config; set `task_acks_late` if durability demands it |

---

## 3. Standing rules

- **Any change to lifespan, uvicorn settings, compose files, or health routes
  runs section 2 that week.** No "small config change" exemption — silent
  chain breakage is exactly why this doc exists.
- **Adding a new owned resource** (client, relay, model runtime) requires: a
  closer in `_shutdown_resources`, AND a paragraph here if it holds durable
  state (outbox pattern: drain-before-cancel).
- **Timeout knobs are edited in pairs**: if you move a number in
  `config/settings.py`, update section 1. If you move a number in compose,
  update section 1. The tables ARE the check.

## 4. Drill history

| Date | Change exercised | Outcome | Follow-ups |
|---|---|---|---|
| — | — | — | — |
