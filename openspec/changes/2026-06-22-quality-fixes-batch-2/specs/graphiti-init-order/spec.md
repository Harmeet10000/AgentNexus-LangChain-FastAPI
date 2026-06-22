# Graphiti Initialization Health Observability

## Scope

`lifespan.py` — Graphiti setup block (lines 150-162), `health_check.py` — check_graphiti/check_neo4j probes

## Problem

Two observable states in lifespan after startup:

| Neo4j driver | Graphiti | Observable health |
|---|---|---|
| OK | OK | All healthy |
| OK | FAIL | `/health`: neo4j=OK, graphiti=DEGRADED ✓ |
| FAIL | OK | `/health`: neo4j=DEGRADED, graphiti=OK ⚠️ inconsistent |
| FAIL | FAIL | `/health`: both DEGRADED ✓ |

State 3 (FAIL/OK) is inconsistent: the Neo4j driver is None, but Graphiti independently established its own connection (Graphiti manages its own driver internally). The `/health` endpoint reports neo4j as "not initialised" and graphiti as "ok" — which is accurate but confusing to an operator who expects Graphiti to depend on Neo4j.

## Solution

No runtime behaviour change. Add a lifespan-level warning when the two states are inconsistent:

```python
# After Graphiti setup (approx line 162):
neo4j_ok = getattr(app.state, "neo4j_driver", None) is not None
graphiti_ok = getattr(app.state, "graphiti", None) is not None
if not neo4j_ok and graphiti_ok:
    logger.warning("State inconsistency: Neo4j driver unavailable but Graphiti initialised independently")
elif neo4j_ok and not graphiti_ok:
    logger.warning("State inconsistency: Neo4j driver available but Graphiti not initialised")
```

This surfaces the inconsistency in structured logs at startup time, letting operators correlate with the health endpoint state.

## Verification

1. Manual: `docker compose stop neo4j`, restart app, check startup logs for "State inconsistency" warning
2. Manual: `/health` endpoint shows Graphiti=DEGRADED, Neo4j=DEGRADED (both or single depending on scenario)
