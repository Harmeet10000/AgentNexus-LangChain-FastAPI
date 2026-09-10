# Review — rag-tree-repair

Sections marked **measured** were established during planning and are reproduced here as the audit
record. Sections marked **pending** are filled during implementation by the task that names them.

---

## Measured — the specification tree was never synced (task 5.3)

Four independent measurements. Each is reproducible from the repository as it stands.

### 1. The size of the gap

`openspec/specs/` holds **28** capabilities. The archives under `openspec/changes/archive/` hold
roughly **108**. Eighty exist only as deltas inside archived changes.

### 2. The September 2026 archives were moved by hand, not applied

`openspec/changes/archive/2026-09-07-ingestion-pipeline-unification/specs/typed-exception-handling/spec.md`
adds four requirements:

- Embedding failures SHALL raise a typed failure rather than substitute a placeholder value
- Retry boundaries SHALL retry only named transient exception types
- Retry boundaries SHALL raise one typed transient failure and their callers SHALL be converted to
  catch it
- Retry boundaries SHALL NOT wrap whole graph nodes or catch control-flow exceptions

Live `openspec/specs/typed-exception-handling/spec.md` holds **thirteen** requirements, and **none of
them is any of those four**. That capability is present in the baseline only because of the earlier
`2026-07-22-noqa-exception-handling-migration` archive.

The delta *form* does not explain the gap: sampled capabilities on both sides of it — synced and
unsynced alike — use an identical `## Purpose` + `## ADDED Requirements` shape. The conclusion is that
the September batch was moved into `archive/` without `openspec archive` being run.

An earlier attempt at this test used `llm-injection` and was **inconclusive**, because that capability
also appears in `2026-06-16-quality-fixes-llm-datetime`, which could have supplied the live
requirements independently. `typed-exception-handling` was chosen precisely because its September
delta is disjoint from the live set.

### 3. The work was declared complete

Task tick counts in the archived `tasks.md` files:

| Archive | Ticked |
|---|---|
| `2026-09-07-cleanup-foundation` | 27 / 27 |
| `2026-09-07-error-handling-foundation` | 141 / 141 |
| `2026-09-07-documents-unified-schema` | 27 / 27 |
| `2026-09-07-ingestion-pipeline-unification` | 24 / 24 |
| `2026-09-07-agent-tools-unification` | 45 / 46 |
| `2026-09-07-cognee-agent-memory` | 45 / 46 |

### 4. But at least two archived requirements are knowingly unimplemented

`src/app/shared/rag/docling/embedder.py:40-48`, verbatim:

```python
# The provider model id this module passes to the API. Deliberately not read from
# ``settings.GEMINI_EMBEDDING_MODEL`` and — as of this task — deliberately no
# longer *named* like it. The two diverge today ("gemini-embedding-001" here
# against "gemini-embedding-2-preview" in configuration), and the old name made
# that divergence read as configuration. Reconciling the model, not just the
# width, belongs to B1, which collapses the four embedding paths into one.
_PROVIDER_EMBEDDING_MODEL = "gemini-embedding-001"
```

`_PROVIDER_EMBEDDING_MODEL` is referenced at lines 48, 137, 197, and 316. This is an in-code deferral
of `unified-embedding`'s requirement *"Every embedding consumer resolves to the single path"* to a
task named "B1" that was never done.

Corroborating: the `AgentToolBundle` docstring in `src/app/shared/rag/graphiti/registry.py` states
that agents are "currently built with empty tool lists in factory.py", which is
`agent-tool-registry`'s requirement *"Every agent role receives the tools assigned to it"*, also
unmet.

**Consequence for this cluster:** the overlapping task groups in later changes do **not** collapse to
verify-only. The behaviour is specified and absent, so it gets implemented — as a repair against a
restored requirement, not as a new feature.

---

## Measured — two capabilities are deliberately not restored (task 5.2)

`graphiti-init-order` (`2026-06-22-quality-fixes-batch-2`) and `embedding-dimension-config`
(`2026-06-22-quality-fixes-batch-2`, and earlier `2026-06-16-tech-debt-reliability`) predate the
requirement grammar. Their `spec.md` files use `## Scope` / `## Problem` / `## Solution` /
`## Verification` headings and contain **zero** `### Requirement:` blocks.

There is nothing to restore. Their content is treated as ordinary source material, not as a
specification baseline. Note that `embedding-dimension-config`'s June-22 delta was itself written
against a base that no longer exists live — a second, older instance of the same sync gap.

---

## Measured — a stale docstring that cost a planning round

`src/app/shared/rag/graphiti/registry.py:11-34` reads as a wiring recipe and is not one. It is a
module docstring, and it is stale in two ways:

- it names `build_tool_registry`; the real symbol is `build_tool_bundle` (`:92`)
- it names `app.state.saul_checkpointer`; the actual reader, `agent_saul/dependencies.py:49`, uses
  `app.state.langgraph_checkpointer`

`graph-lifecycle` wires against the reader and corrects the docstring. Recorded here because planning
off it produced a step for a function that does not exist, and the same trap is still armed for the
next reader.

---

## Pending — the live migration head (task 4.6)

Fill in after task 4.5 succeeds.

- Measured `alembic current`: `0016` (exit `0`, measured 2026-09-10 against the live Timescale instance; matches the `revision` literal at `src/alembic/versions/0016_add_statute_identity_index.py:19`; chain `0016 → 0017` confirmed via 0017's `down_revision`; single head `0017` via `alembic heads`, exit `0`).
- Which of the three claims it contradicts: handover (`b3e7c41d92af` matches no revision on disk — `grep -rl` over `src/alembic/versions/` empty) and project memory (`0004` contradicted — live stamp is `0016`); disk (`0001`–`0017`, seventeen revisions on disk) confirmed. The live database sits exactly one revision behind the disk head (`0016` vs `0017`); the probe was read-only (stamp unchanged, no revision authored, edited, or applied; `src/alembic/env.py` untouched — the hypothesised env defect is unproven, the earlier Errno 111 reads as environmental/transient).
- Consumed by: `ingestion-chunking` task 2.2, as the `down_revision` for the chunk-identity migration.

## Measured — proof-text divergences recorded at 4.5/4.6 close-out (task text untouched)

- Task 4.1 proof text expects `PLC2701` only; measured zero diagnostics (`All checks passed!`, byte-identical to `docs/relay/baseline-ruff-after.txt`). Better direction, not a regression; cause recorded in the 4.1–4.4 commit body (restoring `src/app/features/__init__.py` reclassifies the same-package private imports).
- Task 5.1 proof text expects `openspec validate --specs` exit `0`; measured exit `1` with `36 passed, 2 failed`, both failures pre-existing specs owned by no change (all ten restored specs pass). Neutral; recorded here and in the commit body, never in `tasks.md`.
- Task 4.5 proof text is satisfied with no source change: `src/alembic/env.py` is byte-identical to `main` and `alembic current` exits `0` today, so there was nothing to repair; the 4.4 failure (Errno 111) does not reproduce.

## Pending — features package fallback (task 3.2)

Fill in only if the empty-`__init__.py` route proves unavailable and a scoped `INP001` ignore is used
instead. Record what loaded eagerly and why.
