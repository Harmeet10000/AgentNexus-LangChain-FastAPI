# Verification — seven RAG-cluster OpenSpec proposals

Date: 2026-09-10 · Branch: `main` · Verifier leg, specs-only deliverable.

## Verdict

**RED** — narrowly, and on one thing only: **the scope claim does not hold.**

Every check that belongs to the seven spec proposals passes. All seven validate `--strict`.
The grammar audit is clean. The MODIFIED-block integrity check — the one place a defect
would cause silent data loss — is **perfect on both blocks**: zero dropped scenarios, zero
altered bullet text. Cross-change capability ownership is consistent.

What blocks is that the working tree contains a **source modification and a new test file
that the deliverable's own scope statement says do not exist**. Since the anchor leg commits
on this verdict, a GREEN here would sweep an unreviewed `src/` change into a
documentation-only commit.

---

## 1. Scope claim — **FAILS**

The claim under verification: *"No source file under `src/` was modified by this work."*

```
$ git status --porcelain
 D .github/skills/cognee-quickstart/SKILL.md
 D src/app/shared/rag/document_processing/__init__.py
 D src/app/shared/rag/document_processing/chunker.py
 D src/app/shared/rag/document_processing/docling_enhanced.py
 D src/app/shared/rag/document_processing/embedder.py
 D src/app/shared/rag/document_processing/entity_extractor.py
 D src/app/shared/rag/document_processing/ingest_v2.py
 D src/app/shared/rag/document_processing/models.py
 M src/mcp_core/client/settings.py            <-- NOT in the declared pre-existing set
 M tests/performance/todo.md
?? docs/performance-agent-prompts-2026-09-09.md
?? docs/relay/decisions-rag-cluster.md
?? docs/relay/plan-ingestion.md
?? docs/relay/plan-platform.md
?? docs/relay/plan-retrieval.md
?? docs/relay/research-rag-external.md
?? docs/relay/scout-rag-cluster.md
?? openspec/changes/agentic-retrieval/
?? openspec/changes/graph-lifecycle/
?? openspec/changes/ingestion-chunking/
?? openspec/changes/knowledge-stack/
?? openspec/changes/rag-eval-harness/
?? openspec/changes/rag-tree-repair/
?? openspec/changes/retrieval-sql/
?? src/app/shared/rag/docling/
?? tests/unit/test_mcp_client_settings.py     <-- NOT in the declared pre-existing set
```

### The two out-of-scope entries

**`src/mcp_core/client/settings.py`** — a real behavioural refactor, not a whitespace touch:

```diff
-import json
-from pydantic import ValidationError
+from pydantic import TypeAdapter, ValidationError
+_MCP_CONFIG_ADAPTER = TypeAdapter(list[MCPClientServerConfig])

     try:
-        payload = json.loads(raw)
-    except json.JSONDecodeError as exc:
-        msg = "MCP client server config JSON is invalid"
-        raise ValidationException(msg) from exc
-
-    if not isinstance(payload, list):
-        msg = "MCP client server config must be a JSON array"
-        raise ValidationException(msg)
-
-    try:
-        return [MCPClientServerConfig.model_validate(item) for item in payload]
+        return _MCP_CONFIG_ADAPTER.validate_json(raw)
     except ValidationError as exc:
```

This collapses three distinct error paths into one. The two `ValidationException` messages
`"MCP client server config JSON is invalid"` and `"...must be a JSON array"` no longer exist;
both conditions now surface as `"MCP client server config validation failed"`. That is an
observable contract change, whatever its merits.

**`tests/unit/test_mcp_client_settings.py`** — new, untracked, 5 tests, all passing:

```
$ uv run pytest -q tests/unit/test_mcp_client_settings.py
.....                                                                    [100%]
5 passed in 0.48s
```

### Timestamp evidence

```
23:07:41  src/mcp_core/client/settings.py
23:11:03  tests/unit/test_mcp_client_settings.py
--------  (first spec artifact written below)
23:25:11  openspec/changes/rag-tree-repair/proposal.md
23:29:04  openspec/changes/rag-eval-harness/proposal.md
23:32:07  openspec/changes/graph-lifecycle/proposal.md
23:39:12  openspec/changes/ingestion-chunking/proposal.md
23:43:22  openspec/changes/retrieval-sql/proposal.md
23:47:45  openspec/changes/agentic-retrieval/proposal.md
23:51:11  openspec/changes/knowledge-stack/proposal.md
```

Both land **before** the first spec artifact but **after** `docs/relay/research-rag-external.md`
(22:00) — inside the session window, outside the specs work proper.

**I cannot determine which agent authored them.** I am reporting the ground state, not
attributing intent. The orchestrator must decide whether these were an intentional separate
task or stray work; either way they must not ride along in a docs-only commit unreviewed.

Everything else outside `openspec/` matches the declared pre-existing set. The `docs/relay/*`
additions are prior relay legs' own artifacts (scout, research, decisions, three plans).

---

## 2. OpenSpec validation — **PASSES**

```
$ for c in rag-tree-repair rag-eval-harness graph-lifecycle ingestion-chunking \
           retrieval-sql agentic-retrieval knowledge-stack; do
    /home/harmeet/.bun/bin/openspec validate "$c" --strict; done

Change 'rag-tree-repair' is valid          exit=0
Change 'rag-eval-harness' is valid         exit=0
Change 'graph-lifecycle' is valid          exit=0
Change 'ingestion-chunking' is valid       exit=0
Change 'retrieval-sql' is valid            exit=0
Change 'agentic-retrieval' is valid        exit=0
Change 'knowledge-stack' is valid          exit=0
```

Seven for seven.

```
$ openspec validate --specs
Totals: 26 passed, 2 failed (28 items)
✗ spec/noqa-documentation
✗ spec/transactional-outbox
exit=1
```

**Both failures are pre-existing and are not this work's.** Raw detail:

```
$ openspec validate noqa-documentation --type spec --strict
✗ [ERROR] file: Spec must have a Purpose section. Missing required sections.

$ openspec validate transactional-outbox --type spec --strict
✗ [ERROR] requirements[0]: Requirement "Outbox Table Schema" must contain SHALL or MUST
✗ [ERROR] requirements[1]: Requirement "Outbox Helper" must contain SHALL or MUST
✗ [ERROR] requirements[2]: Requirement "Relay Process" must contain SHALL or MUST
✗ [ERROR] requirements[3]: Requirement "Relay Lifecycle" must contain SHALL or MUST
✗ [ERROR] requirements[4]: Requirement "Dead Letter" must contain SHALL or MUST
✗ [ERROR] requirements[5]: Requirement "Migration" must contain SHALL or MUST
```

Proof of pre-existence — both files are tracked and **clean**, last touched by an unrelated commit:

```
$ git status --porcelain openspec/specs/
(no output — working tree clean for the entire live spec tree)

$ git log --oneline -1 -- openspec/specs/noqa-documentation openspec/specs/transactional-outbox
94d1915 refactor: replace match/case Result unwrapping with isinstance+raise standard
```

`openspec list` confirms all seven register cleanly with zero tasks ticked (expected — planning
artifacts, no implementation yet):

```
knowledge-stack        0/18 tasks
agentic-retrieval      0/16 tasks
retrieval-sql          0/20 tasks
ingestion-chunking     0/24 tasks
graph-lifecycle        0/20 tasks
rag-eval-harness       0/15 tasks
rag-tree-repair        0/23 tasks
```

---

## 3. Project gates — red at baseline, **none attributable to this work**

`uv lock --check` first, per the standing hazard that a bare `uv sync` prunes `pytest-asyncio`:

```
$ uv lock --check
Resolved 593 packages in 129ms
exit=0
```

Lockfile in sync. **No `uv sync` was run.**

| Rung | Result | Count | Attributable to a markdown diff? |
|---|---|---|---|
| `ruff format --check src/` | fail | 4 files would reformat | No |
| `ruff check --no-cache src/` | fail | 24 errors | No |
| `ty check src/` | fail | 15 diagnostics | No |
| `pytest -q` (full) | fail | **collection interrupted, 14 errors** | No |
| `pytest -q` (collectable subset) | fail | 5 failed, 438 passed, 2 errors | No |
| `ast-grep scan src/` | pass | warnings only, exit 0 | No |

### Why none of it can be a markdown diff

The full suite **never runs** — it aborts during collection:

```
ERROR tests/unit/documents/test_hybrid_search_failure.py
ERROR tests/unit/documents/test_index_identity_gate.py
ERROR tests/unit/documents/test_segmentation_hybrid.py
ERROR tests/unit/features/documents/test_parser_offload_and_tables.py
ERROR tests/unit/shared/langgraph_layer/test_entity_canonicalisation.py
ERROR tests/unit/shared/langgraph_layer/test_ingestion_checkpoint_plumbing.py
ERROR tests/unit/shared/langgraph_layer/test_ingestion_degraded_identity.py
ERROR tests/unit/shared/langgraph_layer/test_ingestion_persistence_retarget.py
ERROR tests/unit/shared/langgraph_layer/test_ingestion_state_runtime.py
ERROR tests/unit/shared/langgraph_layer/test_kb_transient_boundary.py
ERROR tests/unit/shared/rag/test_chunker_tokenizer_cache.py
ERROR tests/unit/shared/rag/test_embedder_no_substitution.py
ERROR tests/unit/test_auth_documents_feature_errors.py
ERROR tests/unit/test_generation_with_cb.py
!!!!!!!!!!!!!!!!!!! Interrupted: 14 errors during collection !!!!!!!!!!!!!!!!!!!
39 deselected, 8 warnings, 14 errors in 55.05s
exit=2
```

All fourteen have **one identical cause**:

```
$ uv run pytest -q 2>&1 | grep -E "^E " | sort | uniq -c
     14 E   ModuleNotFoundError: No module named 'app.shared.rag.document_processing'
```

This is the mid-rename `document_processing/` → `docling/` breakage exactly as briefed. `ty`
sees the same thing from the other side:

```
error[unresolved-import]: Cannot resolve imported module `app.shared.rag.document_processing.docling_enhanced`
  --> src/app/shared/langgraph_layer/ingestion_kb/nodes.py:31:6
   |
31 | from app.shared.rag.document_processing.docling_enhanced import table_markdown
```

Ruff's 24 errors cluster in the untracked half-landed rename plus the features tree:

```
      3 src/app/shared/rag/docling/ingest_v2.py
      2 src/app/features/documents/service.py
      1 src/app/shared/rag/docling/{entity_extractor,embedder,docling_enhanced,chunker}.py
      1 each: src/app/features/*/__init__.py (11 packages)
      1 each: src/app/features/documents/{parser,dependencies,classification}.py
```

Format failures — `celery.py`, `classification.py`, `otel/instrument.py`, `otel/logs.py` — none
in the diff.

Excluding only the uncollectable modules, the suite does run:

```
$ uv run pytest -q --ignore=tests/unit/documents --ignore=tests/unit/features/documents \
    --ignore=tests/unit/shared/langgraph_layer --ignore=tests/unit/shared/rag \
    --ignore=tests/unit/test_auth_documents_feature_errors.py \
    --ignore=tests/unit/test_generation_with_cb.py

FAILED tests/unit/celery/test_typed_dispatch.py::test_registered_name_with_a_missing_field_is_refused_at_dispatch
FAILED tests/unit/celery/test_typed_dispatch.py::test_registered_name_with_an_unexpected_field_is_refused_at_dispatch
FAILED tests/unit/celery/test_typed_dispatch.py::test_a_matching_payload_still_reaches_the_send
FAILED tests/unit/celery/test_typed_dispatch.py::test_the_original_validation_detail_is_preserved
FAILED tests/unit/test_throwaway_graph_resilience.py::test_permanent_failure_is_not_retried
ERROR tests/unit/celery/test_task_registration.py::test_every_declared_task_name_has_a_registered_payload_model
ERROR tests/unit/celery/test_task_registration.py::test_every_declared_task_name_is_bound_on_the_task_application
5 failed, 438 passed, 39 deselected, 10 warnings, 2 errors in 45.14s
```

Six of those seven are **the same import again**:

```
_______ test_registered_name_with_a_missing_field_is_refused_at_dispatch _______
E   ModuleNotFoundError: No module named 'app.shared.rag.document_processing'
```

The seventh is an unrelated assertion:

```
____________________ test_permanent_failure_is_not_retried _____________________
E       unit.test_throwaway_graph_resilience.PermanentConfigError: misconfigured index
E       During task with name 'node_a' and id '7dc166ad-b869-7528-7af8-dcce42f23928'
```

**Correction to the brief's expectation:** the ~12 *websocket fixture-drift* failures did **not**
appear in this run. Today's residual set is 4 celery dispatch + 1 graph resilience + 2 celery
registration errors, and 6 of those 7 are the rename import. The websocket figure appears stale.
Flagging it because a future baseline comparison against "~12 websocket" would mislead.

`ast-grep` passes (exit 0) — output is `router-renders-result` **warnings** against
`src/app/features/crawler/router.py:196,209`, pre-existing and non-blocking. Rules present at
`.ast-grep/rules/` (7 files).

**Conclusion for this section:** every single gate failure traces to source files this
deliverable did not author. A markdown-only diff cannot produce a `ModuleNotFoundError`.
Gates are **no worse than baseline**, which is the correct criterion here.

---

## 4. Artifact grammar audit — **CLEAN**

Scripted audit across all spec files in the seven change directories, checking for:
empty requirement bodies before the first scenario; near-miss thin bodies (<40 chars);
`Scenario:` at any depth other than exactly four hashes; scenarios lacking `- **WHEN**` or
`- **THEN**`; requirement bodies missing SHALL/MUST.

```
GRAMMAR ISSUES: 0
```

Zero findings, including zero near-misses. Artifact completeness — all seven carry the six
required artifacts, plus two extras (`.openspec.yaml`, `README.md`) consistently across all seven:

```
['.openspec.yaml', 'README.md', 'adrs.md', 'design.md', 'proposal.md', 'review.md',
 'specs/<capability>/spec.md', 'tasks.md']
```

---

## 5. MODIFIED-block integrity — **CLEAN, both blocks**

The highest-value check. A MODIFIED block replaces its requirement wholesale on archive, so any
original scenario not copied forward is silently deleted and `--strict` cannot see the loss.

First, confirming each requirement has exactly **one** archive source (no later archived change
supersedes the one being diffed against):

```
== Every document kind is chunked structure-aware
    openspec/changes/archive/2026-09-07-ingestion-pipeline-unification/specs/hierarchical-document-chunking/spec.md
== Ranked retrieval and fusion have a single implementation
    openspec/changes/archive/2026-09-07-agent-tools-unification/specs/legal-corpus-retrieval/spec.md
```

One each. The diffs are therefore against the authoritative originals.

### 5a. `ingestion-chunking` → *Every document kind is chunked structure-aware*

| | Archive (3) | MODIFIED (7) |
|---|---|---|
| Legal document chunks carry their heading path | present | **preserved verbatim** |
| Peer sections are merged within the bound | present | **preserved verbatim** |
| Clause boundaries are respected for legal documents | present | **preserved verbatim** |
| Each legal family resolves a distinct policy | — | new |
| The resolved policy is recorded on the chunk | — | new |
| An unclassified document still chunks | — | new |
| Policy resolution performs no input or output | — | new |

```
>> DROPPED (in archive, absent from MODIFIED): none
>> NEW in MODIFIED: 4 scenarios
>> ALTERED scenario: (none reported — byte-identical bullet lists)
```

Requirement body is a strict **append**, original sentence retained word-for-word:

- archive: `"...Splitting a document by blank-line pattern matching SHALL NOT be used for any document kind."`
- modified: `"...Splitting a document by blank-line pattern matching SHALL NOT be used for any document kind. Chunking SHALL further resolve a chunk policy from the document's classified kind, such that each of contract, statute, judgment, and filing resolves a policy distinct from the others, and the identity of the resolved policy SHALL be recorded on every emitted chunk. Policy resolution SHALL be a pure function of the classified kind and SHALL perform no input or output."`

### 5b. `retrieval-sql` → *Ranked retrieval and fusion have a single implementation*

| | Archive (2) | MODIFIED (5) |
|---|---|---|
| Precedent search uses the shared ranked retrieval path | present | **preserved verbatim** |
| Combining ranked lists uses the shared fusion | present | **preserved verbatim** |
| Two doors return one answer | — | new |
| A repository method is not an exemption | — | new |
| Fusion constants have one home | — | new |

```
>> DROPPED (in archive, absent from MODIFIED): none
>> NEW in MODIFIED: 3 scenarios
>> ALTERED scenario: (none reported — byte-identical bullet lists)
```

Body is again a strict append:

- archive: `"...An agent tool SHALL NOT introduce a second ranking or fusion implementation."`
- modified: `"...An agent tool SHALL NOT introduce a second ranking or fusion implementation. No caller SHALL introduce a second ranking or fusion implementation, whether in an agent tool, a repository method, a service, or a graph node. Two callers issuing the same query with the same filters SHALL receive the same ranked identifiers in the same order. The constants governing fusion SHALL have exactly one definition, and SHALL NOT be restated as literals at a call site."`

**Result: zero data loss on either block.** Every original scenario carried forward with
identical WHEN/THEN text; every original body sentence retained. This is the failure mode the
brief was most concerned about, and it is absent.

---

## 6. Cross-change consistency — **CONSISTENT**

| Change | NEW capability | MODIFIED capability | Proposal names it? |
|---|---|---|---|
| rag-tree-repair | `source-tree-integrity` | — | yes |
| rag-eval-harness | `retrieval-evaluation` | — (explicit "None") | yes |
| graph-lifecycle | `compiled-graph-lifecycle` | — (explicit "None") | yes |
| ingestion-chunking | `legal-document-chunking` | `hierarchical-document-chunking` | yes, both |
| retrieval-sql | `postgres-hybrid-retrieval` | `legal-corpus-retrieval` | yes, both |
| agentic-retrieval | `agentic-retrieval-loop` | — (explicit "None") | yes |
| knowledge-stack | `knowledge-extraction-stack` | — (explicit "None") | yes |

Exactly one NEW capability per change. `ingestion-chunking` and `retrieval-sql` are the only two
with `## MODIFIED Requirements`, as specified. No spec directory contains a capability its
proposal does not name. The four changes with no modifications say so explicitly rather than
omitting the section, and distinguish *citing* a capability from *amending* it.

**Note on heading vocabulary:** proposals use `## Capabilities` containing bold
`**New Capabilities**` / `**Modified Capabilities**` sub-labels, not a literal
`## New Capabilities` heading. Uniform across all seven. Not a defect — recording it so a future
automated audit keyed on the literal heading does not report seven false positives.

---

## 7. Observed ordering dependency — not a defect, but load-bearing

Both MODIFIED targets are **absent from the live spec tree**:

```
$ ls openspec/specs/ | grep -E 'hierarchical-document-chunking|legal-corpus-retrieval'
(no output)

hierarchical-document-chunking: NO live spec under openspec/specs/
legal-corpus-retrieval:         NO live spec under openspec/specs/
```

They exist only inside archived changes. So archiving `ingestion-chunking` or `retrieval-sql`
today would apply a MODIFIED against a base that is not there.

**This is anticipated and specified**, not overlooked. `rag-tree-repair` carries a requirement
that makes restoring the baseline a precondition — at
`openspec/changes/rag-tree-repair/specs/source-tree-integrity/spec.md`:

```
### Requirement: Every capability a change amends is present in the specification baseline

Capabilities whose requirements this cluster amends SHALL exist under the live specification tree
before any change declares a modification against them.

#### Scenario: A modification has a base to modify
- **WHEN** a change proposes a modification to a requirement
- **THEN** that requirement SHALL already be present in the live specification baseline

#### Scenario: A restored capability is reproduced without alteration
- **WHEN** a capability is restored from an archived change
- **THEN** its requirement and scenario text SHALL be reproduced without alteration, and its
  delta operation header SHALL be replaced by a plain requirements header
```

The dependency is real and hard: **`rag-tree-repair` must be implemented and archived before
either `ingestion-chunking` or `retrieval-sql` can be archived.** Recording it because
`validate --strict` passes all seven and will not warn anyone about it.

---

## 8. Untested

Nothing to flag on the deliverable — it introduces no executable behaviour, so there is no
uncovered behaviour to name. `tasks.md` sit at 0 ticks across all seven, which is correct for
planning artifacts.

The genuinely untested surface is the **out-of-scope** change: `src/mcp_core/client/settings.py`
has 5 new passing tests, but I did not verify whether they cover the two error messages the
refactor **deleted** (`"MCP client server config JSON is invalid"` and
`"MCP client server config must be a JSON array"`). If any caller or test elsewhere asserts on
those strings, it would now break. Not investigated — out of scope for this leg, and I do not fix.

---

## What must happen before the anchor leg commits

1. **Decide on `src/mcp_core/client/settings.py` + `tests/unit/test_mcp_client_settings.py`.**
   Either commit them separately with their own review, or stash them. They must not ride along
   in a docs-only commit.
2. Re-run scope verification after that decision. Everything else is already green.

No fixes were applied. No files outside `docs/relay/verify-specs.md` were written.
