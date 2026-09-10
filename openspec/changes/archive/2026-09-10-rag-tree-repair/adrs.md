# ADRs — rag-tree-repair

## ADR-001 — The half-finished rename is completed, not reverted

**Status:** accepted.

**Context.** `src/app/shared/rag/document_processing/` is deleted-and-staged; its replacement
`src/app/shared/rag/docling/` is untracked and still imports the package it replaces. Reverting is one
command; completing is seven repoint sites plus a `pyproject.toml` edit.

**Decision.** Complete it.

**Consequences.** Seven sites and five per-file-ignore entries change in this repository. One
exemption is deleted outright rather than repointed, because `document_processing/ingest.py` has no
counterpart among the replacement package's seven modules. The alternative — reverting — would
discard a deliberate rename and guarantee the same work recurs.

## ADR-002 — `src/app/features/__init__.py` is restored empty and stays empty

**Status:** accepted.

**Context.** Twelve `INP001` diagnostics exist because the features directory is an implicit namespace
package. The file was deleted in `8e25352`, deliberately, to sever model-import and router-import
coupling.

**Decision.** Restore it as an empty file, and prove emptiness rather than merely presence.

**Consequences.** The lint diagnostics clear without reintroducing eager imports. The proof is not
`test -f` but `test ! -s` plus an assertion that importing the package loads no router module — a
non-empty `__init__.py` would pass a presence check while restoring the exact coupling the deletion
removed. If the emptiness route turns out to be unavailable, the fallback is a scoped per-file-ignore
with the reason recorded, never content in the file.

## ADR-003 — Ten archived capabilities are restored into the specification baseline

**Status:** accepted. **This ADR records an assumption, not a user ruling.**

**Context.** `openspec/specs/` holds 28 capabilities; the archives hold roughly 108. The September
2026 changes were moved into `archive/` without their deltas being applied — proven by
`typed-exception-handling`, whose archived delta adds four requirements, none of which appear among
the thirteen live ones. Ten of the missing capabilities are ones this cluster's changes must amend.

**Decision.** Restore exactly those ten, verbatim, in this change. Adopt the archived names rather
than coining near-duplicates. Restore none of the other seventy.

**Consequences.** Later changes can write `## MODIFIED` blocks with a real base. The overlaps become
visible as *repairs of violated requirements* rather than as new features — which is what they are:
`embedder.py:44-48` defers `unified-embedding`'s single-path requirement explicitly, and
`AgentToolBundle`'s docstring concedes that `agent-tool-registry`'s role-assignment requirement is
unmet.

Two archived capabilities are excluded because there is nothing to restore: `graphiti-init-order` and
`embedding-dimension-config` predate the requirement grammar and contain zero requirement blocks.

**Reversal cost.** Ten file deletions. If the preference is fresh capability names instead, nothing
else in the cluster has to change except the delta headers.

**Why this is an assumption.** The blocker was found and resolved during a stretch in which the
instruction was to proceed without asking further questions. The user has not seen this decision.
It is the one decision in this cluster that should be checked before the deltas that depend on it are
implemented.

## ADR-004 — The migration environment is repaired here; no revision is authored

**Status:** accepted.

**Context.** `src/alembic/env.py` breaks migration commands. Three sources disagree about the live
head: disk carries `0001`–`0017`, a handover claims `b3e7c41d92af` (matching nothing on disk), project
memory says `0004`. `ingestion-chunking` needs a real `down_revision`.

**Decision.** Repair `env.py` until `alembic current` executes and record the measured identifier.
Authoring, editing, or applying a revision is out of scope for this change.

**Consequences.** A downstream migration derives its parent from a measurement rather than a guess.
The scope line — tooling yes, revisions no — is what keeps this change free of runtime behaviour, so
its recorded gate baseline describes the tree every other change actually starts from.

## ADR-005 — Test outcomes are compared by summary counts, never by exit status

**Status:** accepted, and inherited from the archived ingestion-pipeline-unification tasks.

**Context.** The configured coverage floor is far above measured coverage, so a fully green test suite
still exits non-zero.

**Decision.** Every Proof in this cluster that concerns test outcomes compares summary pass and
failure counts against a recorded baseline file, and re-measures that baseline immediately before use.

**Consequences.** A green suite is never misread as a failure, and a newly-broken test is never hidden
behind a coverage failure that was already there. The cost is that no Proof in this cluster can be a
bare `pytest; echo $?`.
