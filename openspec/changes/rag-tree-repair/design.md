# Design — rag-tree-repair

## The shape that was rejected

`git checkout -- src/app/shared/rag/document_processing/` followed by deleting the untracked
`docling/` tree. One command, instantly green, no repointing.

Rejected because the rename is deliberate and is meant to survive. Reverting it would trade a broken
tree for a lost intent, and the same rename would have to be redone — with the same seven repoint
sites — the next time anyone opened this area. The cost of finishing it is bounded and known; the
cost of relitigating it is not.

## Why the tracking step comes before the repointing

`src/app/shared/rag/docling/` is untracked. Every proof in this change and in the six that follow is
a command run against the working tree, and every one of them would pass identically whether or not
version control can see those files. The failure would surface only at the commit — as a repository
that does not import, with a green task list explaining that it does.

Tracking first converts an invisible failure into an impossible one. It costs one command.

## Why the specification baseline is repaired here and not in each change

Ten capabilities that later changes must amend exist only as deltas inside archived changes. The
mechanical consequence is narrow and absolute: **a `## MODIFIED Requirements` block has no base to
modify if the capability is absent from `openspec/specs/`.** Without the restore, `ingestion-chunking`
could not tighten `hierarchical-document-chunking`, and `retrieval-sql` could not amend
`legal-corpus-retrieval`; both would be forced to write `## ADDED` for behaviour that is already
required, which reads as new work and hides a regression.

Three placements were considered.

| Placement | Why not |
|---|---|
| Each change restores what it needs | Two changes need `hybrid-retrieval-ranking`; whichever runs second finds it present and its restore step becomes a no-op that still has to be written, reviewed, and proven |
| A separate spec-hygiene change | Correct in isolation, but it would sit between this change and every other one as a second blocker, for work that is ten file copies |
| Here, in the change that already owns "the tree is broken" | Chosen. The specification tree being out of sync with its own archive is the same class of defect as the source tree being out of sync with its own rename |

## Why ten and not eighty

Eighty archived capabilities are missing from `openspec/specs/`. Restoring all of them is a
repository-hygiene project with its own risk: it would put seventy capabilities into the live
baseline that no change in this cluster can defend, verify, or repair, and a baseline nobody is
accountable for is worse than an absent one — it invites `## MODIFIED` blocks against requirements
whose implementation status is unknown.

Ten is exactly the set the seven changes touch. It is defensible because each restored capability has
a change that owns it.

## Why restore rather than coin fresh capability names

Coining new names is mechanically easier: no restore step, no archive archaeology, every delta is a
clean `## ADDED`.

It was rejected for one reason. The overlapping requirements are **specified and violated**, not
absent — `src/app/shared/rag/docling/embedder.py:44-48` says so in an in-code deferral, and
`AgentToolBundle`'s docstring says so for agent tools. Writing `## ADDED: Every embedding consumer
resolves to the single path` next to an archived requirement that already says exactly that would
present a regression repair as a new feature, and would fork the lineage so that a future reader
cannot tell which of the two near-identical capabilities is authoritative.

Restoring makes the amendment legible: the requirement exists, it is not met, and a change is
repairing it.

## Why the migration environment is repaired here

The earlier draft of this plan scoped `alembic/env.py` out — measure, record the failure, hand it on.
That reading was overturned by two things. The user ruled the live database fully available (zero
data, zero users), which turns the head revision from an unanswerable question into a command that
must be made to run. And `ingestion-chunking`'s chunk-identity migration needs a real `down_revision`
that cannot be guessed: three sources disagree, and one of them names a revision that does not exist
on disk.

Repairing the environment module is in scope. Authoring or applying a revision is not — that boundary
is what keeps this change free of behaviour.

## What this change deliberately leaves broken

Three `PLC2701` private-import diagnostics, the four divergent embedding paths, every chunking
defect, and both retrieval SQL implementations. Each belongs to a named later change. Fixing any of
them here would make the recorded baseline measure a tree that no other change starts from.
