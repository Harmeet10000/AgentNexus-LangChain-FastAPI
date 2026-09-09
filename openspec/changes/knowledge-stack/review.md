# Review — knowledge-stack

Sections marked **accepted cost** are known limitations recorded deliberately. Sections marked
**pending** are filled during implementation by the task that names them.

---

## Measured 2026-09-09 — both packages have exactly zero runtime call sites

Verified by grepping the whole of `src/` for each package while excluding the package itself:

**Extraction (`shared/rag/langextract/`)** — one reference outside the package:

```
src/app/shared/rag/graphiti/write_clause_episodes.py:42:
    from app.shared.rag.langextract.langextract_to_graph import GraphIngestionContext
```

and line 40 of that file is `if TYPE_CHECKING:`. So the single edge into a complete extraction module is
a **type-checking-only import**. The type checker sees a dependency; the interpreter never executes one.
That is the exact shape that makes a disconnected module read as wired to anyone browsing, and it is why
task 5.3's Proof asserts the guard no longer covers that import rather than merely asserting the module
is imported somewhere.

**External structural client (`shared/rag/pageindex/`)** — references outside the package are the
re-export at `shared/rag/__init__.py:3-18` (six exported symbols), the commented construction at
`lifespan.py:538`, and configuration. No call site.

Two finished modules, zero runtime callers between them. This is not a half-built feature; it is built
and disconnected, and the two halves want opposite treatment — one connected, one deleted.

---

## Measured 2026-09-09 — the retirement Proof was incomplete, and the measurement caught it

Task 3.2 as first written proved the package gone with `rg -n 'pageindex' src/ tests/; test $? -eq 1`.
That Proof **could not have passed**, because the configuration surface survives the package:

```
src/app/config/settings.py:333:    # --- PageIndex Configuration ---
src/app/config/settings.py:334:    PAGEINDEX_API_KEY: SecretStr = Field(default=SecretStr(""))
docs-site/configuration/environment-variables.mdx:156: | `PAGEINDEX_API_KEY` | ...
```

Task 3.3 was added to remove both, and its Proof widens the grep to `docs-site/` and adds a settings
construction check, because a removed field with a surviving reader is a startup failure rather than a
lint finding.

Recorded because of what it says about the general shape: a package retirement is never just the
package. Deleting the code and leaving the setting produces a configuration surface for a capability
that no longer exists — the most inviting possible target for someone later reconnecting a dependency
this change is deliberately removing.

---

## Measured 2026-09-09 — a live provider credential, and where it is not

`.env.development:201` holds a real-looking value for `PAGEINDEX_API_KEY`. Checked:

```
$ git ls-files --error-unmatch .env.development
error: pathspec '.env.development' did not match any file(s) known to git
$ git check-ignore -v .env.development
.gitignore:120:.env.*	.env.development
```

The file is gitignored and untracked, so **the key never entered git history** — this is not an exposure
incident and needs no history rewrite. What it is, once the client is retired, is a dead credential
sitting in a local file. Task 3.4 records removing it locally and revoking it at the provider as an
operator action, because no code change will do it and nothing will later prompt it.

---

## Measured 2026-09-09 — the extraction provider needs no new configuration

`LANGEXTRACT_API_KEY` already exists at `settings.py:252` with the default
`SecretStr("empty-langextract-api-key")`, and `settings.py:24` registers that string in the
placeholder-value list.

Two consequences, both folded into task 5.1. The extraction stage adds **no** settings field. And the
absent-provider case is not "the key is empty" — it is "the key equals a known placeholder", which is a
different predicate, and one a naive `if not key` check gets wrong. A configured-looking placeholder that
reaches the provider produces an authentication error at ingest time instead of clean degradation, so
5.1's Proof now asserts the placeholder is treated as unconfigured.

---

## The finding that shapes the change — and the measurement still outstanding

Whether the parser's structural tree is **persisted** is not established. Task 1.2 settles it, and the
answer changes what this change is:

| 1.2 | Group 2 | This change becomes |
|---|---|---|
| persisted | no-op, recorded via 2.3 | wiring |
| not persisted | storage column plus migration | a storage build |

The proposal states the conditional rather than betting on a branch. Task 2.3 exists so that an unticked
group 2 is never ambiguous between *not needed* and *not done* — an unrecorded no-op looks identical to
abandoned work six months later.

---

## Accepted cost — the navigator is less capable than the service it replaces

The retired client reasons over document structure with whatever sophistication its provider has built.
The replacement is a local pure function over a stored tree.

Accepted, because the trade is not primarily cost or latency: wiring the client sends **document content
to a third party**, and for a legal corpus that is the decision, not a line item. The residual risk is
real — a locally written navigator will handle fewer structural shapes well, and its failures will be
quiet ones (a plausible but wrong section path). Task 4.1's fixture test pins the behaviour that is
claimed; nothing pins the behaviour that was never claimed. A named follow-up: as tier-1 retrieval
metrics accumulate under `rag-eval-harness`, add structural-navigation queries to the golden set so the
navigator's quality becomes measurable rather than asserted.

---

## Accepted cost — extraction adds a per-document provider call to the ingestion path

Ingestion gains a network dependency it did not have, on the critical path of every upload.

Accepted, with the failure semantics doing the work: failure is a typed value, ingestion completes, and
the document is flagged. The residual cost is latency — every ingestion now waits on a provider — and
this change does not make the stage optional or batched. Recorded rather than solved, because making it
asynchronous-and-later means reconciling entities against chunks after the fact, which is the specific
thing ADR-002 rejects.

---

## Note — why an empty extraction and a failed extraction are separated so insistently

Recorded because the distinction looks pedantic and is the most consequential requirement in the change.

Merge them and a document ingested during a provider outage becomes **permanently** indistinguishable
from a document that genuinely contained no clause entities. There is no later signal: the document is
stored, retrieval works, nothing errors, and the graph is simply missing everything that document should
have contributed. Nothing would ever prompt a re-run, because nothing looks wrong.

That is why task 5.2 carries two tests, the second asserting a *negative* — an empty success leaves the
flag unset — and why the spec spends a scenario on it.

---

## Note — this change specifies nothing about canonicalisation, by construction

`graph-entity-canonicalisation` already owns idempotent canonical graph writes. This change's
re-ingestion scenario states what it must *satisfy*; it does not redefine *how*. Task 5.4's Proof greps
this change's own spec directory for canonicalisation vocabulary to keep it that way.

The same discipline as `ingestion-chunking` and its embedder: **restore, cite, and repair rather than
re-specify.** Two definitions of one rule in two capabilities diverge the first time either is edited,
and the divergence is invisible until something depends on the difference.

---

## Pending — the tree-persistence answer (task 1.2)

- `rg -n 'DoclingDocument' src/app/features/documents/` output: _pending_
- `UnifiedDocument.metadata_` contents on a stored row: _pending_
- **Answer, yes or no:** _pending_
- Branch taken, and group 2's disposition: _pending_ (recorded here by task 2.3 either way)

---

## Pending — the ingestion stage boundary (task 1.3)

- Function the extraction stage will precede: _pending_
- File and line: _pending_
- Confirmation the boundary is a stage edge, not mid-stage: _pending_

---

## Pending — the extraction path is reached, not merely importable (task 6.2)

- Stub provider call count during a fixture ingestion: _pending_ (expected: exactly one)
- Confirmation the call preceded chunking: _pending_

Every other Proof in group 5 can pass against a stage that is wired but never invoked, which is the
failure mode this change starts from. This is the one that rules it out.

---

## Pending — retirement completeness (tasks 3.1–3.4)

- `rg -in 'pageindex' src/ tests/ docs-site/` after removal: _pending_ (expected: no matches)
- `app.config.settings.get_settings()` after the field is removed: _pending_ (expected: exit `0`)
- `.env.development` still untracked: _pending_
- Provider-side revocation recorded: _pending_
