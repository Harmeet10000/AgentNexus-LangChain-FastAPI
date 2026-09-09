# Tasks — knowledge-stack

## How to read the Proofs

1. **Never use a test-process exit code as a Proof of test outcome.** Compare **summary pass and
   failure counts** against the baseline captured in task 1.1.
2. **Never prove a schema fact by rendering migrations offline.** Prove it against a database built
   from `alembic upgrade head`.
3. **Never make a Proof depend on a durable outbound event firing.** The outbox tables do not exist.
4. **The live database may be used** — zero data, zero users, ruled available.
5. **No Proof in this change may cite an answer-quality metric.** Only tier-1 retrieval metrics exist.
6. **Task group 2's existence is conditional on task 1.2's measurement.** Do not design it before that
   measurement is taken.

**Blocked by `rag-tree-repair`** (the tree does not import today) and **`ingestion-chunking`** (this
change inserts a stage into the ingestion path that change restructures).

## 1 Baseline, and the question that shapes this change

- [ ] 1.1 Capture `uv lock --check`, `uv run ruff check --no-cache src/ 2>&1 | tail -1`,
  `uv run pytest -q 2>&1 | tail -1`, and `uv run python -c "import app.main"` into `baseline.md`.
  **Proof:** the file exists and the import exits `0`.
- [ ] 1.2 **Determine whether the parser's structural tree is persisted at all.** This is not
  established and will not be assumed.
  **Proof:** `rg -n 'DoclingDocument' src/app/features/documents/` plus an inspection of
  `UnifiedDocument.metadata_` against a stored row; record **yes or no** in `baseline.md`. A *no* makes
  task group 2 mandatory and converts this change from wiring into a storage build; a *yes* makes group
  2 a no-op and group 4 a navigator over data that already exists.
- [ ] 1.3 Record where the ingestion pipeline's stage boundaries sit after `ingestion-chunking` lands,
  so the extraction stage is inserted at a boundary rather than into the middle of a stage.
  **Proof:** `baseline.md` names the function and line the extraction stage will precede.

## 2 Persist the tree — conditional on 1.2 returning *no*

- [ ] 2.1 Store the serialised structural tree on the document row.
  **Proof:** a round-trip test reconstructs the tree from a stored document and asserts a known node
  path survives serialisation and deserialisation.
- [ ] 2.2 Migration for the storage column, with `down_revision` set from the head recorded by
  `ingestion-chunking`, never guessed.
  **Proof:** `uv run alembic check` reports no pending autogenerate diff.
- [ ] 2.3 If 1.2 returned *yes*, record that explicitly rather than leaving the group silently unticked.
  **Proof:** `review.md` states which branch was taken and why, so a later reader does not read an
  unticked group as unfinished work.

## 3 Retire the external structural surface

- [ ] 3.1 Remove the re-export at `src/app/shared/rag/__init__.py:3` and delete the commented
  construction at `src/app/lifecycle/lifespan.py:538`. `graph-lifecycle` deliberately left that line in
  place for this change to remove.
  **Proof:**
  `uv run python -c "import app.shared.rag as r; assert not [n for n in dir(r) if 'ageindex' in n.lower()]"`
  exits `0`; and `rg -n "pageindex" src/app/lifecycle/lifespan.py; test $? -eq 1` → exit `0`.
- [ ] 3.2 Delete `src/app/shared/rag/pageindex/`.
  **Proof:** `uv run python -c "import app.main"` exits `0`; `uv run ruff check --no-cache src/` line
  count ≤ the 1.1 baseline; `rg -in 'pageindex' src/ tests/; test $? -eq 1` → exit `0`.
- [ ] 3.3 Remove the client's **configuration surface** — the settings field and its section comment at
  `src/app/config/settings.py:333-334`, and the row at
  `docs-site/configuration/environment-variables.mdx:156`. Measured 2026-09-09: these are the only
  references outside the package itself, and task 3.2's Proof cannot pass while the settings field
  survives.
  **Proof:** `rg -in 'pageindex' src/ tests/ docs-site/; test $? -eq 1` → exit `0`; and
  `uv run python -c "import app.config.settings as s; s.get_settings()"` exits `0`, confirming no reader
  depended on the removed field.
- [ ] 3.4 Note in `review.md` that `PAGEINDEX_API_KEY` holds a live provider credential in the local
  `.env.development`, which is gitignored and untracked — measured 2026-09-09, so the key never entered
  git history. Retiring the client makes it dead; it should be removed locally and revoked at the
  provider.
  **Proof:** `git ls-files --error-unmatch .env.development` still fails (the file stays untracked), and
  `review.md` records the revocation as an operator action rather than a code change.

## 4 Tree-reasoning retrieval

- [ ] 4.1 A **pure** navigator over the stored tree returning node paths — a function from a tree and a
  query to paths, performing no I/O.
  **Proof:** a unit test over a fixture tree returns the expected section path, with no database and no
  parser constructed.
- [ ] 4.2 Expose it as a retrieval branch **through the repository and the service layer**, not from a
  route handler.
  **Proof:** a service-level test exercises the branch; and `rg -n 'repository' src/app/features/documents/router.py; test $? -eq 1`
  → exit `0`, confirming no new direct repository import reached the router.

## 5 The extraction stage

- [ ] 5.1 An asynchronous extraction service reading its provider key from settings, with its client
  constructed per the lifespan convention. The setting already exists — measured 2026-09-09,
  `LANGEXTRACT_API_KEY` at `src/app/config/settings.py:252`, with a placeholder default registered in the
  empty-value list at `:24` — so no new configuration field is added, and the placeholder default is what
  the degradation path in 5.2 must recognise as *absent*.
  **Proof:** `rg -n 'langextract' src/app/lifecycle/lifespan.py` hits; ruff's async rule set is clean
  over the new module; and a test asserts the placeholder default is treated as an unconfigured provider
  rather than a usable key.
- [ ] 5.2 Insert the stage **before chunking**, at the boundary recorded in 1.3, with failure
  represented as a typed value rather than a caught-and-discarded exception.
  **Proof:** a test injecting a failing provider asserts ingestion **succeeds** and the
  extraction-incomplete flag is set; and a second test asserts a provider that succeeds with zero
  entities leaves the flag **unset** — because yielding nothing and failing are distinct outcomes.
- [ ] 5.3 Feed `src/app/shared/rag/graphiti/write_clause_episodes.py`, turning its `TYPE_CHECKING`
  import at `:42` into a real one.
  **Proof:** a re-ingestion test creates no duplicate episodes; and `rg -n 'TYPE_CHECKING' src/app/shared/rag/graphiti/write_clause_episodes.py`
  no longer guards the extraction import.
- [ ] 5.4 Confirm this change specifies no canonicalisation semantics of its own.
  **Proof:** `rg -in 'canonical|deduplicat' openspec/changes/knowledge-stack/specs/; test $? -eq 1` →
  exit `0`, except where a scenario defers to the existing writer. `graph-entity-canonicalisation`
  already owns idempotent canonical writes; this change satisfies it rather than restating it.

## 6 Close out

- [ ] 6.1 **Proof:** `openspec validate knowledge-stack --strict` exits `0`;
  `uv run ruff check --no-cache src/`, `uv run ty check src/`, and `uv run pytest -q` are each equal to
  or better than the 1.1 baseline.
- [ ] 6.2 Confirm the extraction path is actually reached — not merely importable.
  **Proof:** an ingestion of a fixture document with a stub provider asserts the provider was called
  exactly once, before chunking. Every other Proof in group 5 can pass against a stage that is wired
  but never invoked.
