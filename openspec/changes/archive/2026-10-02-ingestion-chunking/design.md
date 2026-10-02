# Design — ingestion-chunking

## The schema already had the right shape; the code was misusing it

The single most useful finding behind this change is that **no new column is needed**.

`UnifiedChunk` (`src/app/features/documents/model.py:158-184`) already carries:

- `content` — intended to be the chunk's own text
- `preamble` — intended to hold the surrounding context
- `search_text` — a `Computed(persisted=True)` column concatenating `clause_type`, `preamble`, and
  `content`

And `retrieval_kb/reranker.py:56` already reranks over `f"{chunk.preamble}\n\n{chunk.chunk_text}"`.

So the two-column separation this change requires was designed and then not honoured.
`src/app/shared/rag/docling/chunker.py:250-254` writes `content=contextualized_text.strip()` and leaves
`preamble` empty — putting the heading boilerplate into the field citations quote from and leaving the
context field unused.

This changes the character of the work completely. It is not a schema migration with a backfill and a
retrieval-behaviour change; it is a two-line correction at the write site. And it has a property worth
stating explicitly because it is the argument that makes the change safe:

**Because `search_text` is generated from `preamble || content`, moving text between the two fields
leaves `search_text` byte-identical.** Lexical retrieval — BM25, trigram — sees no change at all. Only
what a citation quotes changes, which is the entire point.

That is also why task 5.2's Proof asserts `search_text` is *unchanged* for a fixture chunk. It is the
only assertion in this change that can catch getting the concatenation order or separator wrong.

## Why the version question is the most consequential defect in the cluster

For a general corpus, chunk identity without a version is a housekeeping problem. For a legal corpus it
is a correctness problem with a specific bad outcome: a clause that was amended out of a contract
remains retrievable and indistinguishable from the clause that replaced it. The system will cite
withdrawn language with the same confidence as current language.

`(document_id, document_version, chunk_index)` fixes it structurally rather than by convention. The
alternative — soft-deleting superseded chunks — was not seriously considered, because it makes prior
versions unreachable, and the ability to answer "what did this contract say last quarter" is a genuine
requirement of the domain.

`chunks.user_id` and its index `ix_chunks_user_document` already exist (`model.py:155`), so tenant
scoping is not part of this migration.

## Why the token-budget probe runs before the storage change

Task 5.1 precedes 5.2 for a reason that is easy to lose.

`chunker.py:251` computes the token count **after** `contextualize()`. So the current code has already
been budgeting against the contextualized length — which means the question "does every contextualized
chunk fit in the embedding window?" is answerable today, by a test, against the current tree. It is a
runnable probe, not a research question.

If the probe comes back negative — contextualized chunks overflow — then 5.2 must budget `max_tokens`
against the contextualized length rather than the bare chunk length, or chunks will start being
silently truncated at embedding time. Discovering that *after* changing what is stored would mean
diagnosing a truncation bug through two layers of change at once.

## Rejected shape — dependency drop first

The mechanical, satisfying order is: remove torch and `sentence-transformers` first (it gates image
size, it touches nothing behavioural), then do the chunking work.

Rejected, and the reason is specific to the OCR swap. Moving from EasyOCR to RapidOCR changes
**extraction quality on scanned documents**, which is the filing family — the hardest of the four. The
only honest proof that the swap did not degrade extraction is an ingestion run scored against a
retrieval baseline. And ingestion output is not trustworthy for scoring until 5.2 lands, because until
then the stored `content` is contextualized text and any citation-level comparison is measuring the
wrong field.

So the dependency work goes last, after the storage semantics are correct.

## The OCR swap is the least-defended step in this change

Recorded plainly rather than mitigated, because there is no available mitigation.

Tier-1 retrieval metrics measure OCR quality only indirectly: worse character recognition produces
worse chunk text, which produces worse embeddings and worse lexical matches, which eventually shows up
as lower recall. But the signal is attenuated and the golden set is deliberately small. A
faithfulness-style metric would measure it much more directly — and tier 2 does not exist yet, by
decision.

The task's Proof therefore requires recording the measured delta **whether it improved or regressed**,
rather than requiring an improvement. An honest recorded regression on the filing family is a usable
input to a later decision; a silent one is not.

## Why the embedder collapse writes no new requirement

`unified-embedding` already requires that *"Every embedding consumer resolves to the single path"*, and
the second embedder at `src/app/shared/rag/docling/embedder.py` hardcodes
`_PROVIDER_EMBEDDING_MODEL = "gemini-embedding-001"` while the configured model is different. There is
an in-code IOU at `:44-48` acknowledging it.

That means the behaviour is **specified and absent** — a violated requirement, not a missing one.
Writing an `## ADDED` requirement for it would create a second, near-duplicate statement of the same
rule in a second capability, and the two would drift. Task group 6 is a repair, and task 6.0's Proof
enforces that no requirement in this change claims that ground.

This is the general rule this cluster follows wherever a task group overlaps an archived requirement:
**restore, cite, and repair — do not re-specify.**

## The seam with `agentic-retrieval`

Task 7.2 removes `sentence-transformers`. The only thing importing it is
`retrieval_kb/reranker.py:9`'s `CrossEncoder`, which `agentic-retrieval` replaces with a hosted
reranker.

This change does not touch the reranker. Doing both halves in one change would put the reranker
replacement — a retrieval-quality decision — inside a chunking change, where nobody would look for it.
The cost is a hard ordering dependency between two changes, recorded in both.

## What this change deliberately does not touch

Retrieval SQL, fusion, the reranker, and graph lifecycle. `retrieval-sql` owns `repository.py`'s search
SQL and `fusion.py`. Task 8.2 edits `chunking.py` **write-side only**. And this change writes `chunks`,
never `clauses`.
