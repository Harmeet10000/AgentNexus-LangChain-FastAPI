# Architecture Decision Records

Two decisions in this change outlive it. Everything else is change-local and is recorded in `design.md`; no ADR is
written for those, because missing a non-decision is not a defect.

The schema contract this change's persistence writes against is **not** recorded here — it is the following
change's ADR, accepted before this change implements persistence, and this change depends on it rather than
authoring it.

---

## ADR-1 — The unified embedding contract

### Status

Proposed

### Context

Four embedding paths exist, with two mutually incompatible dimensions. One builds a fresh provider client per call
with a hard-coded width and no cache. One imports that same client, so two features are already a single path. One
duck-types the embedding callable through three candidate method names, embeds a single text per call, caches in the
shared cache, normalises, and retries — but passes **no task type**, so its stored vectors are asymmetric with the
query side, which is a silent relevance defect rather than a style issue. The fourth uses the provider's raw
software development kit, an in-process cache, and declares a width of 1536 against vector columns declared at 768 —
an insert that would fail on data width if it had ever run.

Two later changes consume whatever this settles. One of them re-litigates the same decision from the other end: its
memory subsystem defaults to a third provider at 3072 dimensions against the same 768 columns. Deciding this once,
now, while **no vectors exist anywhere**, is the difference between a column definition and a re-embedding campaign.

### Decision

One embedding path, and it is a contract with five clauses:

1. **Provider and client lifetime.** One provider, one client constructed once per process and reused. Not one per
   call.
2. **Dimension.** Read from a single configuration value, at process start. Every vector column used for retrieval
   or memory declares that same width. No dimension appears as a literal outside historical schema revisions.
   Changing the value is a re-embedding campaign, not a configuration change applied to existing data; where the
   configured width disagrees with stored vectors, new writes are refused and re-embedding is reported as required.
3. **Task type.** Declared explicitly on every request, distinguishing a search query from a stored document, and
   carried through to the provider. No request may omit it.
4. **Cache.** One cross-process cache keyed by a digest of the text together with the model and the task type, with
   a documented expiry. Exactly one cache implementation exists; the two independent implementations in the codebase
   today collapse into it.
5. **Normalisation.** One convention per stored column, applied uniformly and recorded. A column must never mix
   conventions.

Failure raises a typed exception. A placeholder vector is never returned and never persisted.

### Rationale / Alternatives

The clauses are not stylistic. **Task type** is the one with the least visible cost and the largest effect: a
document embedded without it and a query embedded with it occupy subtly different spaces, and retrieval quality
degrades with no error anywhere. **Normalisation uniformity** matters for a reason that is currently invisible:
cosine distance is scale-invariant, so a column mixing normalised and raw vectors ranks correctly today — and
silently mis-ranks the moment anyone switches to an inner-product operator. That is a trap laid for a future
engineer, and it is free to close now.

*Alternatives considered.* (a) **Keep two paths**, one per feature area — the position originally accepted, then
reversed once it became clear that two of the four paths were already one and that the retrieval graph's only
caller lives in the module the carve-out would have frozen. Rejected: it would have rebuilt lexical and fusion in a
second place while the working copy stayed unreachable. (b) **Adopt the framework's cache-backed embeddings
wrapper**, as the reference documentation prescribes. Rejected: the class is importable only from the version-zero
compatibility shim, which the project's import rules forbid, and its prescribed backing store is per-container, so
each replica would silently keep its own cache. The shared cache already provides the same digest-keyed mechanism,
across processes, and already exists in this exact shape twice. (c) **Adopt a framework vector-store abstraction**
and let it own dimension and normalisation. Rejected: retrieval is direct SQL against the vector and lexical
extensions, so this would add a third retrieval path with its own dimension, filter, and identifier conventions.
(d) **Placeholder vectors on failure**, the current behaviour. Rejected as the worst available failure mode for a
legal retrieval product: a zero vector is a *valid* row that ranks against nothing, so a failed embedding becomes
an invisible hole in the corpus.

### Consequences

- Two later changes build on this contract rather than each choosing a dimension. The memory subsystem's provider
  and dimension defaults become a bug to fix against this contract, not an open question.
- The configured dimension becomes a **boot-time** fact. A future engineer who flips it and restarts gets a refusal
  and a clear message, not a data-width error deep in an insert.
- The batch and offline embedding implementation survives as a carve-out, fixed but explicitly **not live**. If it
  ever becomes reachable from a request or an ingestion stage, this contract is violated.
- The token counter used to budget chunks is **not** this provider's counter. That divergence is recorded in the
  chunking contract with the safety margin it implies; it is a known, bounded inconsistency rather than an unknown.
- Cache entries are partitioned by model and task type, so changing either is a cache miss rather than a wrong hit.

---

## ADR-2 — Entity identity is canonical before it reaches the knowledge graph

### Status

Proposed

### Context

The pipeline extracts entities and relationships and writes them to the knowledge graph keyed on **raw extracted
text**. There is no canonicalisation anywhere in the codebase. In a contract graph this means three surface forms of
one company become three party nodes, and the relationships attach to whichever variant appeared nearby.

The property that makes this an architectural decision rather than a bug fix is that it is **not recoverable**. Once
the variants are separate nodes, no later pass can merge them, because the evidence that they were the same entity
was the extraction context — the surrounding clause, the definitions section, the signature block — which has
already been discarded by the time the duplicate is noticeable. A later deduplication pass can only guess from the
strings, and guessing wrong on party identity in a legal product is worse than the duplicate.

Separately, the following change needs idempotency keys for replayed pipeline stages, and the change after that
needs stable entity identity for its memory writes. Three changes need the same mechanism.

### Decision

Every extracted entity resolves to a **stable canonical identity** before anything is written to the knowledge
graph. Canonicalisation is deterministic for a given surface form and is the same in every process. The raw surface
form is **retained as an attribute**, never discarded — it is the audit trail a legal product needs.

Every entity write, relationship write, and episode write keys on the canonical identity. No write path keys on raw
extracted text.

The canonical identity **is** the idempotency key for knowledge-graph writes. Canonicalisation and replay safety are
one mechanism, not two.

Where canonicalisation cannot be performed, the graph write is refused and the document records a terminal failure.
There is no raw-text fallback identity.

### Rationale / Alternatives

*Alternatives considered.* (a) **Deduplicate later, in a maintenance pass.** Rejected on the recoverability argument
above: the disambiguating context is gone by then, so a later pass is a guess. This is the whole reason the decision
is made now rather than deferred with the rest of the graph work. (b) **Let the graph library deduplicate.** Rejected:
its identity is the text it is given, so it deduplicates exact repeats and nothing else — which is precisely the case
that was never the problem. (c) **A separate idempotency key alongside raw-text identity.** Rejected: two keys for one
write is two things to keep in agreement, and the following change would build the second one independently. Stating
that the canonical identity *is* the idempotency key prevents that duplication before it is written. (d) **Fall back
to raw text when canonicalisation fails**, so ingestion never stops. Rejected: that reintroduces exactly the
irreversible defect, in the one situation where it is least likely to be noticed.

### Consequences

- Canonicalisation must land **before** any knowledge-graph write goes live. That is a hard ordering constraint on
  this change, not a preference.
- Its unit test is the most load-bearing new test in the change: variant forms must collapse, and genuinely distinct
  parties must **not** collide. The second half is the harder half and the easier one to get wrong in the unsafe
  direction.
- The three existing graph write sites must each be audited **by reading**, not by pattern search. A missed site
  poisons the graph exactly as thoroughly as no canonicalisation at all.
- The following change inherits an idempotency key rather than designing one, and its rule that structural
  identifiers are hashed while content never is falls out of this decision instead of competing with it.
- Retaining the surface form means the graph carries both identity and provenance, which is what makes a
  canonicalisation error diagnosable after the fact even though it is not automatically reversible.
- A canonicalisation rule that is too aggressive is now the residual risk, and it is a *code* risk rather than an
  architectural one: two genuinely different parties whose names normalise together would merge silently. This is why
  the non-collision half of the test is mandatory.
