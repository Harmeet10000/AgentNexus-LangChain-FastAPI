# RAG external research briefing

**Prepared:** 2026-09-09 · research leg (parallel to codebase scout)
**Target consumer:** the `langchain-fastapi-production` retrieval refactor.

> ## ⚠️ Sourcing caveat — read this first
>
> **Live web tooling was unavailable for roughly the first half of this run.** Firecrawl rejected unauthenticated requests from this IP; `WebFetch`/`WebSearch`/outbound `curl` were refused by the harness safety classifier on ~8 retries. Sections **D, B, E, C were therefore drafted offline from model knowledge (training cutoff May 2026)**. Web access was then restored, and I used it to **retrieve Sections A and F in full and to verify the highest-stakes claims in D and E** — corrections are marked inline.
>
> Confidence is tagged inline:
> - **[VERIFIED 2026-09-09]** — retrieved from the primary source during this run. Trust it.
> - **[STABLE]** — long-settled behaviour, very unlikely to have changed since May 2026.
> - **[VERSIONED]** — correct as of the named version; check the changelog before relying on it.
> - **[CHECK]** / **[UNVERIFIED]** — from memory, or moving fast. Verify before acting.
>
> **Four claims I had wrong offline and corrected against primary sources** (all load-bearing):
> 1. **The lexical decision was framed wrong.** I analysed `tsvector` vs ParadeDB's `pg_search`. The repo actually uses **`pg_textsearch` — Tiger Data's *own* BM25 extension**, a different product from a different vendor, **available on Tiger Cloud** (ParadeDB's is not). This inverts the recommendation: remove `tsvector`, keep the lexical leg. (§D.2)
> 2. **pgvectorscale publishes no benchmark against pgvector HNSW.** Its numbers are vs *Pinecone*. The "2–3× faster than HNSW" figure circulating in blog posts is not Timescale's claim. (§D.1)
> 3. **`langchain_postgres.PGVector` is explicitly deprecated as of v0.0.14** — but this turned out to be moot: the repo has no LangChain vector store at all, so §E.1 is an **adopt-or-stay** question, not a migration. (§E.1)
> 4. **Tiger Cloud enables `pgvector` and `pgvectorscale` by default**; `pg_textsearch` is available but needs `CREATE EXTENSION`. (§D.2)
>
> Sections **B and C are the least verified** — B's chunker semantics and `contextualize()` are confirmed, but the `max_tokens` / tokenizer-wrapper API is not, and C is reasoning from principles rather than from retrieved sources. The Docling API surface churns across releases; check import paths against your installed version.

## Section index

| § | Topic | Status |
|---|---|---|
| **D** | Postgres retrieval stack — pgvector/pgvectorscale, `pg_textsearch` BM25, RRF, index tuning | **[VERIFIED]** — extension availability, params, and `<@>` semantics confirmed |
| **B** | Docling best practice 2026 — pipeline, chunkers, provenance | Chunker semantics **[VERIFIED]**; tokenizer API **[CHECK]** |
| **E** | LangChain — adopt `PGVectorStore` or stay with raw SQL, splitters, dropping `sentence_transformers` | **[VERIFIED]** — deprecation confirmed; reframed for this repo |
| **C** | Chunking strategy for legal documents | Offline reasoning — **[unverified]** |
| **A** | The two cited articles (Uber EAg-RAG; TDS hybrid search + reranking) | **[VERIFIED]** — both retrieved in full |
| **F** | The "195" stack — LangExtract, PageIndex, Graphiti | **[VERIFIED]** — all three READMEs retrieved |
| — | **Decisions this forces** | 15 decisions + a cheapest-first verification queue |

---

# D. Postgres retrieval stack

## D.1 pgvector vs pgvectorscale

### What each one is

**pgvector** (https://github.com/pgvector/pgvector) is the baseline extension: the `vector`, `halfvec`, `bit` and `sparsevec` types, distance operators (`<->` L2, `<=>` cosine, `<#>` negative inner product, `<+>` L1), and two index AMs — `ivfflat` and `hnsw`. **[STABLE]**

**pgvectorscale** (https://github.com/timescale/pgvectorscale) is a Timescale-authored **complement**, not a replacement. It is written in Rust (pgrx) and *depends on* pgvector — it reuses pgvector's `vector` type and operators, and adds one new index access method, **`diskann`**. **[STABLE]** You install both; you do not choose between them.

### What StreamingDiskANN adds over HNSW

Three things, and it's worth being precise because the marketing blurs them:

1. **Disk-first graph layout (DiskANN / Vamana).** HNSW in pgvector is a multi-layer graph designed on the assumption that the graph is in `shared_buffers`. When the index exceeds RAM, HNSW's random-access pattern degrades badly — each hop is a potential page fault, and the multi-layer structure means many hops. DiskANN/Vamana is a **single-layer** graph built with a pruning rule (the `α`-pruning / `max_alpha` parameter) specifically tuned so that a greedy search touches few, large, sequentially-readable nodes. The design target is "index bigger than memory, on SSD". **[STABLE — this is the DiskANN paper's premise, Microsoft Research NeurIPS 2019: https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html]**

2. **Statistical Binary Quantization (SBQ).** Plain binary quantization (BQ) thresholds each dimension at zero: `d` floats → `d` bits, 32× compression. Timescale's claim is that plain BQ loses too much recall on real embedding distributions because the per-dimension mean is not zero and the variance is uneven, so SBQ picks thresholds *statistically per dimension* (and can use more than one bit per dimension — the `num_bits_per_dimension` knob). Compressed vectors drive the graph traversal; the full-precision vectors are then used to **rescore** the candidate set. **[CHECK — I am confident about the mechanism and the parameter name but not about the exact statistical rule; verify against https://github.com/timescale/pgvectorscale and Timescale's SBQ blog post.]**

3. **Streaming / "infinite" candidate retrieval.** This is the operationally most important one and the most under-appreciated. Classic ANN indexes return a *fixed-size* candidate list (`ef_search` for HNSW, `probes` for IVFFlat). If your query has a `WHERE` filter, the filter is applied *after* the ANN returns its candidates, so you can silently get fewer than `LIMIT` rows — or zero. StreamingDiskANN instead exposes the graph as a **stream**: it can keep yielding the next-nearest candidate indefinitely, so Postgres can pull until the filter has been satisfied `LIMIT` times, with **no accuracy cliff from post-filtering**. **[CHECK — mechanism is right; confirm current wording in the repo README.]**
   - Note that pgvector 0.8.0 added `hnsw.iterative_scan` to close exactly this gap (see §D.4). That materially narrows pgvectorscale's advantage for filtered queries — this is the single biggest thing to re-verify.

### Index and query parameters — **[VERIFIED 2026-09-09 against the pgvectorscale README]**

```sql
CREATE INDEX ON items USING diskann (embedding vector_cosine_ops)
WITH (
  storage_layout        = 'memory_optimized',  -- DEFAULT. SBQ-compressed; 'plain' = uncompressed
  num_neighbors         = 50,    -- DEFAULT 50. Max neighbors/node. Higher = more accurate, slower traversal
  search_list_size      = 100,   -- DEFAULT 100. The "S" parameter of the build-time greedy search
  max_alpha             = 1.2,   -- DEFAULT 1.2. Higher = better graph quality, slower build
  num_dimensions        = 0,     -- DEFAULT 0 (all). Limits indexed dims — supports Matryoshka embeddings
  num_bits_per_dimension = 2     -- DEFAULT: 2 below 900 dims, else 1
);

-- query time (session-scoped SET, or SET LOCAL for transaction scope)
SET diskann.query_search_list_size = 100;  -- DEFAULT 100. Extra candidates examined during graph search
SET diskann.query_rescore          = 50;   -- DEFAULT 50. Elements rescored; 0 disables rescoring
```

`diskann.query_rescore` is the recall lever people miss, and the README explicitly flags it as the one to tune for accuracy: with `memory_optimized` storage the graph walk is over *compressed* vectors, so recall without rescoring is mediocre. Raising it costs full-precision distance computations on a small candidate set — usually the cheapest recall you can buy.

**Two things the README revealed that materially change the comparison:**

1. **`diskann` has native label-based filtering.** You declare it in the index itself:
   ```sql
   CREATE INDEX ON items USING diskann (embedding vector_cosine_ops, labels);
   ```
   This is filtering *inside* the ANN graph traversal rather than pre- or post-filtering around it — a genuinely different mechanism from anything pgvector offers, and directly relevant to the §D.4 recall cliff. **This is pgvectorscale's strongest remaining differentiator** now that pgvector has `iterative_scan`.
2. **Parallel index builds** are supported via `diskann.parallel_flush_interval`, `diskann.parallel_initial_start_nodes_count`, `diskann.min_vectors_for_parallel_build` (65536), `diskann.force_parallel_workers` (-1 = auto). Requirements: **SBQ storage, no labels, and enough vectors** to clear the threshold. Note the constraint — **you cannot have both labels and parallel builds**, which is a real tradeoff at ingest time.

Builds are memory-hungry; raise `maintenance_work_mem` well above the 64MB default.

### When pgvectorscale actually wins

| Situation | Winner | Why |
|---|---|---|
| Index fits comfortably in `shared_buffers` (say < ~10–20M vectors at 768d after quantization, on a well-specced box) | **pgvector HNSW** | DiskANN's whole advantage is the disk path. In-memory, HNSW is excellent and has far more eyes on it. |
| Index ≫ RAM | **pgvectorscale** | This is the design point. Timescale's headline benchmarks are on 50M 768-d vectors. |
| Storage cost is the binding constraint | **pgvectorscale** | SBQ at 32× (1 bit/dim) or 16× (2 bits/dim) is a real bill reduction. pgvector can approximate this with `halfvec` (2×) or manual `bit` + reranking, but that's DIY. |
| Heavily filtered queries (`WHERE tenant_id = …`, `WHERE doc_type = 'statute'`) | **pgvectorscale**, via native `labels` | In-graph label filtering vs pgvector's post-filter + `iterative_scan` resume. Different mechanisms; benchmark on your own data. |
| You want the fewest moving parts | **pgvector** | One extension, in every managed Postgres on earth. |

**The benchmark claims — corrected.** I checked the README directly, and it is important to be precise here because the folklore is wrong:

> **The pgvectorscale README contains no benchmark against pgvector's HNSW index at all.** Its published numbers are against **Pinecone**: on 50M 768-dimension Cohere embeddings, pgvector + pgvectorscale claims **28× lower p95 latency** and **16× higher query throughput** vs Pinecone's storage-optimized `s1` index at 99% recall, plus a claimed **75% cost reduction** self-hosted on EC2. **[VERIFIED — these are the actual claims. Methodology lives in Timescale's "pgvector vs Pinecone" blog post, which I did not retrieve.]**

So if you are choosing between pgvector-HNSW and pgvectorscale-DiskANN, **the vendor has not published a head-to-head to help you**, and any "2–3× faster than HNSW" figure you find in a blog post is someone's own benchmark, not Timescale's. You will have to measure it yourself on your own data. That is a more honest framing than the comparison tables circulating online.

### Operational cost of adopting pgvectorscale

- **It is a compiled Rust extension** (pgrx). Local dev and CI need the same extension as prod. Timescale publishes Docker images that bundle it. Building from source needs Rust + `cargo-pgrx`; the README warns **Intel macOS builds are unsupported** (ARM Macs, Linux, or Docker are the workarounds) — worth knowing if anyone on the team is on an Intel Mac.
- **Licence: not stated in the README.** **[UNVERIFIED — check the repo's `LICENSE` file directly. pgvectorscale was originally Timescale License (TSL) and I believe it was relicensed to the PostgreSQL License, but I could not confirm this, and it matters if you ever leave Timescale Cloud.]**
- **Index build is slower and more memory-hungry than HNSW.** Parallel builds help, but only without labels.
- **Fewer users, fewer answers, fewer eyes on correctness** than pgvector.

### Is it on Timescale Cloud? — **the load-bearing question for this user**

**Yes — verified.** The pgvectorscale README documents three install paths, and the third is Timescale Cloud explicitly: create a service and run

```sql
CREATE EXTENSION IF NOT EXISTS vectorscale CASCADE;  -- CASCADE pulls in pgvector
```

**[VERIFIED 2026-09-09 against https://github.com/timescale/pgvectorscale]**

Two caveats from the README worth carrying:
- For **pre-existing services**, the extension only becomes available "after the pgvectorscale release date" on the service's **first maintenance window**. So a long-running instance may not have the newest version yet — check, don't assume.
- Timescale's **vector-optimized database instances** for production workloads are described as **private beta**, gated behind a signup form. If you want the tuned hardware profile rather than just the extension, that's a sales conversation.
- The README never mentions "Tiger Cloud" despite the 2025 TigerData rebrand, so the Timescale Cloud naming is still current in their own docs.

> **Still do this first, it costs one query.** Even with the README verified, run
> ```sql
> SELECT name, default_version, installed_version
> FROM pg_available_extensions
> WHERE name IN ('vector','vectorscale','pg_trgm','pg_search','timescaledb');
> ```
> against the real instance. It tells you the *actual available versions on your service*, which the docs cannot.


**Note on the instance's current state:** the project memory records the live DB as Timescale Cloud stamped at alembic `0004`, with the document/vector/search schema *never created*. So there is no migration burden and no existing index to preserve — this is a greenfield choice on a live-but-empty schema. That materially lowers the cost of picking either option.

## D.2 Full-text: native `tsvector`/GIN vs `pg_textsearch` BM25 (Tiger) vs `pg_search` (ParadeDB) vs `pg_trgm`

> **Read the verified subsections first** — "What a Tiger Cloud instance will actually let you install" and "`pg_textsearch` — Tiger's own BM25 extension" — then the steelmans below. The steelmans were drafted before I knew `pg_textsearch` was on the table, and they analyse `tsvector` vs *ParadeDB*. Their reasoning about `ts_rank`'s weaknesses is still correct and worth reading; their conclusion is superseded.

### What each actually is

**`tsvector` + GIN** — in-core Postgres. Pipeline is: `to_tsvector(config, text)` → parser splits into tokens → dictionaries (stemmer via Snowball, stopwords, synonyms, thesaurus) normalise → a sorted list of lexemes with positions. Query via `@@` against `to_tsquery` / `plainto_tsquery` / `phraseto_tsquery` / `websearch_to_tsquery`. Ranking via `ts_rank` / `ts_rank_cd`. **[STABLE]**

What people get wrong about it:
- **`ts_rank` is not BM25 and is not even close.** It is a term-frequency × weight-class function with an optional length normalisation flag. **It has no IDF term at all** — it does not know that "the" is common and "estoppel" is rare, because it never consults corpus statistics. **[STABLE — this is the single most important fact in this section.]** For a corpus of legal documents where discriminative rare terms are the whole game, this is a genuine quality ceiling, not a nitpick.
- **`ts_rank` cannot use the index for ranking.** The GIN index answers "which rows match"; ranking is then computed per-candidate-row, requiring a heap fetch of the `tsvector`. A high-recall query that matches 200k rows will fetch and score 200k rows. This is *the* reason "Postgres full-text is slow" — it isn't the matching, it's the ranking. **[STABLE]**
- **Positions are needed for phrase search and `ts_rank_cd`**, and `strip()`ing them (or exceeding the 1MB `tsvector` limit, or the 16383-position cap per lexeme) silently breaks proximity. **[STABLE]**

**`pg_search` (ParadeDB)** — https://github.com/paradedb/paradedb. A Rust extension embedding **Tantivy** (a Rust Lucene) as a Postgres index access method (`bm25`). Gives you: real **BM25** scoring with corpus-wide IDF, a proper inverted index with its own segment/merge lifecycle, fast top-N ranked retrieval (score is computed *in* the index, so `ORDER BY score LIMIT 10` is a genuine top-N, not a full scan), plus fuzzy matching, phrase-with-slop, faceting, and a query DSL. **[VERSIONED — ParadeDB moved from a `paradedb.search()` function API to an index-backed `@@@` operator; the operator form is current. CHECK the exact syntax against current docs: https://docs.paradedb.com]**

**`pg_trgm`** — in-core-ish contrib. Trigram similarity for **fuzzy/typo/substring** matching (`similarity()`, `%`, `<->`, and GIN/GiST index support for `LIKE '%foo%'`). It is **not** a full-text engine: no stemming, no IDF, no phrase semantics. It is the right tool for name matching, typo tolerance, and accelerating unanchored `LIKE`. It is a *complement* to either of the above, never a substitute. **[STABLE]**

### Steelman: **keep** `tsvector`

1. **It is already there.** Zero install risk, zero managed-service risk, zero extra binary in CI, works on `postgres:17` in a test container. For a team whose main constraint is shipping, this is decisive.
2. **The hybrid retrieval architecture — not the lexical scorer — is where most of the recall comes from.** In a vector + lexical + RRF pipeline, the lexical leg's job is mostly to catch *exact-token* queries the embedder fumbles: a statute number ("§ 1782"), a party name, a defined term ("Permitted Encumbrance"), a citation. `tsvector` catches all of those. BM25's IDF advantage shows up in *ranking* the lexical results — and RRF then throws most of that ranking information away anyway, keeping only the rank order (see §D.3). **This is the strongest argument for keeping it: RRF is partially insulated from ts_rank's weakness, because RRF consumes ranks, not scores.**
3. **`websearch_to_tsquery` is genuinely good ergonomics** — it parses `"quoted phrases"`, `or`, and `-negation` from raw user input without throwing on malformed syntax, unlike `to_tsquery`. **[STABLE]**
4. **Phrase search works** (`<->` and `<N>` operators, via `phraseto_tsquery`). For legal defined terms this matters and you would lose it with a naive replacement.
5. **Managed-service risk is zero.** No question about whether Timescale Cloud allows it.

### Steelman: **remove** `tsvector`

1. **`ts_rank` has no IDF.** On legal text — heavy boilerplate, huge shared vocabulary, meaning concentrated in rare terms — a scorer that cannot tell "hereinafter" from "escheat" is scoring noise. If lexical *ranking quality* matters to you, `ts_rank` is not a small compromise.
2. **The ranking-scan cost is a real latency wall** at corpus scale, and the usual mitigations (`LIMIT` before rank, materialised rank columns) distort results.
3. **Maintenance surface.** A `tsvector` column means a generated column or trigger, a GIN index, a text-search configuration choice, and a stemmer whose behaviour on legal Latin ("res judicata", "certiorari") and on hyphenated/possessive forms you have to reason about. That's real code and real migrations.
4. **If you are going to run a reranker anyway** (§A.2 / §E), the lexical leg only has to produce *candidates*, and the cross-encoder does the real scoring. In that architecture the case for BM25 weakens too — but so does the case for keeping an elaborate lexical setup at all. **The honest version of "remove tsvector" is usually "remove it and let a reranker over vector-only candidates do the work."** That is a coherent architecture, and it is the one to compare against, not "remove it and hope."
5. **You might replace it with something strictly better** (`pg_search`) rather than with nothing.

### What you actually lose by removing it

| Capability | Lost? | Mitigation |
|---|---|---|
| Exact rare-token recall (statute numbers, party names, defined terms) | **Yes, and this is the big one** | Dense embeddings are systematically bad at exact identifiers. Either keep a lexical leg or add `pg_trgm`/exact-match on an extracted-identifier column. |
| Stemming (`argued` → `argu`) | Yes | Embeddings handle morphology implicitly; this is a small loss. |
| Phrase search (`"reasonable efforts"` as a phrase) | Yes | `pg_trgm` cannot do this. Only BM25-with-positions or `tsvector` can. Real loss for legal defined terms. |
| Ranking quality | You lose a *weak* ranker | If a cross-encoder reranks anyway, near-zero loss. |
| Negation / boolean query syntax | Yes | Rarely used by real users in a RAG frontend. |

### What a Tiger Cloud instance will actually let you install — **[VERIFIED 2026-09-09]**

Retrieved from https://www.tigerdata.com/docs/use-timescale/latest/extensions (the `docs.tigerdata.com` URL 301s here). **This list is the ground truth and it settles most of §D.2.**

| Extension | On Tiger Cloud? | Enabled by default? |
|---|---|---|
| **`pgvector`** | ✅ Yes (Tiger Data extension) | ✅ **Yes, already on** |
| **`pgvectorscale`** | ✅ Yes (Tiger Data extension) | ✅ **Yes, already on** |
| **`pg_textsearch`** | ✅ Yes (Tiger Data extension) | ❌ No — `CREATE EXTENSION` it |
| **`pgai`** | ✅ Yes | ❌ No |
| **`pg_trgm`** | ✅ Yes (Postgres built-in list) | ❌ No |
| **`pg_search` (ParadeDB)** | ❌ **Not on the list at all** | — |

Also available: `timescaledb`, `timescaledb_toolkit`, `pg_stat_statements`, `postgres_fdw`, `pg_cron` (support request required), PostGIS, `pgaudit`, `pg_repack`, and the usual contrib set. No version numbers are published on that page — get those from `pg_available_extensions` on the instance.

> **Three consequences, all decision-relevant:**
> 1. **`pgvector` and `pgvectorscale` are enabled by default.** No `CREATE EXTENSION` needed, no maintenance-window wait, no decision to make. The `diskann` access method is available today. This confirms the repo's existing `diskann` index (`documents/model.py:122`) is on solid ground.
> 2. **`pg_textsearch` is available but not enabled** — one `CREATE EXTENSION pg_textsearch;` away. Migration `0013`'s hard failure if the extension is missing is therefore a *fixable* failure, not a platform limitation.
> 3. **ParadeDB's `pg_search` is not available**, confirming the earlier inference. But that no longer matters, because Tiger ships its own BM25 — see below.

### `pg_textsearch` — Tiger's own BM25 extension (NOT ParadeDB's `pg_search`)

**These are two different products from two different vendors and they must not be confused.** The repo uses **`pg_textsearch`**.

| | `pg_textsearch` | `pg_search` |
|---|---|---|
| Vendor | **Timescale / TigerData** | ParadeDB |
| On Tiger Cloud | ✅ **Yes** | ❌ No |
| Index | `USING bm25(...)` | `USING bm25(...)` (confusingly, same AM name) |
| Rank operator | **`<@>`** | `@@@` |
| Engine | Native C, LSM + Block-Max WAND | Embeds Tantivy (Rust) |

**[VERIFIED 2026-09-09 from https://github.com/timescale/pg_textsearch]** — key facts:

- **"Modern ranked text search for Postgres… BM25 relevance-ranked full-text search. Postgres OSS licensed."** Originally named **Tapir** (Textual Analysis for Postgres Information Retrieval) — that name still appears in the codebase. Status **v1.5.0-dev, "Production ready"**, PostgreSQL 17 and 18.
- **Real BM25 with real tunables** — this is the thing `ts_rank` fundamentally cannot do:
  ```sql
  CREATE INDEX docs_idx ON documents USING bm25(content)
    WITH (text_config='english', k1=1.2, b=0.75);
  ```
  `k1` (default 1.2) is term-frequency saturation; `b` (default 0.75) is length normalisation. `text_config` is **required** and takes a standard Postgres text search config (english/french/german/…) — so **you keep Postgres stemming and stopword handling**, you just get a real scorer on top.
- **Query syntax uses `<@>`, and the sign is inverted:**
  ```sql
  SELECT * FROM documents ORDER BY content <@> 'database system' LIMIT 5;
  ```
  The docs explain `<@>` "returns the **negative** BM25 score since Postgres only supports `ASC` order index scans on operators" — **so lower is better, and a naive `DESC` sort silently returns your worst matches.** This is a genuine footgun and it matters for the RRF query in §D.3, where you must get the rank direction right.
- **The explicit form names the index** and is **mandatory for partial indexes and inside PL/pgSQL**, because the planner hooks that auto-detect the index don't fire there:
  ```sql
  ORDER BY content <@> to_bm25query('database system', 'docs_idx')
  ```
  **[Worth checking in the repo: if the BM25 leg is built in a PL/pgSQL function or against a partial index, the implicit form will silently not use the index.]**
- **Always pair `ORDER BY` with `LIMIT`** to engage Block-Max WAND; otherwise it scores up to `pg_textsearch.default_limit` (**1000**) documents. This is the direct analogue of §D.2's complaint that `ts_rank` can't rank from the index — **`pg_textsearch` fixes exactly that**, computing top-k inside the index via Block-Max WAND.
- **Storage:** LSM-style, on-disk paged memtable as L0, WAL-logged via `GenericXLog`, segments across eight levels, `compaction` policy (`inline` / `background` / `off`). Compression on by default. Parallel index builds. **Partitioned-table support** — which matters for §D.4's multi-tenant partitioning advice.
- **Index flexibility:** expression indexes (JSONB extraction, multi-column concatenation, `lower()`), partial indexes via `WHERE`, and one partial index per language for multilingual tables.
- **`shared_preload_libraries` is required** for self-hosted installs. **On Tiger Cloud managed this is not your problem** — Tiger ships it as a first-party extension and handles preloading; you just `CREATE EXTENSION`. **[The GitHub README does not discuss Tiger Cloud at all; the availability fact comes from the Tiger Cloud extensions page above.]**

### So: keep `tsvector`, or remove it? — **the answer changes given `pg_textsearch`**

The earlier steelman was written assuming the only options were `ts_rank` or nothing. That was wrong for this deployment. The real option set is:

| Option | Verdict |
|---|---|
| `tsvector` + `ts_rank` **only** | **No.** All the weaknesses of §D.2 (no IDF, ranking can't use the index) with a strictly better option sitting unused on the same instance. |
| `pg_textsearch` BM25 **only** | ✅ **Yes — this is the answer, and it is what the repo already does.** Real IDF, tunable `k1`/`b`, index-resident top-k, keeps Postgres stemming via `text_config`. |
| **Both** | Redundant. Two lexical indexes over the same text, two maintenance paths, and RRF would double-count the lexical signal against the single vector leg — quietly skewing fusion toward lexical. |
| ParadeDB `pg_search` | Not available on Tiger Cloud. Moot. |

> **The user's instinct to remove `tsvector` is correct — but for a better reason than they may realise.** It is not "full-text search isn't worth it" (§A.2's RAGAS numbers say the lexical leg is where recall comes from — 0.74 → 0.83). It is that **`tsvector` is the redundant leg once `pg_textsearch` gives you real BM25 on the same box.** Remove `tsvector`; **keep the lexical leg**. Those are different things and conflating them would be the expensive mistake here.

**What you actually lose by dropping `tsvector` specifically (given `pg_textsearch` stays):**

| Capability | Lost? | Why |
|---|---|---|
| Exact rare-token recall | **No** | BM25 does this better — IDF is precisely the mechanism that rewards rare tokens. |
| Stemming | **No** | `text_config='english'` uses the same Postgres text search config and Snowball stemmer. |
| Ranking quality | **No — improves** | Real BM25 with tunable `k1`/`b` replaces an IDF-less scorer. |
| Phrase search (`"reasonable efforts"`) | **[CHECK]** | `ts_vector` has `phraseto_tsquery` and positional `<->`. Whether `pg_textsearch` supports phrase queries is **not** established by what I retrieved. **For legal defined terms this is the one real risk — verify it before deleting the `tsvector` column.** |
| Boolean/negation query syntax (`websearch_to_tsquery`) | **[CHECK]** | Same caveat. Rarely used by real users, but confirm. |

**The one verification that gates the removal:** does `pg_textsearch` support phrase queries and boolean operators? If it does, drop `tsvector` outright. If it does not, consider keeping a narrow `tsvector` column **only** for phrase search, not as a scoring leg.

**Fallback note if you ever leave managed Postgres:** you can implement BM25 in SQL over a lexeme table (`tsvector_to_array` → a `doc_term` table + document-frequency counts + a window function). Roughly 100 lines you then own forever. With `pg_textsearch` available, there is no reason to. **[STABLE — technique; no canonical reference implementation I can point to.]**

## D.3 RRF (Reciprocal Rank Fusion) in SQL

### Origin of `k = 60`

RRF is from **Cormack, Clarke & Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009** (https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf). **[STABLE]**

The formula: `RRFscore(d) = Σ_{r ∈ rankers} 1 / (k + rank_r(d))`.

The honest history of `k = 60`: the authors state they **used k = 60 without tuning it**, chosen so that the contribution of high ranks is damped — it prevents a document ranked #1 by one weak ranker from dominating. It was a reasonable default in a 2009 TREC setting, not a derived optimum. Every "k=60 is optimal" claim you'll read is folklore citing a paper that explicitly declines to claim it. **[STABLE — this is well documented and worth stating plainly in your own docs.]**

**Is tuning it worth anything?** Mostly no, and here is the intuition: `k` controls only *how sharply* rank 1 is favoured over rank 10. With `k = 60`, ranks 1 and 10 score 1/61 vs 1/70 — nearly flat, so the fusion is dominated by *how many lists a doc appears in* rather than *where*. With `k = 1`, ranks 1 and 10 score 1/2 vs 1/11 — steep, so top-1 from either list wins. So:
- **Small `k` (1–10)** ⇒ trusts each ranker's top hits, behaves closer to "interleave the top results".
- **Large `k` (60+)** ⇒ trusts *consensus* across rankers.

If your two legs are of very unequal quality, per-leg **weights** are a far better lever than `k` — and weights are the knob almost nobody exposes but everybody needs. Weighted RRF: `Σ w_r / (k + rank_r(d))`. Tune the weights, leave `k` at 60. **[STABLE reasoning; the "tune weights not k" guidance is my synthesis, not a citation.]**

### Canonical single-statement shape

```sql
WITH
-- Leg 1: dense vector KNN. Must have its own LIMIT so the ANN index is used.
vec AS (
  SELECT
    c.id,
    ROW_NUMBER() OVER (ORDER BY c.embedding <=> :qvec) AS rank
  FROM chunks c
  WHERE c.tenant_id = :tenant          -- see §D.4 on filtering + ANN
  ORDER BY c.embedding <=> :qvec
  LIMIT :n                             -- e.g. 60
),
-- Leg 2: lexical.
txt AS (
  SELECT
    c.id,
    ROW_NUMBER() OVER (
      ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id
    ) AS rank
  FROM chunks c, websearch_to_tsquery('english', :qtext) AS q(query)
  WHERE c.tenant_id = :tenant
    AND c.tsv @@ q.query
  ORDER BY ts_rank_cd(c.tsv, q.query) DESC, c.id
  LIMIT :n
)
SELECT
  ch.id,
  ch.content,
  ch.metadata,
  COALESCE(:w_vec / (:k + vec.rank), 0.0)
  + COALESCE(:w_txt / (:k + txt.rank), 0.0) AS rrf_score
FROM vec
FULL OUTER JOIN txt USING (id)
JOIN chunks ch USING (id)
ORDER BY rrf_score DESC
LIMIT :top_k;
```

Non-obvious points that bite people:

- **`FULL OUTER JOIN`, not `UNION` or `LEFT JOIN`.** A document found by only one leg must survive. `LEFT JOIN` silently drops lexical-only hits — a real bug I have seen in production RRF implementations.
- **`USING (id)` on a `FULL OUTER JOIN`** is what makes the coalesced `id` come out right; with `ON vec.id = txt.id` you get two nullable columns and must `COALESCE(vec.id, txt.id)` yourself. Easy to get wrong.
- **Each leg needs its own `ORDER BY … LIMIT` inside the CTE**, matching the window's `ORDER BY`. If the planner can't see a bare `ORDER BY <-> LIMIT` it will not use the HNSW index — it will seq-scan and sort. Adding a `WHERE` can also cost you the index scan (§D.4).
- **Add a deterministic tiebreaker** (`, c.id`) to both `ORDER BY`s, or ranks are nondeterministic across runs and your evals become non-reproducible.
- **`ROW_NUMBER()` not `RANK()`.** `RANK()` produces ties (two rows both rank 3), which double-weights tied documents in the fusion.
- **Fetch `:n` ≫ `:top_k`.** Typical: `n = 50–100` per leg, `top_k = 10–20` after fusion. If you rerank afterwards, `n` should be whatever your reranker budget allows (see §A.2).
- **Only ranks matter, not scores** — which is RRF's whole point and why it is robust to `ts_rank` and cosine distance being on incomparable scales. It is *also* why RRF discards calibration information a well-tuned score fusion could use. If you later find RRF limiting, the upgrade path is a learned reranker, not weighted score fusion. **[STABLE]**

### The `pg_textsearch` variant — what this repo actually needs

The query above uses `ts_rank_cd`. On Tiger Cloud with `pg_textsearch` (§D.2), the lexical CTE changes shape, and **the sign convention is a trap**:

```sql
txt AS (
  SELECT
    c.id,
    -- <@> returns the NEGATIVE BM25 score (Postgres only supports ASC index
    -- scans on operators), so ASC = best-first. Do NOT write DESC here.
    ROW_NUMBER() OVER (ORDER BY c.content <@> :qtext, c.id) AS rank
  FROM chunks c
  WHERE c.tenant_id = :tenant
  ORDER BY c.content <@> :qtext, c.id
  LIMIT :n                     -- LIMIT is REQUIRED to engage Block-Max WAND
)
```

Four things that will bite:
1. **`<@>` is inverted — lower is better.** A `DESC` here silently returns your *worst* matches, and it will look like a plausible result set rather than an error. This is the single easiest way to ship a broken lexical leg.
2. **`ORDER BY` without `LIMIT` disables Block-Max WAND** and falls back to scoring up to `pg_textsearch.default_limit` (1000) documents. The `LIMIT` inside the CTE is not just for candidate-count control — it is a performance requirement.
3. **Inside PL/pgSQL, or against a partial index, the implicit form does not use the index** — the planner hooks that auto-detect it don't fire. Use the explicit `to_bm25query('...', 'docs_idx')` form there. **Worth auditing the repo's existing BM25 leg for this.**
4. **A three-branch fusion** (the repo's shape) just adds a third CTE and a third `COALESCE(:w_n / (:k + n.rank), 0.0)` term, with `FULL OUTER JOIN … USING (id)` chained across all three. Watch that the weights sum to something you intend — with three legs, an unweighted RRF gives the two lexical-ish legs 2/3 of the influence if two of them are text-derived.

## D.4 Index tuning that matters

### HNSW `m` / `ef_construction` / `ef_search`

```sql
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 64);
SET hnsw.ef_search = 40;  -- default 40, per-session
```

- **`m`** (default 16) — max bidirectional links per node per layer. Drives index *size* and *memory* linearly. Raise to 24–48 for high-dimensional (1536+) or high-recall needs; the returns flatten fast. **[VERSIONED — pgvector 0.8.x defaults.]**
- **`ef_construction`** (default 64) — build-time beam. Higher = better graph = better recall at *every* later `ef_search`, at the cost of build time only. **This is the best-value knob: you pay once at build.** 128–256 is a common production choice. Constraint: `ef_construction >= 2 * m`. **[STABLE]**
- **`hnsw.ef_search`** (default 40) — query-time beam. This is your recall/latency dial at runtime. Must be `>= LIMIT`; setting `LIMIT 100` with `ef_search = 40` quietly gives you garbage. **[STABLE — and a very common bug.]**
- **Build speed:** set `max_parallel_maintenance_workers` and a large `maintenance_work_mem`. If the graph doesn't fit in `maintenance_work_mem`, pgvector spills to disk and the build gets dramatically slower — it logs a hint. Size `maintenance_work_mem` above the expected index size. **[STABLE]**

### pgvector 0.8.0+ `iterative_scan` — the filtering fix — **[VERIFIED 2026-09-09 against the pgvector README]**

Added in **pgvector 0.8.0**.

```sql
SET hnsw.iterative_scan = strict_order;     -- or relaxed_order; disabled unless enabled
SET hnsw.max_scan_tuples = 20000;           -- DEFAULT 20000. Approximate; does NOT affect the initial scan
SET hnsw.scan_mem_multiplier = 1;           -- DEFAULT 1. Multiple of work_mem
-- ivfflat equivalents:
SET ivfflat.iterative_scan = relaxed_order;
SET ivfflat.max_probes = 100;               -- if set below ivfflat.probes, probes wins
```

The problem it solves, in pgvector's own words: with approximate indexes "filters are applied after the index is scanned", so a query can return fewer rows than the `LIMIT`; iterative scans "automatically scan more of the index until enough results are found".

- **`strict_order`** — "ensures results are in the exact order by distance".
- **`relaxed_order`** — "allows results to be slightly out of order by distance, but provides better recall". Fine if you rerank downstream (and you probably do).
- **Recovering strict order from `relaxed_order`:** wrap the query in a **materialized** CTE and re-sort outside it. The docs note `+ 0` is needed on **Postgres 17+** to defeat an optimization that would otherwise flatten the CTE. For distance-threshold queries, keep the distance filter *outside* the materialized CTE and all other filters *inside*. **[VERIFIED — this is a genuinely non-obvious trap and worth a comment in your SQL.]**
- **`hnsw.max_scan_tuples` is a silent truncation point**: hit it and you get fewer rows with no error. It is approximate and does not bound the initial scan. If raising it doesn't improve recall, the docs say to raise `scan_mem_multiplier` instead — the memory cap, not the tuple cap, is often what's binding.

**This is the feature that most changes the pgvector-vs-pgvectorscale calculus** relative to 2024-era blog posts. Any comparison written before Oct 2024 is describing a pgvector that no longer exists.

### Partial / filtered indexes for metadata pre-filtering — **[VERIFIED: this is pgvector's own documented guidance]**

pgvector's README gives an explicit escalation ladder, which is worth following rather than improvising:

1. **Start with a plain B-tree index on the filter column** — `CREATE INDEX ON items (category_id);` — and a multicolumn index for several predicates. This often yields fast **exact** search, which beats approximate search outright when the predicate is selective. **Most teams skip straight to the ANN index and never try this.**
2. **Exact indexes win when the predicate matches a small fraction of rows.** Approximate wins otherwise.
3. **Partial index** when only a handful of distinct filter values exist:
   ```sql
   CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) WHERE (category_id = 123);
   ```
   Perfect for 3–5 stable partitions (doc type, jurisdiction). Useless for 10k tenants.
4. **Partitioning** when there are many distinct values. The docs specifically call out multi-tenant setups: list partitioning or separate tables "keep one tenant's vectors from degrading another's recall". **This is the right answer for multi-tenant legal search**, and note the reason given — it is not just about speed, it is that a shared HNSW graph lets one tenant's data damage another tenant's recall.
5. **`iterative_scan`** for everything else.

Also: **`NULL` vectors are not indexed** (nor zero vectors for cosine distance). A row with a `NULL` embedding is invisible to the ANN path, which is exactly the sort of thing that makes a "missing document" bug take a day to find.

### The recall cliff when `WHERE` meets an ANN index — say this out loud

This is the most important operational fact in §D and the one that surprises people. pgvector's own docs supply the killer illustration:

> **A condition matching 10% of rows, with the default `hnsw.ef_search = 40`, yields roughly 4 matching rows.** **[VERIFIED — pgvector README.]** Not 40. Not "slightly degraded recall". Four. And your query said `LIMIT 10`.

The general shape: Postgres has exactly three plans available and all three are bad in some regime:
1. **Post-filter** (ANN scan → filter): fast, but returns < `LIMIT` rows when the filter is selective. The "cliff" — recall falls off a *ledge*, not a slope.
2. **Pre-filter** (index scan on the filter → exact KNN sort over survivors): exact and correct, but O(matching rows) — fine at 10k rows, fatal at 10M.
3. **Iterative/streaming**: repeatedly resume the ANN until enough rows survive. Correct-ish, but latency varies with selectivity — a query that is 5ms for a common tenant is 500ms for a rare one. **Your p99 becomes a function of your filter's selectivity**, which is a nasty thing to discover in production.

The planner chooses between (1) and (2) on cost estimates that are **poor for vector columns**, because Postgres has no statistics on distance distributions. Check with `EXPLAIN (ANALYZE, BUFFERS)` and look for whether the HNSW index appears at all.

Practical guidance:
- **Selectivity < ~0.1%** → force the pre-filter path; exact KNN over a few thousand rows is genuinely fast and *exactly correct*. Follow pgvector's ladder step 1.
- **Selectivity > ~5%** → post-filter with a generous `ef_search` (4–10× your `LIMIT`).
- **In between, or unknown** → `iterative_scan = relaxed_order` + rerank.
- **Always** set `hnsw.ef_search >= LIMIT`, preferably `>= 2 × LIMIT` when any filter is present. The README notes explicitly that because `ef_search` caps returned candidates, "a query may come back with fewer rows than expected after an HNSW index is added" — i.e. *adding an index changes your result count*, which violates everyone's mental model of what an index does.
- **Measure recall, don't assume it.** Compute ground truth once with an exact scan and diff against the ANN result for a few hundred representative queries — *including the filtered ones*. Most teams never do this and have no idea what their recall is.

---

# B. Docling best practice (2026)

Primary sources to verify against: https://github.com/docling-project/docling and https://docling-project.github.io/docling/ (the project moved into the **LF AI & Data Foundation** from IBM Research in 2025 — the old `DS4SD/docling` URLs redirect). **[CHECK the org path.]**

## B.1 Recommended PDF → structured pipeline

The core object is `DocumentConverter`, configured per input format:

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions, TableFormerMode, EasyOcrOptions,
)

opts = PdfPipelineOptions()
opts.do_ocr = False                       # see below — default True, usually wrong
opts.do_table_structure = True
opts.table_structure_options.mode = TableFormerMode.ACCURATE
opts.table_structure_options.do_cell_matching = True
opts.generate_page_images = False         # only if you need bbox crops / VLM
opts.images_scale = 2.0

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
)
result = converter.convert("contract.pdf")
doc = result.document                      # a DoclingDocument
```
**[VERSIONED — Docling 2.x API. Docling's options module has been reorganised more than once (`pipeline_options` vs `pipeline_options_vlm`); CHECK import paths against the installed version. This is the #1 thing that breaks between Docling releases.]**

### The decisions that actually matter

**1. OCR on/off is the biggest cost lever, and the default is wrong for you.**
`do_ocr` defaults to **True**, and OCR is typically **the dominant cost** in the pipeline — often an order of magnitude more wall-clock than everything else combined. Legal PDFs are overwhelmingly **born-digital with a real text layer** (court filings, executed contracts from a DMS, statute exports). Running OCR on them is pure waste *and can be actively harmful* — the OCR text can replace or conflict with the higher-fidelity embedded text layer.

The correct pattern is **conditional OCR**: detect whether a page has extractable text, and OCR only the pages that don't (scanned exhibits stapled into an otherwise digital filing — extremely common in litigation). Docling exposes `ocr_options.force_full_page_ocr` (default False) and, in recent versions, per-page OCR of bitmap areas only. **[CHECK the exact current knob for "OCR only pages lacking a text layer" — I believe the default `do_ocr=True` already means "OCR bitmap regions only" rather than "OCR everything", with `force_full_page_ocr=True` being the full-page override, but verify.]**

OCR backend options **[VERSIONED]**:
| Backend | Notes |
|---|---|
| **EasyOCR** | Default. Pulls **torch**. Decent quality, slow on CPU, GPU-capable. |
| **RapidOCR** | ONNX-runtime based, no torch. Good CPU speed. Often the best default for a server pipeline. |
| **Tesseract** (`TesseractOcrOptions` / `TesseractCliOcrOptions`) | Needs system `tesseract` + language packs. Fast, lower quality on complex layout. |
| **OcrMac** | macOS Vision framework. Dev-machine only. |

If you turn OCR off for born-digital docs, **you may be able to drop torch from the ingestion image entirely** — see §E.3, this connects.

**2. Table structure: `FAST` vs `ACCURATE`.**
Table extraction runs **TableFormer**, a transformer model. `TableFormerMode.ACCURATE` is meaningfully better on complex/merged-cell tables and meaningfully slower. For legal work — payment schedules, exhibit tables, cap tables, statutory rate tables — tables are often where the answerable facts live, so `ACCURATE` usually earns its cost. `do_cell_matching=True` matches predicted cells back to the PDF's own text tokens rather than re-OCR'ing them; keep it on for born-digital PDFs. **[VERSIONED]**

**3. Standard pipeline vs VLM pipeline.**
Docling added `VlmPipeline` (SmolDocling, and pluggable remote VLMs) as an alternative to the layout-model + TableFormer + OCR assembly. **[VERSIONED — SmolDocling-256M was released mid-2025; the VLM pipeline API and its model zoo are the fastest-moving part of Docling. CHECK https://docling-project.github.io/docling/usage/vision_models/]**

```python
from docling.pipeline.vlm_pipeline import VlmPipeline
from docling.datamodel.pipeline_options import VlmPipelineOptions
converter = DocumentConverter(format_options={
    InputFormat.PDF: PdfFormatOption(
        pipeline_cls=VlmPipeline, pipeline_options=VlmPipelineOptions(...)
    )
})
```

**When the VLM pipeline earns its cost:** documents where *layout semantics* defeat the geometric pipeline — multi-column with sidebars, forms, heavily-stamped court documents, handwriting, scanned-and-skewed exhibits, complex nested tables. **When it does not:** clean born-digital text documents, which is most of a legal corpus. VLM inference is far more expensive per page and introduces hallucination risk into your *extraction* layer, where it is much harder to detect than in generation.

**My recommendation:** standard pipeline as the default, with a VLM fallback triggered by a quality heuristic (very low text yield per page, or a table the standard pipeline failed to parse). Do not make VLM the default path.

**4. Ancillary options worth knowing:** `do_code_enrichment`, `do_formula_enrichment`, `do_picture_classification`, `do_picture_description` — all default off, all add model cost. Formula enrichment is irrelevant for legal; picture description is expensive and rarely worth it. **[VERSIONED]**

**5. Determinism/reproducibility.** Docling downloads models from HF on first use (`docling-tools models download` pre-fetches them). In a container, **pre-bake the models into the image** and set `HF_HUB_OFFLINE=1` — otherwise your first request after a deploy pays a multi-hundred-MB download and your pipeline has an unannounced dependency on huggingface.co being up. **[CHECK exact CLI name; the guidance is sound regardless.]**

## B.2 `HybridChunker` vs `HierarchicalChunker` — and the tokenizer trap

**[VERIFIED 2026-09-09 against https://docling-project.github.io/docling/concepts/chunking/ except where marked]**

**`HierarchicalChunker`** works purely from document structure: using the layout info in the `DoclingDocument` it will "create one chunk for each individual detected document element," attaching metadata such as headers and captions. Its one built-in grouping behaviour is that **list items are combined** — "by default only merging together list items (can be opted out via param `merge_list_items`)." **It has no notion of tokens at all**, so it emits a 4-token chunk for a one-line paragraph and a 6000-token chunk for a long recitals block. Neither is embeddable.

**`HybridChunker`** "uses a hybrid approach, applying tokenization-aware refinements on top of document-based" hierarchical chunking — two passes:
1. **Splitting** — "splits chunks only when needed (i.e. oversized w.r.t. tokens)".
2. **Merging** — "merges chunks only when possible (i.e. **undersized successive chunks with same headings & captions**)". Opt out via `merge_peers` (default `True`).

> **Note that merge condition carefully — it is load-bearing for §C.3.** Peer merging requires *the same headings and captions*, so it will glue short sibling paragraphs together but **will not merge across a section boundary**. That means `merge_peers=True` is **safe for citation provenance**, which is the thing I most wanted to confirm.

**Import paths [VERIFIED]:**
```python
from docling.chunking import HybridChunker           # if you have the `docling` package
# or, on docling-core alone:
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
```
Extras: **`chunking`** for HuggingFace tokenizers, **`chunking-openai`** for tiktoken-based ones. **[VERIFIED — the extras split is worth knowing; it's how you avoid pulling HF machinery when you only need tiktoken. Connects directly to §E.3.]**

**Table-specific options [VERIFIED]** — directly relevant to legal schedules and exhibit tables:
- **`repeat_table_header`** (default `True`) — re-emits table headers on each chunk of a table that spans chunks. Keep this on; a payment-schedule fragment without its column headers is unreadable and un-answerable.
- **`omit_header_on_overflow`** (default `False`) — drops the header for a row that fits without it but overflows with it.

```python
chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
for chunk in chunker.chunk(doc):
    ...
```
**[CHECK — the `max_tokens` parameter and the `HuggingFaceTokenizer` / `OpenAITokenizer` wrapper classes are NOT documented on the concepts page I retrieved. They live in the hybrid-chunking example notebook or the API reference. The code shape below is from memory and is the single most likely thing in this document to be wrong for your installed version. Verify it first.]**

```python
from transformers import AutoTokenizer
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = HuggingFaceTokenizer(
    tokenizer=AutoTokenizer.from_pretrained(EMBED_MODEL),
    max_tokens=512,          # MUST match the embedding model's real limit
)
```

### The tokenizer-vs-embedder alignment trap — state this explicitly

Docling's own guidance is thin here: the concepts page says only that the tokenizer is "typically to be aligned to the embedding model tokenizer." **There is no warning message, no validation, and no error** when you get it wrong. That is precisely why it deserves the emphasis below.

> **The `HybridChunker` counts tokens with the tokenizer you hand it. If that tokenizer is not the *same* tokenizer as your embedding model's, your `max_tokens` bound is fiction — and the failure is silent truncation at embed time, not an exception.**

Three distinct ways to get this wrong, all common:

1. **Different tokenizer family.** You chunk with a BERT WordPiece tokenizer (`all-MiniLM-L6-v2`) and embed with OpenAI `text-embedding-3-small` (cl100k_base BPE). Token counts diverge by 10–30% on ordinary English and much more on legal text with citations, section symbols, and Latin. Chunks you believe are 500 tokens arrive at the API as 620 and get truncated from the tail — where, in a contract clause, the operative proviso often lives.
2. **Right tokenizer, wrong limit.** `all-MiniLM-L6-v2` has `model_max_length = 512`, but that budget must also cover special tokens (`[CLS]`, `[SEP]`) — set `max_tokens = 512` and you overflow by two. More insidiously, some HF tokenizer configs report `model_max_length` as `1000000000000000019884624838656` (the int64 sentinel meaning "unset"), and naive code that reads it produces an unbounded chunker. **[STABLE — a real and famous HF footgun.]**
3. **Forgetting `contextualize()` adds tokens.** See §B.4.

**The rule:** derive `max_tokens` from the embedding model you will actually call, use *that model's* tokenizer, and subtract headroom (~10–15%). Then **assert** it: after chunking, re-tokenize a sample of the serialized text you're about to embed and check the max. A five-line test that fails loudly beats silent tail truncation.

## B.3 `DoclingDocument` / `DocChunk` metadata and provenance

`chunker.chunk()` yields `DocChunk` objects (Pydantic models) shaped roughly:

```
DocChunk
├── text: str                    # raw concatenated text of the chunk
└── meta: DocMeta
    ├── schema_name: str
    ├── version: str
    ├── origin: DocumentOrigin   # filename, mimetype, binary_hash
    ├── headings: list[str]      # ← the heading path. THE valuable field.
    ├── captions: list[str]|None # table/figure captions
    └── doc_items: list[DocItem] # the source elements, each with:
        ├── self_ref: str        # e.g. "#/texts/42" — JSON pointer into the doc
        ├── label: str           # "text" | "section_header" | "table" | "list_item" | ...
        └── prov: list[ProvenanceItem]
            ├── page_no: int
            ├── bbox: BoundingBox (l, t, r, b, coord_origin)
            └── charspan: tuple[int, int]
```
**[VERSIONED — field names from docling-core 2.x. CHECK `DocMeta`'s exact fields; `headings` and `doc_items[].prov` are the stable, load-bearing ones.]**

**What survives chunking:** page number, bounding box, character span, element label, heading path, and the `self_ref` pointer back into the `DoclingDocument`. **This is the single strongest reason to use Docling over PyMuPDF/unstructured for legal work** — pin-cite fidelity (§C.3) needs exactly page + bbox + section, and Docling is one of the few extractors that carries all three through chunking.

**Carrying it into a vector-store payload.** A chunk can span pages and have multiple `doc_items`, so flatten deliberately:

```python
prov = [p for it in chunk.meta.doc_items for p in it.prov]
payload = {
    "text": chunker.contextualize(chunk),     # what you embed — see §B.4
    "raw_text": chunk.text,                   # what you show the user
    "headings": chunk.meta.headings,          # list[str], the section path
    "heading_path": " > ".join(chunk.meta.headings or []),
    "page_start": min(p.page_no for p in prov) if prov else None,
    "page_end":   max(p.page_no for p in prov) if prov else None,
    "bboxes": [
        {"page": p.page_no,
         "l": p.bbox.l, "t": p.bbox.t, "r": p.bbox.r, "b": p.bbox.b}
        for p in prov
    ],
    "labels": sorted({it.label for it in chunk.meta.doc_items}),
    "self_refs": [it.self_ref for it in chunk.meta.doc_items],
    "doc_hash": chunk.meta.origin.binary_hash,   # content-addressed dedupe key
}
```

Two design notes:
- **Persist the whole `DoclingDocument`** (`doc.export_to_dict()` → JSONB or object storage) alongside the chunks. `self_ref` is then a live pointer: given a retrieved chunk you can re-hydrate its parent section, its neighbours, or the whole document, without re-parsing the PDF. This is what makes parent-child retrieval (§C.1) nearly free.
- **`origin.binary_hash`** is your idempotency key for re-ingestion. Use it; don't invent your own.

## B.4 `chunk.text` vs `chunker.contextualize(chunk)`

**[VERIFIED 2026-09-09]** — Docling's `BaseChunker` contract defines both:
- **`chunk()`** yields chunks, each capturing "some part of the document as a string accompanied by respective metadata" — so **`chunk.text` is the raw extracted text alone**.
- **`contextualize()`** returns "the potentially **metadata-enriched serialization** of the chunk, **typically used to feed an embedding model**."

That last clause is Docling's own guidance and it settles the question: **embed the contextualized string; display and cite `chunk.text`.**

Roughly:
```
chunk.text
  "Neither party shall be liable for any failure to perform ..."

chunker.contextualize(chunk)
  "Master Services Agreement\nSection 12. Miscellaneous\n12.4 Force Majeure\nNeither party shall be liable for any failure to perform ..."
```

Reasons, in order of importance:

1. **Isolated clauses are nearly unretrievable.** A force-majeure clause contains none of the words "force majeure" if the heading is stripped — that phrase lives *only* in the heading. A query for "force majeure" would miss it entirely on a dense embedding of the bare text. Legal documents are unusually dependent on this because clause bodies are drafted to be self-contained *legally* while being semantically anonymous.
2. **It disambiguates near-identical boilerplate.** "Section 3.2" appears in every agreement in the corpus; "Acme MSA > Article III > 3.2 Payment Terms" does not.
3. **`HybridChunker` plausibly already accounts for it in token budgeting.** **[CHECK — this is the one claim here I could not verify, and it is the crux of the token-budget argument. If the chunker's `max_tokens` applies to the *bare* text rather than the contextualized serialization, you must subtract the heading-path token cost from `max_tokens` yourself, or every contextualized chunk silently overflows. Test it: chunk a document, then tokenize `contextualize(chunk)` for every chunk and assert the max is ≤ your limit. If that assertion fails, you have your answer.]**

The corollary people miss: **because you embed one string and display another, your chunk row needs both columns.** Storing only the contextualized text pollutes every quoted answer with heading boilerplate; storing only the raw text destroys retrieval quality. Store both.

## B.5 `langchain-docling`

`pip install langchain-docling` → `DoclingLoader`, a `BaseLoader` yielding LangChain `Document`s. **[VERSIONED — this package superseded the earlier community `DoclingLoader` in `langchain_community`. CHECK https://github.com/docling-project/docling-langchain]**

```python
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

loader = DoclingLoader(
    file_path=[...],
    export_type=ExportType.DOC_CHUNKS,   # or ExportType.MARKDOWN
    chunker=HybridChunker(tokenizer=tokenizer),
)
docs = loader.load()
```

**What you get:** LangChain `Document` interop (so `PGVectorStore.add_documents` just works), two export modes, and chunk metadata flattened into `Document.metadata`.

**What you lose vs calling Docling directly** — and this list is why I'd call Docling directly for a legal pipeline:

1. **Metadata gets flattened and, in places, lossy.** The nested `DocMeta` (especially `doc_items[].prov[].bbox`) has to be squashed into LangChain's flat `metadata` dict. Depending on version you may get a nested `dl_meta` blob you then have to re-parse anyway. Bounding boxes are the first casualty, and bboxes are exactly what you need for pin-cites and for highlighting a source in a PDF viewer. **[CHECK how the current version serializes `dl_meta`.]**
2. **No handle on the `DoclingDocument` itself** — so no `self_ref` re-hydration, no parent-child retrieval, no re-serialization at a different granularity. You get chunks and nothing else. Given §B.3's advice to persist the whole document, this is a real loss.
3. **A version-coupling layer you don't need.** `langchain-docling` pins ranges of both `docling` and `langchain-core`. When Docling ships a breaking change you wait for the adapter.
4. **The abstraction saves ~15 lines.** `DocumentConverter` → `HybridChunker` → `Document(page_content=..., metadata=...)` is short, and you control the metadata shape exactly.

**Recommendation: call `docling` directly; write your own 20-line adapter to `Document`. Use `langchain-docling` only for prototyping.** The one thing that would change my mind is if you have no need for bboxes.

> **Specific to this repo:** the scout found `langchain-docling>=2.0.0` declared as a dependency with **zero imports** — the code already calls `docling` directly. That is the correct architecture, arrived at already; the dependency is just dead weight. **Drop `langchain-docling` from `pyproject.toml`.** Given the pin is `>=2.0.0` with no upper bound, it is also a live source of transitive version pressure on `docling` and `langchain-core` for a package nothing imports — the worst kind of dependency. Removing it is a pure win with no behavioural risk (verify with `rg 'langchain_docling|langchain-docling' src/ tests/` returning nothing but the manifest).

---

# E. LangChain specifics

## E.1 `PGVector` vs `PGVectorStore` in `langchain-postgres`

Repo: https://github.com/langchain-ai/langchain-postgres

### The lineage (three generations, and people confuse all three)

1. **`langchain_community.vectorstores.PGVector`** — the original. **Deprecated, do not use.** **[STABLE]**
2. **`langchain_postgres.PGVector`** — the v2 rewrite in the dedicated package. Async-capable via `psycopg3`. **[VERSIONED]**
3. **`langchain_postgres.PGVectorStore`** — the newer class, contributed largely out of the Google Cloud Postgres ecosystem (it shares its design with `langchain-google-cloud-sql-pg` / AlloyDB stores). **[VERSIONED]**

### Real differences

| | `PGVector` | `PGVectorStore` |
|---|---|---|
| **Schema** | Two tables, fixed: `langchain_pg_collection` and `langchain_pg_embedding`. Metadata in a single **JSONB** column. Collections are rows, not tables. | **One table per store**, with **real typed columns**. You declare metadata columns explicitly; `metadata_columns=[...]` promotes fields to first-class columns, with a JSONB catch-all for the rest. |
| **Table creation** | Auto-created on first use (`create_extension`, `pre_delete_collection` flags). | Explicit: `await PGEngine.ainit_vectorstore_table(...)` before use. More ceremony, more control. |
| **Engine** | Takes a connection string or a SQLAlchemy engine; `psycopg3`. | Takes a `PGEngine` wrapper that owns a SQLAlchemy **async** engine and its own background loop. |
| **Async** | Sync + async methods on one class. | Async-first; `PGVectorStore.create()` is async, with a `create_sync()` sibling. |
| **Index control** | Limited. You largely create indexes yourself in SQL/Alembic. | First-class: `HNSWIndex`, `IVFFlatIndex`, `apply_vector_index()`, `reindex()`, `drop_vector_index()`, distance-strategy objects. |
| **Filtering** | JSONB operators — `metadata->>'x'`. Hard to index well, and pushing a JSONB predicate through an ANN query reliably lands you in the §D.4 recall cliff. | Typed column predicates — plain B-tree indexable, composable with partial indexes, and the planner has real statistics on them. |
| **ID type** | UUID/text in the embedding table. | Configurable `id_column`. |

**The schema difference is the whole story.** For legal retrieval you will filter by `jurisdiction`, `doc_type`, `effective_date`, `matter_id`, `tenant_id`. Under `PGVector` those are all JSONB extractions: no useful statistics, no cheap partial index, no partitioning key. Under `PGVectorStore` they are typed columns you can B-tree index, partition on, and build partial HNSW indexes against — i.e. **`PGVectorStore` is the only one of the two that lets you implement §D.4's mitigations.** That connection is the practical reason to care.

### Current status — **[VERIFIED 2026-09-09 against the langchain-postgres README]**

> **`PGVector` is deprecated.** The README carries an explicit warning that **as of v0.0.14 and later, `PGVector` is deprecated** and users should "migrate to `PGVectorStore`" for "improved performance and manageability."

This settles the question. Further verified details:
- `PGVectorStore` is the class demonstrated throughout the docs; Quickstart and how-to notebooks all point at `pg_vectorstore` examples.
- "All synchronous functions have corresponding asynchronous functions" — so `create_sync` / `create`, etc.
- **`PGEngine`** is the connection/setup object: `PGEngine.from_connection_string(url=...)` → `engine.init_vectorstore_table(table_name=..., vector_size=...)` → `PGVectorStore.create_sync(engine=engine, table_name=..., embedding_service=...)`.
- **Migration:** no helper functions are named in the README; it points to the notebook `examples/migrate_pgvector_to_pgvectorstore.ipynb`. **[Open that notebook if you ever need to migrate — but per the project memory, the vector schema was never created on the live DB, so there is nothing to migrate. This is greenfield.]**
- The README does **not** spell out the schema differences; the table comparing "Schema Flexibility" and "Improved metadata handling" is about the Google AlloyDB / Cloud SQL integrations, not `PGVector` vs `PGVectorStore`. So the schema comparison above is inferred from the API surface, not from an official statement. **[PARTIALLY VERIFIED]**

### The finding that changes the recommendation: `PGVectorStore` has built-in hybrid search with RRF

`PGVectorStore.create_sync(...)` accepts a **`hybrid_search_config`** argument, and the README's snippet references `HybridSearchConfig` and **`reciprocal_rank_fusion`**. **[VERIFIED that these exist and are wired into `PGVectorStore`; NOT verified what the config exposes — the README's own snippet is buggy (it assigns to `vs` then calls `vector_store.similarity_search(...)`, and uses both symbols without showing imports), which is a mild signal that this surface is newer and less exercised than the rest.]**

This matters because it undercuts the strongest argument for hand-rolling: I claimed below that "no LangChain vector store can express the RRF query." That is now **wrong as stated** — `PGVectorStore` can do vector + full-text RRF natively. What remains unknown, and what you must check in the source before deciding:
- Does it let you set the RRF `k` and per-leg **weights** (§D.3)?
- Does it use `websearch_to_tsquery` or something less forgiving?
- Can you inject arbitrary `WHERE` predicates into *both* legs (needed for tenant isolation)?
- Does it set `hnsw.ef_search` per query, or leave it at the session default of 40 (§D.4)?

**Read `langchain_postgres/v2/hybrid_search_config.py` (or equivalent) before committing.** If it answers yes to all four, use it. If not, you are back to hand-rolled SQL.

### Which should a new 2026 project pick? — **reframed for this repo**

The scout established that **this repo has no LangChain `VectorStore` object at all.** Vector search is hand-written SQL (`documents/repository.py:452`) using `<=>` cosine with `SET LOCAL diskann.query_search_list_size` / `diskann.query_rescore`, against a `diskann` index (`documents/model.py:122`) and a `bm25` index (`:114`), fused three ways.

**So this is not a migration question. There is nothing to migrate — `PGVector` vs `PGVectorStore` is moot.** The real question is **adopt `PGVectorStore`, or stay with the tuned raw SQL.** And framed that way it is not close.

#### Can `PGVectorStore` even express what this repo already does?

| What the repo does today | Can `PGVectorStore` express it? |
|---|---|
| `CREATE INDEX … USING diskann (…)` (pgvectorscale) | **No.** The store's index abstractions are `HNSWIndex` and `IVFFlatIndex` — both pgvector access methods. There is no pgvectorscale/`diskann` index class. **[HIGH CONFIDENCE — the store originates in the Google Cloud SQL / AlloyDB ecosystem, which uses pgvector, not pgvectorscale. Verify by grepping the installed package for `diskann`; I expect zero hits.]** |
| `SET LOCAL diskann.query_search_list_size` / `diskann.query_rescore` per query | **No.** These are pgvectorscale GUCs that must be set on the *same transaction* as the query. `PGVectorStore` builds and executes its own SQL through `PGEngine`'s own connection; there is no documented hook to run a `SET LOCAL` in that transaction. **[HIGH CONFIDENCE, not directly verified.]** |
| Three-branch RRF fusion | **Partially.** `hybrid_search_config` + `reciprocal_rank_fusion` exist, but they are built for a **two**-leg (vector + full-text) fusion. A third branch is not something the config is shaped for. |
| `pg_textsearch` `<@>` BM25 leg with inverted scores | **Almost certainly not.** Any built-in full-text leg will target `tsvector`/`ts_rank`, not Tiger's `<@>` operator with its negative-score convention. |
| Arbitrary `WHERE` predicates injected into both legs | Unknown — depends on the config's filter surface. |

> **Say it plainly: adopting `PGVectorStore` here would be a downgrade.** It cannot express the `diskann` index, cannot set the pgvectorscale query-time GUCs that are the entire recall/latency tuning surface (§D.1 — `query_rescore` is *the* recall lever when `storage_layout='memory_optimized'`), and cannot drive a `pg_textsearch` BM25 leg. You would be trading a tuned, working, Tiger-native retrieval path for a portable abstraction that targets a different vector extension.

#### The recommendation

**Stay with raw SQL. Do not adopt `PGVectorStore`.** The reasons compound:

1. **Capability loss** (the table above) — this alone settles it.
2. **`PGEngine` would add a second async engine and connection pool** alongside the SQLAlchemy async engine this project already runs, meaning two lifecycles in `src/app/lifecycle/lifespan.py`.
3. **You would inherit a schema you don't control**, when you already own SQLAlchemy models and Alembic migrations (through `0013`).
4. **The abstraction's payoff is portability across vector stores** — a benefit you are explicitly declining by choosing pgvectorscale + `pg_textsearch`, both Tiger-specific.

**Keep LangChain where it earns its place:**
- the **`Embeddings`** interface, for provider abstraction — this is the genuinely valuable part;
- **`Document`** for interop;
- a hand-written **`BaseRetriever`** subclass (~20 lines) wrapping your existing repository query, so LangGraph gets a standard retriever without your SQL moving anywhere.

**The one thing worth revisiting later:** if you ever migrate off Tiger Cloud to a plain-pgvector managed Postgres, `PGVectorStore` becomes reasonable — at that point you'd be on HNSW anyway and the capability gap closes. Note that in the ADR so the decision has an expiry condition rather than being re-litigated from scratch.

**Historical note, since it will come up in review:** `PGVector` is formally deprecated as of `langchain-postgres` v0.0.14, so nobody should propose it. `PGVectorStore` is the live class. Neither is the right choice here.

## E.2 Text splitters — inventory and honest guidance

From `langchain-text-splitters`. **[VERSIONED]**

| Splitter | What it does | Verdict |
|---|---|---|
| `RecursiveCharacterTextSplitter` | Recursively splits on a separator list (`["\n\n","\n"," ",""]`), falling through as chunks stay too big. Character-counted by default. | **The sane default when you have nothing better.** Its virtue is that it's predictable. Its vice is that it counts *characters*, not tokens — `chunk_size=1000` means 1000 chars ≈ 250 tokens, and people routinely conflate the two. Use `.from_tiktoken_encoder()` / `.from_huggingface_tokenizer()` to make it token-aware. |
| `RecursiveCharacterTextSplitter.from_language(Language.MARKDOWN / .PYTHON / …)` | Same, with language-appropriate separator lists. | Useful. Underused. |
| `TokenTextSplitter` | Splits strictly on token counts with `tiktoken`. | **Rarely what you want** — it will happily cut mid-sentence and mid-word. Its legitimate use is as a hard *safety* pass after a semantic splitter, to guarantee no chunk exceeds the model limit. |
| `MarkdownHeaderTextSplitter` | Splits on `#`/`##`/`###` and **attaches the header path to metadata**. Does not enforce a size limit. | **Genuinely good, and the closest LangChain equivalent to `HierarchicalChunker`.** Standard pattern is `MarkdownHeaderTextSplitter` → `RecursiveCharacterTextSplitter` to enforce size. If you export Docling to Markdown instead of using `DocChunk`s, this is the pairing — but you lose bboxes, so don't. |
| `HTMLHeaderTextSplitter` / `HTMLSectionSplitter` | Same idea for HTML. | Fine for HTML sources. |
| `SemanticChunker` (`langchain-experimental`) | Embeds each sentence, then splits where consecutive-sentence embedding distance exceeds a percentile/stddev/interquartile threshold. | **Mostly folklore.** It costs an embedding call per sentence at ingest (expensive at corpus scale), it is sensitive to a threshold nobody knows how to set, and published comparisons have repeatedly failed to show it beating a well-tuned recursive splitter by enough to matter. It is also **worst on legal text**, where adjacent clauses are stylistically near-identical so the embedding-distance signal is flat exactly where the real boundary is. **Do not use it here.** It lives in `langchain-experimental` for a reason. |
| `NLTKTextSplitter` / `SpacyTextSplitter` | Sentence segmentation via NLTK/spaCy. | Only if you need real sentence boundaries and can afford the dependency. spaCy's segmenter mangles legal citations ("v.", "§", "Id.", "F.3d") without customisation. |
| `CharacterTextSplitter` | Splits on one separator. | Legacy. No reason to use it. |

### The honest guidance when Docling already emits structure-aware chunks

> **Don't run a LangChain splitter over Docling `DocChunk`s. You would be destroying structure you paid a transformer to recover.**

`HybridChunker` already does what `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` approximates, and does it with page/bbox provenance intact. Re-splitting throws away `doc_items[].prov` — the pin-cite data — because a LangChain splitter has no idea how to subdivide a bounding box.

The **only** legitimate residual role for a LangChain splitter in a Docling pipeline is as a **guard rail**: a final assertion pass that no chunk exceeds the embedding model's hard token limit, for the pathological cases `HybridChunker` can't split (a single 8000-token table row; an un-paragraphed exhibit). And even there, an explicit assert + a targeted fallback is clearer than a splitter in the main path.

For **non-PDF** sources (Markdown, HTML, plain-text statutes scraped from a legislature site) where you never invoked Docling, the splitters are fine and `MarkdownHeaderTextSplitter` → recursive is the right pairing.

## E.3 Can `sentence_transformers` / `transformers.AutoTokenizer` be dropped?

Short answer: **`sentence_transformers` — probably yes. `transformers` (for `AutoTokenizer`) — probably not, but it can be made cheap.** The distinction matters and is usually missed.

### What each is actually for

| Need | Requires | Notes |
|---|---|---|
| **Calling a hosted embedding API** (OpenAI, Voyage, Cohere, Anthropic-adjacent) | `langchain-openai` / `langchain-voyageai` / `langchain-cohere` — **HTTP clients only** | No torch, no transformers. |
| **Local embedding inference** | `sentence-transformers` **+ torch** (via `langchain-huggingface`'s `HuggingFaceEmbeddings`) | This is the torch-shaped dependency. |
| **Local cross-encoder reranking** | `sentence-transformers` **+ torch** (`CrossEncoder`, via `HuggingFaceCrossEncoder`) | Same. Hosted alternatives exist: Cohere Rerank, Voyage rerank-2, Jina Reranker — all HTTP. |
| **Exact HF token counting for `HybridChunker`** | `transformers.AutoTokenizer` — **tokenizers only, no torch** | ← the key insight, see below. |
| **Token counting for OpenAI models** | `tiktoken` — small, Rust, no torch | Docling exposes `OpenAITokenizer` for exactly this. |

### The dependency-weight reality

**`transformers` does not require torch to tokenize.** `AutoTokenizer.from_pretrained(...)` for any modern model resolves to a **fast** tokenizer backed by the `tokenizers` package — a small Rust extension. `transformers` imports lazily and will happily tokenize in an environment with no deep-learning framework installed at all. So the cost of "keeping `AutoTokenizer`" is roughly `transformers` (pure Python, ~10–20MB) + `tokenizers` (Rust wheel, a few MB) + `huggingface_hub`. **Tens of megabytes, not gigabytes.** **[STABLE — this is a genuine and widely-missed fact. Verify trivially: `pip install transformers tokenizers` in a clean venv and confirm torch is not pulled.]**

**`sentence-transformers` *does* hard-depend on torch.** Torch alone is ~800MB–2.5GB depending on CUDA variant (`torch` CPU-only wheels are far smaller than the default CUDA ones — `--index-url https://download.pytorch.org/whl/cpu` is the mitigation if you must keep it). This is your container-size problem, and it is essentially the *only* one.

**So the decomposition is:**

```
Drop sentence-transformers  ⇒ drop torch  ⇒ container shrinks by ~1–2 GB
Keep transformers/AutoTokenizer ⇒ costs ~30 MB ⇒ keep exact token alignment (§B.2)
```

That is close to a free lunch, and it is the recommendation.

### The three things you genuinely cannot do without them

1. **Local embedding.** If data residency, cost at volume, or offline operation forbids a hosted embedding API, you need torch. There is no way around it in Python. (Alternatives that avoid torch: an ONNX-runtime path — `fastembed` (Qdrant), or `optimum` + `onnxruntime` — which runs many sentence-transformer models with no torch at all. **[CHECK current model coverage; `fastembed` supports a curated list, not everything on HF.]** This is a real escape hatch and worth evaluating if local embedding is mandatory.)
2. **Local cross-encoder reranking.** Same story; same ONNX escape hatch. Reranking is the highest-value quality lever in the whole pipeline (§A.2), so do not simply drop it — move it to a hosted reranker if you drop torch.
3. **Exact HF token counting.** Only `AutoTokenizer` gives you byte-exact counts for a HF embedding model. Approximations (chars/4) are wrong by enough to matter at the boundary — see §B.2's silent-truncation failure.

### The catch that makes the recommendation clean

**If you use a hosted embedding model, you don't need `AutoTokenizer` either — you need `tiktoken`** (OpenAI) or the provider's own counter. `HybridChunker` accepts an `OpenAITokenizer`, so:

```python
import tiktoken
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer

tokenizer = OpenAITokenizer(
    tokenizer=tiktoken.encoding_for_model("text-embedding-3-large"),
    max_tokens=8191,          # text-embedding-3-* context limit
)
```
**[VERSIONED — CHECK the import path and that `OpenAITokenizer` exists in your docling-core version; it was added alongside `HuggingFaceTokenizer`.]**

**Recommendation, concretely:**
- Hosted embeddings + hosted reranker ⇒ **drop `sentence-transformers` AND `torch` AND `transformers`; keep `tiktoken`.** Smallest image, cleanest deps, and §B.2's alignment trap disappears because the tokenizer *is* the embedder's tokenizer.
- Local embeddings required ⇒ keep `sentence-transformers` + CPU-only torch, and keep `transformers` for tokenization; pin the CPU wheel index in `pyproject.toml`.
- **Watch out:** Docling's default OCR backend (EasyOCR) pulls torch anyway. If you're chasing image size, you must *also* switch to RapidOCR/Tesseract or disable OCR (§B.1). Dropping `sentence-transformers` alone will not shrink the image if EasyOCR is still in the graph. **This is the trap — the two decisions are coupled and are usually made by different people.**

---

# C. Chunking strategy for legal documents

## C.1 What works: structure-aware vs fixed-token, hierarchical, contextual, late chunking

### The governing fact about legal text

Legal documents are **explicitly, canonically hierarchical**, and the hierarchy is *normative* rather than cosmetic. A contract's § 12.4 is a unit because the drafters made it one; a statute's subsection (b)(2)(A) is the unit courts cite and amend. Unlike prose, where paragraph boundaries are stylistic, **the document tells you where the chunk boundaries are.** Any chunking strategy that ignores that is discarding free, high-quality supervision.

Two consequences that follow immediately and are worth stating because they invert generic RAG advice:
- **Fixed-token splitting is strictly worse here than it is on generic prose**, because you are overriding a boundary signal that generic prose doesn't have.
- **Overlap is a workaround for not knowing where boundaries are.** Once you know, its main justification evaporates (see §C.2).

### Structure-aware (clause/section) splitting — the primary strategy

Chunk at the smallest *citable* unit that is semantically self-sufficient. Concretely:

| Document type | Chunk unit | Notes |
|---|---|---|
| Contracts / agreements | The numbered **clause** (`12.4`), rolling up to the sub-article if too small | Sub-clauses `(a)`, `(b)` usually stay with their parent — they're often sentence fragments completing the parent's stem, and are meaningless alone. **This is the single most common chunking bug in contract RAG.** |
| Statutes / regulations | **Section**, or subsection for long sections | The citation unit. Amendment granularity too, which matters for versioning. |
| Case law / opinions | Harder — opinions are prose with weak structure | Split on headings where present (syllabus, background, discussion, holding), else paragraph-group with generous size. Paragraph numbers (`¶ 14`) where the reporter provides them are the citation unit. |
| Briefs / filings | Argument headings (the roman-numeral structure) | Heading text is a strong retrieval signal — it's a thesis statement by construction. |

`HybridChunker` over a `DoclingDocument` (§B.2) approximates this well *if* Docling's layout model correctly recovers the numbered hierarchy. **The thing to verify empirically** is whether Docling produces `section_header` labels for contract clause numbers — many contracts style clause headings as inline bold runs rather than as structural headings, and a layout model can miss them. If it does miss them, you need a post-pass: regex the clause numbering (`^\s*(\d+(\.\d+)*)\s`, `^\s*\(([a-z]|[ivx]+)\)`) over `DocItem`s and rebuild the tree yourself. **Budget for this — it is the most likely place a Docling-based legal pipeline needs custom code.**

### Hierarchical / parent-child retrieval — the highest-value structural technique

**Embed small, retrieve large.** Index precise child chunks (a clause) but return the parent (the whole section/article) to the LLM.

- **Why it works:** the two things are in tension — retrieval precision wants small chunks (a focused embedding, not diluted by neighbouring topics), while answer quality wants large context (the definitions, provisos, and cross-references the clause depends on). Parent-child dissolves the tension instead of compromising on chunk size.
- **Why it is *especially* right for legal:** legal clauses are famously non-self-contained. "Permitted Encumbrance" in § 7.1 is meaningless without the definition in § 1.1; a limitation of liability is meaningless without its carve-outs in the next subsection. Returning the parent captures the carve-outs; you still need a definitions-resolution step for the cross-document references (see the note on §F/LangExtract).
- **Implementation:** LangChain's `ParentDocumentRetriever` does this with a docstore + vectorstore. But with Docling you get it nearly free (§B.3): store the `DoclingDocument`, index children with their `self_ref`, and on retrieval walk `self_ref` up to the parent section. **That is strictly better than `ParentDocumentRetriever`** because the parent is a real structural node, not an arbitrary bigger slice.
- **Variant worth knowing:** *multi-vector* retrieval — index several representations of the same parent (the raw clause, an LLM-generated summary, hypothetical questions it answers) all pointing at one parent. Costs an LLM call per chunk at ingest. Helps most when user queries are phrased very differently from the document's language, which in legal is common (a layperson asks "can they cancel on me?" against text saying "termination for convenience").

### Contextual retrieval (Anthropic's context-prefix technique)

Source: https://www.anthropic.com/news/contextual-retrieval (Sept 2024). **[VERSIONED — figures below are from that post; I could not re-verify.]**

The technique: before embedding, prepend to each chunk a 50–100 token, LLM-generated description situating it in the whole document. Generated by prompting a cheap model with (whole document, this chunk) → "give short context to situate this chunk". Then embed **and** BM25-index the contextualized chunk.

Reported results **[CHECK — reconstructed from memory of the post]**:
- Contextual Embeddings alone: **~35% reduction** in top-20 retrieval failure rate (5.7% → 3.7%).
- Contextual Embeddings + Contextual BM25: **~49% reduction** (5.7% → 2.9%).
- Adding a reranker on top: **~67% reduction** (5.7% → 1.9%).

Two things to take from this that are more durable than the exact numbers:
1. **The gains stack** — contextual prefixes, hybrid lexical+dense, and reranking are largely orthogonal. That is the strongest empirical argument for building all three legs rather than betting on one.
2. **Prompt caching is what makes it affordable.** The naive cost is (whole document) × (number of chunks) input tokens. With prompt caching on the document, Anthropic quoted roughly **$1.02 per million document tokens**. Without caching it is ~10× that and the technique is not economic. **[CHECK the current price and cache mechanics — this is exactly the kind of number that has moved since 2024. Consult the `claude-api` skill / current pricing page rather than this document.]**

**How it relates to Docling's `contextualize()`:** they are the same idea at different cost points. Docling's version prepends the **actual heading path** — free, deterministic, no hallucination risk. Anthropic's prepends an **LLM-written summary** — costs money, can hallucinate, but captures things no heading does ("this clause modifies the indemnity in § 9 and applies only after the Closing Date").

**Recommendation: do the free one first.** Ship `contextualize()`, measure, and only add LLM-generated context if the eval shows a gap. For structured legal documents with genuine heading hierarchies, the heading path recovers a large fraction of what the LLM prefix would give you, at zero marginal cost. The LLM version earns its keep mainly on documents with *weak* structure — case law, correspondence, exhibits — which is precisely where Docling's headings are thin. **So: heading-path context everywhere, LLM-generated context selectively on the low-structure subset.** That's the nuanced answer.

### Late chunking

Source: Jina AI, "Late Chunking in Long-Context Embedding Models" (2024), https://arxiv.org/abs/2409.04701 and https://jina.ai/news/late-chunking-in-long-context-embedding-models/. **[VERSIONED]**

Mechanism, and it is genuinely elegant: run the **whole document** (up to the model's long context, e.g. 8192 tokens) through the transformer **first**, producing token embeddings that have attended over the entire document; **then** apply chunk boundaries and mean-pool within each chunk. Each chunk embedding is therefore *contextually informed* by the whole document without any generation step.

- **The problem it solves:** the classic failure where a chunk says "the Company shall indemnify..." and "the Company" is defined 40 pages earlier. Under normal chunking that pronoun-like reference is unresolvable in the embedding; under late chunking the token embeddings for "the Company" have attended to the definition.
- **Why it's attractive for legal specifically:** legal drafting is *saturated* with defined terms and cross-references. This is the anaphora problem at industrial scale, and late chunking attacks it directly and cheaply (one forward pass per document, no LLM).
- **The catch, and it is disqualifying for many stacks:** it requires **token-level access to a long-context embedding model** — i.e. you must run the model yourself (`jina-embeddings-v2/v3`, or another long-context encoder), because no hosted embedding API exposes per-token embeddings. **That directly contradicts §E.3's recommendation to drop torch.** You cannot have both.
- **Also:** documents longer than the model's context window need a sliding-window scheme, which reintroduces boundary effects.

**Verdict: promising, and the most intellectually satisfying option, but it is a torch-shaped commitment.** Treat it as a v2 experiment, not a v1 architecture. If §E.3 lands on "hosted embeddings, no torch", late chunking is off the table and contextual retrieval (which works with any embedding API) is its substitute.

### Summary ranking for this use case

| Technique | Value | Cost | Verdict |
|---|---|---|---|
| Structure-aware chunking (Docling + clause post-pass) | Very high | Medium (custom code likely) | **Do this. Foundation.** |
| Heading-path context (`contextualize()`) | High | ~Zero | **Do this. Free.** |
| Parent-child retrieval via `self_ref` | Very high | Low | **Do this.** |
| Reranking | Very high | Low-medium (hosted) | **Do this.** Biggest single quality lever. |
| Hybrid lexical + dense + RRF | High | Medium | **Do this** (§D). |
| LLM contextual prefixes | Medium-high | Real $ + ingest latency | Selectively, on low-structure docs, after measuring. |
| Late chunking | Medium-high | Requires self-hosted embedding | v2 experiment. Conflicts with dropping torch. |
| Semantic chunking (`SemanticChunker`) | Low | Real $ | **Skip.** Worst-suited to legal text (§E.2). |

## C.2 Practical numbers — and whether overlap is the right tool

### The numbers people converge on

**[STABLE as community consensus; note there is no authoritative study establishing these — they are convergent practice, and anyone citing them as measured optima is overclaiming.]**

- **Chunk size:** 256–512 tokens for the *indexed* (child) chunk; 1000–2000 tokens for the *returned* (parent) context. The 512 figure is not a coincidence — it's the context limit of the BERT-family encoders most retrieval models descend from, so it became the default by construction rather than by measurement.
- **Overlap:** 10–20% of chunk size (50–100 tokens on a 512-token chunk) is the standard recipe.
- **Modern embedding models (`text-embedding-3-large` at 8191 tokens, Voyage, Jina v3) permit much larger chunks**, and larger chunks are being used successfully. But the precision argument still favours small chunks for the *indexed* unit — a big chunk's embedding is an average over multiple topics and matches everything mediocrely. That's why parent-child (§C.1) beats "just use big chunks".

### Is overlap the right tool once you have real structural boundaries?

**Mostly no.** This deserves to be argued rather than asserted:

**What overlap is for:** with arbitrary boundaries, a sentence that answers the query may be cut in half, or its essential antecedent may sit just across the boundary. Overlap is insurance against a boundary you had no principled way to place.

**Why it stops being the right tool here:**
1. **The boundary is no longer arbitrary.** A clause boundary is a *semantic* boundary — the drafters put it there. Bleeding 15% of § 12.5 into § 12.4's chunk doesn't repair a bad cut; it contaminates a good one.
2. **It corrupts citation fidelity** (§C.3). If a chunk contains text from two clauses, "which clause does this chunk cite to?" has no clean answer, and a quote you surface may be attributed to the wrong section. **For legal work this is not a quality nit — it is a correctness failure with professional-liability implications.** This is the decisive argument.
3. **It inflates the index** by the overlap fraction, for both storage and embedding cost.
4. **It creates near-duplicate retrieval results** — the same passage returned twice under two chunk IDs, wasting slots in your top-k and in the LLM's context.

**What to use instead — three tools, each better-targeted than overlap:**
- **Parent-child retrieval** — solves "the answer spans the boundary" completely and correctly, by returning the whole parent. This is overlap's job done properly.
- **Neighbour expansion at retrieval time** — on a hit, also fetch chunk `n-1` and `n+1` (cheap: `WHERE doc_id = ? AND ordinal BETWEEN ? AND ?`). Gives you overlap's benefit *dynamically*, without duplicating anything in the index, and without corrupting provenance because each retrieved chunk keeps its own citation.
- **Heading-path context** (§B.4) — solves "the chunk lacks its antecedent" for the structural case.

**Concrete recommendation:**
- Indexed chunk: the clause/subsection, target **~400–600 tokens**, hard cap at the embedder's limit minus headroom.
- **Overlap: 0**, when structural boundaries are reliable.
- **Overlap: 10–15%**, only on the fallback path where a chunk had to be split *mid-clause* because it exceeded the token cap — there the boundary genuinely is arbitrary and overlap is genuinely the right fix. Mark those chunks with a flag so you know which ones have impure provenance.
- Retrieval-time: parent expansion, plus ±1 neighbour.

That "overlap only on arbitrary splits" rule is the crisp version of the answer, and it's the one to put in the code.

## C.3 How citation / pin-cite fidelity constrains chunking

This is a **hard constraint, not a nice-to-have**, and it should be treated as an architectural invariant. A legal answer that cannot be traced to "§ 12.4, page 37" is not merely less useful — it is unusable, because the practitioner must verify it and cannot.

### The invariant to enforce

> **Every chunk must map to exactly one citable locus** — one document, one section/clause identifier, one page range — and that mapping must survive the entire pipeline: extraction → chunking → embedding → retrieval → reranking → generation.

### What this forbids

| Practice | Why it's forbidden |
|---|---|
| Overlap across clause boundaries | A chunk with two homes cannot be cited (§C.2). |
| Merging peers across a section boundary | `HybridChunker(merge_peers=True)` merges *siblings under one parent*, so it should be safe — **but verify this on your documents**, because if it merges across parents your provenance is silently broken. Test it explicitly. |
| Discarding `prov`/bbox in a LangChain adapter | §B.5 — the main reason to avoid `langchain-docling`. |
| Re-splitting Docling chunks with a text splitter | §E.2 — bboxes cannot be subdivided. |
| Reformatting/normalising text before storage | Destroys the `charspan` offset mapping into the original. Store raw; normalise only in a derived column. |

### What this requires you to store, per chunk

```
document_id           -- FK
document_version      -- statutes are amended; contracts are amended and restated
citation_label        -- "§ 12.4" / "Art. III(b)(2)" / "¶ 47"  ← human-facing
heading_path          -- ["MSA", "Article XII", "12.4 Force Majeure"]
page_start, page_end
bboxes                -- JSONB, per page, for viewer highlighting
char_span             -- offsets into the canonical extracted text
self_ref              -- pointer into the stored DoclingDocument
ordinal               -- position within document, for neighbour expansion
```

### Two under-appreciated consequences

1. **Versioning is part of citation.** A statute cited without its effective date is potentially wrong, and a contract clause cited without knowing it was superseded by Amendment No. 2 is actively dangerous. Your chunk identity should be `(document_version, locus)`, not `(document, locus)`. **This constrains your data model far more than it constrains your chunker**, and it is the thing most RAG designs discover too late — after ingesting an amended agreement and silently retrieving the superseded clause. Decide it now.

2. **Generation must be constrained to cite, and the citation must be checkable.** Do not let the model synthesise a section number — it will, and it will be plausible and wrong. Pass the `citation_label` as structured metadata alongside each chunk, require the model to emit chunk IDs, and **post-hoc resolve** IDs to citations in code. Then verify that quoted spans actually appear in the cited chunk (a substring check is enough to catch most fabrication). This turns a "trust the model" problem into a "verify in code" problem, which is the only version of it you can ship to lawyers.

---

# A. The two cited articles

Both **[VERIFIED — retrieved 2026-09-09]**.

## A.1 Uber — "Enhanced Agentic RAG" (EAg-RAG)

https://www.uber.com/en-IN/blog/enhanced-agentic-rag/

### Setting
**Genie** is Uber's internal on-call copilot: answers engineering questions in Slack with citations from internal docs. Applied to ~40+ engineering **security and privacy policy** PDFs, it failed an SME-curated golden set — answers were "incomplete, inaccurate, or failed to retrieve relevant information in correct detail." EAg-RAG is the rebuild.

### Component 1 — Offline enriched document processing

The framing quote is Judea Pearl: *"The quality of a model depends on the quality of its assumptions and the quality of its data."* Uber's diagnosis was that **the extraction layer, not the retrieval layer, was the bottleneck.**

- **Source migration: PDFs → Google Docs (HTML).** Two reasons: cleaner text extraction, and built-in access control whose metadata can be indexed and enforced at answer time.
- **Loaders they explicitly rejected:** `SimpleDirectoryLoader` (LlamaIndex), `PyPDFLoader` (LangChain), PdfPlumber, PyMuPDF, LlamaParse — no single one handled all policy docs. **Multi-page and nested tables lost formatting, orphaning cells from their row/column context** and corrupting both chunking and semantic search. `html2text`/Markdownify + `MarkdownTextSplitter` also fell short.
- **Custom loader** on the Google Python API, recursively pulling paragraphs, tables, and the table of contents.
- **LLM enrichment at ingest:** an LLM rewrites extracted tables as **markdown tables**; a metadata identifier flags table-bearing chunks so table-aware chunking keeps them whole; each table gets a **two-line summary plus keywords** to lift semantic-search relevance.
- **Metadata schema:** standard fields (title, URL, IDs) plus **document summary, FAQs, and keywords**. FAQs/keywords are generated *after* chunking so they bind to individual chunks; the summary is shared across all chunks from one document.
- **Dual persistence:** chunk embeddings → vector store; document-list artifacts (titles, summaries) and FAQs → an **offline feature store**, for reuse at generation time.

### Component 2 — Answer generation pipeline

Motivating problem: policy chunks differ *subtly* within and across documents (retention rules, data classification, sharing protocols varying by persona and geography), so plain vector similarity retrieves near-misses.

**Pre-retrieval agents** (both read the document-list artifact from the offline store):
1. **Query Optimizer** — disambiguates under-specified queries, decomposes complex ones into sub-queries.
2. **Source Identifier** — narrows to the subset of policy documents likely to hold the answer, using few-shot examples for in-context learning.

Output = an optimized query **+ a document-title allowlist that constrains the search space**.

**Hybrid retrieval** — vector search **plus a BM25 retriever running over the enriched metadata** (summaries, FAQs, keywords), combined as "the union of results from the vector search and the BM25 retriever."

**Post-processing agent** — de-duplicates chunks, then **re-orders context by the positional order of chunks within the original documents.**

**Generation** — original query + auxiliary optimized queries + post-processed context + construction instructions → LLM → Slack.

### Control flow — be precise, because the title oversells it

> **There is no agentic loop.** The article is explicit that the shipped implementation is **strictly sequential**. LangGraph was adopted so the graph *can later* become cyclic. **Self-critique is not in the system** — it is listed under Next Steps, alongside iterative **Chain-of-RAG** for multi-hop queries, multi-modal (image) enrichment, and exposing these capabilities as **tools** an agent selects per query complexity.

So "how retrieval is planned and critiqued" resolves to: **planned** by Query Optimizer + Source Identifier; **critiqued only offline**, by the evaluation harness, never in-loop. The actual flow is a five-stage pipeline:

```
query
  → Query Optimizer      (rewrite + decompose)
  → Source Identifier    (document-title allowlist)   ─┐ both read offline feature store
  → hybrid retrieve      (vector ∪ BM25-over-metadata) ─┘  constrained to the allowlist
  → post-process         (dedupe + restore document order)
  → generate             (orig query + optimized queries + context)
```

**Take the honest lesson:** Uber got a substantial win from **better extraction, LLM-generated retrieval metadata, and a planning step** — not from an agent loop. That is a much cheaper and more reproducible result than the title implies, and it is the part worth copying.

### Evaluation

Two problems drove the redesign of evaluation itself: SME review consumed **weeks per experiment**, and gains were "marginal" and plateauing.

**LLM-as-a-Judge** (citing Gu et al. 2024), three stages: (1) a one-time SME pass producing reference answers and feedback; (2) a batch run of the current chatbot; (3) an LLM scores each response against context C = user query + SME response + evaluation instructions + **freshly retrieved source content from the live RAG pipeline** (this last piece is the clever bit — it boosts the judge's domain awareness rather than relying on the judge's priors).

**Metrics:** 0–5 quality scale plus written reasoning that feeds the next experiment. Evaluation latency dropped "**from weeks to minutes**."

**Measured results:** against a golden set of **100+ queries** — "increasing the percentage of acceptable answers by a **relative 27%** and reducing incorrect advice by a **relative 60%**."

> **Caveat, stated plainly:** these are *relative* deltas with **no published absolute baseline**, so you cannot tell whether acceptable answers went 40%→51% or 70%→89%. Downstream claims (reduced on-call load, productivity) are asserted qualitatively — "a measurable reduction" with no figure.

### Uber-specific vs portable

| Uber-specific | Portable |
|---|---|
| Genie; the **Michelangelo** ML platform; **Langfx** (Uber's internal LangChain service) | The pattern: **plan → constrain → hybrid retrieve → post-process → generate** |
| Internal corpora (engineering wiki, Terrablob PDFs, the policy set) | **Query Optimizer / Source Identifier** as reusable agent roles |
| The Slack help-channel surface and on-call workflow | **Vector ∪ BM25 where BM25 targets LLM-generated metadata, not raw text** ← the most transferable idea in the piece |
| SME golden set + domain-tuned judge instructions | LLM table→markdown enrichment; metadata-identifier-driven **table-aware chunking** |
| Offline feature store for document artifacts | **Chunk-level FAQs/keywords generated post-chunking** |
| | LLM-as-a-Judge harness with SME references |
| | LangChain/LangGraph orchestration; custom API-based loader |

Uber notes a useful second-order effect: proving that better source docs yield better bot answers **pushed teams to maintain cleaner documentation.**

### What to steal for a legal pipeline

1. **The Source Identifier is the highest-value idea here.** For legal, the analogue is a **jurisdiction/document-type/matter router** that constrains the search space before retrieval runs. It also happens to be the fix for §D.4's recall cliff — a narrow allowlist turns a hard filtered-ANN problem into a small exact-search problem.
2. **BM25 over LLM-generated metadata rather than raw text** is a genuinely non-obvious inversion. It also **partially rehabilitates `ts_rank`** (§D.2): if the lexical leg searches short, dense, keyword-rich summaries rather than long boilerplate-laden clause text, the absence of IDF hurts much less. That is a real argument for keeping `tsvector` — pointed at a *summary/keyword* column, not at the full chunk text.
3. **Restoring document order in post-processing** is cheap and directly addresses the "lost in the middle" problem (§A.2). For legal it is close to mandatory — clauses read in document order, and a jumbled context invites the model to misread a proviso as freestanding.
4. **The evaluation harness matters more than any single retrieval trick.** Uber's real unlock was cutting eval from weeks to minutes so they could iterate at all. Build the golden set first.

## A.2 Towards Data Science — "Hybrid Search and Re-Ranking in Production RAG"

https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/

Context: an IT-helpdesk RAG on Weaviate + LlamaIndex.

### Fusion method — **not RRF**

The article uses **Weaviate's "Relative Score Fusion"** — a *linear score blend* controlled by a single `alpha`, **not** a rank-based formula. **No RRF `k` constant appears anywhere in the piece.** (Worth flagging explicitly, since the article is often cited as an RRF reference. It isn't one.)

- `alpha = 1.0` → pure dense; `alpha = 0.0` → pure BM25.
- **Production value chosen: `alpha = 0.5`** (equal weighting).

### How the weight was chosen — the useful part

Tuned empirically against **150 labeled query-document pairs** from helpdesk history, scored on Hit Rate and MRR:

| Alpha | Hit Rate | MRR |
|---|---|---|
| 0.00 (pure BM25) | 0.71 | 0.58 |
| 0.25 | 0.80 | 0.66 |
| **0.50** | **0.83** | **0.69** |
| 0.75 | 0.81 | 0.67 |
| 1.00 (pure dense) | 0.73 | 0.61 |

**Both pure modes underperformed every blend.** That is the single most useful number in the article: hybrid beat pure-dense by **+0.10 hit rate** and pure-BM25 by **+0.12**, and the curve is broad and flat between 0.25 and 0.75 — so *some* blending matters a lot and the exact weight matters little.

Corpus-dependent guidance from the author: long-form narrative documentation may favour **0.65–0.75**; corpora heavy in **error codes and product names** tend toward **0.35–0.5**. "No universal correct value."

> **Read across to legal:** legal text is closer to the "error codes and product names" end — statute numbers, docket numbers, defined terms, citations. That argues for **more lexical weight, not less**, and is a direct evidence-backed argument *against* removing the lexical leg (§D.2).

### Candidate counts — the two-stage funnel

- **Stage 1 (retrieval):** `similarity_top_k = 20` candidates from hybrid search.
- **Stage 2 (rerank):** `top_n = 5` passed to generation — "sufficient for multi-part questions without cluttering the prompt."

Rationale: **cross-encoders cannot precompute.** A bi-encoder embeds documents once at index time; a cross-encoder needs a forward pass per (query, document) pair, so full-corpus scoring is infeasible. Hence funnel, always.

### Reranker choice

**`cross-encoder/ms-marco-MiniLM-L-6-v2`** via `sentence-transformers`, or LlamaIndex's `SentenceTransformerRerank`. Chosen as MS MARCO-trained and the most widely used open-source cross-encoder for general retrieval. Fine-tuning on domain-labeled pairs is suggested for specialised content.

> **Note the dependency collision with §E.3:** this recipe's reranker is a local `sentence-transformers` cross-encoder, i.e. **torch**. If you drop torch, you need a hosted reranker (Cohere Rerank, Voyage `rerank-2`, Jina Reranker) instead. Do not read this article as an argument for keeping torch — read it as an argument for *having a reranker*, by whatever means.

### Latency / cost

- Cross-encoder rerank of **20 documents on CPU** with a lightweight model: **~80–120 ms** added query latency. (Cheap. This is the point.)
- `response_mode="compact"` merges the 5 chunks into one LLM call instead of per-chunk calls, cutting latency substantially; `tree_summarise` is the small-context-window alternative.
- **No dollar figures are given.**

### Retrieval quality impact (RAGAS)

| Config | Ctx Precision | Ctx Recall | Answer Rel. | Faithfulness |
|---|---|---|---|---|
| Dense only (α=1.0) | 0.61 | 0.74 | 0.78 | 0.82 |
| Hybrid (α=0.5) | 0.71 | 0.83 | 0.81 | 0.85 |
| Hybrid + rerank (top 5) | **0.79** | 0.84 | **0.87** | **0.89** |

**The decomposition is the lesson, and it is worth internalising:**
- **Recall gains came almost entirely from adding BM25** (0.74 → 0.83). Reranking barely moved recall (0.83 → 0.84) — *because a reranker can only reorder what retrieval already found.*
- **Precision is where reranking paid off** (0.71 → 0.79).

> **These two levers are not substitutes; they fix different failures.** Lexical search fixes "the right document was never retrieved." Reranking fixes "it was retrieved at rank 11." Removing the lexical leg and adding a reranker does **not** recover the lost recall — the reranker never sees the missing document. This is the cleanest available refutation of "just use vectors + a reranker," and it bears directly on the §D.2 decision.

### Metadata pre-filtering

Applied before scoring to shrink the candidate pool: department equality, `updated_at` > 365-day cutoff, classification ≠ confidential, ANDed. **The stated risk is over-filtering** — excluding the answer document yields a *confident wrong answer* from what remains. (For legal, substitute: filtering to the wrong jurisdiction, or to a superseded contract version — see §C.3's versioning point. Same failure mode, higher stakes.)

### The motivating anecdote

The failing query ranked its correct document at **position 11**, just outside the top 10. At `alpha = 0.5` it moved to **rank 4**, pulled up by the BM25 signal. This matters because of **"lost in the middle"** — a chunk at position 8 of 10 sits in a zone models attend to less reliably. Connects directly to Uber's "restore document order" post-processing step (§A.1).

---

# F. The "195" stack — LangExtract, PageIndex, Graphiti

All three **[VERIFIED 2026-09-09 from their READMEs]**; the *composition* sketch at the end is my synthesis and is speculative.

## F.1 LangExtract (Google)

https://github.com/google/langextract · Apache 2.0 · **"not an officially supported Google product."**

**What it is:** a Python library that uses LLMs to pull **structured records out of free-form text** "based on user-defined instructions," while "ensuring the extracted data corresponds to the source text."

**API — three pieces:**
```python
import langextract as lx

result = lx.extract(
    text_or_documents=contract_text,      # raw string or a URL
    prompt_description=textwrap.dedent("""..."""),   # plain-language instruction
    examples=[lx.data.ExampleData(
        text="...",
        extractions=[lx.data.Extraction(
            extraction_class="governing_law",
            extraction_text="the laws of the State of Delaware",   # VERBATIM from example text
            attributes={"jurisdiction": "DE"},
        )],
    )],
    model_id="gemini-3.5-flash",
    extraction_passes=2,       # recall on long docs
    max_workers=10,            # parallelism
    max_char_buffer=8000,      # chunk size for extraction
    output_schema=...,         # harder constraints (enum-restricted attrs) on Gemini/OpenAI
)
```
- **`examples` do the heavy lifting** — "Examples drive model behavior." `extraction_text` must be copied **verbatim** from the example text, not reworded; the library emits "Prompt alignment" warnings otherwise.
- **Output:** `result.extractions`, saveable to JSONL via `lx.io.save_annotated_documents(...)`; `lx.visualize(...)` renders a self-contained interactive HTML review page (scales to thousands of entities).
- **Models:** Gemini (default `gemini-3.5-flash`; Flash-Lite for volume, Pro for hard reasoning), OpenAI (`langextract[openai]`, with a Batch API mode), **Ollama** for local (no `output_schema`), plus a plugin system (`@router.register(...)`) for custom providers.

**The headline feature — source grounding:** every extraction gets a **`char_interval`** mapping it to its exact location in the source text. When the model invents content or lifts it from the few-shot examples, the span **cannot be resolved and the field becomes `None`** — so `[e for e in result.extractions if e.char_interval]` filters to only anchored results.

> **That is a hallucination detector that costs nothing.** For legal work this is the whole ballgame: it converts "did the model make this up?" from a judgement call into a boolean. It is the same idea as §C.3's "verify quoted spans appear in the cited chunk," shipped as a library.

**Where it sits in the pipeline: beside chunking and embedding, not in the chain.** LangExtract does not chunk *for* retrieval and does not embed. It reads the extracted text (post-Docling) and emits **structured metadata**. That metadata then:
- populates the **typed metadata columns** a `PGVectorStore` gives you (§E.1) — `governing_law`, `effective_date`, `parties`, `term_end` — which is exactly what makes §D.4's partial indexes and pre-filters possible;
- feeds the **Source Identifier**-style router (§A.1);
- feeds the **lexical leg over enriched metadata** (§A.1's most portable idea).

**Honest assessment:** the most immediately useful of the three, and the least speculative. It has a real, novel, verifiable feature (`char_interval` grounding). Risks: it's an LLM call per document (cost scales with corpus), it's Gemini-first, and "not an officially supported Google product" means no support SLA.

## F.2 PageIndex (VectifyAI)

https://github.com/VectifyAI/PageIndex · open source **and** hosted (PageIndex Cloud).

**The claim:** "similarity ≠ relevance," and establishing relevance "requires **reasoning**." For dense professional documents, similarity search "misses what is relevant but not similar." It describes itself as "a **vectorless**, **reasoning-based RAG** engine that **mirrors how humans read**."

**How it works** — "inspired by AlphaGo," two stages:
1. **Index:** build a **tree-structure index** per document. Crucially, the **skeleton "is extracted from the document layout without an LLM"**; the model only summarizes and refines nodes — so a cheap model suffices at index time.
2. **Retrieve:** "agentically **search that tree** with LLM reasoning." Here the guidance flips: "use the best model you can afford."

Units are **natural document sections**, not fixed-size chunks. Results trace back to "explicit references" rather than "vibe retrieval."

**Claimed benchmarks** — treat as vendor claims:
- **FinanceBench: 98.7% accuracy**, "vastly outperforming vector-based RAG" (chart alt-text cites vector RAG at 50%).
- **Indexing: ~$0.001/page** locally; 9-page to 1,098-page documents indexed in **13 seconds to 4.5 minutes**.
- **vs. stuffing the whole PDF into context:** native PDF input costs **2.1× more at 52 pages, 16.6× more at 420**; at 805 pages it exceeds the context window.
- **PageIndex-OSS-Benchmark:** 62 lookup questions across 34 PDFs (1,945 pages), designed so errors indicate retrieval/reading failure rather than flawed reasoning.

**Local vs Cloud:** `pip install -U pageindex` runs indexing, retrieval and chat locally with your own LLM key. Cloud adds OCR, image understanding, managed storage, **line-level rather than page-level citations**, an MCP server, and a "PageIndex File System" for reasoning across millions of documents.

**How it runs *parallel* to a vector store — the framing that matters.** Ignore the "vectorless" marketing; you don't have to choose. The two fail in **opposite** ways:

| | Vector/hybrid retrieval | PageIndex tree search |
|---|---|---|
| Good at | "Find passages about indemnification caps" — semantic breadth, unknown location | "What does § 7.3(b) of *this* agreement say, and what does it cross-reference?" — navigation within a known document |
| Bad at | Precise navigation; cross-references; "the relevant but not similar" | Corpus-wide search — it reasons *within* a document tree, so it needs to already know which document |
| Latency | ~10s of ms | Seconds (LLM reasoning per hop) |
| Cost/query | ~$0 | Real LLM cost per query |

**The natural composition:** hybrid retrieval **selects the documents**; PageIndex **navigates within them**. That is exactly Uber's Source Identifier → retrieval pattern (§A.1) with a smarter second stage, and it maps onto legal work precisely — "find the agreements with a Delaware choice of law" (vector/metadata) then "walk this agreement's termination provisions and their cross-references" (tree).

Note also that PageIndex's tree is **the same structure Docling already gives you** (§B.3). If you persist the `DoclingDocument` and index by `self_ref`, you can implement tree navigation over your own data without adopting PageIndex at all. **That is probably the right call for a first version** — the idea is more valuable than the dependency.

**Honest assessment: the benchmark claims are the least credible thing in this section.** 98.7% on FinanceBench vs 50% for "vector RAG" is a comparison against an unspecified and probably weak baseline. The *architectural idea* is sound and worth stealing; the numbers are marketing.

## F.3 Graphiti (Zep)

https://github.com/getzep/graphiti · paper: *Zep: A Temporal Knowledge Graph Architecture*, arXiv:2501.13956.

**What it is:** a framework for **temporal context graphs** that "continuously integrates user interactions, structured and unstructured enterprise data, and external information" into one queryable graph. Underpins Zep's commercial product.

**Graph model:**

| Element | Role |
|---|---|
| **Episodes** | Raw ingested data — "the ground truth stream"; every derived fact traces back to it |
| **Entity nodes** | People, products, policies, concepts — with summaries that update as new data arrives |
| **Edges (facts)** | Entity → Relationship → Entity triplets, each with a **validity window** |
| **Custom types** | Developer-supplied ontology via **Pydantic models** |

**Bi-temporality is the differentiator.** Versus GraphRAG's "basic timestamp tracking," Graphiti does "explicit bi-temporal tracking with **automatic fact invalidation**." Each fact records when it became true and when it was superseded, so you can ask what holds *now* or what held *then*. **Superseded facts are invalidated, not deleted.** New episodes fold in incrementally without recomputing the graph.

> **This is a direct match for §C.3's versioning problem.** "What did this contract's termination clause say before Amendment No. 2?" and "which statute was in force on the date of breach?" are bi-temporal queries, and they are the questions legal RAG most often gets silently wrong. Of the three tools in §F, Graphiti addresses the most severe *correctness* gap.

**Ingestion cost profile** — and here the README is unhelpfully quiet:
- Ingestion **depends on the LLM for entity/edge extraction and deduplication**, and **requires structured (JSON) output**. Providers that genuinely enforce schemas (OpenAI, Anthropic, Gemini) are recommended; weaker/smaller models emit malformed JSON that surfaces as extraction failures. `OpenAIGenericClient` has a `structured_output_mode` toggle (native `json_schema` by default, or `json_object` which injects the schema into the prompt — better for local servers that accept schema requests without enforcing them).
- **No token counts, per-episode pricing, or ingestion latency figures are published.** **[This is a real gap — you cannot budget a Graphiti ingestion from the docs. Prototype on 10 documents and extrapolate.]**
- The only concurrency signal is `SEMAPHORE_LIMIT`, **default 10** parallel operations, to avoid provider 429s.
- Query side: "typically sub-second latency" vs GraphRAG's "seconds to tens of seconds"; managed Zep advertises sub-200ms.

**What it needs from an upstream extraction step:** clean, well-scoped **episodes**. Feeding it raw PDF text produces a noisy graph. It wants text that is already segmented into coherent units with an ontology to target — which is **exactly LangExtract's output**, and exactly Docling's chunks. Defining custom Pydantic entity/edge types up front (rather than letting the ontology emerge) is what separates a useful legal graph from a pile of noise.

**Backends:** Python 3.10+, plus **Neo4j 5.26, FalkorDB 1.1.2, or Amazon Neptune** (Neptune additionally needs an OpenSearch Serverless collection for full-text search). **Kuzu 0.11.2 ships but is deprecated** — upstream unmaintained, driver raises `DeprecationWarning`. `pip install graphiti-core` with extras `[falkordb]`, `[neptune]`, `[anthropic]`, `[google-genai]`. Expects `OPENAI_API_KEY` by default.

> **The operational cost is a second database.** This is the single biggest objection: a project on Timescale Cloud Postgres would be adding Neo4j or FalkorDB — a new datastore, backup story, migration story, and failure mode. Weigh that against the bi-temporal capability, which Postgres can also express (a `valid_from`/`valid_to` range on a facts table with an exclusion constraint) if you don't need graph traversal.

**Retrieval:** blends **semantic embeddings + keyword BM25 + graph traversal**, avoiding LLM summarization at read time. Reranking by **graph distance**, plus predefined "search recipes" for node search. Pluggable cross-encoder rerankers: `OpenAIRerankerClient`, `GeminiRerankerClient` (defaults to `gemini-2.5-flash-lite`, scoring relevance via boolean classification over **log probabilities** — a neat, cheap trick worth knowing independently).

## F.4 How they compose — and what's speculative

```
                        ┌──────────────────────────────────────┐
   PDF ──► Docling ─────┤ DoclingDocument (tree + prov/bbox)    │
          (§B)          └───┬──────────────┬───────────────┬───┘
                            │              │               │
        ┌───────────────────┘              │               └──────────────┐
        ▼                                  ▼                              ▼
  HybridChunker                      LangExtract                    persist whole doc
  (§B.2) chunks                      (§F.1) entities                (enables tree nav,
        │                            + char_interval                 parent-child, §C.1)
        │                                  │
        ▼                                  ├──────► typed metadata columns ──► §D.4 filters
   embed contextualize()                   │        (PGVectorStore, §E.1)
        │                                  │
        ▼                                  ├──────► lexical leg over summaries/keywords
   Postgres (pgvector[scale])              │        (§A.1's portable idea, §D.2)
   + tsvector / metadata                   │
        │                                  └──────► Graphiti episodes (§F.3)
        │                                             [optional, 2nd DB]
        ▼
   RRF fusion (§D.3) ──► rerank (§A.2) ──► generate with citations (§C.3)
                                    ▲
   PageIndex-style tree nav (§F.2) ──┘   [runs parallel, on the selected documents]
```

**Blunt assessment of maturity:**

| Piece | Maturity | Verdict |
|---|---|---|
| Docling + HybridChunker | **Mature.** Real project, LF AI & Data, wide use. | Build on it. |
| pgvector / pgvectorscale + RRF + rerank | **Mature.** Boring, well-understood. | This *is* the system. Everything else is optional. |
| LangExtract | **Young but low-risk.** Small surface, Apache 2.0, degrades gracefully (worst case: bad metadata you ignore). `char_interval` grounding is genuinely valuable. | **Adopt second.** Highest value/risk ratio of the three. |
| PageIndex | **Idea: sound. Product: young. Benchmarks: marketing.** | **Steal the idea, skip the dependency** — implement tree navigation over your stored `DoclingDocument`. |
| Graphiti | **Real, with a paper — but a second database, unpublished ingestion cost, and an LLM call per episode.** | **Defer.** Revisit only when you have a concrete bi-temporal requirement (contract amendments, statutes over time) that a Postgres validity-range table demonstrably cannot serve. |

**The thing to say out loud:** the "195 stack" is three tools solving three *different* problems (structured extraction, intra-document navigation, temporal facts), and none of them is a retrieval system. **None replaces §D.** A team that adopts all three before getting hybrid retrieval + reranking + a golden-set eval working will have a very sophisticated pipeline and no way to tell whether it is any good. Uber's actual lesson (§A.1) points the other way: fix extraction, add a planning step, build the eval harness, and measure.

---

# Decisions this forces

> Decisions 1–3 were rewritten after the scout established the deployment is **Tiger Cloud**, that the repo already builds `bm25` + `diskann` indexes (`documents/model.py:114`, `:122`), and that there is **no LangChain `VectorStore` in the codebase** — retrieval is hand-written SQL (`documents/repository.py:452`).

1. **Remove `tsvector`, or keep it — given `pg_textsearch` is already in play.**
   **Recommendation: remove `tsvector`; keep the lexical leg on `pg_textsearch` BM25.** These are different things and conflating them is the expensive mistake. `pg_textsearch` is a Tiger Data first-party extension, **available on Tiger Cloud** (verified), giving real BM25 with tunable `k1`/`b`, index-resident top-k via Block-Max WAND, and Postgres stemming retained through `text_config='english'`. Running `tsvector` alongside it is redundant and would double-count the lexical signal in RRF.
   *Tradeoff:* **one open question gates this — does `pg_textsearch` support phrase queries and boolean operators?** `tsvector` has `phraseto_tsquery` and positional `<->`; for legal defined terms ("reasonable efforts", "Permitted Encumbrance") phrase search is a real capability. Verify before dropping the column; if phrase search is missing, keep a narrow `tsvector` for that purpose only, never as a scoring leg. Do **not** read this as "full-text isn't worth it" — §A.2's RAGAS numbers show the lexical leg is where recall comes from (0.74 → 0.83), and a reranker cannot recover a document retrieval never found.

2. **pgvector HNSW vs pgvectorscale `diskann` — already settled, keep it settled.**
   **Recommendation: stay on `diskann`.** Both `pgvector` and `pgvectorscale` are **enabled by default** on Tiger Cloud (verified) — no extension decision to make — and the repo already tunes `diskann.query_search_list_size` and `diskann.query_rescore` per query, which is the correct surface. `query_rescore` is *the* recall lever under `storage_layout='memory_optimized'`, since the graph walk runs over SBQ-compressed vectors.
   *Tradeoff:* you are locked to Tiger-specific extensions, and **Timescale publishes no head-to-head benchmark against pgvector HNSW** — their numbers are vs Pinecone (28× p95, 16× throughput at 99% recall on 50M vectors). So the choice rests on the storage/scale argument, not on a published speed claim. Also note `diskann`'s native `labels` filtering and parallel builds are **mutually exclusive** — pick one per index.

3. **Adopt `PGVectorStore`, or stay with the tuned raw SQL.**
   **Recommendation: stay with raw SQL. Do not adopt `PGVectorStore`.** It is not a migration question — there is nothing to migrate. `PGVectorStore` **cannot express a `diskann` index** (its index classes are `HNSWIndex`/`IVFFlatIndex`, both pgvector), **cannot set `SET LOCAL diskann.*` GUCs** in its own query transaction, and its `hybrid_search_config`/`reciprocal_rank_fusion` is shaped for a two-leg `tsvector` fusion, not a three-branch fusion with `pg_textsearch`'s `<@>` operator. Adopting it would be a capability downgrade.
   *Tradeoff:* you forgo a portable abstraction — but portability across vector stores is a benefit you already declined by choosing pgvectorscale + `pg_textsearch`. Keep LangChain's `Embeddings` interface and `Document`, and wrap the existing repository query in a ~20-line `BaseRetriever` for LangGraph. Record an expiry condition in the ADR: *if we ever leave Tiger Cloud for plain-pgvector managed Postgres, revisit — the capability gap closes there.* (`PGVector` is formally deprecated as of v0.0.14, so nobody should propose it either.)

4. **Drop `langchain-docling` from `pyproject.toml`?**
   **Recommendation: yes, drop it.** It is declared as `>=2.0.0` with **zero imports** — the code already calls `docling` directly, which is the right architecture. An unbounded pin on a package nothing imports is pure transitive version pressure on `docling` and `langchain-core`.
   *Tradeoff:* none material. Confirm with `rg 'langchain_docling|langchain-docling' src/ tests/` returning only the manifest. Calling `docling` directly is also the correct choice on the merits: `langchain-docling` flattens `DocMeta` and loses bounding boxes, which pin-cites and PDF highlighting need (§C.3).

5. **Docling `HybridChunker` vs a custom legal clause splitter.**
   **Recommendation: `HybridChunker` as the engine, plus a clause-numbering post-pass.** Docling's layout model often misses contract clause numbers styled as inline bold runs rather than structural headings; regex the numbering and rebuild the tree. Verified bonus: peer merging only combines chunks with **the same headings & captions**, so `merge_peers=True` is safe for citation provenance.
   *Tradeoff:* custom code you maintain — but there is no way around it, and it is the most likely place a Docling legal pipeline needs work. Budget for it rather than discovering it.

6. **Embed `chunk.text` vs `chunker.contextualize(chunk)` — and store which?**
   **Recommendation: embed `contextualize(chunk)`, display and cite `chunk.text`, store both columns.** Docling's own docs say `contextualize()` is "typically used to feed an embedding model."
   *Tradeoff:* ~15% storage overhead and two columns to keep in sync — but embedding bare text makes clauses whose only topical signal lives in the heading ("Force Majeure") nearly unretrievable, and displaying contextualized text pollutes every quote with heading boilerplate. **Verify that `max_tokens` applies to the contextualized serialization, not the bare text** — if not, every contextualized chunk silently overflows.

7. **Drop `sentence_transformers` + torch, or keep local models?**
   **Recommendation: drop `sentence-transformers` and torch; hosted embeddings + hosted reranker; keep `tiktoken`.** Saves ~1–2 GB of image. `transformers`/`AutoTokenizer` costs only ~30 MB and does **not** pull torch — cheap if you need it. Docling's `chunking-openai` extra exists precisely for the tiktoken path.
   *Tradeoff:* per-query reranker cost and a network dependency in the hot path; and **you must also move Docling's OCR off EasyOCR (→ RapidOCR/Tesseract) or torch returns anyway.** These two decisions are coupled and usually made by different people. It also forecloses **late chunking**, which needs a self-hosted long-context encoder.

8. **Overlap: fixed 10–20% everywhere vs zero-on-structural-boundaries.**
   **Recommendation: overlap 0 where clause boundaries are reliable; 10–15% only on fallback splits that cut mid-clause, flagged as impure.**
   *Tradeoff:* you must trust your structural boundaries (decision 5) — but overlap across clause boundaries corrupts citation attribution, which for legal work is a correctness failure, not a quality nit. Use parent-child retrieval and ±1 neighbour expansion instead; they do overlap's job without duplicating the index.

9. **Free heading-path context vs LLM-generated contextual prefixes (Anthropic's technique).**
   **Recommendation: ship `contextualize()` first (free, deterministic, no hallucination risk), measure, then add LLM prefixes selectively on the *low-structure* subset** — case law, correspondence, exhibits — where headings are thin.
   *Tradeoff:* you leave some of Anthropic's reported ~35% failure-rate reduction on the table initially — but LLM prefixes cost real money per chunk (economic only with prompt caching) and heading paths recover much of the benefit on well-structured documents at zero cost.

10. **RRF vs weighted score fusion, and do you tune `k`?**
    **Recommendation: RRF with `k = 60` and per-leg *weights*; do not tune `k`.** `k` only controls how sharply rank 1 beats rank 10; weights are the lever that matters. Cormack et al. (2009) used `k=60` untuned and never claimed it optimal. **With three branches, set weights deliberately** — an unweighted three-way RRF hands 2/3 of the influence to the text-derived legs.
    *Tradeoff:* RRF discards score calibration a weighted blend could exploit — §A.2 got its best result from an explicit `alpha` blend (0.5, hit rate 0.83 vs 0.73 pure-dense). But RRF is robust to `<@>` and cosine being on incomparable scales, which a linear blend is not. Legal corpora (identifiers, citations, defined terms) argue for **more lexical weight**, per §A.2's own corpus guidance.

11. **Do you build the reranking stage at all in v1?**
    **Recommendation: yes — the cheapest large quality win available.** §A.2 measured ~80–120 ms on CPU for 20 candidates, context precision 0.71 → 0.79.
    *Tradeoff:* adds a stage and a vendor (if hosted, per decision 7). But it fixes a *different* failure than hybrid search: reranking barely moved recall (0.83 → 0.84). Build both legs; they are not substitutes.

12. **Chunk identity: `(document, locus)` vs `(document_version, locus)`.**
    **Recommendation: version it now.** Make `document_version` part of chunk identity from the first migration.
    *Tradeoff:* more complex ingestion and dedupe up front — but retrieving a clause superseded by Amendment No. 2, or a statute not in force on the date of breach, is a silent wrong answer with professional-liability consequences. Cheapest decision to make now, most expensive to retrofit.

13. **Multi-tenant / jurisdiction isolation: shared index vs partial indexes vs partitioning.**
    **Recommendation: partition (list or hash) with local indexes** once you have more than a handful of distinct filter values; partial indexes for 3–5 stable categories. `pg_textsearch` supports partitioned tables, so both legs can follow the same partitioning scheme.
    *Tradeoff:* partition management and cross-partition fan-out — but pgvector's own docs give the decisive reason: a shared graph lets **one tenant's vectors degrade another tenant's recall**. That is a correctness issue, and it is invisible until someone audits it. Note that `diskann`'s `labels` filtering is an alternative here, at the cost of parallel index builds.

14. **Adopt any of the "195" stack in v1?**
    **Recommendation: LangExtract yes (second, after core retrieval works); PageIndex as an idea implemented over your stored `DoclingDocument`, not as a dependency; Graphiti deferred.**
    *Tradeoff:* you forgo Graphiti's bi-temporal fact invalidation, which genuinely matches decision 12's problem — but it costs a **second database** (Neo4j/FalkorDB), an LLM call per episode, and has **no published ingestion cost figures**. A Postgres validity-range table serves the same need until you need graph traversal.

15. **Build the golden-set evaluation harness before or after the retrieval work?**
    **Recommendation: before.** Uber's actual unlock was cutting evaluation from weeks to minutes; every other decision in this list is unfalsifiable without it.
    *Tradeoff:* delays visible feature work by days and needs SME time — but every recommendation above is a prior, not a measurement, and without an eval you cannot tell which ones were wrong on your corpus.

---

## Verification queue — cheapest first

Run these against the live Tiger Cloud instance and the installed packages before acting on anything above:

```sql
-- 1. Ground truth on extensions and versions (settles §D.1, §D.2)
SELECT name, default_version, installed_version
FROM pg_available_extensions
WHERE name IN ('vector','vectorscale','pg_textsearch','pg_trgm','timescaledb');
```
2. **Does `pg_textsearch` support phrase and boolean queries?** — gates decision 1.
3. **Is the repo's BM25 leg using the implicit `<@>` form inside PL/pgSQL or against a partial index?** — if so it is silently not using the index (§D.3).
4. **Is the BM25 `ORDER BY` sorted `ASC`?** — `<@>` returns *negative* scores; `DESC` silently returns worst-first.
5. `rg -n 'diskann' $(python -c "import langchain_postgres,os;print(os.path.dirname(langchain_postgres.__file__))")` — expect zero hits, confirming decision 3.
6. **Does `HybridChunker`'s `max_tokens` bound the contextualized serialization or the bare text?** — chunk a document, tokenize `contextualize(chunk)` for all chunks, assert max ≤ limit (§B.4).
7. `rg 'langchain_docling|langchain-docling' src/ tests/` — expect manifest-only, confirming decision 4.




