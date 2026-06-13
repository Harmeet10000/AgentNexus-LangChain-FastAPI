# RAG Decisions

## Source

- Title: `How to Design a RAG Pipeline for 10 Million Documents with Zero Hallucination (Live Demo)`
- Author: `Vishal Mysore`
- Format: cleaned and structured notes from the provided text

## Core Thesis

At small scale, naive RAG works:

- chunk documents
- embed them
- run similarity search
- pass top chunks to the LLM

At 10 million documents, that breaks down.

The hard problems are not mysterious. They are predictable engineering problems:

- retrieval latency
- index maintenance cost
- precision degradation at scale
- hallucinations from weak grounding
- lack of verifiability
- silent production failure without evaluation and observability

The main claim is simple:

> At large scale, retrieval quality matters more than the frontier model itself.

Five well-retrieved chunks with a mid-tier model will outperform a stronger model reasoning over bad retrieval.

---

## Why 10 Million Documents Changes Everything

At 1,000 documents, brute force is still tolerable.

At 10 million documents:

- brute-force vector search takes minutes, not milliseconds
- no frontier model can directly inspect a meaningful fraction of the corpus
- one bad retrieval step poisons everything downstream
- hallucinations compound from one wrong fact into many plausible false extensions

This is why the retrieval layer becomes the real foundation of the system.

---

## Decision 01: Ingest and Normalize Documents

### Problem

Large corpora come from mixed sources:

- PDFs
- Word docs
- HTML
- scanned images
- database exports
- Markdown
- internal wikis

These all carry different noise, encodings, and structural quirks.

### Required actions

- strip HTML tags, PDF control characters, and footnote noise
- normalize Unicode with NFC
- remove non-printable characters and control sequences
- standardize whitespace and newline handling
- detect multi-language content and route it separately
- attach metadata during ingest

### Required metadata

- source
- date
- author
- domain
- version

### Why this matters

Silent normalization failures destroy recall. Example:

- `Vishal Mysore` with a non-breaking space will not BM25-match the same text using a regular space

At 10M scale, the recommendation is a distributed, idempotent ingest pipeline using tools like Kafka plus Spark or Flink, with a content hash so unchanged documents are skipped.

---

## Decision 02: Use Hybrid Retrieval, Not Embeddings Alone

### Problem

Embedding-only retrieval fails on exact identifiers.

Example:

> What did Clause 4.2.1 of the NDA say about termination?

Dense retrieval may return chunks about termination in general. BM25 is far better at exact clause references.

### BM25

BM25 is useful because it rewards rare exact terms and normalizes for document length.

High-level form:

```text
score(q, d) = sum over query terms of:
IDF(t) * (tf(t, d) * (k1 + 1)) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

k1 = 1.2
b = 0.75
```

### Vector retrieval

Suggested examples:

- `all-MiniLM-L6-v2` for speed
- `text-embedding-3-large` for higher accuracy

Suggested practices:

- mean pooling
- L2 normalization
- cosine similarity on normalized vectors

### Fusion

```text
fused_score = alpha * cosine_similarity + (1 - alpha) * normalized_bm25
```

Where `alpha` depends on the domain:

- legal documents: lean BM25, for example `0.3 vector / 0.7 BM25`
- conceptual knowledge bases: lean vector, for example `0.7 vector / 0.3 BM25`

### Recommended production pattern

- run vector retrieval and BM25 retrieval in parallel
- return top 30 from each branch
- union candidates
- fuse scores
- pass top 15 to reranking

---

## Decision 03: Use Two-Stage Retrieval with ANN and Reranking

### Stage 1: ANN retrieval

Exact nearest-neighbor search over 10M high-dimensional vectors is too slow for real-time systems.

Use ANN indexes instead.

Common options:

- HNSW: strongest recall/speed tradeoff in many production systems
- IVF-PQ: lower memory footprint, common in FAISS
- ScaNN: high throughput at extreme scale

The article's recommendation favors HNSW for production maturity and practical performance.

### Stage 2: Cross-encoder reranking

ANN is fast but imprecise because query and chunk are scored independently.

Cross-encoders score them jointly:

```text
CrossEncoder([query, chunk]) -> relevance_score in [0, 1]
```

Examples:

- `ms-marco-MiniLM-L-6-v2`
- `ms-marco-MiniLM-L-12-v2`

### Why reranking matters

Initial retrieval may surface the right chunk at rank 17. The reranker can move it to rank 1 after reading the query and chunk together.

At scale:

- rerank only the top 15 to 30 candidates
- never rerank the whole corpus

---

## Decision 04: Score Source Confidence Before Generation

Every retrieved chunk should carry a confidence score before it reaches the prompt.

### Confidence components

- retrieval confidence from the fused retrieval score
- source freshness
- source authority
- cross-chunk agreement

### Example weighting

```text
confidence =
  0.5 * retrieval_score +
  0.2 * freshness_score +
  0.2 * authority_score +
  0.1 * agreement_score
```

### Threshold gate

If all retrieved chunks fall below a threshold like `0.65`, do not generate an answer.

Return:

> Insufficient information found in the knowledge base.

This is not a product failure. It is a correctness feature.

---

## Decision 05: Constrain Generation Hard

This is the central anti-hallucination decision.

The model must be explicitly forbidden from using knowledge outside the provided context.

### Example prompt contract

```text
System: You are a citation-backed AI assistant. Answer using ONLY the provided Context sections below.

Rules:
1. Every claim you make must be supported by the provided Context.
2. Cite every assertion with [Source N] where N is the context section number.
3. If the Context does not contain the answer, respond with exactly:
   "The provided documents do not contain sufficient information to answer this question."
4. Do NOT use any knowledge from your training data to fill gaps.
5. Do NOT speculate, extrapolate, or make inferences beyond what the Context explicitly states.
```

### Model settings

Use low temperature for RAG:

- `0.0`
- `0.1`

High temperature increases creativity, which is the opposite of what grounded retrieval systems need.

---

## Decision 06: Make Every Response Citation-Backed

Enterprise RAG requires claim-level traceability.

### Example answer style

```text
Vishal Mysore joined the company in 2019 [Source 1] and led the cloud migration initiative [Source 3], which reduced infrastructure costs by 40% [Source 1, Source 2].
```

### Store per response

- exact chunk text used
- source document ID
- source document version
- page number
- character offset
- retrieval score at generation time
- timestamp of the document version used

### Why this is mandatory

Legal, compliance, and audit teams need to trace every generated claim back to a source. Without that, the system is not enterprise-ready.

Citations also create a repair loop:

- user disputes a claim
- system identifies the exact chunk
- fix source content or retrieval behavior
- invalidate related cache entries

---

## Decision 07: Add a Hallucination Fallback Layer

Even with constrained prompting and citations, failures still slip through.

### Three-pass verification

#### Pass 1: assertion extraction

Extract factual claims from the answer:

- numbers
- names
- dates
- percentages
- named entities

Use regex plus NER.

#### Pass 2: grounding check

For each assertion, verify that it appears in the retrieved context using fuzzy matching, not just exact string matching.

#### Pass 3: faithfulness score

```text
faithfulness = verified_assertions / total_assertions
```

If `faithfulness < 0.8` and there are flagged assertions, surface a warning.

### Fallback actions

- show response with inline warning on unverified claims
- rerun with `temperature = 0.0` and stricter prompt
- return `cannot verify` if the second pass still fails
- escalate to human review for high-risk cases

### Production note

This can run as an async post-processor:

- stream answer first
- verify in parallel
- display warning shortly after completion if needed

---

## Decision 08: Run Continuous Evaluations

RAG systems must be evaluated continuously in production, not just offline.

### Three core metrics

#### Context relevance

Are the retrieved chunks actually relevant?

```text
context_relevance = overlap(query_tokens, context_tokens) / |query_tokens|
```

Low score usually means a retrieval problem.

#### Faithfulness

Is the answer grounded in the retrieved context?

```text
faithfulness = verified_claims / total_claims
```

Low score usually means a generation or grounding problem.

#### Answer relevance

Does the answer actually answer the question?

```text
answer_relevance = overlap(query_tokens, answer_tokens) / |query_tokens|
```

Low score despite strong context relevance suggests the model is ignoring the retrieved context.

### Additional production metrics

- latency p50, p95, p99 for each pipeline stage
- cache hit rate
- retrieval diversity
- user rejection rate

### Suggested pipeline

- compute eval scores per query
- log to a time-series store
- visualize on dashboards
- alert on regressions

Example:

- if rolling one-hour faithfulness drops below `0.75`, page someone

---

## Decision 09: Use Caching and Memory Intentionally

At scale, repeated queries are normal. Recomputing the full pipeline each time wastes both time and money.

### Two-level cache

#### Level 1: exact query result cache

Cache key shape:

```text
hash(query + retrieval_config + model)
```

Cache the full response plus citations.

Invalidate when any source document used in the answer changes.

#### Level 2: semantic near-duplicate cache

- cache query embeddings
- if a new query is highly similar to a prior one, for example cosine `> 0.97`, reuse the prior result

### Memory layer

#### Session memory

Conversation systems should remember prior references.

If a user discussed `Clause 4.2.1` three turns ago, they should not need to restate it.

#### Long-term memory from human feedback

When a human expert corrects the system, store the correction with:

- query topic
- source document
- domain keywords

Then retrieve those corrections for future similar queries.

Example:

```text
[Retrieved expert correction from prior session]
Note: Previous answer on termination clauses was incorrect.
Clause 4.2.1 applies only to fixed-term contracts, not at-will.
```

This is one of the main ways a RAG system improves without retraining the base model.

---

## Decision 10: Add End-to-End Observability

At 10M scale, failures are guaranteed. Observability determines whether you debug them in minutes or after customer damage.

### Example trace output

```text
[INGEST LAYER]       Document parsed -> 847 chunks generated in 2.3s
[VECTOR LAYER]       ANN search -> 30 candidates in 8ms (HNSW index)
[BM25 LAYER]         Keyword search -> 12 candidates in 3ms
[FUSION LAYER]       Hybrid merge -> 38 unique candidates, top 15 selected
[RERANK LAYER]       Cross-encoder scored 15 chunks in 180ms
[CONFIDENCE LAYER]   Top chunk: 0.847, threshold: 0.65 -> PASS
[GENERATION LAYER]   LLM call -> 1240ms, 387 tokens generated
[EVAL LAYER]         Faithfulness: 0.91, Relevance: 0.84 -> OK
[CACHE LAYER]        Result cached. Key: a3f9b2c1...
```

### Trace per query

- timing breakdown per stage
- retrieved documents and scores
- chunks selected and rejected by reranker
- final prompt sent to the model
- raw model output before citation post-processing
- eval scores
- cache hit or miss

### Suggested infra

- OpenTelemetry for tracing
- Prometheus plus Grafana for metrics
- Elasticsearch or Loki for structured logs

Everything should be queryable by:

- document ID
- query hash
- session
- time range

---

## Main Takeaway

Most teams optimize the wrong thing.

They spend too much time comparing frontier models and not enough time improving retrieval.

The real production lesson:

> At 10 million documents, retrieval quality matters more than the frontier model itself.

Your retrieval pipeline is the foundation. The LLM is the finishing layer.

If the retrieval layer is weak, no model can rescue the answer reliably.

---

## Demo Notes from the Provided Text

The source text describes a browser-based `Advanced Local RAG Demo` that maps each architectural choice to a visible system behavior.

### Claimed visible behaviors

- hybrid retrieval trace with vector and BM25 running in parallel
- adjustable fusion weights between vector and keyword retrieval
- stage 2 reranking with either syntactic reranking or neural cross-encoder reranking
- per-source confidence display
- citation badges that jump directly to the supporting chunk
- automatic hallucination alerting when faithfulness drops below `0.8`
- live display of context relevance, faithfulness, answer relevance, and latency
- caching behavior visible through cache hit traces
- side-by-side comparison of vector-only, BM25-only, and fused hybrid retrieval

---

## Practical Summary for This Repo

If we adapt these ideas into repo-level architecture decisions, the strongest reusable principles are:

- normalize aggressively at ingest
- prefer hybrid retrieval over dense-only retrieval
- add reranking as a mandatory precision layer
- gate low-confidence retrieval before generation
- enforce citation-only grounded prompting
- verify faithfulness after generation
- measure retrieval and generation separately
- build cache invalidation around document versioning
- treat observability as part of the product, not just platform plumbing

---


# RAG & Tools

### Google Docs API gave better performance for converting docs to markdown than lamaparse, PdfPlumber, PyMuPDF

 pypdfium has the highest score for for matching docs/PDF parsing
metaData includes:
   source: filePath
   page_no: 0

custom_metadata includes:
   source: filePath
   page_no: 0
   document_summary:
   chunk_id:
   chunk_faqs:
   chunk_keywords:

1. Knowledge Processing (Chunking & Embeddings)
The Problem: Fixed-length chunking (e.g., 30 tokens) fragments context, destroying the interconnected nature of information (02:46).
The Solution: Use Semantic Chunking based on document structure (e.g., sections, headers) and Hierarchical Chunking to maintain parent-child relationships between text chunks (06:23).
Enhanced Embeddings: Move beyond single-vector embeddings to multi-vector embeddings (capturing embeddings at the token level) for richer semantic representation (12:23).
2. Query Understanding
The Problem: Naive similarity search often fails to understand the user's true intent, leading to irrelevant search results (15:39).
The Solution: Enhance queries with user context, meta-information, and entity extraction to identify the true intent, urgency, and relevant domain (19:45).
Knowledge Orchestration: Implement planning mechanisms to determine necessary permissions and data freshness requirements before retrieving information (22:09).
3. Hybrid Retrieval Systems
The Problem: relying solely on cosine similarity for retrieval is insufficient for complex queries (25:51).
The Solution: Implement a Hybrid Retrieval system that combines parallel searches across a Vector Store (semantic), Document Store (keyword/BM25/splade), and Graph Store (knowledge graph entities) (30:30).
Fusion Ranking: Combine results from these different methods using algorithms like Reciprocal Rank Fusion (RRF) to determine the best final chunks for the language model (33:04).

### Accuracy and reliability

## V1

You can evaluate how correct, truthful, and complete your agent’s answers are. For example:
Hallucination. Do responses contain facts or claims not present in the provided context? This is especially important for RAG applications.Faithfulness. Do responses accurately represent provided context?

Content similarity. Do responses maintain consistent information across diﬀerent phrasings?
Completeness. Do response includes all necessary information from the input or
context? Answer relevancy. How well do responses address the original query?

You can evaluate how well the model delivers its ﬁnal answer in line with requirements around format, style, clarity, and alignment.
Tone consistency. Do responses maintain the correct level of formality, technical complexity, emotional tone, and style?
Prompt Alignment. Do responses follow explicit instructions like length restrictions, required elements, and speciﬁc formatting requirements?
Summarization Quality. Do responses condense information accurately?
Consider eg information retention, factual accuracy, and conciseness?
Keyword Coverage. Does a response include technical terms and terminology use?

 Retrieval Metrics (1:32 - 2:48)

Context Precision: The fraction of retrieved chunks that are actually useful.
Context Recall: The percentage of relevant information from the entire corpus that was successfully found.
Mean Reciprocal Rank (MRR): Measures the placement of the most relevant chunk; early placement (rank 1) is critical since most systems only ingest the top few results.
4. Generation Metrics (2:48 - 3:49)

Faithfulness: A reference-free metric checking if every claim in the output is supported by the retrieved context.
Answer Relevance: A reference-free check to ensure the response answers the specific question asked.
Answer Correctness: A comparison against ground truth labels using semantic similarity.
5. Building Your Evaluation Test Set (3:49 - 4:47)

The foundation involves creating triplets: (Question, Answer, Relevant Chunk IDs).
Test sets must include edge cases such as ambiguous queries, multi-hop reasoning, and out-of-scope questions.
The test set is a living artifact that must be updated as user patterns evolve.
Faithfulness Check with LLM-as-Judge
Extract atomic claims (single indivisible facts) from the answer
Classify each claim as supported or unsupported by context
Score = supported claims / total claims
Production target: 0.85 to 0.95 depending on use case

Retrieval Metrics Calculation
Precision = true positives / total retrieved
Recall = true positives / total relevant in corpus
Reciprocal Rank = 1 / position of first relevant chunk
MRR = average reciprocal rank across test set

1. Step 2: Code Examples (4:47 - 7:04)

Faithfulness Check (4:52): Uses an LLM-as-a-Judge to extract atomic claims and classify them as supported or unsupported. Targets are typically 0.85–0.95.
Retrieval Metrics (6:06): Demonstrates how to programmatically calculate precision, recall, and reciprocal rank to identify performance bottlenecks.

1. Document Processing & Recursive Chunking (3:23 - 6:41)
This phase establishes the system's foundation. The speaker advocates for structure-aware recursive chunking over fixed-size splitting. By prioritizing document boundaries (chapters, sections, paragraphs), the system preserves semantic coherence, which is crucial for retrieval precision. He recommends a target of 300 to 800 tokens per chunk with a 100-token overlap to prevent information loss at boundaries.
2. Metadata & Embedding Strategy (6:42 - 12:47)
Metadata should be a first-class citizen (6:42). Storing source IDs, versions, and document types allows for pre-retrieval filtering, drastically reducing search space. For embeddings, the strategy is to batch API requests to minimize overhead, use persistent storage for embeddings, and track model versions to allow for selective re-embedding as models improve.
3. Vector Database Architecture (12:48 - 20:28)
For scale, the speaker discusses HNSW (Hierarchical Navigable Small World) indices to achieve logarithmic search complexity. At 10 million+ documents, he emphasizes:

Keeping the index memory-resident for millisecond latency.
Implementing date-based or type-based sharding to route queries efficiently.
Using primary-replica topologies to handle high read throughput.
4. Hybrid Retrieval & Re-ranking (20:29 - 27:09) To maximize recall, the system combines dense vector search (semantic) with sparse keyword search/BM25 (lexical). These are merged via Reciprocal Rank Fusion (RRF). Post-retrieval, a cross-encoder re-ranker is used to refine the top 100 candidates into the final top 10, typically improving precision by 20–30%.
5.  Methods like TF-IDF and BM25 rely on term frequency. BM25 is particularly noted for its ability to normalize document length and prevent common words from dominating results.
6. LLM Integration & Context Assembly (27:10 - 35:15) Context assembly focuses on token budget management. Even with large context windows, the speaker suggests focusing on high-relevance chunks rather than raw volume to avoid diluting the LLM's performance. The prompt should explicitly require source citations to ensure trust and allow for graceful fallbacks when the context is insufficient.
7. Multi-Layer Caching & Cost Optimization (35:16 - 45:59) Caching is the most significant lever for cost reduction (70–80% for FAQ workloads). The architecture utilizes three layers:
Query Cache: For identical questions.
Embedding Cache: To avoid redundant API costs.
Retrieval Cache: For semantically similar queries. Proper TTL (Time-To-Live) and content-based invalidation patterns are required to maintain data freshness.
1. Operations & Evaluation (46:00 - 58:00) Production-grade RAG requires continuous evaluation using a golden dataset (100–500 diverse queries). The pipeline should measure precision, recall, and answer correctness automatically. Centralized monitoring for latencies and costs is mandatory for long-term stability and to prevent cascading failures via circuit breakers.

Vector Database Internals: HNSW, Sharding & Scaling (15:38 - 20:28)

The speaker explains that vector databases use HNSW (Hierarchical Navigable Small World) graphs to enable efficient approximate nearest neighbor search. Unlike brute-force linear search—which would require 10 million distance calculations for 10 million vectors—HNSW creates hierarchical structures that reduce this to a few thousand calculations, achieving logarithmic-like search complexity (16:05-16:45).

Key operational details include:

Memory Residency: To hit millisecond-level latency at the 95th percentile, the index structure must be in memory or memory-mapped with ample page cache (16:54-17:15).
Parameter Tuning: Users must balance recall, memory, and query speed by tuning parameters like the 'M' value (connections) and 'EF search.' The speaker warns against using default settings, which are designed for small demo datasets (17:23-18:05).
Sharding Strategies: For systems exceeding 10 million documents, sharding becomes necessary. Date-based sharding is highly effective for time-sensitive content: recent data is kept in 'hot' shards, while historical data resides in 'cold' shards, preventing the system from wasting cycles searching irrelevant history (18:30-19:35).
Scaling: The architecture supports linear scaling; doubling the shards roughly doubles the capacity, making capacity planning more predictable (20:17-20:25).The Retrieval Engine: Architecting Hybrid Search (20:29 - 24:42)
Hybrid search is presented as a method to combine the strengths of two distinct retrieval worlds:

Dense Vector Search: Excels at semantic understanding, matching intent even when terminology differs (20:48-20:56).
Sparse Keyword Search (BM25): Excels at exact matches, such as error codes or specific technical identifiers that might lose signal strength during embedding (21:00-21:12).
The implementation strategy involves running both searches in parallel (using tools like promise.all) to ensure the total latency is the maximum of the two search times rather than their sum (21:17-21:20). The results are then merged using Reciprocal Rank Fusion (RRF). RRF is preferred over score normalization because vector and keyword scores operate on different, non-comparable scales; RRF instead relies on the ranking position, which is more robust and requires no parameter tuning (21:21-23:32). After merging, the system performs deduplication by chunk ID before passing the final candidates to the re-ranker (23:33-23:40).Improving Precision: Re-ranking with Cross-Encoder Models (24:43 - 27:09)

Re-ranking is described as a critical refinement step that typically yields a 20-30% improvement in retrieval quality (24:47-24:52).

The speaker draws a sharp technical distinction between bi-encoder models (used for initial embedding generation) and cross-encoder models:

Bi-Encoders: Process queries and documents independently; they never 'see' the relationship between the two until the final dot-product or cosine similarity comparison (25:34-25:47).
Cross-Encoders: Process the query and document as a single, combined input. This allows the model to analyze the semantic relationship directly, making them far more accurate for assessing relevance (25:54-26:10).
While cross-encoders add 200-300 milliseconds of latency due to their higher computational cost, the speaker argues this is a necessary trade-off for production applications where quality directly dictates user satisfaction and reduces the need for expensive, secondary LLM calls (26:27-26:50).

Improving Precision: Re-ranking with Cross-Encoder Models (24:43 - 27:09)
Cross-Encoders vs. Bi-encoders: While bi-encoders (used for initial retrieval) process queries and documents independently, cross-encoders process them as a joint pair. This allows the model to deeply evaluate the relationship between the query and document, leading to significantly higher accuracy.
Precision Gains: Re-ranking the top 100 retrieval candidates typically improves precision by 20% to 30%.
Latency Trade-off: Cross-encoders are computationally expensive, adding roughly 200-300ms of latency. However, for applications where quality is paramount, this refinement stage is essential for filtering out noise.

Context Assembly & Token Budget Management (27:10 - 29:58)
Beyond Volume: More context does not always equate to better performance; excessive context can dilute the model's focus. The focus should be on relevance, not just filling the context window.
Assembly Strategy: Chunks should be assembled in descending order of relevance. Always include metadata to enable source attribution and citations.
Diversity & Fallbacks: To prevent redundancy, implement diversity constraints to avoid selecting similar chunks from the same document section. If retrieval scores are consistently low, implement a fallback strategy to avoid generating hallucinations.

LLM Integration & Production Prompt Engineering (29:59 - 35:15)
System Prompts: These are vital for defining behavioral constraints, such as forcing the model to answer only from the provided context and citing sources.
Streaming & Performance: Using streaming responses improves perceived latency, ensuring the user sees the answer being generated in real-time. Streaming also allows for early termination if a user cancels the query, which saves on token costs.
Reliability: Always implement exponential backoff and retries for LLM API calls, as service interruptions are common in production environments.
Performance & Cost: Multi-Layer Caching (35:16 - 42:22)

The Three-Layer Approach:
Query Cache: Stores final responses for identical user questions. This provides the fastest path to a response.
Embedding Cache: Stores query embeddings to avoid redundant API calls.
Retrieval Cache: A more advanced layer that caches results for semantically similar queries to skip the expensive vector search.
Cost Efficiency: In high-repetition environments (like FAQs), this approach can reduce operational costs by 70% to 80%.
Freshness: Use Time-To-Live (TTL) settings and content-based invalidation keys to ensure that cache entries are cleared when underlying documents are updated.

Cost Optimization Framework (42:23 - 45:59)
Economic Sustainability: Unoptimized RAG systems can become prohibitively expensive. The goal is to monitor cost-per-query as a primary KPI.
Optimization Tactics:
Model Routing: Direct simple queries to cheaper, smaller models, and reserve high-cost, advanced models for complex reasoning tasks.
Granular Monitoring: Track costs by individual components (embeddings vs. retrieval vs. generation) to identify where spend is concentrated.

Document Processing & Chunking
Recursive chunking strategies vs fixed-size splitting
Maintaining semantic coherence at scale
Metadata extraction and schema design

Vector Database Architecture
HNSW index internals and configuration
Sharding strategies for 10M+ documents
Primary-replica topology for high availability

Hybrid Search Implementation
Combining dense vector and sparse keyword retrieval
Reciprocal Rank Fusion for result merging
BM25 scoring and when keyword search matters

Re-ranking with Cross-Encoders
Bi-encoder vs cross-encoder architectures
Precision-latency trade-offs
Production model selection

Multi-Layer Caching
Query cache, embedding cache, retrieval cache
Cost reduction strategies (70-80% in FAQ workloads)
Cache invalidation patterns

## V2

 Advanced Document Chunking
Semantic chunking with embedding similarity detection
Hierarchical (parent-child) chunking for context preservation
Contextual chunking with LLM-generated preambles
Vision-based extraction for complex document formats

Query Understanding & Transformation
Query rewriting for vocabulary alignment
Query decomposition for multi-part questions
HyDE (Hypothetical Document Embeddings)
Building adaptive transformation pipelines

Agentic RAG Architectures
Router pattern for query classification
Tool-use and function calling for dynamic retrieval
Self-reflection and adaptive retrieval loops
Multi-hop reasoning across documents
Multi-agent orchestration patterns

Knowledge Graphs for RAG
Entity and relationship extraction
Hybrid graph + vector retrieval
When graph retrieval adds value vs. overhead

Multi-Modal RAG
Vision-language models for document understanding
Cross-modal embedding and retrieval (CLIP)
Complete multi-modal architecture design

Evaluation Deep-Dive
Component-level metrics (Recall@K, MRR, NDCG)
LLM-as-Judge patterns and limitations
Synthetic test set generation
Continuous online evaluation
Hallucination detection and faithfulness verification

Fine-Tuning for RAG
Embedding model fine-tuning with hard negatives
Reranker fine-tuning for domain precision
Generator fine-tuning for faithfulness

Production Hardening
Access control and multi-tenancy
Incremental indexing and cache invalidation
Guardrails and safety
Latency and cost optimization
Quality-focused monitoring

### advanced chunking strategies designed to overcome the limitations of standard approaches in production-grade RAG systems

1. Recursive Chunking Limitations (06:30 - 08:28)
While recursive chunking is a common starting point due to its simplicity and ability to respect natural text boundaries, it often fails in production because it optimizes for size rather than semantic coherence. Key failure modes include:

Context Loss: Separating a paragraph from its referenced table or a legal clause from its definition (e.g., spanning across multiple chunks).
Fragmented Logic: Splitting code examples mid-function or separating methodology descriptions from their key abbreviations.
Semantic Blindness: Because it relies on arbitrary size limits, it does not understand the underlying relationships between different sections of a document.

2. Semantic Chunking (08:28 - 10:23)
This approach uses embedding similarity to identify natural topic transitions rather than hard size limits.

Process: The document is split into sentences, which are then embedded. The system calculates the cosine similarity between consecutive sentences; sharp drops in similarity scores indicate topic boundaries.
Pros/Cons: It produces more coherent, self-contained chunks but leads to unpredictable chunk sizes and higher upfront computational costs for embedding every sentence.

3. Hierarchical (Parent-Child) Chunking (10:23 - 12:19)
This strategy balances the conflicting needs of retrieval (which prefers small, precise chunks) and generation (which requires broad, contextual chunks).

The Structure: A system stores small child chunks for efficient retrieval, each containing a pointer to a larger parent chunk (a section or full page).
The Workflow: Upon retrieving a child chunk, the system retrieves the full parent chunk to provide the LLM with the necessary surrounding context, effectively solving the "referenced table" problem mentioned earlier.

4. Contextual Chunking (12:19 - 14:04)
Contextual chunking uses an LLM to prepend a brief preamble to each chunk before embedding, explicitly situating that specific text within the larger document.

Value: A generic chunk like "revenue increased by 12%" is transformed into "This section discusses Q3 2024 financial results for the North American division: revenue increased by 12%." This ensures the embedding captures document-level context.

5. Handling Complex Document Formats (14:04 - 15:53)
Traditional text extraction often fails on enterprise documents like PDFs with multi-column layouts, tables, and charts.

Vision-based Extraction: The recommended solution is to pass pages as images to a vision-language model. This model performs implicit OCR, interprets the layout, and outputs structured data (e.g., Markdown tables) that preserve the integrity of the original format.

6. Chunking Strategy Decision Framework (15:53 - 17:46)
Mukul provides a helpful guide for choosing the right strategy based on document type:

**Narrative Documents** (Articles/Reports): Use Semantic Chunking combined with Hierarchical Chunking.
**Highly Structured Documents** (Legal/Specs): Use structure-aware Recursive Chunking with Contextual Preambles.
**Mixed Media** (Tables/Charts): Use Vision-based Extraction.
**General Knowledge Bases**: Simple recursive splitting is often sufficient for independent, well-defined articles.
**Pro-tip**: Always validate these strategies empirically by examining the resulting chunks and testing them against representative queries before finalizing your pipeline.

### raw user queries are often insufficient for high-performance RAG systems and outlines advanced transformation techniques to bridge the semantic gap

1. The Raw Query Problem (17:51 - 19:50)
The primary issue is that the user's natural language question is rarely the optimal search query. Challenges include:

Vocabulary Mismatch: Users may use colloquial terms (e.g., "firing employees") that do not align with formal document language (e.g., "termination procedures").
Implicit Context: Users often omit crucial details like jurisdiction or policy specifics, assuming the system knows the background.
Multi-part Questions: Complex queries requiring synthesis from multiple sources often fail if processed as a single embedding.
Ambiguous References: In a conversational flow, pronouns like "the other option" lose meaning without context.
2. Query Rewriting (19:50 - 21:33)
This technique uses an LLM to reformulate the user's input into an optimized retrieval query. It helps by:

Aligning Vocabulary: Automatically expanding terms with synonyms or domain-specific language.
Expanding Clarification: Using dialogue history to turn vague questions into specific ones (e.g., changing "What is the deadline?" to "What is the Q3 project deadline?").
Generating Multiple Interpretations: For ambiguous intents, generating several variations of the query can increase recall.
Information Selection: Using LLMs to filter, summarize, or rank retrieved documents to avoid overwhelming the system.
Reciprocal Rank Fusion (RRF): Combines rankings from multiple retrieval algorithms (e.g., embeddings + BM25) to surface the most consistent results.
FLARE: An iterative strategy where the LLM actively identifies where it needs more information during the generation process and triggers further retrieval as needed.
3. Query Decomposition (21:33 - 23:00)
For complex questions containing multiple parts (e.g., comparing 2024 pricing vs. 2023), a single pass is inadequate. Decomposition breaks these into:

Independent Subqueries: Each part is queried in parallel.
Unified Synthesis: The system retrieves chunks for each subquery and the LLM synthesizes them into one coherent answer.
4. HyDE: Hypothetical Document Embeddings (23:00 - 24:52)
HyDE addresses the asymmetry between interrogative questions and declarative documents. Instead of embedding the question, the system:

Generates a Hypothetical Answer: Uses an LLM to create a plausible (even if potentially imperfect) answer.
Embeds the Answer: Since the answer is declarative, it exists in the same semantic space as the source documents, leading to more accurate retrieval.
Rejected for V1
Why: HyDE is a massive hallucination risk for the legal domain. If a user asks about a highly specific, obscure penalty clause, HyDE asks an LLM to hallucinate a "plausible" answer first, embeds that hallucination, and searches. In law, plausible but fake wording will retrieve the wrong real clauses. Rely on exact hybrid search instead.
5. Query Transformation Pipeline (24:52 - 26:34)
Effective systems don't apply these techniques blindly; they use a classification step to determine the best approach:

Routing: Simple factoid queries pass through unchanged, while complex or ambiguous ones are routed to decomposition or rewriting.
Pipeline Order: Typically, decomposition occurs first, followed by rewriting specific subqueries.
Empirical Validation: It is essential to measure the impact of each transformation and disable those that degrade performance on specific query types.

### the transition from fixed, static RAG architectures to Agentic RAG, which uses LLMs to dynamically orchestrate retrieval and reasoning

1. From Static Pipelines to Adaptive Agents (26:39 - 28:11)
Static V1 pipelines process every query identically, regardless of complexity. Agentic RAG replaces this with an LLM-powered agent that operates in a reasoning loop:

Dynamic Decision Making: The agent decides what to retrieve, evaluates if the context is sufficient, and iterates if necessary.
Capability Shift: Unlike static systems, agents can reformulate queries based on initial results, synthesize information from multiple sources, and even decide when retrieval is unnecessary.
Trade-off: This introduces higher latency and cost due to multiple LLM calls per query.

2. The Router Pattern (28:11 - 29:31)
Routing is the simplest form of agentic RAG. A classifier or LLM analyzes incoming queries to direct them to the most effective retrieval backend:

Factoid Questions: Routed to vector search.
Analytical/Structured Queries: Routed to SQL databases.
Relational/Entity Queries: Routed to knowledge graph traversal.
Conversational/Other: Routed to dialogue history or answered directly without retrieval.

3. Tool Use for Retrieval (29:31 - 30:56)
By giving the LLM access to retrieval as a callable tool, the system becomes more flexible:

Definition: Tools are defined with specific capabilities (e.g., searching a specific index or fetching a document by ID) and instructions on when to use them.
Dynamic Parameters: The agent can set metadata filters or request more results if initial findings are insufficient, allowing the search strategy to change mid-reasoning.
Guardrails: It is critical to limit the number of tool calls per query and implement timeouts to avoid infinite, costly loops.

4. Self-Reflection and Adaptive Retrieval (30:56 - 32:15)
This pattern enables the agent to critique its own retrieved context before generating a final answer:

Judgment: The LLM acts as a judge, classifying the context as sufficient, insufficient, or contradictory.
Adaptive Loop: If the context is insufficient, the agent reformulates the query and tries again. If contradictory, it performs additional searches to resolve the conflict.

5. Multi-Hop Retrieval (32:15 - 33:41)
Some queries require chaining information across multiple documents (e.g., finding a policy in a memo, then finding the revenue data linked to that policy).

Sequential Reasoning: Unlike decomposition (which runs in parallel), multi-hop is inherently sequential because each step depends on the result of the previous one.
State Tracking: The agent must maintain state across hops to track identified entities and remaining unknowns.

6. Multi-Agent Architectures (33:41 - 34:49)
For complex enterprise systems, a planning agent can orchestrate specialized sub-agents:

Retrieval Agent: Focuses on search and context gathering.
SQL Agent: Focuses on database queries.
Summarization/Verification Agents: Handle condensing data or checking for hallucinations.
Recommendation: Start with a single-agent architecture (hub-and-spoke) and only graduate to multi-agent when query complexity demonstrably exceeds the capacity of a single agent.

7. Agentic RAG Decision Framework (34:49 - 36:12)
Use this framework to choose the right complexity level:

Static Pipeline: Use for uniform, low-complexity queries where latency and cost are the primary concerns.
Routing: Use when you have varied query types that can be mapped to predictable backends.
Single-Agent with Tools: Use for adaptive, multi-source retrieval requirements.
Multi-Agent: Reserved for the most complex systems where sub-tasks require specialized reasoning.

### Knowledge Graphs (KGs) in RAG systems, explaining how they complement vector search to handle complex relational queries

1. What Vector Search Cannot Do (36:14 - 37:34)
While vector search is excellent for finding semantically similar documents, it has fundamental limitations when it comes to structured relational data:

Lack of Intersection Logic: Vector search cannot perform set intersections required to find entities sharing multiple specific relationships (e.g., finding suppliers that contract with both Company A and Company B).
No Hierarchy Traversal: It cannot navigate organizational structures or citation networks, making questions about reporting lines or paper lineages impossible to answer directly.
Inability to Reason: Vector search lacks an understanding of entity properties and edge types, which is required for multi-step relational reasoning.

2. Building a Knowledge Graph (37:34 - 39:09)
Constructing a KG involves transforming raw text into a structured network of nodes and edges:

Entity Extraction: Identifying people, organizations, products, and domains using Named Entity Recognition (NER) or LLM-based extraction.
Relationship Extraction: Determining how entities interact (e.g., "John Smith is the CEO of Acme Corp" or "Acme Corp acquired Beta Co in 2023").
Graph Construction: Creating nodes for entities (with metadata) and edges for relationships (with types and temporal attributes).
Entity Resolution: A critical step to ensure that different names for the same entity (e.g., "John Smith" vs. "J. Smith") are merged into a single canonical identity.

3. Hybrid Graph + Vector Retrieval (39:09 - 40:17)
Optimal performance is achieved by combining the strengths of both methods:

Query Classification: The system determines if a user's prompt requires relational reasoning (KG) or content retrieval (vector).
Graph Traversal: The system executes a graph query to identify relevant entities and their connections.
Ranking Boost: The results from the graph traversal are used to provide a ranking boost to the relevant chunks retrieved by vector search. This ensures that the context provided to the LLM is both semantically relevant and relationally accurate.

4. Knowledge Graphs at Scale (40:17 - 41:31)
Scaling KGs requires managing complexity and costs effectively:

Incremental Extraction: Only process new or updated documents to avoid re-running expensive LLM extractions on the entire corpus.
Community Detection: Identify clusters of densely connected entities to create hierarchical summaries. This allows the system to answer "global" questions about entire topics by retrieving summaries instead of individual chunks.
Maintenance: Ensure the system handles updates and deletions, as changes in source documents must trigger updates to nodes and edges within the graph.

### Multi-Modal RAG how to extend RAG systems to process visual content, which is essential for enterprise documents containing tables, charts, diagrams, and scanned images

1. The Multi-Modal Challenge (41:33 - 42:56)
Standard text-only RAG pipelines often fail when encountering complex documents because they extract only text and discard visual information. This leads to:

Data Loss: Tables become scrambled, charts lose their labels, and diagrams disappear entirely.
Query Failure: Users asking about specific visual content (e.g., "What does the network topology show?") cannot be answered if the visual context was never preserved.

2. Vision Models for Document Understanding (42:56 - 44:15)
The recommended approach is to process document pages as images using Vision-Language Models (VLMs):

Process: At ingestion time, each page is sent to a VLM that extracts text, converts tables into Markdown, describes charts with trends and data points, and explains diagrams.
Benefits: This transforms visual content into structured text that is searchable through standard pipelines, capturing information that traditional OCR often misses.
Trade-off: This is a one-time, upfront cost during document ingestion, but it is necessary for high-quality retrieval.

3. Multi-Modal Embedding and Retrieval (44:15 - 45:32)
To enable cross-modal retrieval, the system uses models like CLIP to map images and text into the same shared vector space:

Embedding: Images are processed through a vision encoder to produce vectors, which are stored alongside text chunk vectors in the index.
Modality Filtering: At query time, the system can search across both text and image vectors. Metadata tags are used to filter results (e.g., searching only for images or only for text).
Limitation: Current multi-modal embeddings capture high-level semantics but may miss fine-grained visual details (like specific node labels in a diagram).

4. Multi-Modal Architecture (45:32 - 46:47)
A complete multi-modal architecture integrates these components:

Ingestion: Use format-appropriate extraction. Pages with complex layouts are sent to a VLM for structural extraction.
Retrieval: Depending on the user's query, the system retrieves either text chunks, images, or a combination of both.
Generation: The final response uses a Multi-Modal LLM that accepts both text and retrieved visual evidence, allowing it to synthesize answers that reference both types of data.

### Evaluating RAG systems is notoriously difficult because they are modular architectures where individual components can fail independently. The video highlights how end-to-end metrics often mask the root cause of a system failure, necessitating a shift toward component-level evaluation

1. Why Evaluation is Hard (46:51 - 48:06)
System failures are difficult to diagnose because an "incorrect" answer could stem from multiple points in the pipeline:

Retrieval failure: The query was misformulated or relevant content is missing.
Reranking failure: Good candidates were discarded by the reranker.
Generation failure: The model hallucinated or failed to synthesize provided context. End-to-end metrics tell you if the answer is bad, but component-level metrics are required to localize why it is bad.
2. Retrieval Evaluation Metrics (48:06 - 49:28)
To assess if the retriever is effectively surfacing relevant information, use these standard metrics:

Recall@K: The fraction of relevant documents found within the top K results.
Precision@K: The fraction of retrieved results that are actually relevant.
Mean Reciprocal Rank (MRR): Measures how high the first relevant document appears in the result list. Retrieval quality is the most critical diagnostic; if relevant information never reaches the generator, downstream improvements will not compensate.
3. Generation Evaluation (49:28 - 50:35)
This isolates the LLM's performance by providing it with "gold standard" context that definitely contains the answer. Key dimensions include:

Faithfulness: Does the answer rely only on the provided context?
Completeness: Does the answer cover all aspects of the query?
Relevance: Does the response address the user's intent?
Conciseness: Is the answer appropriately brief?
4. LLM-as-Judge (50:35 - 51:54)
Using an LLM to act as an automated grader is scalable but requires caution due to inherent biases:

Position/Verbosity Bias: Models may favor answers that appear first or are simply longer.
Self-Preference Bias: Models tend to favor outputs from their own family.
Calibration: It is essential to calibrate the judge by comparing its scores to human-annotated data and refining prompts accordingly.
5. Synthetic Test Set Generation (51:54 - 53:12)
To avoid the labor-intensive process of manual annotation, you can use LLMs to generate test sets:

Process: Sample chunks from your corpus and use an LLM to generate synthetic question-answer pairs.
Best Practices: Use different models for generation and evaluation to prevent circularity, and ensure your test set includes adversarial, edge-case questions rather than just "easy" factoid queries.
6. Continuous Online Evaluation (53:12 - 54:30)
Performance should be monitored post-deployment using both implicit and explicit signals:

Implicit: Look at query reformulation rates, conversation abandonment, and click-through rates on citations.
Explicit: Use user-provided thumbs up/down feedback.
Drift Detection: Longitudinal tracking helps identify if model quality is declining due to corpus changes or external data drifts.
7. Hallucination Detection (54:30 - 55:41)
To ensure groundedness, the system can perform claim verification:

Claim Decomposition: Break the generated response into atomic factual claims.
NLI (Natural Language Inference): Use models to classify each claim as supported, contradicted, or neutral relative to the retrieved context.
Mitigation: Post-generation filtering can block ungrounded answers, or the system can provide visual confidence indicators to the user.

### fine-tuning is a critical step for adapting general-purpose models to specific domain vocabularies, writing styles, and information patterns. It highlights three key components that can be fine-tuned to resolve specific bottlenecks in a RAG pipeline

1. Why Fine-Tuning Matters (55:43 - 56:41)
Off-the-shelf models are trained on general internet data. Fine-tuning allows your system to bridge the gap between this general knowledge and your organization's unique domain. The key is to start with evaluation to identify the specific failure point—whether it's poor retrieval recall, imprecise reranking, or subpar generation—and fine-tune only that component.

2. Fine-Tuning Embeddings (56:41 - 57:48)
Fine-tuning embedding models is used to improve retrieval recall.

Data: Uses query-passage pairs where the passage provides the answer. Data can be sourced from QA logs, manual annotations, or synthetic generation.
Hard Negative Mining: A crucial technique where you identify passages that are semantically similar but do not contain the answer. Training on these helps the model make finer distinctions.
Objective: Uses contrastive learning to pull positive pairs together while pushing negative ones apart. The video also mentions Matryoshka representation learning, which allows for embeddings that work well across multiple dimensions to trade off storage vs. quality.

3. Fine-Tuning Rerankers (57:48 - 58:32)
When retrieval returns relevant documents but they are ranked too low, fine-tuning the reranker acts as a precision filter.

Cross-Encoders: The model is trained to score query-document pairs jointly. This is often more effective than embedding fine-tuning for increasing precision.
Graded Relevance: Training often involves binary or graded labels (e.g., highly vs. somewhat relevant), teaching the model domain-specific relevance signals.

4. Fine-Tuning Generators (58:32 - 59:18)
When retrieved context is accurate but the generated response is poor, fine-tuning the generator is required.

Data Triples: Training uses (Question, Context, Answer) triples.
Negative Examples: It is essential to include examples where the context is insufficient, teaching the model to acknowledge uncertainty or abstain from answering rather than hallucinating.
Alignment: Proper citation training ensures the model explicitly links claims to supported chunks. Refusal training is also included to handle out-of-scope questions.

## V3

Context Strategy & Decision Framework
Long-context vs RAG trade-off analysis
Cost, latency, and precision comparisons
Hybrid retrieval + large context architectures
Decision framework for architecture choice

Multi-Turn Conversational RAG
Conversation memory and token budgeting
Query reformulation for follow-ups
Coreference resolution and entity tracking
Session state management with Redis
History compression strategies

Failure Diagnostics & Debugging
Retrieval vs generation failure taxonomy
Hallucination types and root causes
Lost-in-the-middle attention issues
Systematic diagnostic workflow
Building debugging tools

Edge Case Handling
No results and conflicting sources
Ambiguous and out-of-scope queries
Staleness detection and graceful degradation

Production Observability
Quality metrics beyond latency
Alerting and threshold calibration
Trace-based debugging

### technical analysis of the architectural trade-offs between Long-Context models and Retrieval-Augmented Generation (RAG) systems. Below is a detailed summary of the requested chapters

When Long-Context Wins Over RAG (04:09 - 06:55)
Long-context architectures are often superior when your entire dataset fits within the model's context window (e.g., up to 100k tokens).

Operational Simplicity: You avoid the engineering overhead of chunking, embedding management, and vector database maintenance.
Cross-Document Synthesis: Since the model has the entire corpus in view, it excels at global tasks like summarizing themes across multiple reports or identifying contradictions between documents—tasks where retrieval might fragment the necessary context.
Best for Small Teams: Startups with limited resources should prioritize the simplicity of long-context models over building complex RAG pipelines.
When RAG Remains Essential (06:55 - 09:21)
Despite the power of long-context, RAG is mandatory for production at scale:

Massive Corpora: If your data exceeds the context window (e.g., millions of records in healthcare or legal firms), RAG is your only option.
Continuous Freshness: If your documents update hourly, a RAG pipeline’s incremental indexing allows for real-time updates. Long-context approaches require re-uploading the entire dataset, which is operationally prohibitive.
Cost Analysis: Concrete Numbers (09:21 - 12:23)
Cost is often the primary driver for architectural choice.

The Math: Using GPT-4 pricing, a 200k token input costs roughly $2 per query. At 10,000 queries/day, that is $600k per month.
RAG Efficiency: By retrieving only relevant chunks (e.g., 5k tokens), your input cost drops significantly to pennies per query, leading to massive monthly savings even after accounting for vector database and embedding infrastructure.
Latency Comparison: Time to First Token (12:23 - 14:36)
Real-time constraints: If your application requires sub-second responses, long-context is typically non-viable because the model must process a massive prompt before generating the first token.
User Experience: RAG allows for faster latency by keeping the context prompt small and focused.
Precision Trade-offs: Needle in Haystack (14:36 - 17:03)
Attention Degradation: Research indicates models suffer from "lost-in-the-middle" phenomena, where they struggle to retrieve information buried in the center of long contexts.
Controlled Focus: RAG improves precision by pre-selecting the most relevant information and placing it in high-attention positions, effectively solving the "needle in the haystack" problem.
Hybrid Architectures: Best of Both Worlds (17:03 - 19:45)
Modern production systems combine the two. The pattern is:

Broad Retrieval: Use fast nearest-neighbor search to retrieve a larger set (20-50 chunks).
Reranking: Use cross-encoders to select the top 20 most relevant chunks.
Synthesis: Feed these chunks into the context window for the LLM to synthesize. This allows for deep reasoning over a focused, high-precision context.
Decision Framework: Choosing Your Approach (19:45)
Use these four filters to guide your architecture:

Corpus Size: <50k tokens = Long Context; >500k tokens = RAG.
Query Volume: >10,000 queries/day = RAG is mandatory for unit economics.
Latency: Sub-second requirement = RAG.
Update Frequency: Real-time/High frequency = RAG.

### the technical transition from simple retrieval systems to sophisticated, multi-turn conversational architectures. Below is a detailed breakdown of these core components

Implementation Patterns (22:12 - 25:06)
To build a robust production RAG system, engineers should follow these hierarchical and iterative patterns:

Hierarchical Retrieval: When a chunk is retrieved, include its parent section or document summary to provide framing and prevent context fragmentation.
Iterative Retrieval: For complex queries that cannot be answered in one pass, the system should generate an initial response, identify knowledge gaps, perform a second retrieval pass to fill those gaps, and then regenerate a final, more informed answer.
Single-Turn Limitations (25:06 - 27:28)
Single-turn systems, which treat every query as independent, are generally unsuitable for production chatbots because:

They cannot resolve pronouns (e.g., "that," "it") or positional references (e.g., "the second one") that rely on previous conversation history.
They lack "conversational memory," forcing users to re-state constraints or preferences, which creates a frustrating, unnatural user experience.
Conversation Memory Architectures (27:28 - 30:18)
To maintain continuity, systems must track the sequence of messages using:

Full History: Simple for short interactions but fails quickly due to token budget overflow as conversations lengthen.
Sliding Window: Retains only the last 5–10 turns. While this prevents overflow, it risks losing critical constraints established early in the chat.
Summarization: The most scalable approach; periodically compressing older history into a concise summary while keeping the most recent turns in high-fidelity detail.
Context Window Budgeting (30:18 - 32:47)
Proper budgeting is critical to avoid silent failures or API rejections. Developers should:

Establish a hard ceiling based on the model's limits.
Prioritize allocation: Reserve space for system prompts, conversation history, and an output buffer first; then allocate the remaining tokens to retrieved context.
Use dynamic budgeting to adjust retrieval depth (e.g., shorter conversations get richer context, while longer ones are forced to be more selective).
Query Reformulation (32:47 - 37:17)
Since vector databases cannot search for pronouns like "the second one," systems must use an LLM to rewrite user queries into standalone forms. This process uses recent conversation history to transform ambiguous statements (e.g., "What about that?") into searchable, context-rich queries (e.g., "Tell me the pricing for product B").

Coreference Resolution (37:17 - 40:35)
This is the process of mapping references to specific entities discussed previously. It requires:

Explicit Entity Tracking: Maintaining a structured list of products, people, or concepts mentioned in the chat.
Position-Based Handling: Specifically mapping ordinal references (e.g., "the first option") back to the exact list structure presented in the previous turn.
Re-Retrieval Strategy (40:35 - 43:04)
Retrieval should not always be a one-time event per turn. Systems should trigger re-retrieval when:

The conversation topic shifts significantly.
Retrieval confidence scores for the current context drop below an acceptable threshold.
The system detects that the user is attempting to clarify or dive deeper into a previously retrieved result, necessitating a fresh look at the corpus.

### the operational infrastructure required for production RAG systems, specifically focusing on state management, multi-turn handling, and diagnostic workflows

Session Architecture with Redis (43:04 - 45:30)
To support stateless application scaling, session data must be stored externally rather than in instance memory.

Redis Implementation: Use Redis for sub-millisecond session state management. History is stored as a list keyed by session ID.
Persistence: Use Rpush for adding new turns and Lrange for retrieving context, with TTL settings to automatically handle cleanup of inactive sessions.
Security: Strict key isolation is mandatory to prevent cross-user contamination.
History Compression Strategies (45:30 - 48:15)
To avoid token budget overflow in long conversations, implement tiered compression:

Recursive Summarization: Keep recent turns (e.g., the last 5) in high fidelity, while condensing middle turns into paragraphs and older turns into brief thematic notes.
Importance-Based Compression: Annotate turns with scores. Keep high-value turns containing critical constraints or decisions intact, while aggressively compressing low-importance social or clarifying turns.
Follow-Up Detection (48:15 - 50:18)
Efficient resource allocation depends on knowing when to trigger new retrieval:

Lexical Signals: Use keyword/pattern matching (e.g., "also," "what about," or pronouns) for cheap, initial classification.
Semantic Similarity: Compare query embeddings to recent history. Low similarity scores (e.g., < 0.4) strongly suggest a topic shift requiring a fresh retrieval.
Hybrid Approach: Use heuristics for clear cases and reserve LLM-based classification for ambiguous, complex queries.
Citation Handling Across Turns (50:18 - 52:24)
Maintaining trust requires accurate source attribution even when using old context:

State Tracking: Store retrieved document IDs and relevant snippets in your session state.
Consistency: Cite the original source document version retrieved at the time of the claim. Avoid vague attribution; provide specific document titles, section headers, or page numbers to facilitate user verification.
Conversation Boundaries (52:24 - 54:12)
Defining the scope of a conversation prevents context pollution:

Time-Based Expiration: Standardize breaks (e.g., 30 minutes for support, 24 hours for assistants).
Topic-Based Segmentation: Detect shifts in the conversation to segment context internally, ensuring that irrelevant topics do not interfere with retrieval accuracy for new subjects.
Failure Taxonomy: Retrieval vs Generation (54:12 - 56:35)
Effective debugging starts with correctly classifying the failure:

Retrieval Failure: The system fails to retrieve relevant documents because of embedding mismatches, poor chunking, or restrictive filters. Fixes involve improving embeddings, reranking, or metadata handling.
Generation Failure: The model ignores relevant retrieved context or hallucinates despite having the correct data. Fixes involve strengthening grounding instructions or adjusting the system prompt.
Retrieval Failure Patterns (56:35 - 58:50)
Embedding Mismatch: The user's vocabulary differs from the document's (e.g., "cancel subscription" vs "service termination"). Use query expansion or domain-specific fine-tuning.
Chunk Boundary Issues: Information is split across two chunks. Use overlapping windows or hierarchical chunking to ensure completeness.

### the critical task of diagnosing RAG failures, managing hallucinations, and building production-ready observability

Hallucination Types (58:50 - 1:01:05)
Effective debugging requires classifying the specific type of hallucination occurring:

Intrinsic Hallucination: The model directly contradicts retrieved context (e.g., claiming a 60-day refund window when the document says 30). This points to weak grounding or attention issues.
Extrinsic Hallucination: The model adds information not present in the context. This usually comes from the model's parametric knowledge and can be misleading when the user expects strictly grounded answers.
Fabricated Citations: The model invents sources or points to incorrect section numbers. This is particularly damaging to user trust.
Confidence Uncertainty: The model presents guesses as established facts. Ideally, the system should be calibrated to express uncertainty (e.g., "I don't see this in the documents") when appropriate.
Context Ignored: Lost in the Middle (1:01:05 - 1:03:12)
Models often exhibit "lost-in-the-middle" attention degradation, where information buried in the middle of a large prompt is ignored.

The Fix: Prioritize context ordering. Always place the highest-scoring retrieved chunks at the very beginning and the very end of the prompt to maximize the model's attention.
Instruction Reinforcement: Explicitly instruct the model to treat the provided context as the authoritative source and to quote passages directly when answering.
Diagnostic Workflow (1:03:12 - 1:05:40)
Never generalize a failure until you have diagnosed the specific instance:

Inspect Retrieval: Did the system retrieve relevant information? If not, stop—you have a retrieval problem (e.g., embedding mismatch, poor chunking).
Assess Ranking: Was the relevant document ranked too low (e.g., at rank 5 when it should be 1)? This suggests a need for better reranking.
Examine Utilization: If relevant docs were provided, did the model use them? If the model hallucinated anyway, it is a generation failure requiring better prompting or grounding.
Document findings: Maintain a pattern library of failures to inform future architecture improvements.
Building Debugging Tools (1:05:40 - 1:08:25)
Manual inspection does not scale. You must build internal tooling to:

Visualize retrieved chunks alongside the query and their associated relevance scores.
Isolate the generation step by feeding the exact retrieved context into the model again to see if it correctly synthesizes the answer.
Perform claim extraction, breaking responses into verifiable units to see which parts are grounded and which are hallucinated.
Root Cause Patterns (1:08:25 - 1:10:45)
Identifying recurring patterns helps prioritize engineering efforts.

Partial Correctness: Note that models often get some claims right while hallucinating others; segmenting these claims allows for more granular improvement.
Attribution Tracing: Automatically map model claims to source documents. If no link exists, flag the claim as a potential hallucination.
Prompt Sensitivity: If changing the prompt fixes a hallucination, the model was likely already capable of the answer but needed clearer instructions.
Edge Case Handling (1:10:45 - 1:14:20)
Handle the "known unknowns" gracefully:

No Results: When retrieval returns nothing, avoid forcing a hallucinated answer. Configure the model to state that the information is unavailable.
Conflicting Sources: If different documents contain contradictory information, the model should ideally present both perspectives rather than picking one arbitrarily.
Ambiguous Queries: Implement detection for out-of-scope or vague questions to avoid misleading the user.


### agent specific tool needs

Sources you must support:

Scanned PDFs (stamp papers, annexures)
Handwritten addendums
Poorly formatted Word/PDF files
Multi-language (English + Hindi + regional spillover)

Tools:

OCR (Indic language aware)
Layout-aware parsing (tables, schedules, annexures)
Metadata extraction (stamp duty, execution place, jurisdiction clause)

Data ingestion ≠ file upload
It includes:

Versioning (draft v/s executed v/s amended)
Annexure linking
Cross-reference resolution (“as per Clause 7.2(b)”)

If you skip this, lawyers won’t trust outputs.

B. Data Analysis Tools (beyond basic NLP)

NER alone is table stakes.

You need:

Entity normalization
(“Rs. 10 lakhs”, “₹10,00,000”, “Ten Lakh Rupees” → same value)
Temporal reasoning
“within 30 days of receipt” → receipt date + calendar + holidays
Conditional logic extraction
(“If X happens, Y obligation triggers”)
Jurisdictional mapping
Arbitration Act vs CPC
State stamp laws
Sectoral regulations

This is where LangGraph helps (multi-step reasoning).

C. Legal Knowledge Tools
Precedent linking (case law embeddings)
Statute grounding (section-level references)
Circulars / notifications (RBI, IRDAI, SEBI)

## Cost Optimization for RAG systems

GenAI Cost Fundamentals
Three primary cost domains: training, storage/retrieval, inference
Cost structure differences from traditional cloud computing
7B-13B parameter model economics (baseline for all estimates)

Fine-Tuning Economics
Upfront costs: $5K-$50K training compute for efficient methods (LoRA, QLoRA)
Monthly operating costs: $500-$3K per GPU for self-hosted serving
Break-even analysis: when $25K training investment pays off
Data requirements: 500-50K examples depending on task complexity
Parameter-efficient fine-tuning: 75-95% cost reduction vs full fine-tuning

RAG System Economics
Setup costs: $100-$1K for document ingestion and embedding
Monthly operating costs: $500-$5K vector database + query processing
Scaling behavior: linear cost growth with documents and queries
When RAG struggles: beyond 5M queries/month, beyond 50M documents
RAG advantages: zero training cost, faster time-to-production

Inference Optimization Techniques
Token optimization: prompt compression, context pruning, structured outputs
Caching strategies: 15-70% hit rates depending on query patterns
Cost reduction: 20-50% through semantic caching in production
Tiered model routing: 45% cost savings with complexity-based routing
Batch processing: 40-60% per-query savings (self-hosted only)

Cost Comparison & TCO Analysis
Monthly costs at 500K queries: $2.5K-$6K depending on architecture
Monthly costs at 5M queries: $8K-$60K depending on architecture
12-month TCO at different scales: $36K-$288K range
Break-even points: fine-tuning vs base models vs RAG
When fine-tuning achieves cost parity: 20-30M cumulative queries

Decision Framework
Query volume analysis: low (under 5M), medium (5M-20M), high (20M+) cumulative
Knowledge update frequency: static vs evolving vs hybrid
Quality requirements: high threshold vs good enough vs explainability needs
Hybrid architectures: combining fine-tuning + RAG for 35% cost savings

Production Implementation
Budget controls: per-team quotas, rate limiting, alerts
Cost monitoring: per-query metrics, token trends, cache hit rates
Reserved capacity strategy: 30-50% discounts with 60-70% baseline commitment
Continuous optimization: weekly reviews and iterative improvements

This detailed breakdown explores architectural patterns and cost-optimization strategies for GenAI, emphasizing that the best approach depends on scale, data volatility, and performance requirements.

Architectural Patterns

Architecture 1: Base Model + Optimizations (18:07)
Uses general-purpose foundation models with added caching and token optimization.
Pros: Fastest time to market and zero infrastructure maintenance (no training or retrieval pipelines).
Ideal for: Rapid prototyping, experimentation, or low-volume applications where the cost of custom development isn't justified.

Architecture 2: RAG (Retrieval-Augmented Generation) System + Optimizations (18:48)
Combines a foundation model with a vector database for external context.
Pros: Zero upfront training costs and excellent for applications requiring source attribution/citations.
Ideal for: Frequently changing knowledge bases where retraining would be prohibitive.

Architecture 3: Fine-tuned Model + Optimizations (19:28)
Involves high-upfront training costs, but results in a specialized, smaller, and more efficient model.
Pros: Lower per-query inference costs at scale due to model efficiency.
Ideal for: High cumulative query volume and stable, domain-specific requirements.

Cost Comparison & TCO
500,000 Queries/Month (20:18): Base models ($3K–$5K) and RAG ($3.5K–$6K) are generally cheaper than fine-tuning at this scale when considering the amortization of a $25K training investment ($2.5K–$4K monthly operational, excluding training costs).
5 Million Queries/Month (21:12): The economic advantage shifts. Base models scale linearly ($25K–$40K), while RAG becomes expensive due to retrieval overhead ($35K–$60K). Fine-tuning shines here ($8K–$18K) as the high upfront cost is diluted by high volume.
12-Month TCO Analysis (22:16): Fine-tuning achieves cost parity with other methods around 12–15 million cumulative queries. Beyond 24 million queries, fine-tuning can deliver 25–50% lower total costs compared to RAG or base models.

Decision Framework
Query Volume (24:55): <5M total queries favors base models; 5M–20M total queries is the "middle ground" for RAG; >20M total queries is the primary territory for fine-tuning.
Knowledge Update Frequency (25:46): Static knowledge favors fine-tuning (train once), while evolving knowledge requires RAG (re-embed documents rather than retrain).
Quality & Data Requirements (26:17): High-quality, specific task requirements often demand fine-tuning, provided you have sufficient training data (e.g., 500–50k examples). If explainability/citations are non-negotiable, RAG is the standard choice.

Hybrid Architectures (27:30)
Most high-performing production systems combine these approaches for efficiency:

Fine-tuning + RAG: Fine-tune for core domain language/jargon and use RAG for real-time data retrieval.
Tiered Routing + Caching: Use small, fast models for simple queries (via a router) and larger, expensive models only for complex tasks, while using semantic caching to eliminate redundant processing.
Example: A technical documentation system might use fine-tuning for code logic, RAG for API updates, and semantic caching for common developer questions, potentially reducing costs by 35% compared to a pure RAG

**pgvector vs. tsvector in PostgreSQL: Detailed Comparison and Relevance to Hybrid RAG Retrieval**

In the context of **Retrieval-Augmented Generation (RAG)** systems — especially the hybrid retrieval setups emphasized in the notes you shared (dense vector search + sparse keyword search + Reciprocal Rank Fusion/RRF) — `pgvector` and `tsvector` serve **complementary but fundamentally different** roles. They are not alternatives to each other; the most powerful production pattern is to use **both together** (dual-write strategy), exactly as the snippet you quoted highlights.

Below is a thorough breakdown covering **differences**, **strengths/weaknesses**, **performance characteristics**, **implementation nuances**, **edge cases**, and **which is more useful (or why both are essential)** for your RAG use case, including legal/enterprise document chunks with exact terms, clauses, codes, proper nouns, and multi-language elements.

### 1. Core Differences: What Each Actually Does

| Aspect                  | **pgvector** (Dense Vector Search)                                                                 | **tsvector** (PostgreSQL Built-in Full-Text Search)                                                                 |
|-------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| **Purpose**            | Semantic / meaning-based similarity search using embeddings (e.g., from text-embedding-3-large, voyage, etc.) | Lexical / keyword-based search using tokenized, stemmed, stop-word-removed text representations. |
| **Data Type**          | `vector(n)` — fixed or variable-length array of floats (e.g., 1536 dims for OpenAI).              | `tsvector` — specialized type storing lexemes (normalized words) + positions + weights. |
| **How Search Works**   | Approximate Nearest Neighbor (ANN) via cosine similarity, L2, or inner product. Uses HNSW (or IVFFlat) indexes for logarithmic speed. | Boolean matching (`@@` operator) + ranking via `ts_rank` or `ts_rank_cd`. Uses GIN (inverted) indexes. |
| **Strength**           | Captures synonyms, paraphrases, intent ("how to terminate a contract" matches "termination procedures" even with different words). Handles fuzzy semantics extremely well. | Excellent at **exact matches**, proper nouns, error codes, clause numbers, acronyms, product IDs, legal terms ("Section 7.2(b)", "Arbitration Act", "Rs. 10 lakhs"). Fast boolean filtering. |
| **Weakness**           | Can miss or down-rank exact rare terms, codes, or proper nouns (embeddings dilute signal for infrequent tokens). Vocabulary mismatch between query and docs. | Purely lexical — struggles with synonyms, paraphrases, or conceptual similarity. No native understanding of meaning. Ranking (`ts_rank`) is **not true BM25** (lacks global corpus statistics and proper length normalization). |
| **Ranking Quality**    | Semantic relevance (cosine distance). Good for intent but can return noisy results on exact-term queries. | `ts_rank_cd` (cover density) is decent but limited. True **BM25** (industry standard for relevance) requires extensions like `pg_textsearch`, ParadeDB, or pg_search. |
| **Indexing**           | HNSW (Hierarchical Navigable Small World) for fast ANN. Memory-resident recommended for ms latency at scale. | GIN index on the `tsvector` column (very efficient for keyword lookup). |
| **Language Support**   | Depends on the embedding model (many support multilingual; fine-tune for Hindi/regional).          | Built-in dictionaries for many languages (including 'english', 'hindi' via extensions). Stemming/stop-words configurable. |
| **Storage Overhead**   | High (each embedding is ~6–12 KB depending on dimensions).                                        | Low (tsvector is compact). |
| **Query Latency**      | Fast with HNSW (sub-10ms for top-K on millions of rows when indexed properly).                    | Very fast for filtering; ranking can slow down on large result sets without optimizations. |

**Key Takeaway on "BM25 / pg_textsearch Dual-Write"**:
- Native `tsvector` + `ts_rank` is **TF-IDF-like**, not full BM25. It doesn't normalize for document length properly or apply saturation (diminishing returns on term frequency) as effectively as true BM25.
- Many production RAG implementations still call the `tsvector` path "**BM25**" informally because it delivers the sparse/keyword precision that vectors lack.
- For stricter BM25 scoring, add extensions (pg_textsearch, ParadeDB pg_search, etc.). These integrate cleanly with pgvector and often deliver better relevance for hybrid setups.

### 2. Why Hybrid (pgvector + tsvector) is the Recommended Pattern in RAG
The original notes stress: **"relying solely on cosine similarity is insufficient for complex queries"** → Implement **Hybrid Retrieval** combining Vector Store (semantic) + Document Store (keyword/BM25).

**Real-World Gains** (from community benchmarks and implementations):
- Pure vector: Often ~60–70% precision (misses exact codes, clauses, names).
- Hybrid + RRF: Jumps to ~80–90%+ precision. The lexical path catches what embeddings dilute; semantic path catches paraphrases.
- Improvements of 8–30% in retrieval quality are commonly reported before even adding re-ranking (cross-encoder).
- RRF is preferred because it is **rank-based** (1 / (k + rank)) → no need to normalize incompatible scores (cosine 0–1 vs. ts_rank arbitrary scale). Robust and requires minimal tuning.

**In Your Legal-Domain Context** (scanned PDFs, clauses, annexures, "as per Clause 7.2(b)", stamp duty, jurisdiction, Hindi/English mix):
- **tsvector shines** for exact clause references, legal terminology, monetary amounts (after normalization), proper nouns, and section numbers.
- **pgvector shines** for conceptual queries ("what are the obligations if breach occurs?" matching similar but differently worded clauses across documents).
- Multi-language spillover: Use language-specific dictionaries or multilingual embeddings + careful preprocessing.
- Vision-extracted tables/charts: Text from VLMs still benefits from both (exact numbers vs. semantic description).

**Edge Cases Where One Dominates**:
- Exact match heavy (error codes, IDs, legal citations) → tsvector/keyword path critical.
- Ambiguous/intent-heavy or synonym-rich queries → pgvector dominant.
- Very long vs. short chunks: tsvector ranking can favor longer docs (mitigated somewhat by `ts_rank_cd`); true BM25 handles this better.
- Noisy OCR/handwritten text → Both can struggle; preprocessing (entity normalization, cleaning) + fuzzy trigram (`pg_trgm`) as a third leg helps.
- Scale (> few million chunks): HNSW for vectors + GIN for tsvector both scale well in Postgres if sharded/tuned; pure vector DBs may still win on extreme scale but lose ACID/Joins.

### 3. Implementation: Addressing the "Current Postgres Write Only Populates pgvector" Issue
The note you quoted is spot-on — you **must** dual-write/populate the `tsvector` column at ingestion time for hybrid queries to work.

**Recommended Schema Migration & Setup** (GENERATED ALWAYS is cleanest for maintenance):

```sql
-- 1. Enable extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- Optional: for fuzzy fallback

-- 2. Add columns (if not present)
ALTER TABLE your_chunks_table 
  ADD COLUMN IF NOT EXISTS embedding vector(1536),  -- or your dim
  ADD COLUMN IF NOT EXISTS text_search_vector tsvector 
    GENERATED ALWAYS AS (to_tsvector('english', coalesce(chunk_text, ''))) STORED;

-- For multi-language or weighted fields (title vs body):
-- GENERATED ALWAYS AS (
--   setweight(to_tsvector('english', coalesce(title, '')), 'A') ||
--   setweight(to_tsvector('english', coalesce(chunk_text, '')), 'B')
-- ) STORED

-- 3. Indexes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_chunks_embedding_hnsw 
  ON your_chunks_table 
  USING hnsw (embedding vector_cosine_ops);  -- Tune m/ef_search for recall vs speed

CREATE INDEX IF NOT EXISTS idx_chunks_tsv_gin 
  ON your_chunks_table 
  USING gin(text_search_vector);

-- Optional: metadata filters (source, page_no, document_type, jurisdiction) as btree/GIN
```

**At Write Time (Ingestion)**:
- Compute embedding (your embedding model).
- Let Postgres auto-generate `text_search_vector`.
- For custom weighting or multi-language: Compute manually and store (or use triggers).

**Hybrid Query Example with RRF** (simplified; adapt to your top-K and k parameter):

```sql
WITH semantic_search AS (
  SELECT id, chunk_text, 
         RANK() OVER (ORDER BY embedding <=> $1) AS rank  -- $1 = query embedding
  FROM your_chunks_table
  ORDER BY embedding <=> $1
  LIMIT 50
),
keyword_search AS (
  SELECT id, chunk_text,
         RANK() OVER (ORDER BY ts_rank_cd(text_search_vector, plainto_tsquery('english', $2)) DESC) AS rank
  FROM your_chunks_table, plainto_tsquery('english', $2) q
  WHERE text_search_vector @@ q
  LIMIT 50
),
combined AS (
  SELECT id, chunk_text, 
         (1.0 / (50 + COALESCE(semantic.rank, 999))) + 
         (1.0 / (50 + COALESCE(keyword.rank, 999))) AS rrf_score
  FROM semantic_search s
  FULL OUTER JOIN keyword_search k USING (id)
)
SELECT id, chunk_text, rrf_score
FROM combined
ORDER BY rrf_score DESC
LIMIT 20;
```

**Nuances & Best Practices**:
- Run the two searches **in parallel** (CTE or application-level) so total latency ≈ max(single latencies).
- Deduplicate by `id` before final ranking.
- Add metadata filters (e.g., `WHERE document_type = 'contract' AND jurisdiction = 'Delhi'`) early to shrink both candidate sets.
- For true BM25: Swap in an extension like `pg_textsearch` and use its ranking function.
- Over-fetch (top 40–100 per path) then RRF → better fusion.
- Monitor: Use `EXPLAIN ANALYZE` to confirm both indexes are hit. At 10M+ rows, tune HNSW parameters (`M`, `ef_search`) and ensure sufficient memory for indexes.
- Cost/Latency: Dual indexes add write overhead (minor) and storage, but read performance gains and simpler architecture (single DB) usually outweigh this vs. separate vector + search DB.

### 4. Which is More Useful Here? (For Your RAG Setup)
**Neither alone is sufficient for production-grade RAG** — especially in legal/enterprise contexts with precise references, mixed languages, and high trust requirements.

- **Use pgvector as the primary semantic backbone** (intent, concepts, multi-hop reasoning support via better context).
- **Use tsvector (or enhanced BM25) as the precision booster** for exactness, recall on specific terms, and to compensate for embedding weaknesses.
- **Hybrid + RRF + (optional cross-encoder re-ranking)** is the sweet spot → aligns perfectly with the notes' recommendations on hybrid systems, fusion ranking, and overcoming limitations of pure cosine similarity.

**When to Prioritize One**:
- Early prototyping or purely conceptual queries → Start with pgvector only (simpler).
- Production with exact-match sensitivity (your legal docs, clauses, amounts, jurisdictions) → **Hybrid is mandatory**.
- Very high scale or extreme latency needs → Consider specialized extensions or sharding, but Postgres + pgvector + tsvector handles millions of chunks very well for most teams.

**Additional Enhancements for Robustness** (from the broader notes):
- Hierarchical chunking + metadata (source, page_no, chunk_keywords, document_summary) → filter before vector/keyword search.
- Query transformation (rewriting, decomposition, HyDE) → feed better inputs to both paths.
- Post-retrieval re-ranking and faithfulness checks.
- For legal specifics: Preprocess with entity normalization, temporal/conditional extraction, and VLM for tables before creating chunks/embeddings/tsvector.

**Bottom Line**: The "dual-write" migration you mentioned is **essential** — without the `tsvector` column and GIN index, your hybrid RRF query simply cannot run. Implementing it gives you the best of both worlds with minimal added complexity, directly addressing the "naive similarity search fails on true intent or exact terms" problem highlighted throughout the material.

This setup keeps everything in one reliable Postgres instance (ACID, joins, backups, multi-tenancy via Row Level Security), which is a huge operational win for most RAG deployments. Test empirically on your query distribution and golden dataset — measure Context Precision/Recall and end-to-end faithfulness before/after adding the keyword path. You will almost certainly see meaningful gains. 

**For long-term production use with Gemini Embedding 2 + pgvector + pgvectorscale, the recommended dimension size is 768.**

### Why Gemini Embedding 2 and Its Dimension Flexibility Matter
Gemini Embedding 2 (the model you’re using — `gemini-embedding-2-preview` or equivalent) is Google’s latest multimodal embedding model.  
- Default output → **3072 dimensions**.  
- It natively supports **Matryoshka Representation Learning (MRL)**. This means the model is *trained* so that the most important semantic information is packed into the earliest dimensions.  
- You can request **any size from 128 to 3072** at inference time via the `output_dimensionality` parameter.  
- Google’s own official recommendation (and what they ship in all their example notebooks) is: **768, 1536, or 3072**.

Truncating with MRL is *not* the same as naively slicing a normal embedding — quality drop is minimal. Many production teams report that **768 dimensions from Gemini Embedding 2 performs at or above the level of older 1536-dimension models** (e.g., OpenAI text-embedding-ada-002 or text-embedding-3-small).

### Production Recommendation: 768 Dimensions (Start Here)

**This is the consensus sweet spot among teams running large-scale RAG in production with pgvector + pgvectorscale.**

| Dimension | Storage per vector (float32) | Relative storage vs 3072 | Query/index speed | Accuracy (relative to full 3072) | Best for (production reality) |
|-----------|------------------------------|---------------------------|-------------------|----------------------------------|-------------------------------|
| **768**   | ~6 KB                       | ~25%                     | Fastest           | Near-peak (excellent for most RAG) | **Long-term default** — most teams stay here |
| 1536      | ~12 KB                      | ~50%                     | Good              | Slightly higher in very nuanced cases | Only if benchmarks show clear win |
| 3072      | ~24 KB                      | 100%                     | Slowest           | Maximum                           | Rarely used at scale; only for tiny corpora or ultra-critical accuracy |

**Key reasons 768 wins for long-term use + accuracy**:
- **Accuracy is preserved extremely well** thanks to MRL. The first 768 dimensions already capture the vast majority of the signal. On your legal-domain data (clauses, jurisdiction, temporal logic, cross-references, scanned PDFs), 768 gives you high Context Precision/Recall and Faithfulness with almost no measurable regression vs. full size.
- **pgvectorscale loves it**. DiskANN (the streaming/disk-based index in pgvectorscale) and HNSW both scale dramatically better with smaller vectors. You get lower p95 latency, faster index builds, lower memory pressure, and cheaper hardware scaling. Teams report 2–4× faster queries and much cheaper storage/index size at 768 vs 1536/3072.
- **Long-term operational sanity**: Your corpus will grow (versioned contracts, annexures, amendments, multi-language docs). Smaller vectors = dramatically lower storage costs, faster re-embedding when models improve, easier sharding, and lower overall TCO. Many production RAG systems (legal, enterprise knowledge bases) deliberately choose 768–1024 even with models that support higher.
- **Real-world production feedback** (from teams running exactly this stack):  
  - Google’s own guides and Colab examples for Gemini Embedding 2 explicitly say “start with 768”.  
  - pgvector + pgvectorscale users consistently land on 768 as the balance point — it matches or beats older 1536 models while cutting storage/index overhead by ~75%.  
  - Higher dimensions (1536+) only deliver measurable gains on very specific, highly nuanced domains *after* you have already benchmarked on your golden dataset. Most teams never need to go beyond 768.

### When You Might Consider 1536 (or Rarely 3072)
Only bump up if, **after testing on your own data**, you see a clear, repeatable improvement in:
- Context Recall@10 / MRR on your legal query set.
- Faithfulness score (LLM-as-Judge on atomic claims).
- Handling of very subtle multi-hop or cross-document reasoning (e.g., “as per Clause 7.2(b) in conjunction with Schedule A”).

Even then, **1536 is usually sufficient** — 3072 is almost never worth it at production scale unless your corpus is <100k chunks and accuracy is life-or-death.

### How to Implement It (Code + SQL)

**1. When calling the Gemini API (Python example)**
```python
from google import genai

client = genai.Client()
response = client.models.embed_content(
    model="gemini-embedding-2-preview",
    contents=[your_text_or_pdf_or_image],
    config={
        "task_type": "RETRIEVAL_DOCUMENT",   # or RETRIEVAL_QUERY for queries
        "output_dimensionality": 768          # ← THIS IS THE KEY
    }
)
embedding = response.embedding  # now 768 dims
```

**2. Postgres schema (pgvector + pgvectorscale)**
```sql
-- Column definition
ALTER TABLE chunks 
  ADD COLUMN IF NOT EXISTS embedding vector(768);   -- ← match exactly

-- Recommended index with pgvectorscale (DiskANN) or classic HNSW
CREATE INDEX IF NOT EXISTS idx_chunks_embedding 
  ON chunks 
  USING vectorscale_diskann (embedding vector_cosine_ops);   -- or hnsw

-- Or classic pgvector HNSW (still excellent at 768)
-- CREATE INDEX ... USING hnsw (embedding vector_cosine_ops);
```

**3. Best practice**  
- Always **normalize** the vector after truncation (especially for cosine similarity) — many teams do `embedding = embedding / np.linalg.norm(embedding)` in code.  
- Store the dimension used in metadata so you can re-embed consistently later.  
- Keep your hybrid setup (pgvector + tsvector/BM25) — the 768 vector gives excellent semantic signal, while tsvector handles exact clause numbers, legal terms, etc.

### Final Advice for Your Setup
**Use 768 dimensions from day one.**  
It is the production-proven choice for long-term maintainability, cost efficiency, and accuracy when paired with Gemini Embedding 2, pgvector, and pgvectorscale. You will get near-maximum retrieval quality while keeping your system fast, cheap to scale, and easy to operate as your legal document corpus grows.

Test it on your golden dataset (the same (Question, Answer, Relevant Chunk IDs) triplets you should already have). If 768 meets your precision/recall/faithfulness targets (which it almost certainly will), **never go higher** — you will thank yourself six months from now when you’re at 5–10 million chunks.

If you share your current table schema or a couple of sample queries + expected chunks, I can give you the exact SQL and Python snippets tailored to your hybrid RAG pipeline.

## Contract KB Implementation Correctness Checklist

Use this checklist instead of a unit-test plan for the current implementation pass.

1. **Layout-Aware Parsing Node**
   - `parse_document` is the first ingestion node.
   - Uses Docling.
   - Outputs structured markdown.
   - Tables preserve row/cell structure and are not flattened into unreadable strings.

2. **Parent-Child Chunk Architecture**
   - One parent document row exists per uploaded file.
   - Multiple child `clauses` rows exist per parent.
   - `clauses.parent_doc_id` is a UUID FK.
   - `clauses.chunk_index` is populated.
   - Retrieval fetches child chunks and can return parent/surrounding context.

3. **Contextual Preamble Injection**
   - A separate `contextualize_chunks` node exists.
   - Uses LangGraph `Send` fan-out.
   - Each child chunk receives a preamble like: `"This is [clause_type] from [contract_name] between [party_a] and [party_b], effective [date]."`
   - Uses `.with_structured_output(ContextualizedChunk)`.
   - `ContextualizedChunk` includes `preamble`, `text`, and `tokens`.

4. **langextract Structured Extraction Pass**
   - A dedicated `extract_schema` node runs on full document text, not chunks.
   - Extracts `effective_date`, `parties`, `contract_value`, `jurisdiction`, `governing_law`, `termination_notice_days`, and `liability_cap`.
   - Uses `.with_structured_output(ContractMetadata)` against Gemini.
   - Output populates `metadata_ JSONB`.

5. **BM25 / pg_textsearch Dual-Write**
   - `pg_textsearch` is used exclusively.
   - No `tsvector` column exists for this pipeline.
   - No GIN full-text index exists for this pipeline.
   - `clauses_bm25_idx` exists on `search_text`.
   - `search_text` includes clause type, preamble, and chunk text.
   - Retrieval uses `<@> to_bm25query(:query_text, 'clauses_bm25_idx')`.

6. **JSONB Metadata Column for Pre-Filtering**
   - `clauses.metadata_` exists as JSONB.
   - Populated with jurisdiction, contract type, year, party names, and required retrieval metadata.
   - Retrieval can hard-filter with expressions like `metadata_->>'jurisdiction' = 'Delaware'`.

7. **Immediate 3-Attempt Retry Handling**
   - External I/O in ingestion/retrieval nodes has retry handling.
   - Gemini, Postgres writes, Neo4j writes, Graphiti upserts, Redis calls, and embedding calls are covered where retry is safe.
   - Retry policy is exactly 3 immediate attempts with no exponential wait.
   - Transient errors are separated from validation/permanent errors.

8. **MessagesState Standardization**
   - Retrieval graph state inherits from `MessagesState`.
   - Retriever and generator pass `HumanMessage` / `AIMessage` objects through state.
   - Ingestion can remain specialized typed state.
   - Handoff boundary from ingestion to retrieval is typed and message-compatible.

9. **Structured Output Throughout**
   - No raw JSON parsing for structured LLM outputs.
   - Entity extraction uses `EntityExtractionResult`.
   - Clause classification uses `ClauseClassification`.
   - Preamble generation uses `ContextualizedChunk`.
   - Schema extraction uses `ContractMetadata`.

10. **Query Analyzer & Router Node**
    - Retrieval graph starts with Query Analyzer.
    - Performs coreference rewrite.
    - Decomposes multi-part questions.
    - Routes to `hybrid_postgres`, `graph_neo4j`, or `both`.
    - Emits runtime RRF weights.
    - Uses `.with_structured_output(QueryPlan)`.

11. **In-Database RRF via CTE**
    - RRF is computed inside Postgres.
    - Vector and `pg_textsearch` ranks are CTE/subquery branches.
    - BM25 ordering accounts for negative scores.
    - Outer query fuses with weighted `1/(60 + rank)` terms.
    - Python receives only top-K fused rows.

12. **Cross-Encoder Reranker Node**
    - Retrieval candidates top 20 are reranked.
    - Default model is `BAAI/bge-reranker-v2-m3` or configured fallback.
    - Output is cut to top 5.
    - Node is documented as a Celery task candidate for V2.

13. **Context Grader Node**
    - Uses `.with_structured_output(ContextGrade)`.
    - Includes `sufficient`, `missing_aspects`, and `rewrite_suggestion`.
    - If insufficient, loops back to Query Analyzer.
    - Loop is capped at 2 iterations.
    - After cap, returns hardcoded fallback instead of improvising.

14. **Grounded Generator Node**
    - Final generator requires citation for every factual claim.
    - Citations include exact `chunk_id` and `clause_type`.
    - Output schema is `GeneratedAnswer`.
    - If confidence is `uncertain`, fallback disclaimer is appended.

15. **Redis Exact-Match Query Cache**
    - Query Analyzer checks cache before retrieval work.
    - Key is `sha256(rewritten_query + doc_ids_filter)`.
    - Cache hit returns `GeneratedAnswer` directly.
    - TTL is 1 hour normally, 24 hours for immutable historical documents.

16. **Embedding Cache**
    - Embedding API calls are wrapped in Redis cache.
    - Key is `sha256(text_to_embed)`.
    - Repeated boilerplate clauses reuse cached embeddings.

17. **Additional Graphiti Node/Edge Types**
    - Adds nodes: `Party`, `Obligation`, `RightOrPermission`, `PenaltyClause`.
    - Adds edges: `SIGNED_BY`, `SUBSIDIARY_OF`, `OBLIGATED_TO`, `GOVERNED_BY`, `SUPERSEDES`, `REFERENCES_CLAUSE`.
    - `REFERENCES_CLAUSE` carries `postgres_chunk_id`.
    - Graph traversal can return Postgres IDs as hard retrieval filters.

18. **Graphiti Episode Upsert for Contract Events**
    - Ingestion writes event episodes for `contract_signed`, `amendment_effective`, and `expiry_date`.
    - Temporal queries can use Graphiti's time-aware graph search instead of scanning Postgres.


# Advanced RAG Utilities

This folder now exposes one reusable module for the numbered strategy demos that previously lived in separate files.

## Module Layout

- `strategies.py`: reusable ingestion, retrieval, reranking, routing, and answer-generation helpers for advanced RAG strategies
- `rag_agent_advanced.py`: older demo-oriented agent script that can consume the reusable helpers if needed
- `graphiti/`, `langextract/`, `pageindex/`, `multimodal/`: strategy-adjacent integrations that remain separate because they solve distinct problems

## Runtime Conventions

The consolidated strategy module is aligned with the application runtime:

- uses `app.state.db_engine` and `app.state.db_session_local`
- writes SQL with SQLAlchemy `text(...)`
- uses `app.utils.logger`
- uses `orjson` for metadata serialization
- avoids `psycopg2`, module-level DB connections, and hardcoded local credentials

## Public API

Core entrypoint:

```python
from app.shared.rag.strategies import RAGStrategyService

service = RAGStrategyService.from_app(app)
```

Key reusable methods:

- `ingest_document()`
- `ingest_contextual_document()`
- `ingest_context_aware_document()`
- `ingest_late_chunked_document()`
- `ingest_hierarchical_document()`
- `vector_search()`
- `search_with_reranking()`
- `search_with_query_expansion()`
- `search_with_multi_query()`
- `search_hierarchical()`
- `search_knowledge_graph()`
- `run_agentic_rag()`
- `run_self_reflective_rag()`
- `retrieve_full_document()`

Useful pure helpers:

- `parse_query_variants()`
- `semantic_chunk_text()`
- `late_chunk_text()`
- `mean_pool_embeddings()`
- `serialize_metadata()`
- `deserialize_metadata()`

## Strategy Reference

| Strategy | Support in `strategies.py` | Notes |
|---|---|---|
| Re-ranking | `search_with_reranking` | Two-stage vector retrieval plus cross-encoder reranking |
| Agentic RAG | `run_agentic_rag` | Routes between vector, SQL, and web resolvers |
| Knowledge graphs | `search_knowledge_graph` | Uses injected graph searcher instead of hardcoded Graphiti credentials |
| Contextual retrieval | `ingest_contextual_document` | Adds LLM-generated chunk prefixes before embedding |
| Query expansion | `expand_query`, `search_with_query_expansion` | Generates alternate phrasings and unions results |
| Multi-query RAG | `search_with_multi_query` | Same expansion path, tuned for broader retrieval |
| Context-aware chunking | `semantic_chunk_text`, `ingest_context_aware_document` | Splits on semantic boundaries rather than only size |
| Late chunking | `late_chunk_text`, `ingest_late_chunked_document` | Blends document and chunk embeddings |
| Hierarchical RAG | `ingest_hierarchical_document`, `search_hierarchical` | Stores parent/child relationships in `meta_data` |
| Self-reflective RAG | `run_self_reflective_rag` | Grades retrieval quality, refines query, verifies answer support |
| Fine-tuned embeddings | `ingest_document(..., use_fine_tuned_embeddings=True)` | Uses optional SentenceTransformer path supplied to the service |

## Storage Model

The old demos assumed ad hoc tables such as `chunks`, `parent_chunks`, and `child_chunks`. The merged implementation stores strategy variants in the existing `document_vectors` table and encodes strategy-specific state inside `meta_data`.

Examples:

- hierarchical parent rows use `{"strategy":"hierarchical","level":"parent","hierarchy_id":"..."}`
- hierarchical child rows use `{"strategy":"hierarchical","level":"child","hierarchy_id":"..."}`
- contextual retrieval rows use `{"strategy":"contextual_retrieval", ...}`

## Example

```python
service = RAGStrategyService.from_app(app, fine_tuned_model_path="./fine_tuned_model")

await service.ingest_contextual_document(
    user_id="user-123",
    document_id="doc-1",
    title="Refund Policy",
    text_value=document_text,
)

results = await service.search_with_reranking(
    user_id="user-123",
    query="What is the refund window?",
    limit=5,
)
```

## Notes

- The consolidated module is reusable application code, not a turnkey “all strategies at once” production pipeline.
- Graph-based retrieval still depends on the caller to provide a graph search function.
- Cross-encoder and SentenceTransformer models are loaded lazily to avoid import-time side effects.
- The old numbered files were educational demos and have been removed in favor of the single reusable module.


# Re-ranking
add all the contents of this file in the README.md


## Resource
**Rerankers and Two-Stage Retrieval | Pinecone**
https://www.pinecone.io/learn/series/rag/rerankers/

## What It Is
Re-ranking uses a cross-encoder model to refine initial retrieval results by scoring query-document pairs more accurately. After a fast retriever (like vector search) returns candidate documents, the re-ranker evaluates each candidate with the query simultaneously, capturing richer semantic interactions. This two-stage approach balances speed and accuracy.

## Simple Example
```python
# Stage 1: Fast retrieval
candidates = vector_search(query, top_k=100)

# Stage 2: Re-rank with cross-encoder
reranker = CrossEncoder('ms-marco-MiniLM')
scored_results = []
for doc in candidates:
    score = reranker.predict([query, doc])
    scored_results.append((doc, score))

# Return top re-ranked results
final_results = sorted(scored_results, key=lambda x: x[1], reverse=True)[:10]
```

## Pros
Significantly improves retrieval precision by understanding query-document relationships. Works well as a refinement layer on top of existing systems.

## Cons
Computationally expensive compared to embedding models. Adds latency as each document must be processed with the query.

## When to Use It
Use when accuracy is more important than speed. Ideal for narrowing down a large candidate set to the most relevant documents.

## When NOT to Use It
Avoid when real-time performance is critical. Skip if you have limited compute resources or very large result sets to re-rank.


# Agentic RAG

## Resource
**What is Agentic RAG? Building Agents with Qdrant**
https://qdrant.tech/articles/agentic-rag/

## What It Is
Agentic RAG empowers autonomous agents with multiple tools to explore knowledge dynamically. Unlike traditional RAG's single vector search, agents can write SQL queries for structured data, perform web searches, read entire files, or query multiple vector stores based on query complexity. The agent reasons about which tools to use and in what order, adapting its strategy to the task.

## Simple Example
```python
# Agent decides which tools to use
agent = RAGAgent(tools=[vector_search, sql_query, web_search, file_reader])

query = "What were Q2 sales for ACME Corp?"

# Agent reasoning:
# 1. Checks if structured data needed → use SQL tool
result = agent.sql_tool("SELECT revenue FROM quarterly_sales WHERE company='ACME' AND quarter='Q2'")

# If insufficient, agent tries another tool
if not result.complete:
    result += agent.vector_search("ACME Q2 financial performance")
```

## Pros
Highly flexible and adapts to query complexity. Can access heterogeneous data sources (SQL, files, web, vectors) intelligently.

## Cons
Increased complexity and unpredictability in behavior. Higher latency and cost due to multi-step reasoning and tool calls.

## When to Use It
Use for complex queries requiring multiple data sources or exploration strategies. Ideal when you have diverse knowledge types (structured and unstructured).

## When NOT to Use It
Avoid for simple lookups where traditional RAG suffices. Skip when you need predictable, fast responses or have limited tool infrastructure.


# Knowledge Graphs

## Resource
**RAG Tutorial: How to Build a RAG System on a Knowledge Graph | Neo4j**
https://neo4j.com/blog/developer/rag-tutorial/

## What It Is
Knowledge Graph RAG (GraphRAG) combines vector search with graph databases to capture both semantic meaning and explicit relationships between entities. Instead of just retrieving similar text chunks, the system queries a graph of interconnected entities (nodes) and relationships (edges), providing structured, contextual information. This grounds LLM responses in factual relationships and prevents hallucinations.

## Simple Example
```python
# Knowledge graph structure
graph = {
    "ACME Corp": {
        "type": "Company",
        "relationships": {
            "HAS_CEO": "Jane Smith",
            "REPORTED_REVENUE": "$314M",
            "LOCATED_IN": "California"
        }
    }
}

# Query combining vector + graph
query = "Who runs ACME Corp?"

# 1. Vector search finds relevant entity
entity = vector_search(query)  # Returns "ACME Corp"

# 2. Traverse graph for relationships
result = graph.query(
    "MATCH (c:Company {name: 'ACME Corp'})-[:HAS_CEO]->(ceo) RETURN ceo"
)  # Returns "Jane Smith"

# 3. LLM generates answer with structured facts
answer = llm.generate(query, context=result)
```

## Pros
Captures explicit relationships that vectors miss. Reduces hallucinations by providing structured, factual connections.

## Cons
Requires building and maintaining a knowledge graph. Complex setup and querying compared to simple vector search.

## When to Use It
Use when relationships between entities are crucial to answers. Ideal for domains with complex interconnections (healthcare, finance, research).

## When NOT to Use It
Avoid when data lacks clear entities and relationships. Skip if you need rapid prototyping without graph infrastructure investment.


# Contextual Retrieval

## Resource
**Introducing Contextual Retrieval | Anthropic**
https://www.anthropic.com/news/contextual-retrieval

## What It Is
Contextual Retrieval, introduced by Anthropic, prepends chunk-specific explanatory context to each chunk before embedding and indexing. An LLM generates a brief description explaining what each chunk is about in relation to the entire document. This technique includes Contextual Embeddings and Contextual BM25, reducing retrieval failures by 49% alone and 67% when combined with re-ranking.

## Simple Example
```python
# Original chunk (lacks context)
chunk = "The company's revenue grew by 3% over the previous quarter."

# Generate contextual prefix with LLM
context = llm.generate(
    f"Document: {full_document}\n\nChunk: {chunk}\n\n"
    "Provide brief context for this chunk:"
)
# Returns: "This chunk is from ACME Corp's Q2 2023 SEC filing;
# previous quarter revenue was $314M."

# Embed with context
contextualized_chunk = context + " " + chunk
embedding = embed_model.encode(contextualized_chunk)

# Also create contextual BM25 index with the same contextualized chunks
bm25_index.add(contextualized_chunk)
```

## Pros
Dramatically improves retrieval accuracy by adding document context to chunks. Works with both vector embeddings and BM25 keyword search.

## Cons
Significantly increases indexing time and cost due to LLM calls for every chunk. Larger index size due to additional context text.

## When to Use It
Use when chunks lack standalone meaning without document context. Ideal for technical documents, financial reports, or dense reference materials.

## When NOT to Use It
Avoid when chunks are already self-contained and clear. Skip if indexing budget or time constraints are tight, or corpus updates frequently.



# Query Expansion

## Resource
**Advanced RAG: Query Expansion | Haystack**
https://haystack.deepset.ai/blog/query-expansion

## What It Is
Query Expansion enhances user queries by generating multiple variations or adding related terms before retrieval. An LLM automatically generates additional queries from different perspectives, capturing various aspects of the user's intent. This addresses vague or poorly formed queries and helps cover synonyms and similar meanings.

## Simple Example
```python
# Original query
user_query = "What is RAG?"

# LLM generates expanded queries
expanded_queries = [
    "What is Retrieval Augmented Generation?",
    "How does RAG work in AI systems?",
    "Explain RAG architecture and components"
]

# Retrieve documents for all queries
for query in expanded_queries:
    results = vector_search(query)
```

## Pros
Improves retrieval recall by capturing multiple interpretations of the query. Handles vague queries and terminology variations effectively.

## Cons
Increases latency and cost due to multiple LLM calls and retrievals. May introduce noise if expanded queries drift from original intent.

## When to Use It
Use when users provide short, ambiguous, or poorly-worded queries. Ideal for keyword-based retrieval systems that need semantic variations.

## When NOT to Use It
Avoid when queries are already specific and well-formed. Skip if latency is critical or when operating under strict cost constraints.


# Multi-Query RAG

## Resource
**Advanced RAG: Multi-Query Retriever Approach | Medium**
https://medium.com/@kbdhunga/advanced-rag-multi-query-retriever-approach-ad8cd0ea0f5b

## What It Is
Multi-Query RAG generates multiple reformulations of the original query, executes parallel searches, and aggregates results. An LLM creates diverse perspectives of the same question to overcome the limitations of distance-based retrieval. Results from all queries are combined (typically taking the unique union) to create a richer, more comprehensive result set.

## Simple Example
```python
# Generate multiple query perspectives
original_query = "How do I deploy a model?"

reformulated_queries = llm.generate([
    "What are model deployment steps?",
    "Best practices for deploying ML models",
    "Model deployment infrastructure options"
])

# Execute searches in parallel
all_results = []
for query in reformulated_queries:
    results = vector_search(query, top_k=20)
    all_results.extend(results)

# Deduplicate and return unique results
final_results = deduplicate(all_results)
```

## Pros
Mitigates single query bias and improves result diversity. Increases recall by capturing different interpretations of user intent.

## Cons
Higher computational cost due to multiple retrievals. May retrieve redundant or less relevant documents if queries overlap poorly.

## When to Use It
Use when user queries may have multiple valid interpretations. Ideal for improving recall on ambiguous or broad questions.

## When NOT to Use It
Avoid for very specific queries with clear intent. Skip when latency and cost are major constraints, or retrieval corpus is small.



# Context-Aware Chunking

## Resource
**Semantic Chunking for RAG | Medium**
https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

## What It Is
Context-Aware Chunking (also called semantic chunking) intelligently determines chunk boundaries based on semantic similarity rather than fixed sizes. It generates embeddings for sentences, compares their similarity, and groups semantically related content together. This ensures chunks contain coherent topics, improving embedding quality and retrieval accuracy.

## Simple Example
```python
# Split document into sentences
sentences = document.split_sentences()

# Generate embeddings for each sentence
embeddings = [embed_model.encode(s) for s in sentences]

# Calculate similarity between consecutive sentences
similarities = [cosine_similarity(embeddings[i], embeddings[i+1])
                for i in range(len(embeddings)-1)]

# Group sentences where similarity > threshold
chunks = []
current_chunk = [sentences[0]]
for i, sim in enumerate(similarities):
    if sim > 0.8:  # High similarity = same topic
        current_chunk.append(sentences[i+1])
    else:  # Low similarity = new topic
        chunks.append(" ".join(current_chunk))
        current_chunk = [sentences[i+1]]
```

## Pros
Creates semantically coherent chunks that improve embedding quality. Preserves topic continuity and contextual meaning within chunks.

## Cons
Computationally expensive due to embedding every sentence. Slower indexing process compared to simple fixed-size chunking.

## When to Use It
Use when document topics are diverse and intermingled. Ideal for complex documents where topic boundaries are important for retrieval.

## When NOT to Use It
Avoid when processing speed is critical or documents are already well-structured. Skip for homogeneous documents with consistent topics throughout.



# Late Chunking

## Resource
**Late Chunking in Long-Context Embedding Models | Jina AI**
https://jina.ai/news/late-chunking-in-long-context-embedding-models/

## What It Is
Late Chunking processes entire documents (or large sections) through the embedding model's transformer before splitting into chunks. Traditional "naive chunking" splits text first, losing long-distance context. Late Chunking embeds all tokens together, then applies chunking after the transformer but before pooling, preserving full contextual information in each chunk's embedding.

## Simple Example
```python
# Traditional chunking (loses context)
chunks = split_document(doc, chunk_size=512)
embeddings = [embed_model.encode(chunk) for chunk in chunks]

# Late Chunking (preserves context)
# 1. Process entire document through transformer
full_doc_embeddings = transformer_layer(doc)  # 8192 tokens max

# 2. Chunk the token embeddings (not the text)
chunk_boundaries = [0, 512, 1024, 1536, ...]
chunk_embeddings = []
for i in range(len(chunk_boundaries)-1):
    start, end = chunk_boundaries[i], chunk_boundaries[i+1]
    # Mean pool the token embeddings for this chunk
    chunk_emb = mean_pool(full_doc_embeddings[start:end])
    chunk_embeddings.append(chunk_emb)
```

## Pros
Maintains full document context in chunk embeddings, improving accuracy. Leverages long-context models (8K+ tokens) effectively.

## Cons
Requires long-context embedding models with high token limits. More complex implementation than standard chunking approaches.

## When to Use It
Use when document context is crucial for understanding chunks. Ideal for documents where meaning depends on long-distance relationships.

## When NOT to Use It
Avoid when using standard embedding models with small context windows. Skip if documents are already short and context is local.



# Hierarchical RAG

## Resource
**Document Hierarchy in RAG: Enhancing AI Efficiency | Medium**
https://medium.com/@nay1228/document-hierarchy-in-rag-boosting-ai-retrieval-efficiency-aa23f21b5fb9

## What It Is
Hierarchical RAG organizes documents in parent-child relationships, retrieving small chunks for accurate matching while providing larger parent contexts for generation. Child chunks are embedded and searched, but when a match is found, the system returns the parent chunk (containing broader context) to the LLM. Metadata maintains relationships between chunks, enabling efficient navigation of the hierarchy.

## Simple Example
```python
# Index structure
document = {
    "parent": "Q2 Financial Report - Full Section",
    "children": [
        "Revenue increased 3% to $314M",
        "Operating costs decreased 5%",
        "Net profit margin improved to 12%"
    ]
}

# Embed only child chunks
for child in document["children"]:
    index.add(embed(child), metadata={"parent_id": document["parent"]})

# Retrieval
query = "What was Q2 revenue?"
child_match = vector_search(query)  # Finds "Revenue increased 3%..."

# Return parent context instead of just the child
full_context = get_parent(child_match.metadata["parent_id"])
# LLM sees entire Q2 section for better reasoning
```

## Pros
Balances retrieval precision with generation context. Reduces noise in search while providing sufficient context for reasoning.

## Cons
Requires careful design of parent-child relationships. Adds complexity to indexing and retrieval logic.

## When to Use It
Use when small chunks match better but lack sufficient context for answers. Ideal for structured documents with natural hierarchies (sections, chapters).

## When NOT to Use It
Avoid when documents lack clear hierarchical structure. Skip if simple flat chunking provides adequate context for your use case.




# Self-Reflective RAG

## Resource
**Self-Reflective RAG with LangGraph | LangChain**
https://blog.langchain.com/agentic-rag-with-langgraph/

## What It Is
Self-Reflective RAG (including Self-RAG and Corrective RAG/CRAG) adds self-assessment and iterative refinement to retrieval. The system evaluates whether retrieved documents are relevant, grades response quality, and refines queries or retrieves additional information when necessary. It creates a feedback loop where the system critiques its own outputs and adapts until producing a satisfactory answer.

## Simple Example
```python
# Initial retrieval
query = "What is quantum computing?"
docs = vector_search(query)

# Self-reflection: Grade document relevance
grades = []
for doc in docs:
    grade = llm.evaluate(f"Is this document relevant to '{query}'? {doc}")
    grades.append(grade)

# If relevance is low, refine and retry
if avg(grades) < 0.7:
    refined_query = llm.refine(query, docs)
    docs = vector_search(refined_query)

# Generate answer
answer = llm.generate(query, docs)

# Self-reflection: Verify answer quality
if not llm.verify_answer(answer, docs):
    # Retrieve more context or refine further
    additional_docs = web_search(query)
    answer = llm.generate(query, docs + additional_docs)
```

## Pros
Improves answer quality through self-correction and validation. Adapts dynamically to poor retrieval results by refining approach.

## Cons
Significantly higher latency due to multiple LLM calls and iterations. Increased cost and complexity compared to single-pass RAG.

## When to Use It
Use when answer accuracy is critical and errors are costly. Ideal for complex queries where initial retrieval may be insufficient.

## When NOT to Use It
Avoid when real-time responses are required. Skip for simple queries or when operating under strict latency/cost budgets.



# Fine-tuned Embedding Models

## Resource
**Fine-tune Embedding models for Retrieval Augmented Generation (RAG) | Philipp Schmid**
https://www.philschmid.de/fine-tune-embedding-model-for-rag

## What It Is
Fine-tuning embedding models adapts pre-trained models to domain-specific data, improving retrieval accuracy for specialized vocabularies and contexts. Instead of using generic embeddings trained on broad data, you train the model on your specific corpus with relevant query-document pairs. This teaches the model to recognize domain jargon, relationships, and semantic patterns unique to your use case.

## Simple Example
```python
# Prepare domain-specific training data
training_data = [
    ("What is EBITDA?", "positive_doc_about_EBITDA.txt"),
    ("Explain capital expenditure", "capex_explanation.txt"),
    # ... thousands of query-document pairs
]

# Load pre-trained model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Fine-tune on domain data
fine_tuned_model = base_model.fit(
    train_data=training_data,
    epochs=3,
    loss=MultipleNegativesRankingLoss()
)

# Use fine-tuned model for retrieval
query_embedding = fine_tuned_model.encode("What is working capital?")
# Better matches domain-specific financial documents
```

## Pros
Significantly improves retrieval accuracy for specialized domains (5-10% gains typical). Can achieve better performance with smaller model sizes after fine-tuning.

## Cons
Requires domain-specific training data (query-document pairs or synthetic data). Additional time and resources needed for training and evaluation.

## When to Use It
Use when working with specialized domains (medical, legal, technical) with unique terminology. Ideal when retrieval accuracy is suboptimal with generic embeddings.

## When NOT to Use It
Avoid when working with general knowledge or common topics. Skip if you lack training data or can't afford the fine-tuning investment.


## 🎯 Strategy Overview

| # | Strategy | Status | Use Case | Pros | Cons |
|---|----------|--------|----------|------|------|
| 1 | [Re-ranking](#1-re-ranking) | ✅ Code Example | Precision-critical | Highly accurate results | Slower, more compute |
| 2 | [Agentic RAG](#2-agentic-rag) | ✅ Code Example | Flexible retrieval needs | Autonomous tool selection | More complex logic |
| 3 | [Knowledge Graphs](#3-knowledge-graphs) | 📝 Pseudocode Only | Relationship-heavy | Captures connections | Infrastructure overhead |
| 4 | [Contextual Retrieval](#4-contextual-retrieval) | ✅ Code Example | Critical documents | 35-49% better accuracy | High ingestion cost |
| 5 | [Query Expansion](#5-query-expansion) | ✅ Code Example | Ambiguous queries | Better recall, multiple perspectives | Extra LLM call, higher cost |
| 6 | [Multi-Query RAG](#6-multi-query-rag) | ✅ Code Example | Broad searches | Comprehensive coverage | Multiple API calls |
| 7 | [Context-Aware Chunking](#7-context-aware-chunking) | ✅ Code Example | All documents | Semantic coherence | Slightly slower ingestion |
| 8 | [Late Chunking](#8-late-chunking) | 📝 Pseudocode Only | Context preservation | Full document context | Requires long-context models |
| 9 | [Hierarchical RAG](#9-hierarchical-rag) | 📝 Pseudocode Only | Complex documents | Precision + context | Complex setup |
| 10 | [Self-Reflective RAG](#10-self-reflective-rag) | ✅ Code Example | Research queries | Self-correcting | Highest latency |
| 11 | [Fine-tuned Embeddings](#11-fine-tuned-embeddings) | 📝 Pseudocode Only | Domain-specific | Best accuracy | Training required |

