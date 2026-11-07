"""Constants for the unified documents feature.

Two of these are not tuning knobs but **identifier names**, and they exist for the schema
identifier gate to compare against: `CHUNKS_BM25_INDEX_NAME` and
`CHUNKS_UNIQUE_CONSTRAINT_NAME`. Decision 10 rejects interpolating an identifier into query
text, so the literals stay literals at their use sites and these constants are what the gate
asserts them equal to. Do not "DRY" a use site by importing one of these into an f-string —
that defeats both the decision and the gate, which scans query text statically.

`SEARCH_CHUNKS_BM25_INDEX_NAME` from the superseded feature is deliberately not carried over:
it named an index on `search_chunks`, a relation this change consolidates away.
"""

# --- pagination and budgets --------------------------------------------------

DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
DEFAULT_RAG_TOKEN_BUDGET = 2_000
DEFAULT_SEARCH_CACHE_TTL_SECONDS = 900

# --- retrieval ---------------------------------------------------------------

HYBRID_CANDIDATE_LIMIT = 50
RRF_K = 60
# Per-leg fusion weights for `reciprocal_rank_fusion`, weighted toward the lexical
# legs. These are starting priors, not measurements — tune them against the
# retrieval eval harness (`evals/reports/baseline.json`) before treating any value
# as settled. The fusion default stays unweighted; callers opt in explicitly.
RRF_WEIGHT_BM25 = 1.5
RRF_WEIGHT_VECTOR = 1.0
RRF_WEIGHT_TRIGRAM = 1.2
# Relationship-graph fusion branch (optional fourth leg, default-off via
# DOCUMENTS_GRAPHITI_FUSION_ENABLED). Measured, not folk: 0.1 is the argmax of
# the recorded weight sweep on the grown golden corpus — recall@10, MRR, and
# nDCG@10 all peak at 0.1 and the branch's leave-one-branch-out ablation at
# that weight is positive on every metric. Report (measurement method, sweep
# table, proxy caveat, contribution deltas):
# evals/reports/graphiti_branch_legal_retrieval_v2_wave4.json and
# evals/reports/graphiti_branch_legal_retrieval_v2_wave4.measurements.json.
# Flipping the branch default-on is a later change that cites that report.
RRF_WEIGHT_GRAPHITI = 0.1
# Keyword-leg phrase over-fetch multiple: the LIKE post-filter narrows, so the
# leg reads this many times the candidate budget and truncates after filtering.
PHRASE_OVERFETCH_MULTIPLE = 5
TRIGRAM_SIMILARITY_THRESHOLD = 0.1
DISKANN_QUERY_SEARCH_LIST_SIZE = 100
DISKANN_QUERY_RESCORE = 50

# --- ingestion ---------------------------------------------------------------

INGEST_EMBEDDING_BATCH_SIZE = 200
ANALYZE_THRESHOLD_CHUNKS = 10_000

# Explicit provenance for the canonical chunk pipeline.  This is kept in the
# schema (rather than inferred from JSON metadata) so cache keys and durable
# evidence handles can reject stale bytes deterministically.
CANONICAL_PIPELINE_VERSION = "documents-v1"

# --- schema identifiers (asserted by the gate; see the module docstring) ------

CHUNKS_BM25_INDEX_NAME = "chunks_bm25_idx"
CHUNKS_UNIQUE_CONSTRAINT_NAME = "uq_chunks_document_version_chunk_index"
