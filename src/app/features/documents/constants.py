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
TRIGRAM_SIMILARITY_THRESHOLD = 0.1
DISKANN_QUERY_SEARCH_LIST_SIZE = 100
DISKANN_QUERY_RESCORE = 50

# --- ingestion ---------------------------------------------------------------

INGEST_CHUNK_SIZE = 512
INGEST_CHUNK_OVERLAP = 64
INGEST_EMBEDDING_BATCH_SIZE = 200
ANALYZE_THRESHOLD_CHUNKS = 10_000

# --- schema identifiers (asserted by the gate; see the module docstring) ------

CHUNKS_BM25_INDEX_NAME = "chunks_bm25_idx"
CHUNKS_UNIQUE_CONSTRAINT_NAME = "uq_chunks_document_chunk_index"
