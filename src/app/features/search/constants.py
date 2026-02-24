# Pagination
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

# Reciprocal Rank Fusion constant (60 is the industry standard for RRF)
RRF_K = 60

# The number of candidates to fetch from each index before fusing them.
# At scale, you don't score the whole database; you score the top N.
HYBRID_CANDIDATE_LIMIT = 100
