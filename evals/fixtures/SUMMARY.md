# Test Fixtures Setup Summary

## Task Completion: 0.3.1 Set up test fixtures

This document summarizes the test fixtures infrastructure created for the evaluation infrastructure extension.

## Created Structure

```
evals/fixtures/
├── __init__.py                                    # Fixture loading utilities
├── README.md                                      # Comprehensive documentation
├── SUMMARY.md                                     # This file
├── golden_sets/                                   # Sample golden set data
│   ├── ingestion_basic.jsonl                     # 5 rows covering core ingestion cases
│   └── coverage_gap_examples.jsonl               # 3 rows for coverage audit testing
├── tickets/                                       # Sample support tickets
│   ├── retrieval_failures.json                   # 10 diverse failure patterns
│   └── clustered_failures.json                   # 6 similar failures for clustering
└── research_outputs/                              # Sample deep research reports
    ├── complete_report.json                      # High-quality baseline
    ├── missing_citations_report.json             # Citation gaps
    ├── low_quality_report.json                   # Shallow research
    └── contradictory_claims_report.json          # Contradictory claims
```

## Fixtures Coverage

### Golden Sets (8 total rows across 2 files)

**ingestion_basic.jsonl** - 5 rows covering:
1. `direct_answer` with ready document
2. `no_evidence_vs_ingestion_failure` with failed ingestion
3. `index_not_ready` case
4. `partial_coverage_answer` with multi-document partial ingestion
5. Unauthorized document access

**coverage_gap_examples.jsonl** - 3 rows covering:
1. `direct_answer` (candidate tier)
2. `multi_hop` (candidate tier)
3. `wrong_date_version_authority` with stale version

### Support Tickets (16 total across 2 files)

**retrieval_failures.json** - 10 diverse patterns:
- Indexing lag (2 tickets)
- OCR quality issues
- Semantic chunking problems
- Permission sync issues
- Stale version retrieval
- Wrong jurisdiction filtering
- Table extraction failures
- Annexure/appendix linking
- Footnote misclassification

**clustered_failures.json** - 6 similar patterns:
- All related to indemnification clause retrieval
- Same root cause (indexing lag)
- Spread across 15 days
- Exceeds 5-occurrence threshold for taxonomy candidate generation

### Deep Research Outputs (4 files)

**complete_report.json**:
- 3 well-structured sections
- 9 total citations
- 6 unique documents
- 285-word report from 20 chunks
- Healthy metrics (5 follow-ups, 8 transitions)

**missing_citations_report.json**:
- Citation gaps in 2 of 3 sections
- Low coverage ratio (150 words from 8 chunks)
- Only 2 follow-up queries
- >3 consecutive paragraphs without citations

**low_quality_report.json**:
- NO citations in any section
- Generic content
- 120 words from only 3 chunks
- NO follow-up queries
- Only 1 iteration

**contradictory_claims_report.json**:
- Contradictory warranty claims
- "Limited warranties" vs "as-is disclaimer"
- Both sections cite evidence
- 180 words with 6 references

## Fixture Loading Utilities

The `fixtures/__init__.py` module provides three loader functions:

1. **`load_golden_set_fixture(fixture_name: str) -> list[dict[str, Any]]`**
   - Loads `.jsonl` files from `golden_sets/`
   - Returns list of golden set row dictionaries

2. **`load_ticket_fixture(fixture_name: str) -> list[dict[str, Any]]`**
   - Loads `.json` files from `tickets/`
   - Returns list of support ticket dictionaries

3. **`load_research_output_fixture(fixture_name: str) -> dict[str, Any]`**
   - Loads `.json` files from `research_outputs/`
   - Returns deep research output dictionary

All loaders raise `FileNotFoundError` for missing fixtures.

## Validation Tests

Created `test_fixtures_loading.py` with 14 tests organized in 4 classes:

1. **TestGoldenSetFixtures** (3 tests)
   - Load basic ingestion fixture
   - Load coverage gap examples
   - Handle missing fixtures

2. **TestTicketFixtures** (3 tests)
   - Load retrieval failures
   - Load clustered failures
   - Handle missing fixtures

3. **TestResearchOutputFixtures** (5 tests)
   - Load complete report
   - Load missing citations report
   - Load low quality report
   - Load contradictory claims report
   - Handle missing fixtures

4. **TestFixturesDirectory** (3 tests)
   - Verify directory structure
   - Verify subdirectories exist
   - Verify README exists

**All 14 tests pass ✓**

## Code Quality

All code passes:
- ✓ `uv run ruff format evals/fixtures/ tests/evals/`
- ✓ `uv run ruff check evals/fixtures/ tests/evals/`
- ✓ `uv run ty check tests/evals/`
- ✓ `uv run pytest tests/evals/test_fixtures_loading.py -v` (14 passed)

## Use Case Coverage

The fixtures support testing for:

### Iteration 1: Ingestion & Retrieval
- ✓ Ingestion state validation
- ✓ Policy enforcement testing
- ✓ Coverage ratio calculations
- ✓ Coverage audit gap detection
- ✓ Candidate query generation
- ✓ CI gate validation

### Iteration 2: Deep Research Process Metrics
- ✓ Process metrics computation
- ✓ Spot-check harness validation
- ✓ Citation density checks
- ✓ Source attribution validation
- ✓ Topic coverage validation
- ✓ Claim consistency scoring

### Iteration 3: Signal Ingestion Pipeline
- ✓ Ticket parsing testing
- ✓ Similarity scoring (80% threshold)
- ✓ Cluster generation
- ✓ Failure threshold validation (5 in 30 days)
- ✓ Taxonomy candidate generation
- ✓ End-to-end pipeline testing

## Next Steps

With fixtures in place, the following tasks can now proceed:

1. **Task 1.2.3**: Write property tests for ingestion state comparison
2. **Task 1.3.3**: Write property tests for coverage metrics
3. **Task 2.2.3**: Write property tests for ratio metrics
4. **Task 2.3.6**: Write unit tests for spot-check rules
5. **Task 3.3.3**: Write property tests for deduplication
6. **Task 3.4.3**: Write property tests for cluster threshold

All test tasks can now reference these fixtures using the loader utilities.

## References

- **Design Document**: `.kiro/specs/evaluation-infrastructure-extension/design.md`
- **Requirements**: `.kiro/specs/evaluation-infrastructure-extension/requirements.md`
- **Tasks**: `.kiro/specs/evaluation-infrastructure-extension/tasks.md`
- **Fixture Documentation**: `evals/fixtures/README.md`
