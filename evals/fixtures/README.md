# Evaluation Infrastructure Test Fixtures

This directory contains test fixtures for the evaluation infrastructure extension, organized into three categories: golden sets, support tickets, and deep research outputs.

## Directory Structure

```
evals/fixtures/
├── __init__.py                          # Fixture loading utilities
├── README.md                            # This file
├── golden_sets/                         # Sample golden set data
│   ├── ingestion_basic.jsonl           # Basic ingestion test cases (5 rows)
│   └── coverage_gap_examples.jsonl     # Examples for coverage gap testing (3 rows)
├── tickets/                             # Sample support tickets
│   ├── retrieval_failures.json         # Diverse retrieval failure patterns (10 tickets)
│   └── clustered_failures.json         # Similar failures for clustering tests (6 tickets)
└── research_outputs/                    # Sample deep research reports
    ├── complete_report.json            # High-quality report with citations
    ├── missing_citations_report.json   # Report with citation gaps
    ├── low_quality_report.json         # Shallow report without evidence
    └── contradictory_claims_report.json # Report with contradictory claims

```

## Golden Sets

### `ingestion_basic.jsonl`

Contains 5 golden set rows covering core ingestion case types:

1. **ing_001**: `direct_answer` - Document ready, successful retrieval
2. **ing_002**: `no_evidence_vs_ingestion_failure` - Document failed ingestion
3. **ing_003**: `index_not_ready` - Document not yet indexed
4. **ing_004**: `partial_coverage_answer` - Multi-document with partial ingestion
5. **ing_005**: `no_evidence_vs_ingestion_failure` - Unauthorized document access

**Use Cases**:
- Testing ingestion state validation
- Verifying policy enforcement (mechanically_verified vs graph_indexed_only)
- Testing coverage ratio calculations

### `coverage_gap_examples.jsonl`

Contains 3 golden set rows representing different quality tiers:

1. **gap_001**: `direct_answer` - Candidate tier
2. **gap_002**: `multi_hop` - Candidate tier
3. **gap_003**: `wrong_date_version_authority` - Candidate tier with stale version

**Use Cases**:
- Testing coverage audit gap detection
- Testing candidate query generation
- Testing quality tier handling

## Support Tickets

### `retrieval_failures.json`

Contains 10 support tickets with diverse failure patterns:

- Indexing lag (TICKET-001, TICKET-004)
- OCR quality issues (TICKET-002)
- Semantic chunking problems (TICKET-003)
- Permission sync issues (TICKET-005)
- Stale version retrieval (TICKET-006)
- Wrong jurisdiction filtering (TICKET-007)
- Table extraction failures (TICKET-008)
- Annexure/appendix linking (TICKET-009)
- Footnote misclassification (TICKET-010)

**Use Cases**:
- Testing ticket parsing from different systems
- Testing failure pattern categorization
- Testing taxonomy candidate generation

### `clustered_failures.json`

Contains 6 support tickets with similar failure patterns (indexing lag for indemnification clauses):

- All tickets relate to indemnification/indemnity clause retrieval failures
- All resolved with same root cause (indexing lag)
- Spread across 15 days (March 1-15, 2024)
- 6 occurrences > 5 threshold for taxonomy candidate generation

**Use Cases**:
- Testing ticket similarity scoring (80% threshold)
- Testing cluster generation
- Testing failure threshold validation (5 occurrences in 30 days)

## Deep Research Outputs

### `complete_report.json`

High-quality deep research report with:

- 3 well-structured sections
- Comprehensive citations in each section (9 total chunk references)
- 6 unique documents retrieved
- 285-word report from 20 retrieved chunks
- 5 follow-up queries for thorough research
- 8 node transitions

**Use Cases**:
- Baseline for spot-check harness validation
- Testing process metrics computation (all ratios should be healthy)
- Testing claim consistency scoring

### `missing_citations_report.json`

Report with citation quality issues:

- Introduction and recommendations sections have NO citations
- Findings section has only 1 citation
- 150-word report from 8 retrieved chunks (low coverage ratio)
- Only 2 follow-up queries (shallow depth)
- >3 consecutive paragraphs without citations in findings section

**Use Cases**:
- Testing citation density spot check (should trigger alert)
- Testing source attribution validation (should flag missing citations)
- Testing brief-to-report-coverage-ratio (should be low)

### `low_quality_report.json`

Shallow report with minimal evidence:

- NO citations in any section
- Generic content without specific evidence
- 120-word report from only 3 retrieved chunks (very low coverage)
- NO follow-up queries (no research depth)
- Only 1 iteration (superficial research)

**Use Cases**:
- Testing spot-check harness error detection (should fail multiple checks)
- Testing follow-up depth metric (should be 0.0)
- Testing topic coverage validation (claims not backed by evidence)

### `contradictory_claims_report.json`

Report with contradictory warranty claims:

- Section "limited_warranties" claims 90-day warranties with functionality guarantees
- Section "disclaimer" claims products are 'as-is' with all warranties disclaimed
- Both sections cite evidence, but claims contradict each other
- 180-word report with 6 chunk references

**Use Cases**:
- Testing claim consistency scoring (should detect contradiction)
- Testing claim pair analysis with shared evidence
- Testing quality metrics for conflicting information

## Usage Examples

### Loading Fixtures in Tests

```python
from evals.fixtures import (
    load_golden_set_fixture,
    load_ticket_fixture,
    load_research_output_fixture,
)

# Load golden set rows
rows = load_golden_set_fixture("ingestion_basic")
assert len(rows) == 5

# Load support tickets
tickets = load_ticket_fixture("retrieval_failures")
assert len(tickets) == 10

# Load deep research output
output = load_research_output_fixture("complete_report")
assert len(output["report_sections"]) == 3
```

### Using Fixtures in Property Tests

```python
from hypothesis import given, strategies as st
from evals.fixtures import load_golden_set_fixture

# Property: all ingestion states must match expected states
@given(st.sampled_from(load_golden_set_fixture("ingestion_basic")))
def test_ingestion_state_validation(golden_row):
    for expected_state in golden_row["expected_ingestion_states"]:
        actual_state = get_ingestion_state(expected_state["document_id"])
        assert actual_state == expected_state["state"]
```

### Using Fixtures in Integration Tests

```python
from evals.fixtures import load_ticket_fixture, load_golden_set_fixture

def test_signal_ingestion_pipeline_e2e():
    """Test complete pipeline from tickets to golden set rows."""
    tickets = load_ticket_fixture("clustered_failures")
    
    # Parse tickets
    parsed = [parse_ticket(t) for t in tickets]
    
    # Cluster similar failures
    clusters = cluster_tickets(parsed)
    
    # Generate taxonomy candidates (6 tickets > 5 threshold)
    candidates = generate_candidates(clusters, threshold=5)
    assert len(candidates) > 0
    
    # Convert to golden set rows
    golden_rows = [candidate_to_golden_row(c) for c in candidates]
    assert all(row["quality_tier"] == "candidate" for row in golden_rows)
```

## Fixture Maintenance

### Adding New Fixtures

1. **Golden Sets**: Add new `.jsonl` files following the golden set schema in `src/app/ai/evaluation/schema.py`
2. **Tickets**: Add new `.json` files following the `SupportTicket` TypedDict structure
3. **Research Outputs**: Add new `.json` files with required fields: `operation_id`, `initial_query`, `report_sections`, `retrieved_documents`, `process_metrics`

### Schema Validation

All fixtures should validate against their respective schemas:

```python
from pydantic import TypeAdapter
from app.ai.evaluation.schema import GoldenQuery
from evals.fixtures import load_golden_set_fixture

# Validate golden set fixture
rows = load_golden_set_fixture("ingestion_basic")
adapter = TypeAdapter(list[GoldenQuery])
validated_rows = adapter.validate_python(rows)
```

### Fixture Quality Standards

- **Golden Sets**: Must include all required fields per schema version
- **Tickets**: Must include realistic symptom descriptions and resolution notes
- **Research Outputs**: Must include process metrics for metrics computation tests

## References

- **Golden Set Schema**: `src/app/ai/evaluation/schema.py`
- **Requirements**: `.kiro/specs/evaluation-infrastructure-extension/requirements.md`
- **Design**: `.kiro/specs/evaluation-infrastructure-extension/design.md`
- **Tasks**: `.kiro/specs/evaluation-infrastructure-extension/tasks.md`
