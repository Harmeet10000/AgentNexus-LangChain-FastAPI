# Evaluation Infrastructure

## Overview

This module provides comprehensive evaluation capabilities for the AgentNexus platform, covering:

1. **Retrieval Quality Evaluation**: Golden set-based testing for search and retrieval systems
2. **Ingestion Quality Measurement**: Independent validation of document ingestion states
3. **Deep Research Process Metrics**: Instrumentation for supervisor-researcher loop quality
4. **Signal Ingestion Pipeline**: Automated parsing and categorization of failure signals from support tickets

The infrastructure builds upon the `legal_retrieval_v2` golden set (105 cases, 23 taxonomy types) and supports CI integration for regression detection.

---

## Architecture

```
src/app/ai/evaluation/
├── schema.py              # Core data models (GoldenQuery, GoldenSet, etc.)
├── runner.py              # Evaluation orchestration and execution
├── metrics.py             # Metric computation (precision, recall, coverage)
├── validate.py            # Validation logic for golden sets and results
├── report.py              # Report generation and formatting
├── exceptions.py          # Evaluation-specific exceptions
├── experiment.py          # Experimentation framework
└── README.md              # This file

evals/
├── golden/                # Golden set data (.jsonl files)
├── corpus/                # Reference documents (SRC-* corpora)
├── reports/               # Evaluation reports and results
├── benchmarks/            # Performance benchmarks
├── experiments/           # Experiment tracking
├── ingestion/             # Ingestion-specific golden sets and fixtures
└── ADJUDICATION-RUNBOOK.md # Guide for resolving ambiguous evaluation results
```

---

## Core Modules

### 1. Schema Module (`schema.py`)

Defines the core data models for evaluation:

#### Golden Set Models

```python
from app.ai.evaluation.schema import (
    GoldenQuery,              # Individual golden set row
    GoldenSet,                # Collection of golden queries
    GoldenProvenanceManifest, # Corpus provenance metadata
    GoldenEvidencePolicy,     # Policy for evidence inclusion
)
```

**Key Types:**

- `GoldenQuery`: Represents a test case with query, expected documents, case type, and metadata
- `GoldenSet`: Collection of golden queries with versioning and provenance
- `GoldenEvidencePolicy`: Controls evidence inclusion (`"mechanically_verified"`, `"graph_indexed_only"`)

**Golden Query Fields:**

```python
@dataclass
class GoldenQuery:
    case_id: str                           # Unique identifier
    query: str                             # Test query
    case_type: str                         # Taxonomy type (e.g., "partial_coverage_answer")
    expected_document_ids: list[str]       # Expected document IDs
    expected_chunk_ids: list[str] | None   # Expected chunk IDs (optional)
    expected_ingestion_states: list[ExpectedIngestionState] | None
    expected_coverage: int | None          # Expected document coverage count
    expected_coverage_ratio_min: float | None  # Minimum coverage ratio
    policy: GoldenEvidencePolicy           # Evidence inclusion policy
    metadata: dict[str, object] | None     # Additional metadata
```

#### Evaluation Result Models

```python
from app.ai.evaluation.schema import (
    EvaluationResult,      # Result for a single query
    EvaluationReport,      # Aggregated report
    MetricSnapshot,        # Point-in-time metrics
)
```

---

### 2. Runner Module (`runner.py`)

Orchestrates evaluation execution and manages golden set processing.

#### Key Functions

```python
async def run_evaluation(
    *,
    golden_set: GoldenSet,
    retrieval_client: AsyncRetrievalClient,
    policy: GoldenEvidencePolicy = "mechanically_verified",
    batch_size: int = 10,
) -> EvaluationReport:
    """Run evaluation against a golden set."""
    pass
```

**Usage Example:**

```python
from app.ai.evaluation.runner import run_evaluation
from app.ai.evaluation.schema import GoldenSet, load_golden_set

# Load golden set
golden_set = load_golden_set("evals/golden/legal_retrieval_v2.jsonl")

# Run evaluation
report = await run_evaluation(
    golden_set=golden_set,
    retrieval_client=retrieval_client,
    policy="mechanically_verified",
    batch_size=10,
)

print(f"Precision: {report.precision:.2%}")
print(f"Recall: {report.recall:.2%}")
```

---

### 3. Metrics Module (`metrics.py`)

Computes evaluation metrics from results.

#### Key Metrics

```python
from app.ai.evaluation.metrics import (
    compute_precision,      # Precision: relevant retrieved / total retrieved
    compute_recall,         # Recall: relevant retrieved / total relevant
    compute_f1_score,       # F1: harmonic mean of precision and recall
    compute_coverage,       # Coverage: documents ingested / expected
)
```

**Supported Metrics:**

- **Precision**: `relevant_retrieved / total_retrieved`
- **Recall**: `relevant_retrieved / total_relevant`
- **F1 Score**: `2 * (precision * recall) / (precision + recall)`
- **Coverage Ratio**: `documents_ingested / documents_expected`
- **Brief-to-Report Coverage**: `retrieved_docs / report_word_count` (deep research)
- **Follow-Up Depth**: `unique_follow_up_queries / primary_queries` (deep research)
- **Breadth vs Depth**: `unique_docs_per_iteration` (deep research)

---

### 4. Validate Module (`validate.py`)

Validates golden sets, provenance, and evaluation results.

#### Key Functions

```python
from app.ai.evaluation.validate import (
    validate_golden_set_schema,     # Schema validation
    validate_provenance,            # Provenance and manifest validation
    validate_ingestion_states,      # Ingestion state validation
)
```

**Validation Types:**

1. **Schema Validation**: Ensures golden set rows conform to required fields
2. **Provenance Validation**: Checks for `.manifest.json` sibling files and schema versions
3. **Ingestion State Validation**: Verifies document ingestion states match expectations

---

### 5. Report Module (`report.py`)

Generates evaluation reports in multiple formats.

#### Key Functions

```python
from app.ai.evaluation.report import (
    generate_report,           # Generate evaluation report
    format_report_json,        # JSON format
    format_report_markdown,    # Markdown format
)
```

**Report Formats:**

- **JSON**: Machine-readable format for CI integration
- **Markdown**: Human-readable format for documentation
- **HTML**: Interactive format with charts and tables (future)

---

### 6. Exceptions Module (`exceptions.py`)

Evaluation-specific exceptions.

```python
from app.ai.evaluation.exceptions import (
    EvaluationError,              # Base exception
    GoldenSetValidationError,     # Golden set validation failure
    ProvenanceValidationError,    # Provenance validation failure
    IngestionValidationError,     # Ingestion state validation failure
    MetricComputationError,       # Metric computation failure
)
```

---

## New Modules (Iteration 1-3)

### Module 1: Ingestion Evaluation

**Purpose**: Validate document ingestion states independently from retrieval performance.

#### Key Types

```python
from app.ai.evaluation.ingestion import (
    GoldenIngestionState,         # Ingestion state enum
    ExpectedIngestionState,       # Expected state for a document
    IngestionValidationResult,    # Validation result for one document
    IngestionValidationReport,    # Aggregated validation report
)
```

**Ingestion States:**

- `READY`: Document successfully ingested and indexed
- `COMPLETED_WITH_WARNINGS`: Ingested with non-fatal warnings
- `INDEX_NOT_READY`: Ingestion completed but index not yet available
- `FAILED`: Ingestion failed
- `UNAUTHORIZED`: Access denied during ingestion

#### Interface

```python
async def validate_ingestion_states(
    *,
    golden_rows: list[GoldenQuery],
    ingestion_state_provider: Callable[[str], Awaitable[GoldenIngestionState]],
    policy: GoldenEvidencePolicy = "mechanically_verified",
) -> list[IngestionValidationResult]:
    """Validate each row's declared ingestion states against actual state."""
    pass
```

**Usage Example:**

```python
from app.ai.evaluation.ingestion import validate_ingestion_states

# Load golden set
golden_set = load_golden_set("evals/golden/ingestion_v1.jsonl")

# Validate ingestion states
results = await validate_ingestion_states(
    golden_rows=golden_set.queries,
    ingestion_state_provider=get_ingestion_state,
    policy="mechanically_verified",
)

# Check validation results
passed = sum(1 for r in results if r.match)
print(f"Passed: {passed}/{len(results)}")
```

---

### Module 2: Coverage Auditor

**Purpose**: Analyze golden sets for ingestion coverage gaps and generate candidate queries.

#### Key Types

```python
from app.ai.evaluation.coverage import (
    CoverageAuditReport,        # Coverage audit results
    CoverageGap,                # Gap in case type coverage
    IngestionCaseTemplate,      # Template for candidate queries
)
```

#### Interface

```python
class IngestionCoverageAuditor:
    """Analyze golden set coverage and generate candidate queries."""
    
    async def analyze(self, golden_rows: list[GoldenQuery]) -> dict[str, CoverageAuditorResult]:
        """Map rows to ingestion case types and identify gaps."""
        pass
    
    def generate_candidate_queries(
        self,
        gap_report: dict[str, CoverageAuditorResult],
    ) -> list[GoldenQuery]:
        """Generate templated queries for missing case types."""
        pass
```

**Usage Example:**

```python
from app.ai.evaluation.coverage import IngestionCoverageAuditor

auditor = IngestionCoverageAuditor()

# Analyze coverage
analysis = await auditor.analyze(golden_set.queries)

# Identify gaps
gaps = [case_type for case_type, result in analysis.items() if result.rows_present < 5]
print(f"Coverage gaps: {gaps}")

# Generate candidates
candidates = auditor.generate_candidate_queries(analysis)
print(f"Generated {len(candidates)} candidate queries")
```

---

### Module 3: Process Metrics Instrumentation

**Purpose**: Track deep research operation health through node transitions and coverage ratios.

#### Key Types

```python
from app.ai.evaluation.process_metrics import (
    NodeTransition,           # Single node transition record
    ProcessMetrics,           # Computed metrics
    ProcessMetricsTracker,    # Instrumentation tracker
    ProcessMetricSnapshot,    # Point-in-time snapshot
)
```

**Process Metrics:**

- `brief_to_report_coverage_ratio`: `retrieved_docs / report_word_count`
- `follow_up_depth`: `unique_follow_up_queries / primary_queries`
- `breadth_vs_depth_ratio`: `unique_docs_per_iteration`
- `total_node_transitions`: Count of supervisor-researcher transitions
- `total_tool_calls`: Count of tool invocations
- `total_duration_seconds`: Total operation duration

#### Interface

```python
class ProcessMetricsTracker:
    """Instrument supervisor-researcher loop for process metrics."""
    
    def start_tracking(
        self,
        initial_query: str,
        node_limit: int,
    ) -> None:
        """Begin tracking a deep research operation."""
        pass
    
    def record_node_transition(
        self,
        transition: NodeTransition,
    ) -> None:
        """Record a supervisor-researcher node transition."""
        pass
    
    def compute_metrics(self) -> ProcessMetrics:
        """Compute final metrics from tracked transitions."""
        pass
```

**Usage Example:**

```python
from app.ai.evaluation.process_metrics import ProcessMetricsTracker

tracker = ProcessMetricsTracker()

# Start tracking
tracker.start_tracking(initial_query="What are the key contract terms?", node_limit=10)

# Record transitions during operation
tracker.record_node_transition(NodeTransition(
    node_name="supervisor",
    timestamp=datetime.now(),
    input_state={"query": "..."},
    output_state={"decision": "..."},
    tool_calls=["search", "retrieve"],
))

# Compute final metrics
metrics = tracker.compute_metrics()
print(f"Brief-to-report coverage: {metrics.brief_to_report_coverage_ratio:.2f}")
print(f"Follow-up depth: {metrics.follow_up_depth:.2f}")
```

---

### Module 4: Spot-Check Harness

**Purpose**: Lightweight quality validation for deep research outputs without full SME review.

#### Key Types

```python
from app.ai.evaluation.spot_check import (
    SpotCheckResult,           # Single check result
    ClaimConsistencyResult,    # Claim consistency analysis
    ClaimPairAnalysis,         # Pairwise claim comparison
)
```

**Spot-Check Rules:**

1. **Source Attribution**: Report sections must reference at least one chunk ID
2. **Topic Coverage**: Claimed topics must appear in at least one retrieved document
3. **Missing Citations**: Flag claims without source attribution
4. **Citation Density**: Trigger alert if >3 consecutive paragraphs without citation
5. **Claim Consistency**: Compute score as fraction of claim pairs with shared evidence

#### Interface

```python
async def run_spot_checks(
    *,
    report_sections: list[ReportSection],
    retrieved_chunks: list[Chunk],
    index_client: AsyncIndexClient,
) -> list[SpotCheckResult]:
    """Run lightweight quality checks on deep research output."""
    pass

async def compute_claim_consistency(
    *,
    claims: list[Claim],
    retrieved_chunks: list[Chunk],
) -> ClaimConsistencyResult:
    """Compute consistency score across claims."""
    pass
```

**Usage Example:**

```python
from app.ai.evaluation.spot_check import run_spot_checks

# Run spot checks
results = await run_spot_checks(
    report_sections=report.sections,
    retrieved_chunks=retrieved_chunks,
    index_client=index_client,
)

# Check for issues
errors = [r for r in results if r.severity == "error"]
if errors:
    print(f"Found {len(errors)} quality issues")
    for error in errors:
        print(f"  - {error.check_name}: {error.issue_description}")
```

---

### Module 5: Signal Ingestion Pipeline

**Purpose**: Parse support tickets, extract failure patterns, cluster similar failures, and generate taxonomy candidates.

#### Key Types

```python
from app.ai.evaluation.signal_ingestion import (
    SupportTicket,             # Parsed support ticket
    GoldenQueryCandidate,      # Candidate golden query from tickets
    TaxonomyCandidate,         # Taxonomy candidate from clusters
    TicketParser,              # Parser protocol
    TicketClusterer,           # Clustering engine
    TaxonomyCandidateGenerator,# Candidate generator
)
```

#### Interface

```python
class TicketParser(Protocol):
    """Parse support tickets from various ticketing systems."""
    
    async def parse(self, raw_ticket: object) -> SupportTicket:
        """Parse a raw ticket into standardized format."""
        pass

class TicketClusterer:
    """Cluster similar failure tickets and identify patterns."""
    
    def cluster(self, tickets: list[SupportTicket]) -> dict[str, list[SupportTicket]]:
        """Group tickets by similarity."""
        pass
    
    def compute_cluster_centroid(
        self,
        ticket_group: list[SupportTicket],
    ) -> dict[str, object]:
        """Compute cluster centroid for similarity scoring."""
        pass

class TaxonomyCandidateGenerator:
    """Convert ticket clusters to golden set candidates."""
    
    def generate_candidates(
        self,
        cluster: dict[str, list[SupportTicket]],
        threshold: int = 5,
        window_days: int = 30,
    ) -> list[TaxonomyCandidate]:
        """Generate candidates for clusters exceeding failure threshold."""
        pass
```

**Usage Example:**

```python
from app.ai.evaluation.signal_ingestion import (
    TicketParser,
    TicketClusterer,
    TaxonomyCandidateGenerator,
)

# Parse tickets
parser = TicketParser()
tickets = [await parser.parse(raw) for raw in raw_tickets]

# Cluster similar failures
clusterer = TicketClusterer()
clusters = clusterer.cluster(tickets)

# Generate candidates
generator = TaxonomyCandidateGenerator()
candidates = generator.generate_candidates(clusters, threshold=5, window_days=30)

print(f"Generated {len(candidates)} taxonomy candidates from {len(clusters)} clusters")
```

---

### Module 6: Taxonomy Candidate Feeding

**Purpose**: Convert approved taxonomy candidates into golden set rows with A/B testing support.

#### Interface

```python
class TaxonomyCandidateFeeder:
    """Convert taxonomy candidates to golden set rows."""
    
    def to_golden_row(
        self,
        candidate: TaxonomyCandidate,
        source_golden_row: GoldenQuery | None = None,
    ) -> GoldenQuery:
        """Convert candidate to golden set row."""
        pass
    
    def generate_pair_for_ab_test(
        self,
        base_row: GoldenQuery,
        variable_name: str,
        treatment_value: str,
    ) -> tuple[GoldenQuery, GoldenQuery]:
        """Generate control-treatment pair for A/B testing."""
        pass
```

**Usage Example:**

```python
from app.ai.evaluation.taxonomy_feeding import TaxonomyCandidateFeeder

feeder = TaxonomyCandidateFeeder()

# Convert candidate to golden row
golden_row = feeder.to_golden_row(
    candidate=taxonomy_candidate,
    source_golden_row=existing_row,  # Optional
)

# Generate A/B test pair
control, treatment = feeder.generate_pair_for_ab_test(
    base_row=golden_row,
    variable_name="query_formulation",
    treatment_value="What are the key contract terms?",
)

print(f"Control: {control.query}")
print(f"Treatment: {treatment.query}")
```

---

## CI Integration

### Golden Set Validation

The evaluation infrastructure includes CI gates to catch regressions before merging.

#### Provenance Validation

```bash
# Validate golden set provenance
uv run python -m app.ai.evaluation.validate_provenance \
  --golden-set evals/golden/ingestion_v1.jsonl \
  --manifest evals/golden/ingestion_v1.manifest.json
```

**Requirements:**

- Golden set must have sibling `.manifest.json` file
- Manifest must include `golden_provenance_v1` schema version
- All referenced documents must exist in corpus

#### Coverage Thresholds

```bash
# Audit golden set coverage
uv run python -m app.ai.evaluation.audit_coverage \
  --golden-set evals/golden/legal_retrieval_v2.jsonl \
  --minimum-per-type 5 \
  --report evals/reports/coverage_audit.json
```

**Thresholds:**

- Minimum 20 rows for `ingestion_v1` golden set
- At least 5 rows per major ingestion case type
- Coverage ratio > 0.95 for mechanically verified policy

#### Quality Metrics

```bash
# Run ingestion validation
uv run python -m app.ai.evaluation.validate_ingestion \
  --golden-set evals/golden/ingestion_v1.jsonl \
  --manifest evals/golden/ingestion_v1.manifest.json \
  --report evals/reports/ingestion_validation.json
```

**Quality Gates:**

- Ingestion state validation pass rate > 95%
- Spot-check harness error count < 3
- Process metrics within acceptable ranges

---

## Usage Patterns

### Basic Evaluation Workflow

```python
from app.ai.evaluation import (
    load_golden_set,
    run_evaluation,
    generate_report,
)

# 1. Load golden set
golden_set = load_golden_set("evals/golden/legal_retrieval_v2.jsonl")

# 2. Run evaluation
report = await run_evaluation(
    golden_set=golden_set,
    retrieval_client=retrieval_client,
    policy="mechanically_verified",
)

# 3. Generate report
markdown_report = generate_report(report, format="markdown")
print(markdown_report)
```

### Ingestion Quality Audit

```python
from app.ai.evaluation.ingestion import validate_ingestion_states
from app.ai.evaluation.coverage import IngestionCoverageAuditor

# 1. Load golden set
golden_set = load_golden_set("evals/golden/ingestion_v1.jsonl")

# 2. Validate ingestion states
validation_results = await validate_ingestion_states(
    golden_rows=golden_set.queries,
    ingestion_state_provider=get_ingestion_state,
)

# 3. Audit coverage
auditor = IngestionCoverageAuditor()
coverage_analysis = await auditor.analyze(golden_set.queries)

# 4. Identify gaps
gaps = [case for case, result in coverage_analysis.items() if result.rows_present < 5]
print(f"Coverage gaps: {gaps}")
```

### Deep Research Process Metrics

```python
from app.ai.evaluation.process_metrics import ProcessMetricsTracker
from app.ai.evaluation.spot_check import run_spot_checks

# 1. Start tracking
tracker = ProcessMetricsTracker()
tracker.start_tracking(initial_query=query, node_limit=10)

# 2. Run deep research (with instrumentation)
async for transition in supervisor_researcher_loop():
    tracker.record_node_transition(transition)

# 3. Compute metrics
metrics = tracker.compute_metrics()

# 4. Run spot checks
spot_check_results = await run_spot_checks(
    report_sections=report.sections,
    retrieved_chunks=chunks,
    index_client=index_client,
)

# 5. Report
print(f"Process metrics: {metrics}")
print(f"Spot check issues: {len([r for r in spot_check_results if r.severity == 'error'])}")
```

### Signal Ingestion Pipeline

```python
from app.ai.evaluation.signal_ingestion import (
    TicketParser,
    TicketClusterer,
    TaxonomyCandidateGenerator,
)
from app.ai.evaluation.taxonomy_feeding import TaxonomyCandidateFeeder

# 1. Parse tickets
parser = TicketParser()
tickets = [await parser.parse(raw) for raw in raw_tickets if raw["issue_category"] == "retrieval_failure"]

# 2. Cluster similar failures
clusterer = TicketClusterer()
clusters = clusterer.cluster(tickets)

# 3. Generate taxonomy candidates
generator = TaxonomyCandidateGenerator()
candidates = generator.generate_candidates(clusters, threshold=5, window_days=30)

# 4. Convert to golden rows (for approved candidates)
feeder = TaxonomyCandidateFeeder()
golden_rows = [feeder.to_golden_row(c) for c in candidates if c.approval_status == "approved"]

# 5. Add to golden set
golden_set.queries.extend(golden_rows)
```

---

## Golden Set Structure

### `legal_retrieval_v2` Golden Set

**Location**: `evals/golden/legal_retrieval_v2.jsonl`

**Statistics:**
- 105 cases
- 23 taxonomy types
- Coverage: retrieval quality, evidence selection, abstention behavior

**Sample Row:**

```json
{
  "case_id": "lr_v2_001",
  "query": "What are the termination clauses in the service agreement?",
  "case_type": "partial_coverage_answer",
  "expected_document_ids": ["SRC-12345", "SRC-67890"],
  "expected_chunk_ids": ["chunk_abc", "chunk_def"],
  "expected_coverage": 2,
  "expected_coverage_ratio_min": 0.8,
  "policy": "mechanically_verified",
  "metadata": {
    "difficulty": "medium",
    "corpus": "legal_contracts"
  }
}
```

### `ingestion_v1` Golden Set

**Location**: `evals/golden/ingestion_v1.jsonl`

**Statistics:**
- Minimum 20 rows
- Coverage: ingestion quality, document processing, state validation

**Sample Row:**

```json
{
  "case_id": "ing_v1_001",
  "query": "Document ingestion state validation",
  "case_type": "index_not_ready",
  "expected_document_ids": ["SRC-11111"],
  "expected_ingestion_states": [
    {
      "document_id": "SRC-11111",
      "state": "index_not_ready"
    }
  ],
  "policy": "mechanically_verified",
  "metadata": {
    "ingestion_test": true
  }
}
```

---

## Error Handling

### Exception Hierarchy

```
EvaluationError (Base)
├── GoldenSetValidationError
│   ├── SchemaValidationError
│   └── ProvenanceValidationError
├── IngestionValidationError
│   ├── MissingProvenanceError
│   ├── StateMismatchError
│   └── PolicyValidationError
├── ProcessMetricsError
│   ├── MissingTransitionError
│   └── MetricComputationError
└── SignalIngestionError
    ├── TicketParseError
    └── ClusterNotFoundError
```

### Error Handling Patterns

```python
from app.ai.evaluation.exceptions import (
    GoldenSetValidationError,
    IngestionValidationError,
    ProcessMetricsError,
)

try:
    report = await run_evaluation(golden_set, retrieval_client)
except GoldenSetValidationError as e:
    logger.error(f"Golden set validation failed: {e}")
    # Handle validation error
except IngestionValidationError as e:
    logger.error(f"Ingestion validation failed: {e}")
    # Handle ingestion error
except ProcessMetricsError as e:
    logger.error(f"Process metrics computation failed: {e}")
    # Handle metrics error
```

---

## Testing Strategy

### Unit Tests

Located in `tests/evals/`:

- `test_schema.py`: Schema validation and serialization
- `test_runner.py`: Evaluation orchestration
- `test_metrics.py`: Metric computation
- `test_validate.py`: Validation logic
- `test_ingestion.py`: Ingestion validation
- `test_coverage.py`: Coverage auditor
- `test_process_metrics.py`: Process metrics tracker
- `test_spot_check.py`: Spot-check harness
- `test_signal_ingestion.py`: Signal ingestion pipeline

### Integration Tests

Located in `tests/evals/integration/`:

- `test_golden_set_pipeline.py`: Full ingestion validation from golden set
- `test_deep_research_flow.py`: Supervisor-researcher instrumentation
- `test_ticket_to_golden_pipeline.py`: Ticket parsing to golden row conversion

### Property-Based Tests

Located in `tests/evals/property/`:

- Coverage ratio invariants (bounded [0, 1])
- Process metrics determinism
- Ingestion state validation correctness

---

## Iteration Plan

### Iteration 1: Ingestion & Retrieval Golden Sets (CI-Critical)
**Timeline**: 2-3 weeks

**Deliverables:**
- Coverage audit of `legal_retrieval_v2` for ingestion gaps
- `ingestion_v1` golden set with minimum 20 rows
- CI gates for ingestion quality validation
- Report on missing coverage areas

**Modules:**
- Module 1: Ingestion Evaluation
- Module 2: Coverage Auditor
- Module 6: Taxonomy Candidate Feeding (partial)

### Iteration 2: Deep Research Process Metrics (Post-MVP)
**Timeline**: 3-4 weeks

**Deliverables:**
- Process metrics instrumentation in supervisor-researcher loop
- Spot-check harness with 5 core checks
- Lightweight CI gate for deep research quality
- Sample spot-check report for 10 deep research operations

**Modules:**
- Module 3: Process Metrics Instrumentation
- Module 4: Spot-Check Harness

### Iteration 3: Signal Ingestion Pipeline (Strategic)
**Timeline**: 4-6 weeks

**Deliverables:**
- Support ticket parser for 3 major ticketing systems
- Clustering engine for failure pattern detection
- Taxonomy candidate generator
- Automated golden set updates from accepted candidates

**Modules:**
- Module 5: Signal Ingestion Pipeline
- Module 6: Taxonomy Candidate Feeding (full)

---

## References

### Internal Documentation

- **Adjudication Runbook**: `evals/ADJUDICATION-RUNBOOK.md`
- **Golden Set Schema**: `src/app/ai/evaluation/schema.py`
- **Evaluation Runner**: `src/app/ai/evaluation/runner.py`
- **Metrics Module**: `src/app/ai/evaluation/metrics.py`

### Golden Sets and Corpora

- **legal_retrieval_v2**: `evals/golden/legal_retrieval_v2.jsonl`
- **ingestion_v1**: `evals/golden/ingestion_v1.jsonl`
- **Corpus Manifest**: `evals/golden/*.manifest.json`
- **Reference Documents**: `evals/corpus/`

### Reports and Benchmarks

- **Evaluation Reports**: `evals/reports/`
- **Performance Benchmarks**: `evals/benchmarks/`
- **Experiment Tracking**: `evals/experiments/`

---

## Contributing

### Adding New Case Types

1. Define case type in taxonomy (kebab-case convention)
2. Add at least 5 golden rows covering the case type
3. Document expected behavior in `ADJUDICATION-RUNBOOK.md`
4. Add validation logic if needed
5. Update CI gates to include new case type

### Adding New Metrics

1. Define metric formula in documentation
2. Implement computation in `metrics.py`
3. Add unit tests with edge cases
4. Add property-based tests for invariants
5. Update report generation to include new metric

### Extending Golden Sets

1. Validate provenance schema version
2. Ensure sibling `.manifest.json` exists
3. Run coverage audit to identify gaps
4. Add rows for missing case types
5. Run CI validation pipeline
6. Update documentation

---

## Troubleshooting

### Common Issues

#### Golden Set Validation Failures

**Symptom**: `ProvenanceValidationError: Missing .manifest.json`

**Solution**:
1. Ensure `.manifest.json` sibling file exists
2. Validate schema version is `golden_provenance_v1`
3. Check all referenced documents exist in corpus

#### Ingestion State Mismatches

**Symptom**: `StateMismatchError: expected ready, got index_not_ready`

**Solution**:
1. Check ingestion pipeline logs for document
2. Verify corpus manifest reflects actual state
3. Allow time for index to catch up (eventual consistency)
4. Re-run validation after delay

#### Process Metrics Computation Errors

**Symptom**: `MetricComputationError: Division by zero in ratio computation`

**Solution**:
1. Check for empty node transition lists
2. Verify all required fields are populated
3. Add zero-division guards in metric computation

#### Signal Ingestion Clustering Issues

**Symptom**: `ClusterNotFoundError: No clusters exceed threshold`

**Solution**:
1. Lower similarity threshold (try 0.75 instead of 0.8)
2. Increase window duration (try 45 days instead of 30)
3. Verify tickets have sufficient symptom description text

---

## Glossary

- **Golden Set**: Curated collection of human-annotated queries with expected outcomes
- **Case Type**: Classification for golden set queries based on retrieval challenge
- **Coverage Ratio**: Proportion of expected documents successfully ingested/indexed
- **Process Metrics**: Measurements of system behavior during deep research operations
- **Signal Ingestion**: Automated parsing and categorization of external failure signals
- **Provenance**: Metadata tracking origin, version, and lineage of golden set data
- **Spot Check**: Lightweight quality validation without full SME review
- **Taxonomy Candidate**: Proposed golden set row derived from clustered failure signals

---

## Version History

- **v1.0** (Current): Initial documentation covering existing evaluation infrastructure
- **v1.1** (Planned): Iteration 1 deliverables (ingestion evaluation, coverage auditor)
- **v1.2** (Planned): Iteration 2 deliverables (process metrics, spot-check harness)
- **v1.3** (Planned): Iteration 3 deliverables (signal ingestion pipeline, taxonomy feeding)

---

## Contact

For questions or issues with the evaluation infrastructure:

- **Slack**: `#evaluation-infrastructure`
- **Email**: `platform-eng@example.com`
- **Documentation**: This README and `evals/ADJUDICATION-RUNBOOK.md`
