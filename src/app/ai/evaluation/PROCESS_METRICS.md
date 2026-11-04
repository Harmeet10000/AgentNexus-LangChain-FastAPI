# Deep-research process metrics

`ProcessMetricsTracker` records bounded supervisor/researcher node transitions and derives deterministic metrics from their state. Each transition's `output_state` may carry `retrieved_document_ids`, `follow_up_queries`, `primary_queries` or `primary_query_count`, and `report_word_count`.

## Formulas

Let `D` be the unique retrieved document IDs across transitions, `W` the final report word count, `F` the unique follow-up query strings, `P` the primary query count, and `Dᵢ` the set of document IDs retrieved in transition/iteration `i`:

- **brief_to_report_coverage_ratio** = `|D| / W`.
- **follow_up_depth** = `|F| / P`.
- **breadth_vs_depth_ratio** = mean `|Dᵢ|` across recorded transitions (the implemented ratio divides this breadth by one).
- **NodeTransition** stores `node_name`, timestamp, input/output state, and tool-call names. **ProcessMetricsTracker** enforces the configured node limit, preserves transition order, counts tool calls, and derives elapsed time as `max(timestamp) - min(timestamp)` (zero for fewer than two transitions or incompatible timestamps). This max-minus-min rule makes duration independent of transition arrival order.

All ratios guard zero denominators and return `0.0`; resulting ratios and durations are required to be finite and non-negative. A zero or missing report word count/query count therefore never produces NaN or infinity. When no primary-query count is declared, unique `primary_queries` are counted, falling back to one.

## Instrumentation

Create a tracker at the start of one bounded deep-research run with its initial query and configured node limit. Record one `NodeTransition` for each completed subgraph node, including available retrieval/query details in `output_state` and tool names in `tool_calls`. Compute metrics after the run. `DeepResearchInstrumentation` provides the integration seam for wrapping the bounded deep-research subgraph; keep instrumentation observational and pass the subgraph's configured node bound through to the tracker.

```python
from datetime import UTC, datetime

from app.ai.evaluation.process_metrics import NodeTransition, ProcessMetricsTracker

tracker = ProcessMetricsTracker()
tracker.start_tracking("Compare the vendor agreements", node_limit=8)
tracker.record_node_transition(
    NodeTransition(
        node_name="researcher",
        timestamp=datetime.now(UTC),
        input_state={},
        output_state={
            "retrieved_document_ids": ["doc-1", "doc-2"],
            "primary_query_count": 1,
            "follow_up_queries": ["termination exceptions"],
            "report_word_count": 200,
        },
        tool_calls=["search"],
    )
)
metrics = tracker.compute_metrics()
```
