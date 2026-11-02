> Change class: **M**. The proposal covers *why* and *what*; this covers *how*. Reference the proposal — do not restate it.

## Context

The traversal tools are consumers of two things that do not fully exist yet: typed relationship edges (the gate — retired plan Phase D / harden Phase 4) and the handle-bound evidence read path with run-scoping (`deep-research-bounded-subgraph` consumes the same seam). Authoring now means the unblocking work lands against an agreed contract instead of inventing one under pressure.

## Goals / Non-Goals

**Goals:**
- Mission 2 (portfolio interrogation) becomes reachable through three named, bounded tools.
- The cap discipline is testable data, not a convention someone remembers.
- Empty and gated states are honest statuses, never empty successes.

**Non-Goals:**
- Building or unblocking the edges (the gate's own change).
- Narrative portfolio answers (qna's engine owns answering; traversal returns structure).
- Any new write path to the graph.

## Decisions

### Decision 1 — traversal and content are different tools' jobs
Traversal returns identifiers + provenance + handles; `read_evidence_by_id` remains the only content materializer. One read path means one scoping story and one handle contract; duplicating content access in traversal tools would fork both.

### Decision 2 — the gate is a binding-time check, not a runtime condition
Binding a portfolio tool while the corpus lacks relationship edges fails the graph build with an error naming the gate. A runtime "no edges" would ship tools that return empty successes — the exact defect class the plan's P2 exists to prevent. The check is cheap (an edge-existence probe at build) and unambiguous.

### Decision 3 — swap, don't add, is enforced by test not by review
Each role at the cap gains a portfolio tool only alongside a same-role removal, asserted by the toolset-identity test. Which tool each role swaps out is an apply-time decision recorded in that change's design (candidates: for risk, a specialist lookup that the traversal subsumes).

### Decision 4 — three tools, not a generic query tool
A generic "query the graph" tool would push Cypher-shaped thinking into the model and make result contracts unenforceable. Three named tools with typed envelopes keep each result contract checkable and the projection legible.

## Risks / Trade-offs

- **The gate could stay closed a long time** → the cost of authoring now is small; the cost of inventing contracts after the edges land is rework under pressure.
- **Traversal latency on large corpora** → bounded envelopes and the step budget pattern from D5 apply; portfolio queries are structured graph queries, not scans.

## Migration Plan

None until the gate opens. When it does: land the tools behind the binding check, run the projection-consistency suite, apply the swaps, and only then expose to roles.

## Open Questions

- Which exact same-role tool each specialist swaps out (apply-time decision, per Decision 3).

## Addendum (apply-time, gate open — tasks 2.x-3.x)

- **No swap required (task 3.2's rule binds only at the cap).** At apply time
  risk was already at 6 tools (readiness parity D3 appended
  `get_ingestion_state` to both specialists, contra the 5-tool counts in the
  change brief) and compliance at 5. Binding the traversal tools into the
  slots the cap allows — risk gains `cross_document_obligations`, compliance
  gains `find_conflicting_clauses` + `party_obligation_map` — lands both
  projections at exactly the ≤7 cap with no same-role removal. Binding the
  full trio to both roles (risk at 9) would breach the cap and was rejected;
  the swap rule stays armed for any future widening beyond 7.
- **Binding path.** `build_agent_registry` accepts
  `traversal_graphiti_service` + `traversal_gate_probe`; the async caller
  (lifespan wiring) probes the corpus via `relationship_edges_available`
  before the sync builder. A missing probe is treated as a scopeless corpus
  and fails closed; a `no_edges` probe fails the build with
  `TraversalGateError` naming gate `relationship-edges`. The catalog itself
  always carries backend-less instances (registered by
  `register_default_tools`) so the policy's `portfolio.traversal` grants
  name a declared capability at every startup; the factory replaces them
  with backend-bound instances only behind the gate. Lifespan wiring to pass
  the provisioned backend + corpus probe is the remaining integration step.
- **Evidence handles on matches are deterministic clause locators**
  (`portfolio:<document_id>:<clause_id>:<edge_id>`), not `EvidenceHandle`
  tokens: `EvidenceService` issues handles only out of fused retrieval, and
  a traversal match carries clause identity, not chunk identity. Content
  still materializes only through the evidence seam (Decision 1); upgrading
  locators to run-scoped handle issuance needs a small `EvidenceService`
  extension and belongs to the seam's own change. The envelope names the
  field `evidence_locator` (never `evidence_handle`) so no consumer can
  mistake a locator for a resolvable handle token. Review requirement on
  the future extension: a locator must become resolvable through the
  run-scoped evidence read path — the locator's document/clause must
  resolve only within the run's authorized scope, and until that seam
  exists a read attempt must answer `invalid_handle` (never `unauthorized`
  or a partial answer), which the unit suite pins.
- **Runtime gate probe** behind the binding-time check (defense in depth):
  a corpus that lost its typed edges returns a typed `refused` envelope
  naming the gate — never an empty success. Refusals (no runtime context,
  no scope, no backend, closed gate) reuse the deep-research envelope
  convention (`status: "refused"`) and never masquerade as traversal
  outcomes.
