> Change class: **M** (normal checklist). Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

- [x] Each spec requirement has at least one scenario (tool-capability-catalog: 4 requirements / 9 scenarios; agent-tool-registry: 2 requirements / 2 scenarios)
- [x] Edge and error cases named: unknown role, unknown capability identifier, tag/metadata disagreement, high-relevance unauthorized tool, no-grant web search, granted web search
- [x] Delta operations correct: new `tool-capability-catalog` capability; `agent-tool-registry` deltas are ADDED requirements consistent with the enforcement change's (both may be open at once — see cross-change interactions)

## Correctness

- [x] Requirements observably testable — policy equality, subset invariant, refusal messages, schema surface; no internal class names in specs
- [x] "Selector shrinks and never widens" is phrased as an invariant over any turn, not as an implementation mechanism
- [x] The egress requirement generalizes (classification-driven) rather than naming one tool, while the scenario pins the concrete case

## Verification matrix

| Requirement | Proof |
|---|---|
| Structured metadata authoritative | registration test asserting tag/metadata agreement; lossless migration test for the five tools |
| One policy source | policy-loading test: context capability set == policy entry; unknown role → empty set; unknown capability id → fail-fast |
| Selector shrink-only | rank-an-unauthorized-tool test asserting absence; subset invariant test over all roles |
| Egress gating | web-search no-grant refusal naming `web.search`; granted-path parity test; schema-clean assertion |

## Decisions to defend in review

1. **Static policy over computed rules** (design Decision 1) — challenge with the first dynamic requirement anyone提出; until then, static + fail-fast is the auditable choice.
2. **Intersection before ranking** (Decision 3) — post-filtering is simpler; confirm the reviewer accepts the telemetry-ordering argument for the stronger form.
3. **Initial policy encodes today's effective grants** (Migration Plan) — the consistency test must pass on day one *before* any narrowing; narrowing is a separate reviewed diff.

## Cross-change interactions

- `agent-tool-capability-enforcement`: consumes its executor contract wholesale; migrates its tags into structured fields; its suite must pass unmodified after migration (delta scenario pins this).
- `deep-research-bounded-subgraph`: the wrapper's context construction reads the policy-populated grant set; no interface change.
- `agent-portfolio-traversal-tools` (D2): its tools register through this catalog; its gating is independent of it.
- `saul-validation-adoption`: unaffected (validation layer is value-level, not tool-level).
