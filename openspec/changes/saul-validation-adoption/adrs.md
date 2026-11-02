# ADRs — saul-validation-adoption

## ADR-1: Adoption is type-level; the gates do not move

**Decision:** mechanical predicates and findings adopt the shared shapes; gate logic stays in place, with the existing suite passing unmodified as the acceptance gate.
**Alternatives:** rewriting the gates "properly" onto the pipeline (rejected: rewrites void the unmodified-suite guarantee and risk the most security-sensitive path in the application).
**Consequence:** zero behaviour change, provable; the vocabulary unifies anyway.

## ADR-2: The review node is reached through the sink, never around it

**Decision:** `HumanReviewSink` is a thin adapter over the existing human-review node; every Saul-side ESCALATE arrives through it.
**Alternatives:** letting seams call the node directly (rejected: that is the fourth escalation mechanism this layer exists to prevent).
**Consequence:** one routing story; the interrupt-through-sink test makes drift loud.

## ADR-3: Transport retry and semantic validation stay on opposite sides of a tested line

**Decision:** tool-calling seams keep `ModelRetryMiddleware` / `ToolRetryMiddleware` for transport failures; the validation layer owns semantic gates over parsed values; an assertion test pins the middleware stack.
**Alternatives:** folding transport retry into the pipeline (rejected: double-counted attempts, blurred ownership — reference 14's boundary).
**Consequence:** each mechanism owns its failure class; the boundary survives refactors because it is tested.

## ADR-4: Sequencing over typing twice

**Decision:** this change lands after the qna grade-loop refactor (`mount-retrieval-graph` + seam adoption), not before.
**Alternatives:** adopting early (rejected: the qna seam would be typed once inline and again post-cutover).
**Consequence:** each seam is typed exactly once, in its final shape.
