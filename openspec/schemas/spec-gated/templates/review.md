# Review: <change>

<!-- Reviewer stance: act as a reviewer, NOT the author. Read proposal.md, specs/*, design.md, contracts/ first. S change: short review (verdict + checks that matter). L: full checklist. Never inflate an S review - matching risk is the point. -->

**Class:** S | M | L

## Completeness
- [ ] Every requirement has ≥1 scenario (####, WHEN/THEN)
- [ ] Edge and error cases named, not just happy path
- [ ] Quality attributes quantified: Scale | Meter | MUST — no bare adjectives
- [ ] Decision tables (if any) cover every condition combination, no contradictory rows

## Correctness
- [ ] Requirements observably testable (SHALL/MUST, no implementation details baked in)
- [ ] Delta operations right against current main specs (ADDED/MODIFIED full-block/REMOVED with Reason+Migration/RENAMED FROM:/TO:)
- [ ] `openspec validate <change> --type change --strict` passes (run it — don't assume)

## Standards
- [ ] Layering fits: thin routers, service-layer orchestration, persistence-only repos; no HTTP concerns in repos
- [ ] Client state via app.state + lifespan; deps via Annotated Depends; async-only I/O
- [ ] Secrets behind SecretStr (.get_secret_value, never str()/repr)
- [ ] Tasks run real gates: `uv run ruff format src/ tests/ && uv run ruff check src/ tests/`, `uv run ty check src/`
- [ ] Coherent with the decisions recorded in this change's `adr.md`, including any explicitly referenced decisions from earlier changes

## Risk
- [ ] Security / performance / data integrity / breaking changes + migration surfaced before code

## Contracts
- [ ] Every discovery domain event implemented has an asyncapi.yaml channel tracing to a spec scenario
- [ ] Payloads carry stable identity fields; breaking payload changes declare subscriber migration
- [ ] (skipped only if contracts holds the one-line no-async note)

## VERDICT
<!-- Exactly one: APPROVED | CHANGES-REQUESTED (numbered must-fix list) | INFO (concern, no blocker) -->
<!-- CHANGES-REQUESTED blocks tasks.md until items are fixed in the artifacts. -->
