# Teaching Notes (agent scratchpad)

## Learner profile
- Level: **Beginner** at testing (self-assessed, session 1)
- Stack preference: **generic/universal** examples — no language-specific lessons yet
- Pace: short lessons (~10 min), one topic per session
- Motivation: all four goals — reliability for customers, senior judgment, career, apply to AgentNexus

## Working rules
- Every lesson: retrieval quiz with **equal-length answer options** (no formatting clues)
- Every lesson cites RESOURCES.md sources; primary source called out
- Interleave review questions from earlier lessons once 3+ lessons exist
- Glossary terms in lessons link to `reference/glossary.html`
- Community recommendation (Ministry of Testing) surfaces ~lesson 6

## Roadmap state
1. ✅ Why we test
2. Unit tests
3. Test doubles
4. Integration tests
5. E2E tests
6. Pyramid/trophy + community intro
7. What NOT to test
8. Senior anti-patterns
9. Tests for customer-facing products

## Session log
- Session 1 (2026-08-24): mission captured, workspace scaffolded, lesson 1 built.
- Session 2 (2026-08-24): L1 quiz 4/4 perfect → strong acquisition. Built lesson 2 (unit scope/shape, AAA, behavior-not-class). Interleaved 1 review question from L1 in L2 quiz — keep this pattern every lesson. Watch: does learner absorb "behavior not class" or just pattern-match AAA? Probe in conversation before lesson 3.
- Session 3 (2026-08-24): L2 quiz 5/5. Built lesson 3 (doubles taxonomy, stub-vs-mock queries/commands distinction, over-mocking trap, don't-mock-what-you-don't-own). 2 interleaved questions. Probe next session: can learner articulate WHY mocks on internals are brittle? If yes → ready for integration tests + possibly accelerated ZPD toward applied lessons on real codebase.
- Session 4 (2026-08-24): L3 quiz 5/5. Built lesson 4 (integration = real boundaries, DB-backed + API-level shapes, determinism discipline, Testing Trophy framing). Interleaved L3 brittleness question — answered correctly, so the behavior-vs-implementation concept has transferred. ZPD note: learner is pacing ahead of beginner baseline; consider offering applied mini-exercise on real code after lesson 6 instead of lesson 9.
- Session 5 (2026-08-24): L4 quiz 5/5. Built lesson 5 (E2E definition, cost table, ice-cream-cone anti-pattern, critical-journey selection, smoke tests, flaky=broken). All three layers now taught — learner holds full vocabulary for pyramid/trophy synthesis next session. Plan L6 as synthesis + Ministry of Testing community introduction per roadmap.
- Session 6 (2026-08-24): L5 quiz 5/5. Built lesson 6 (pyramid vs trophy, agree/disagree analysis, codebase-shape decision table, five-minute suite audit, Ministry of Testing + r/softwaretesting community intro). Quiz is synthesis-heavy: 4/5 interleaved across L2–L5. Halfway checkpoint — invited learner to state own position for pressure-testing. Next: probe whether they joined/visited a community before L8 (wisdom layer).
- Session 7 (2026-08-24): L6 quiz 5/5. Learner asked for deep explanation of ice-cream-cone diagnosability line — good sign of genuine comprehension-seeking, not just pattern matching (ZPD confirmed ahead of baseline). Explained diagnostic-information principle: failure locality ∝ test scope narrowness. Built lesson 7 (test economics equation, do-not-test table, two senior questions, coverage-as-detector-not-target). Remaining: L8 anti-patterns, L9 production testing. Consider post-L9 applied capstone on real codebase given accelerated pace.
- Session 8 (2026-08-24): L7 quiz 5/5. Built lesson 8 (six anti-patterns: tolerated flakiness/auto-retry, assertion roulette, mystery guest, logic-in-tests, over-DRY test code, erasing the red; unifying frame = trading long-term signal for short-term convenience). Homework suggestion: spot an anti-pattern in a public repo and report back. One lesson remains; prep applied capstone proposal for session 10.
- Session 9 (2026-08-24): L8 quiz 5/5. Built lesson 9 — curriculum complete (CI gates as ordered cost-of-defect table, canaries, feature flags, error budgets, observability/synthetic monitors). Added R6 (Google SRE book) to RESOURCES.md. Wrote learning record 0002: acquisition phase done, next phase = applied capstone on real code (AgentNexus), coaching mode thereafter. Outstanding: homework anti-pattern find + community visit probe.
- Session 10 (2026-08-24): L9 quiz 5/5 — learner asked "what's next". Discovered AgentNexus has real suite (tests/unit, integration, e2e, property, performance dirs). Built lesson 10 = applied capstone: Part 1 five-minute audit of this repo's actual suite, Part 2 read existing conventions (test_circuit_breaker.py, test_health.py, conftest.py), Part 3 write one unit + one integration test with see-it-fail verification. Deliverables: audit verdict + two test files for my code review. Teaching now in coaching mode.
- Session 11 (2026-08-24): Full audit performed FOR the learner → reference/test-suite-audit-agentnexus.html. Key verified findings: E2E layer empty; integration thin (6 files vs 56); sleeps at ws bug_conditions:156/preservation:116/parser_offload:94; except Exception swallow at bug_conditions:225; bare assert True at ws_integration:324; tautology OR-assert saul_persist:75; 104 MagicMock vs 2 spec=; stub-registry drift between two conftests (integration still lists retired langgraph_layer stubs). Overall ≈5.8/10. Audit written in Incident-Why-Outcome HTML-comment framework per user preference (recorded below). Next coaching moves: learner executes P0 fixes as capstone part 3 instead of greenfield tests? Ask.
- PREFERENCE (user, session 11): documentation should use block-level HTML comments in Incident–Why–Outcome format (incident/failure → why rule helps → outcome/evidence) plus commit references where traceable; vague comments banned. Apply to future audit/reference docs in this workspace.
- Session 12 (2026-08-24): User asked for additions to audit → second-pass CI/pipeline deep-dive added as Addendum to test-suite-audit-agentnexus.html. New verified findings: A1 requires_db test (test_schema_orm_matches_database.py) NEVER runs in CI (addopts excludes it; -m integration step skips it); A2 markers sparse (~12/66 files) + dir/marker split-brain; A3 unit suite runs twice per push (3 pytest steps); A4 no pytest-timeout (hang exposure on async ws suite); A5 htmlcov artifact never generated, cache key omits uv.lock, no concurrency group, migrations-before-lint ordering, factory-boy installed but unused; A6 positives (service health checks, alembic-in-CI, hypothesis). L9 grade revised 8→6. Overall ≈5.8→5.5.
