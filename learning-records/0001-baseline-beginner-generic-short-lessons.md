# 0001: Baseline — Beginner, Generic Stack, Short-Lesson Pace

**Date:** 2026-08-24 · **Status:** current

## Context
First session. Learner requested a full testing curriculum: unit / integration / e2e, best practices, anti-patterns, senior habits, test placement, what not to test, and using tests for customer-facing reliability.

## Insight / Decision
Learner self-assessed **beginner**: has written basic asserts; no fluent grasp of mocking, fixtures, or strategy. Chose:
- **Generic/universal examples** (not language-specific) — principles first, project application later
- **Short lessons**, one tightly-scoped topic per session (~10 min)
- Mission spans four goals: customer reliability → senior judgment → career → transfer to real projects

## Implications
- Curriculum ordered bottom-up (why → unit → doubles → integration → e2e → strategy → anti-patterns → production).
- Do not assume pytest/Jest fluency in early lessons; use pseudocode-flavored generic snippets.
- Interleaved retrieval review begins at lesson 4 (once 3+ topics exist).
- Watch for ZPD jump after lesson 6 — learner may be ready for applied lessons on a real codebase sooner than the roadmap assumes.
