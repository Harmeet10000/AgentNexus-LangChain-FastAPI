# Review: agent-message-standard

## Completeness

* All 13 requirements (5 + 4 + 3... see below) carry at least one scenario; edge cases named:
  retry duplication, budget-trim orphans, migration-period dual validity.
  Count correction: 5 + 4 + 3 = 12 requirements. Each is discrete (one SHALL family each).
* Cross-spec uniformity scenarios reference the shared constructor normatively — acceptable
  here because the constructor requirement itself lives in `agent-saul-delegation`.

## Correctness

* Requirements are observable (transcript contents, linkage ids, trim outcomes) and testable.
* All deltas are ADDED against a tree with no prior delegation standard — correct operation.
* No implementation details baked in (no function/file names in requirements; constructor
  referenced by capability, not module path).

## Standards

* No secrets, no HTTP envelope, no `Result` handling in scope — RESULT-PATTERN/EXCEPTION-RULES
  vacuous except: pair construction is pure sync code (no I/O), node appends go through normal
  state updates (no transport concerns in nodes beyond existing patterns).
* PYTHON-TYPING-RULES: constructor returns a precise tuple type; frozen models for any new
  DTOs; `add_note` applies to the boundary extraction only if new except blocks appear.
* BREAKING marking present and migration story explicit (dual-write, deletion-rollback).

## Risk

* Transcript growth: mitigated by pair-aware summarization (spec'd).
* `tool_call_id` collision: mitigated by fresh ids per invocation (spec'd).
* Test churn: expected and marked BREAKING; tasks update tests per graph.
* No security/data-integrity surface: transcripts already persisted; pairs add no new PII
  channel beyond existing message content.

VERDICT: APPROVED
