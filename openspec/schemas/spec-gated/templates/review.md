> Change class: **S** (short review) · **M** (normal checklist) · **L** (full checklist + verification matrix).
> Role: reviewer, not author. Read proposal.md, specs/, design.md before completing anything.

## Completeness

<!-- - [ ] Each spec requirement has at least one scenario
     - [ ] Edge and error cases are named, not just the happy path
     - [ ] Delta operations (ADDED/MODIFIED/REMOVED/RENAMED) are correct against the current main spec -->

## Correctness

<!-- - [ ] Requirements are observably testable: one SHALL/MUST per statement, no implementation details
     - [ ] Scenarios are concrete (WHEN/THEN), not restatements of the requirement
     - [ ] skip_specs: true used only when behavior genuinely does not change -->

## Standards

<!-- - [ ] Artifacts follow .opencode/instructions/ (RESULT-PATTERN, EXCEPTION-RULES, PYTHON-TYPING-RULES, ARCHITECTURE-RULES)
     - [ ] Secrets via SecretStr + .get_secret_value(); envelope via APIResponse + http_error()
     - [ ] No match/case on Success/Failure; no isinstance on already-typed data -->

## Risk

<!-- - [ ] Security, performance, data integrity, breaking/migration concerns surfaced
     - [ ] Durable decisions that belong in adrs.md are flagged -->

## Verdict

**VERDICT:** `APPROVED` | `CHANGES-REQUESTED` | `INFO`

<!-- CHANGES-REQUESTED: numbered list of must-fix items to address before tasks/apply.
     INFO: no blocker, but note the concern. -->