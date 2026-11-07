# Composition contract — extracted specs → what actually runs

**This file is the definition.** `javascript/openspec-effective-paths.cjs` and `python/openspec_effective_spec.py` are bindings of it; neither is the definition, and a port in a third language implements what is written here rather than reading either source. When behavior changes, this file changes first.

See [EXTRACTION.md](EXTRACTION.md) for how the `.feature` files it consumes are produced, and the skill's "Port parity" note for the cross-stack obligation.

## The effective spec

The default run executes the **effective spec**: every `openspec/specs/**/spec.md` (source of truth) with every **active** change's delta (`openspec/changes/*/specs/**/spec.md`) applied — i.e. exactly what the source of truth will become once those changes are synced and archived.

Per requirement in an active delta:

| Requirement's section in an active delta | Source-of-truth version | Delta version  |
|------------------------------------------|-------------------------|----------------|
| (not mentioned in any delta)             | runs                    | —              |
| `## ADDED Requirements`                  | —                       | runs           |
| `## MODIFIED Requirements`               | **not discovered**      | runs           |
| `## REMOVED Requirements`                | **not discovered**      | (no scenarios) |
| `## RENAMED Requirements`                | runs (name change only) | (no scenarios) |

Superseded rules (MODIFIED/REMOVED by an active delta) must **not reach the runner** and must **not be reported as skipped**. Never edit or tag the spec files to skip them — the source of truth stays pristine until sync — and never let them show up as skipped counts, which pollute the zero-pending completion signal: a superseded rule is replaced, not unfinished.

## Procedure

1. **Extract** — every `spec.md` under `openspec/specs/` and `openspec/changes/*/specs/` (archive excluded) becomes `.extracted/<same-path>/spec.feature` with identical line numbers. See [EXTRACTION.md](EXTRACTION.md).
2. **Collect active deltas** — every `.extracted/changes/*/specs/**/*.feature`, with a defensive `changes/archive/` filter.
3. **Extract superseded rules** — scan each extracted delta for `# @openspec: <OP>` comments (which extraction emitted from the `## <OP> Requirements` headings) and record `(capability, rule name)` for MODIFIED and REMOVED.

   **An operation applies to every `Rule:` until the next marker**, because it came from a section heading covering all the requirements beneath it. Do not reset it after the first rule, or only the first requirement of each section will supersede — and the rest will run twice, once from the source of truth and once from the delta.

   The capability is the path segment after the delta's last `specs/`. ADDED and RENAMED supersede nothing. **Fail** if two active changes supersede the same rule; **warn** if a superseded rule doesn't exist in the source of truth (delta drift).
4. **Exclude the superseded scenarios from the run.** They must not reach the runner and must not be reported as skipped. Two sanctioned bindings, chosen by what the runner actually supports:
   - **Line-targeted discovery**, where the runner genuinely filters before loading — cucumber-js `spec.feature:12:19` loads only the scenarios starting at those lines. Nothing is written back to `.extracted/`.
   - **Pruning the extracted tree**, where line selection would merely runtime-skip — behave reports every unselected scenario as skipped, and no flag suppresses that count, so the superseded `Rule:` blocks are blanked out of `.extracted/` and whole files are passed instead.

   Pruning is not a spec edit: `.extracted/` is a generated artifact, gitignored and rebuilt every run, so the source of truth stays pristine — which is what this protects. Blank the lines rather than deleting them, so line fidelity survives. Omit a file entirely if no scenario survives; include untouched files whole.
5. **Add every delta file whole.**
6. **Print the composition report** to stderr — each superseded rule, the change that superseded it, and every left-out scenario with its **source `spec.md`** `file:line` (identical line numbers, by line fidelity), plus a summary count, so an excluded scenario is never silently absent from results or reports. **This format is identical across stacks:**

   ```
   [effective-spec] user-signup / Rule: A signup SHALL require an email address and a password
   [effective-spec]   superseded by change: signup-email-verification
   [effective-spec]   left out: Signing up with valid details (../openspec/specs/user-signup/spec.md:12)
   [effective-spec]   left out: Rejecting a signup with no email (../openspec/specs/user-signup/spec.md:37)
   [effective-spec] 2 source-of-truth scenario(s) excluded; delta versions run from openspec/changes/
   ```

## Archive exclusion

Specs under `openspec/changes/archive/` must **never** execute. Archived changes are historical deltas already merged into `openspec/specs/`; running them re-executes stale duplicates. This is non-negotiable. Extraction already skips the archive, and composition filters it again — defense in depth.
