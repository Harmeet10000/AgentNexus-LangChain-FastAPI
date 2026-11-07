# Tasks: <change>

<!-- Read review.md first: VERDICT CHANGES-REQUESTED → fix artifacts before writing tasks. Check design.md Open Questions: anything that changes what gets built gets resolved with the user FIRST. Checkbox format is parsed by apply - only `- [ ] X.Y ...` tasks track. -->

## 1. <Section>

- [ ] 1.1 <Task>
      Implements: <Requirement/Scenario> · Channel: <asyncapi operation, if async>
      verify: <command> — <expected outcome>
- [ ] 1.2 <Task>
      verify: <command> — <expected outcome>

## N. Acceptance Gate

- [ ] N.1 Validate the change's spec deltas strictly
      verify: openspec validate <change> --type change --strict — all deltas pass, exit 0

<!-- Rules: small enough for one session; ordered by dependency. Every task cites Requirement/Scenario IDs and ends with an explicit verify line. Async tasks ALSO cite the channel/operation from contracts/asyncapi.yaml. A task without scenario citation or verify is not ready for apply. The validate --strict task is ALWAYS present when specs changed. -->

## Task Dependency Graph

<!-- Waves: same wave = no dependencies between them, safe to parallelize. Waves run strictly in order. Every task ID appears in exactly one wave. -->

- **Wave 1:** 1.1, 1.2
- **Wave 2:** 2.1, 2.2
