---
name: test-driven-development
description: "Use when implementing a change with test-driven development (TDD): work test-first through focused red-green cycles."
---

# Test-Driven Development

Use the repository's existing test tooling and conventions.

One test, one behaviour, one reason to fail. A test name describes that one
behaviour; never use joiners such as "and", "or", or "etc." to combine
independent outcomes. Multiple assertions are acceptable only when together
they establish that same behaviour.

## Red

- Select one observable behaviour from the requirement.
- Write one test before changing implementation code.
- Run the test and confirm it fails because the behaviour is missing, not because the test is broken.

## Green

- Write only enough implementation code to pass the failing test.
- Run the test, then run the relevant suite.
- Do not add behaviour without first seeing its test fail.

## Refactor

- Refactor only while the relevant suite is green, after completing a red-green slice.
- Refactor either the tests or the implementation in one pass, not both.
- Reuse shared setup, teardown, helpers, and parameterized structure when they clarify intent without combining behaviours.
- Rerun the relevant suite after each refactor pass.

## Patterns and anti-patterns

Read [patterns and anti-patterns](references/best-practices.md) when choosing a test shape, writing examples, or deciding whether cases belong together.

## Collaborators

Read [collaborators](references/collaborators.md) before substituting a dependency. Mock system boundaries, not code owned by the application.
