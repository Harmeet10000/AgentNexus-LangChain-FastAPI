---
name: documentation
description: Create, structure, and review technical documentation using the Diataxis framework. Use when Codex needs to write or reorganize tutorials, how-to guides, reference pages, API documentation, explanation pages, user guides, or documentation structure; also use when the request mentions Diataxis, tutorials vs how-to guides, reference docs, technical writing, or organizing docs.
---

# Documentation

Organize technical documentation into the correct Diataxis type before writing. Keep each page single-purpose, cross-link related pages, and optimize structure for the user's immediate need.

Ask clarifying questions about audience, context, and goals before drafting documentation when those details are missing or materially affect the output.

## Identify the Documentation Type

Choose the documentation type from the user's intent:

- User is new and wants to learn by doing: write a tutorial.
- User wants to complete a known task: write a how-to guide.
- User wants parameters, options, syntax, or exact behavior: write reference.
- User wants rationale, concepts, or trade-offs: write an explanation.

Use this quick decision rule:

- Learning for the first time: tutorial.
- Solving a specific problem: how-to guide.
- Looking up facts: reference.
- Building understanding: explanation.

## Write Tutorials

Use tutorials for beginners learning by doing.

- Start the title with a verb such as `Build your first X` or `Create a Y from scratch`.
- Structure as goal, prerequisites, numbered steps, visible result after each step, and final outcome.
- Minimize theory. Emphasize concrete actions and checks.
- Ensure each step produces a verifiable result.
- Validate that a beginner can finish end-to-end without outside help.

Example signal: "Walk me through building my first API."

## Write How-To Guides

Use how-to guides for users who already understand the domain and need to accomplish one task.

- Title as a task, such as `How to configure X`.
- Structure as goal, assumptions, numbered steps, and expected result.
- Skip conceptual background unless it prevents correct execution.
- Note meaningful alternatives only when they affect the task.
- Validate that an experienced user can complete the task without backtracking.

Example signal: "How do I deploy this service to production?"

## Write Reference

Use reference for information lookup.

- Title with the subject name, such as `Configuration options` or `API endpoints`.
- Use a repeatable format for every entry.
- Prefer fields such as name, type, default, description, and example where applicable.
- State facts directly. Keep usage examples minimal.
- Keep the page easy to scan so a user can find a fact in under 30 seconds.

Example signal: "What flags does this CLI command support?"

## Write Explanations

Use explanations for conceptual understanding and rationale.

- Title around the concept, such as `How X works` or `Why Y is designed this way`.
- Structure as context, core concept, alternatives or trade-offs, and broader perspective.
- Avoid turning the page into instructions or a spec.
- Validate that the reader can explain the why in their own words after reading.

Example signal: "Why does CAP matter for AI agents?"

## Maintain Separation

- Keep each document as one Diataxis type.
- Do not mix tutorials with reference tables or explanations with procedural steps.
- Cross-link between related pages when needed.
- Keep terminology and headings consistent across the documentation set.

## Validate Before Delivering

Apply the correct check before finalizing:

- Tutorial: a beginner can complete it unaided.
- How-to guide: it solves the stated task for an experienced user.
- Reference: a specific fact is easy to find quickly.
- Explanation: it improves understanding of the rationale, not just the mechanics.
