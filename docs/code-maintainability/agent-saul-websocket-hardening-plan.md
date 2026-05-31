# Agent Saul Websocket Hardening Plan

Canonical source: `docs/superpowers/plans/2026-05-31-agent-saul-websocket-hardening.md`

This file exists so the maintainability-related materials from the last two prompts live together under one docs folder.

## Summary

- Decouple websocket runtime dependencies from unused checkpointer state.
- Fix unreachable websocket state-update emission logic.
- Fix websocket rate-limit exception construction.
- Fix the session bootstrap websocket URL contract.
- Add focused seam-level regression tests.

## Maintainability Angle

The important architectural point is not just fixing the bugs. It is making the websocket path a deeper module with a smaller public surface:

- routers should not know unused infrastructure details
- event-to-frame mapping should be owned in one place
- public contracts must be truthful
- tests should lock behavior at the seam callers rely on

Read the canonical plan for the full task breakdown and file-level steps.
