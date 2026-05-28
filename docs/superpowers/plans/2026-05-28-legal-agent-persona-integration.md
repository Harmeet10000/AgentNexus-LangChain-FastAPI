# Legal Agent Persona Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the approved disciplined-hybrid Indian appellate legal persona into the reusable legal prompt constant.

**Architecture:** Keep the change minimal and centered on the existing `LAWYER_SYSTEM_PROMPT` definition in `src/app/shared/langchain_layer/prompts.py`. Update the prompt content to reflect the approved senior-advocate core, defensibility-first policy, internal war-room tone, and limited Saul-inspired tactical reframing without expanding scope into unrelated legal-agent wiring.

**Tech Stack:** Python, Pydantic, LangChain prompt layer, Ruff

---

### Task 1: Update the legal prompt constant

**Files:**
- Modify: `src/app/shared/langchain_layer/prompts.py`

- [ ] Replace the current `LAWYER_SYSTEM_PROMPT` content with the approved disciplined-hybrid legal persona.
- [ ] Preserve the existing `SystemPromptParts` structure.
- [ ] Encode the following behaviors:
  - senior advocate practicing in the Supreme Court of India and major High Courts
  - defensibility-first analysis
  - internal war-room strategic voice
  - limited creative reframing and leverage spotting
  - explicit prohibition on impropriety, bluffing, and unsupported cleverness

### Task 2: Verify prompt syntax and style

**Files:**
- No source changes required unless checks fail

- [ ] Run `uv run ruff check src/app/shared/langchain_layer/prompts.py`
- [ ] Fix any lint errors introduced by the prompt update

## Chosen Ones

The critical implementation decision is to keep the Saul-inspired layer inside `execution_policy` rather than `identity`. That preserves institutional credibility while still giving the model permission to surface leverage and reframing moves when they are legally supportable.
