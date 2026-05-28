# Model Config Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize Gemini generation defaults in settings, collapse duplicate chat model builders into one configurable entrypoint, and expose async-only public model helpers.

**Architecture:** Keep `src/app/shared/langchain_layer/models.py` as the repo's model hub. Move runtime defaults into `Settings`, route chat model construction through a single factory, and convert public cache/build helpers to async wrappers while preserving current behavior through explicit arguments.

**Tech Stack:** Pydantic Settings, LangChain, langchain-google-genai, asyncio, Ruff, ty

---

### Task 1: Centralize model defaults in settings and env

**Files:**
- Modify: `src/app/config/settings.py`
- Modify: `.env.example`
- Modify: `.env.development`

- [ ] Add validated settings fields for `GEMINI_TOP_P`, `GEMINI_TOP_K`, and `GEMINI_CONTEXT_CACHE_TTL`.
- [ ] Add matching env entries to `.env.example` and `.env.development`.
- [ ] Keep existing `GEMINI_TEMPERATURE` and `GEMINI_MAX_TOKENS` as the default source of truth.

### Task 2: Collapse chat model creation to one factory

**Files:**
- Modify: `src/app/shared/langchain_layer/models.py`

- [ ] Replace `build_chat_model(...)` and `build_chat_google_genai_model(...)` with a single configurable factory.
- [ ] Make the factory default to `settings` values when overrides are omitted.
- [ ] Preserve explicit selection of flash/vision/pro models via `model_name` instead of separate helper names.

### Task 3: Make public helper surface async-only

**Files:**
- Modify: `src/app/shared/langchain_layer/models.py`

- [ ] Convert Gemini context cache creation into an async wrapper.
- [ ] Move sync-only SDK calls behind `asyncio.to_thread(...)` where needed.
- [ ] Keep public text, multimodal, and embedding helpers async-only.

### Task 4: Update call sites to the new factory

**Files:**
- Modify: `src/app/shared/langchain_layer/agents/factory.py`
- Modify: any other internal imports of removed builder names

- [ ] Replace old builder references with the new factory.
- [ ] Preserve existing behavior with explicit overrides where current code depends on them.

### Task 5: Verify the refactor

**Files:**
- No source changes required unless checks fail

- [ ] Run `uv run ruff check src/`
- [ ] Run `uv run ty check src/`
- [ ] Fix any issues introduced by the refactor.

## Chosen Ones

The winning pattern is not “one helper function.” It is “one behavioral authority.” Once settings own all default knobs, you can replay a bad run with the same effective parameters instead of reverse-engineering which helper happened to inject a different `top_p` or cache TTL.
