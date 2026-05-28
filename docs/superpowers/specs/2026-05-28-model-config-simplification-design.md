# Model Config Simplification Design

## Goal

Simplify Gemini model configuration and model factory usage so that:

- `temperature`, `top_p`, `top_k`, and model cache TTL come from environment-backed settings
- chat model creation goes through one configurable function instead of separate builders for different behaviors
- public model helper APIs are async-only

This change should be minimal and local to the existing model layer, without introducing a new abstraction stack.

## Current Problems

The current model layer has three issues:

1. Defaults are split between settings and hard-coded module constants.
   - `GEMINI_TEMPERATURE` comes from settings
   - `top_p`, `top_k`, and context cache TTL are hard-coded in `src/app/shared/langchain_layer/models.py`

2. Chat model construction is duplicated.
   - `build_chat_model(...)`
   - `build_chat_google_genai_model(...)`
   These overlap heavily and force callers to know which implementation to reach for.

3. The module exposes a mixed surface.
   - async inference helpers are already present
   - sync construction helpers and sync cache creation remain public entry points
   - multimodal helpers still depend on sync file reading internally

## Design Summary

We will keep `src/app/shared/langchain_layer/models.py` as the single model entry point and refactor it around one configurable chat-model factory plus async helper APIs.

### Scope

Files in scope:

- `src/app/config/settings.py`
- `src/app/shared/langchain_layer/models.py`
- `.env.example`
- `.env.development`
- small internal call sites that currently use the old duplicated builder names

Files out of scope:

- broad business-logic retuning of hard-coded per-feature temperatures
- embedding architecture changes
- agent architecture redesign

## Settings Changes

Add validated settings fields in `src/app/config/settings.py`:

- `GEMINI_TOP_P: float`
- `GEMINI_TOP_K: int`
- `GEMINI_CONTEXT_CACHE_TTL: str`

Existing field retained as default source of truth:

- `GEMINI_TEMPERATURE: float`

Validation intent:

- `GEMINI_TEMPERATURE` should remain bounded to a safe model range
- `GEMINI_TOP_P` should be in `[0.0, 1.0]`
- `GEMINI_TOP_K` should be `> 0`
- `GEMINI_CONTEXT_CACHE_TTL` should remain a non-empty duration string because Gemini cache creation currently expects string TTL input like `3600s`

Environment files will be updated to include these keys so the behavior is explicit and reproducible.

## Model Factory Design

Replace the duplicate chat builders with one function that handles all chat-model behavior through arguments plus settings fallbacks.

Proposed shape:

```python
def get_chat_model(
    *,
    model_name: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    max_tokens: int | None = None,
    streaming: bool = False,
    cached_content: str | None = None,
    implementation: Literal["generic", "google_genai"] = "generic",
    **kwargs: Any,
) -> BaseChatModel:
    ...
```

Behavior rules:

- if an argument is passed, use it
- otherwise fall back to `settings.<...>`
- if no `model_name` is passed, use the configured default model from settings
- callers can still request different behavior by passing a different `model_name`
- callers that truly need the concrete Google model implementation can request `implementation="google_genai"`

This keeps one public factory while still allowing the only meaningful distinction in the current code: generic LangChain model init vs direct `ChatGoogleGenerativeAI` use.

## Async-Only Public Surface

The public helper surface in `models.py` should be async-only.

### Keep

- `ainvoke_text(...)`
- `abatch_text(...)`
- `astream_text(...)`
- `ainvoke_multimodal(...)`
- `abatch_multimodal(...)`
- `aembed_text(...)`
- `aembed_batch(...)`

### Refactor

- context cache creation should become `acreate_gemini_context_cache(...)`
- multimodal local file encoding should become async-safe internally by moving file reads behind `asyncio.to_thread(...)`

### Internal-only sync allowance

Some SDK operations are sync-only. Those are still acceptable internally if wrapped with `asyncio.to_thread(...)`.

The rule is:

- no public sync model helper function
- internal sync SDK calls may remain if bridged safely in async wrappers

## Function Simplification Plan

`models.py` will be simplified around these concerns:

1. one chat model factory
2. async text inference helpers
3. async multimodal inference helpers
4. async structured-output helper setup
5. async embedding wrappers
6. async Gemini context cache creation

Notes:

- `with_structured_output(...)` currently returns a configured chain immediately. That is not itself an I/O operation. To satisfy the async-only public surface requirement, this should be replaced by an async entrypoint that returns the configured model/chain or be made private if the repo does not actually depend on public direct access.
- The preferred minimal approach is to keep the synchronous composition private and expose an async helper for invocation-time usage.

## Call Site Migration

Minimal call-site work:

- replace internal references to `build_chat_model(...)` and `build_chat_google_genai_model(...)`
- keep behavior unchanged where possible by forwarding the same explicit args
- preserve current fast/vision/pro behavior by passing `model_name=settings.GEMINI_FLASH_MODEL`, `settings.GEMINI_VISION_MODEL`, or `settings.GEMINI_PRO_MODEL` instead of using separate helper functions

This preserves flexibility without separate factory names.

## Error Handling

Validation failures should happen at settings load time for invalid env values.

Runtime failures should remain explicit:

- invalid multimodal input should still raise a `ValueError`
- unknown MIME type should still raise a `ValueError`
- cache creation errors should bubble up instead of being silently swallowed

## Testing and Verification

Minimum verification after implementation:

1. `uv run ruff check src/`
2. `uv run ty check src/`
3. targeted import sanity for settings/model module

Behavioral checks:

- confirm the new env-backed settings load successfully
- confirm text invocation still works with default settings
- confirm multimodal invocation still works with explicit `model_name` override
- confirm context cache default TTL now comes from settings

## Recommended Minimal Implementation

The safest implementation sequence is:

1. add settings fields and env entries
2. introduce the single chat-model factory
3. convert cache creation to async and settings-backed TTL
4. update async helpers to use the single factory
5. migrate internal call sites away from old builder names
6. remove old duplicated public builder functions

## Non-Goals

- no new “model role” enum or global strategy object
- no broad prompt or temperature retuning across feature code
- no migration of every sync helper in the repo, only this model layer surface

## Expected Outcome

After this refactor:

- model defaults are controlled from `.env` via `Settings`
- there is one configurable chat model creation path
- callers express behavior through arguments, not separate helper names
- public model utilities are async-only
- the module becomes easier to reason about and less likely to drift across environments

## Chosen Ones

The subtle win here is not fewer functions. It is eliminating hidden behavior forks. Once `top_p`, `top_k`, and cache TTL live in settings, retries, workers, background jobs, and local dev all stop inventing their own “almost the same” model behavior. That is how you make agent runs reproducible enough to debug.
