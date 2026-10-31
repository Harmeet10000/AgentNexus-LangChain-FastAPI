# Output Validation And Retry

This reference covers what happens *after* a model call: malformed structured output,
schema-error retries, output-parser fix loops, and where a custom post-LLM validation
pipeline must live instead of middleware. It complements `04-model-selection-and-structured-output.md`
(how to request structured output) and `05-middleware-and-guardrails.md` (how middleware
wraps the model call).

## The Boundary: Transport Retry Vs Content Validation

ModelRetryMiddleware retries failed *model calls* — it wraps the call and retries when an
exception is raised, with exponential backoff:

```python
ModelRetryMiddleware(
    max_retries=2,              # retries after the initial call
    retry_on=default_retry_on,  # retryable model errors + unclassified exceptions
    on_failure="continue",      # "continue" | "error" | callable -> AIMessage content
    backoff_factor=2.0,
    initial_delay=1.0,
    max_delay=60.0,
    jitter=True,                # ±25% to avoid thundering herd
)
```

- `retry_on` is either a tuple of exception types or a callable taking the exception and
  returning whether to retry. Exceptions that do not match propagate immediately and are
  not handled by `on_failure`. [1]
- `on_failure` decides what happens when retries are exhausted: `"continue"` returns an
  `AIMessage` with error details so the agent can keep going; `"error"` re-raises;
  a custom callable formats the error into the `AIMessage` content. [1]

The critical property: **middleware retries only see exceptions, never successful output.**
A model call that returns well-formed-but-wrong, or JSON that fails Pydantic validation,
is a *success* from the middleware's point of view. Content/semantic validation therefore
cannot live in `ModelRetryMiddleware` — it belongs in an explicit post-LLM pipeline after
the output exists (see "Decision Guidance" below).

Inside `create_agent`, structured-output schema errors are handled by the *response format
strategy*, not by retry middleware: with `ToolStrategy`, a schema validation failure is
caught, and the error text is fed back to the model as a `ToolMessage`
("Please fix your mistakes"), prompting an in-loop retry. The `handle_errors` parameter
controls this: `True` (catch all, default template), a custom `str` message, an exception
type or tuple (only retry those, raise others), a `Callable[[Exception], str]`, or `False`
(propagate). [2]

## `with_structured_output`: Malformed-Output Behavior

The base-model method has a well-defined error contract that hinges on `include_raw`:

- `include_raw=False` (default): only the parsed structured output is returned. **If an
  error occurs during model output parsing it will be raised.** [3]
- `include_raw=True`: the output is always a dict with keys `'raw'` (the `BaseMessage`),
  `'parsed'` (`None` if parsing failed), and `'parsing_error'` (the caught exception). The
  parsing error is caught and returned, not raised. [3]

The idiom: use `include_raw=True` whenever the caller needs to *observe* parse failures
(log it, count it, escalate it) instead of catching exceptions — it converts a raise into
a typed `'parsing_error'` field. This is the pattern the repository's validation layer
generalizes: absence/failure is a typed state, never an accidental pass.

Schema typing matters for validation strength: when `schema` is a Pydantic class, the
model output is validated into an instance of it; with a `TypedDict`, JSON Schema dict, or
OpenAI tool schema, the output is a plain dict and **is not validated**. [3]

Method variants (provider-dependent signature, e.g. ChatAnthropic):

```python
with_structured_output(
    schema,
    include_raw=False,
    method="function_calling",  # or "json_schema"
)
```

`function_calling` (the default) drives the schema through tool calling; `json_schema`
uses provider-native structured output where supported. [4]

## Agent Structured Output: Strategy-Level Error Handling

For `create_agent(response_format=...)` there are two strategies:

- `ProviderStrategy` — provider-native structured output (OpenAI, Anthropic, xAI, Gemini).
  The provider enforces the schema; described as the most reliable method when available.
  `strict=True` requires `langchain>=1.2`. Passing a bare schema type auto-selects
  `ProviderStrategy` when the model supports it, falling back to `ToolStrategy` otherwise. [2]
- `ToolStrategy` — structured output via tool calling for models without native support.
  Its error-handling surface is the `handle_errors` parameter described above. [2]

Two documented failure modes that the agent handles by feeding errors back as
`ToolMessage`s and re-prompting:

1. **Multiple structured outputs error** — the model calls two structured-output tools in
   one turn; each gets an error `ToolMessage` naming both schemas, and the model retries. [2]
2. **Schema validation error** — the model's tool-call args fail Pydantic validation (e.g.
   `rating: 10` against `le=5`); the validation error text is returned in the
   `ToolMessage`, and the model retries with corrected args. [2]

Note this retry is *feedback-carrying by construction*: the exact validation error is in
the conversation before the retry. Any custom retry loop should preserve that property —
a blind re-invocation without the error is a dice roll.

## Output Parsers And Their Fix/Retry Semantics (Legacy Path)

The parser family parses *text* output into structures. The official concept guidance is:
as more models support function/tool calling, "It is recommended to use function/tool
calling rather than output parsing." [5]

The two LLM-calling fixers differ in what they pass to the repair call:

| Parser | Passes to the fix LLM | Limitation |
|---|---|---|
| `OutputFixingParser` | the parser's error message + the bad output | Does **not** see the original prompt/instructions; the fix can be confused when the output is partially complete (the docs' example: given `{"action": "search"}` with a missing required field, it invents a placeholder `action_input='input'`). [6] [7] |
| `RetryOutputParser` / `RetryWithErrorOutputParser` | the original prompt, the bad output, and the error message | Re-asks the model with `parse_with_prompt(bad_response, prompt_value)` so the repair is grounded in the original instructions. [6] [7] |

```python
fix_parser = OutputFixingParser.from_llm(parser=parser, llm=ChatOpenAI())
retry_parser = RetryOutputParser.from_llm(parser=parser, llm=OpenAI(temperature=0))
retry_parser.parse_with_prompt(bad_response, prompt_value)
```

The `Retry*` parsers' re-run-with-prompt semantics are the direct ancestor of
feedback-carrying retries: the retry prompt includes *what went wrong* and *the original
task*, not just the malformed text. [6] [7]

## LangGraph Idiom: Validating Node Output Before Commit

LangGraph's prebuilt `ValidationNode` validates the tool calls of the last `AIMessage`
against Pydantic schemas **without running the tools** — it returns `ToolMessage`s
containing either validated content or formatted error messages, and a conditional edge
routes back to the model node when any message carries `is_error`:

```python
builder.add_node("model", llm.bind_tools([SelectNumber]))
builder.add_node("validation", ValidationNode([SelectNumber]))

def should_reprompt(state) -> Literal["model", "__end__"]:
    for msg in state[::-1]:
        if msg.type == "ai":
            return END
        if msg.additional_kwargs.get("is_error"):
            return "model"
    return END
```

Its stated use case is extraction-style structured output where you need schema
conformance "without losing the original messages and tool IDs (for use in multi-turn
conversations)". [8] Note: **`ValidationNode` is deprecated** — the documented successor is
`create_agent` with custom tool error handling (`ToolStrategy.handle_errors`). [8]

For nodes that are not tool-shaped (schema-locked generation nodes), the LangGraph
equivalent is an explicit validation node with a conditional edge back to the generator —
which is exactly the shape of a post-LLM validation pipeline: generate → validate →
ACCEPT / retry-with-feedback / escalate.

## Decision Guidance: Which Tool For Which Failure

| Failure | Right tool | Not the right tool |
|---|---|---|
| Model call raises (rate limit, timeout, provider 5xx) | `ModelRetryMiddleware` / transport-level retry [1] | Anything that inspects content — there is no content yet |
| Tool-call args fail the Pydantic schema inside an agent | `ToolStrategy.handle_errors` (in-loop `ToolMessage` feedback) [2] | Middleware — the call succeeded |
| Parse failure you must observe, not catch | `with_structured_output(..., include_raw=True)` → `'parsing_error'` [3] | `include_raw=False` (raises) |
| Legacy text parsing, cheap format repair | `OutputFixingParser` [5] [7] | Anything needing the original task context — it never sees the prompt |
| Legacy text parsing, repair needs the original instructions | `RetryOutputParser.parse_with_prompt` [6] [7] | `OutputFixingParser` (invents plausible filler on partial output) |
| Semantic/content gates (grounding, entailment, faithfulness, business predicates) | An explicit post-LLM validation pipeline with typed decisions and feedback-carrying retries | Middleware (never sees content), `handle_errors` (schema-level only), parsers (format-level only) |

The last row is the load-bearing one. Schema enforcement — however strict — cannot decide
whether a generated answer is *grounded in the retrieved evidence* or whether an extracted
clause *actually appears in the source document*. Those are mechanical or model-evaluated
content predicates over values that already parsed successfully, and they need a decision
vocabulary (accept / retry-with-feedback / escalate) that middleware and parsers do not
have.

## Repository Context

This repository is building that last layer: `PLAN-result-validation-layer.md` (repo root)
specifies `src/app/ai/validation/` — a pydantic schema gate, mechanical + LLM semantic
evaluators, a typed `ValidationDecision` (ACCEPT/RETRY/ESCALATE), feedback-carrying
retries (`RetryFeedback` appended to the prompt, mirroring the `RetryOutputParser`
semantics above), budget-ledger integration, and typed escalation sinks. Its contract
"unavailable evaluator ⇒ ESCALATE, never ACCEPT" and "retry only retryable finding codes"
have no counterpart in any LangChain primitive above — middleware and parsers cannot
express either. Saul-side adoption (which seams adopt it, which stay on guardrails
middleware) is specified in `PLAN-agent-tools.md` § D7.

## References

1. *ModelRetryMiddleware*. https://reference.langchain.com/python/langchain/agents/middleware/model_retry/ModelRetryMiddleware
2. *Structured output (create_agent response_format, ProviderStrategy/ToolStrategy, error handling)*. https://python.langchain.com/oss/python/langchain/structured-output
3. *with_structured_output (BaseChatModel)*. https://reference.langchain.com/python/langchain-core/language_models/chat_models/BaseChatModel/with_structured_output
4. *with_structured_output (ChatAnthropic — method variants)*. https://reference.langchain.com/python/langchain-anthropic/chat_models/ChatAnthropic/with_structured_output
5. *Output parsers (concept guide — prefer tool calling; parser comparison table)*. https://python.langchain.com/docs/concepts/output_parsers/
6. *How to retry when a parsing error occurs*. https://python.langchain.com/v0.2/docs/how_to/output_parser_retry
7. *OutputFixingParser.from_llm*. https://reference.langchain.com/python/langchain-classic/output_parsers/fix/OutputFixingParser/from_llm
8. *ValidationNode (langgraph.prebuilt — deprecated)*. https://reference.langchain.com/python/langgraph.prebuilt/tool_validator/ValidationNode
