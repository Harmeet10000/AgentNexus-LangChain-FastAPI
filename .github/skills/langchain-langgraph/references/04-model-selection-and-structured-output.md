# Model Selection And Structured Output

## Dynamic Model Selection

Select models based on State:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from langchain.chat_models import init_chat_model
from typing import Callable

large_model = init_chat_model("claude-sonnet-4-6")
standard_model = init_chat_model("gpt-4.1")
efficient_model = init_chat_model("gpt-4.1-mini")

@wrap_model_call
def state_based_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    message_count = len(request.messages)

    if message_count > 20:
        model = large_model
    elif message_count > 10:
        model = standard_model
    else:
        model = efficient_model

    request = request.override(model=model)
    return handler(request)
```

64. Dynamic models are selected at runtime based on the current state and context. This enables sophisticated routing logic and cost optimization.

```python
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

basic_model = ChatOpenAI(model="gpt-4.1-mini")
advanced_model = ChatOpenAI(model="gpt-4.1")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])
    model = advanced_model if message_count > 10 else basic_model
    return handler(request.override(model=model))
```

## Structured Output

59. `create_agent` handles structured output automatically.

```python
def create_agent(
    ...,
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
)
```

Use `response_format` to control how the agent returns structured data:

- `ToolStrategy[StructuredResponseT]`
- `ProviderStrategy[StructuredResponseT]`
- `type[StructuredResponseT]`
- `None`

Provider strategy notes:

- Some providers support structured output natively.
- `ProviderStrategy.strict` requires `langchain>=1.2`.

49. It can be useful to return the raw `AIMessage` object alongside the parsed representation to access response metadata such as token counts. To do this, set `include_raw=True` when calling `with_structured_output`.

50. LangChain chat models can expose a dictionary of supported features through a `profile` attribute:

```python
model.profile
{
  "max_input_tokens": 400000,
  "image_inputs": True,
  "reasoning_output": True,
  "tool_calling": True,
}

model = init_chat_model("...", profile=custom_profile)
```

51. Implicit prompt caching and explicit caching may reduce cost depending on provider. Cache usage is reflected in response usage metadata.

## Dynamic Response Formats

State-based response format:

```python
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from pydantic import BaseModel, Field
from typing import Callable

class SimpleResponse(BaseModel):
    answer: str = Field(description="A brief answer")

class DetailedResponse(BaseModel):
    answer: str = Field(description="A detailed answer")
    reasoning: str = Field(description="Explanation of reasoning")
    confidence: float = Field(description="Confidence score 0-1")

@wrap_model_call
def state_based_output(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    message_count = len(request.messages)
    if message_count < 3:
        request = request.override(response_format=SimpleResponse)
    else:
        request = request.override(response_format=DetailedResponse)
    return handler(request)
```

Store-based response format and runtime-context response format follow the same pattern, using stored preferences or user role to select the schema.
