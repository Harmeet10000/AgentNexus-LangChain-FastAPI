# Prompts And Messages

## Prompt Types

13. `PromptTemplate` is used for single string inputs.

`ChatPromptTemplate` is used for structured chat with system, human, and AI messages.

`MessagesPlaceholder` is a slot in a template where a list of messages, such as history, is injected.

## Preserve Thought Signatures

22. Ensure your message history logic preserves the `extras["signature"]` field in `AIMessage` objects. When a model thinks, it generates a Thought Signature. If you fail to send this signature back in the next turn, the model may re-reason from scratch, increasing latency.

## Text Prompts Vs Message Prompts

52. Use text prompts when:

- you have a single standalone request
- you do not need conversation history
- you want minimal code complexity

Use message prompts when:

- managing multi-turn conversations
- working with multimodal content such as images, audio, or files
- including system instructions

```python
messages = [
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a haiku about spring"),
    AIMessage("Cherry blossoms bloom...")
]
response = model.invoke(messages)
```

## Multimodal Inputs

53. Chat models can accept multimodal data as input and generate it as output.

From URL:

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}
```

From base64 data:

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",
        },
    ]
}
```

From provider-managed file ID:

```python
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
```

## Dynamic Prompt Examples

State-aware system prompt:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dynamic_prompt
def state_aware_prompt(request: ModelRequest) -> str:
    message_count = len(request.messages)

    base = "You are a helpful assistant."
    if message_count > 10:
        base += "\nThis is a long conversation - be extra concise."
    return base

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[state_aware_prompt]
)
```

Store-aware prompt:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@dynamic_prompt
def store_aware_prompt(request: ModelRequest) -> str:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    user_prefs = store.get(("preferences",), user_id)

    base = "You are a helpful assistant."
    if user_prefs:
        style = user_prefs.value.get("communication_style", "balanced")
        base += f"\nUser prefers {style} responses."
    return base

agent = create_agent(
    model="gpt-4.1",
    tools=[...],
    middleware=[store_aware_prompt],
    context_schema=Context,
    store=InMemoryStore(),
)
```

Runtime-context prompt:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

@dataclass
class Context:
    user_role: str
    deployment_env: str

@dynamic_prompt
def context_aware_prompt(request: ModelRequest) -> str:
    user_role = request.runtime.context.user_role
    env = request.runtime.context.deployment_env

    base = "You are a helpful assistant."
    if user_role == "admin":
        base += "\nYou have admin access. You can perform all operations."
    elif user_role == "viewer":
        base += "\nYou have read-only access. Guide users to read operations only."

    if env == "production":
        base += "\nBe extra careful with any data modifications."
    return base
```

## Message Injection Examples

Inject uploaded file context from State:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable

@wrap_model_call
def inject_file_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    uploaded_files = request.state.get("uploaded_files", [])

    if uploaded_files:
        file_descriptions = []
        for file in uploaded_files:
            file_descriptions.append(
                f"- {file['name']} ({file['type']}): {file['summary']}"
            )

        file_context = f"""Files you have access to in this conversation:
{chr(10).join(file_descriptions)}

Reference these files when answering questions."""

        messages = [
            *request.messages,
            {"role": "user", "content": file_context},
        ]
        request = request.override(messages=messages)

    return handler(request)
```

Inject writing style from Store:

```python
from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable
from langgraph.store.memory import InMemoryStore

@dataclass
class Context:
    user_id: str

@wrap_model_call
def inject_writing_style(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    user_id = request.runtime.context.user_id
    store = request.runtime.store
    writing_style = store.get(("writing_style",), user_id)

    if writing_style:
        style = writing_style.value
        style_context = f"""Your writing style:
- Tone: {style.get('tone', 'professional')}
- Typical greeting: \"{style.get('greeting', 'Hi')}\"
- Typical sign-off: \"{style.get('sign_off', 'Best')}\"
- Example email you've written:
{style.get('example_email', '')}"""

        messages = [
            *request.messages,
            {"role": "user", "content": style_context}
        ]
        request = request.override(messages=messages)

    return handler(request)
```
