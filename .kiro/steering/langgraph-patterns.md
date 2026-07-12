---
inclusion: fileMatch
fileMatchPattern: "src/app/shared/langgraph_layer/**/*"
---

# LangGraph Integration Patterns

## Graph Design

- Define clear node responsibilities
- Use typed state objects
- Implement proper error handling
- Use conditional edges for routing

## State Management

Use Pydantic models for state:

```python
from pydantic import BaseModel

class AgentState(BaseModel):
    messages: list[str]
    context: dict
    result: str | None = None
```

## Nodes

Keep nodes focused:
- Single responsibility
- Clear inputs/outputs
- Proper error handling
- Async-first

## Edges

Use conditional edges for routing:
- Route based on state
- Handle errors
- Implement fallbacks

## Persistence

Store graph state for recovery:
- Use checkpointing
- Implement recovery logic
- Monitor state size

## Human-in-the-Loop

Implement human approval:
- Pause at decision points
- Collect human input
- Resume execution
- Log decisions
