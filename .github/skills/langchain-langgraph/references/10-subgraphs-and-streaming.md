# Subgraphs And Streaming

## Subgraph Patterns

32. Two main subgraph patterns:

Call a subgraph inside a node:

- use when parent and subgraph have different schemas
- write a wrapper function that invokes the subgraph and maps state in and out

Add a compiled subgraph as a node:

- use when parent and subgraph share state keys
- add the compiled subgraph directly with `add_node`

34. Subgraph persistence modes:

- per-thread persistence when a subagent needs multi-turn memory
- stateless mode when the subgraph should behave like a plain function call

Use `checkpointer=True` for per-thread behavior.

Use `checkpointer=False` for stateless behavior.

## Subgraphs And Interrupts

48. When invoking a subgraph inside a node, the parent graph resumes from the beginning of the parent node where the subgraph was invoked, and the subgraph resumes from the beginning of the interrupted subgraph node.

```python
def node_in_parent_graph(state: State):
    some_code()
    subgraph_result = subgraph.invoke(some_input)

def node_in_subgraph(state: State):
    some_other_code()
    result = interrupt("What's your name?")
```

## Stream Version 2

35. With `version="v2"`, subgraph events use the same `StreamPart` format.

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode=["values", "updates", "messages", "custom", "checkpoints", "tasks", "debug"],
    version="v2",
):
    ...
```

Every chunk is shaped like:

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),
    "data": ...,
}
```

`invoke(..., version="v2")` returns a `GraphOutput` object with `.value` and `.interrupts`.
