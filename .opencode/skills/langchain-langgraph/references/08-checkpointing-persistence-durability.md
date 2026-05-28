# Checkpointing Persistence Durability

## Why Persistence Exists

LangGraph persistence saves graph state as checkpoints organized into threads. It enables:

- human-in-the-loop
- conversation memory
- time travel
- fault-tolerant execution

## Thread And Checkpoint Rules

When invoking a graph with a checkpointer, pass:

```python
{"configurable": {"thread_id": "..."}}
```

21. `checkpoint_id` is your best friend after pause and resume workflows.

36. The latest state of a graph can be viewed with:

```python
graph.get_state(config)
```

37. Full history can be viewed with:

```python
graph.get_state_history(config)
```

38. You can edit state using `update_state`. This creates a new checkpoint, not an in-place modification.

31. Delete all checkpoints for a thread:

```python
thread_id = "1"
checkpointer.delete_thread(thread_id)
```

## Checkpoint Namespace

Each checkpoint has a `checkpoint_ns` field.

- `""` means root graph
- `"node_name:uuid"` means subgraph checkpoint
- nested subgraphs join namespaces with `|`

Example:

```python
def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
```

## Serializers

39. Checkpointers serialize state channel values using serializer objects.

`JsonPlusSerializer` is the default and handles many built-in LangChain and LangGraph types.

If needed, enable pickle fallback for unsupported types.

## Durable Execution Guidance

40. Durable execution exists to avoid repeating work and to replay non-deterministic behavior consistently.

Guidelines:

- wrap multiple side-effecting operations into separate tasks when replay matters
- isolate non-deterministic operations
- make side effects idempotent when possible

## Durability Modes

41. Durability modes from least to most durable:

- `exit`
- `async`
- `sync`

`exit` is best for performance but loses mid-execution recovery.

`async` is a good middle ground.

`sync` gives the strongest guarantees and the highest overhead.
