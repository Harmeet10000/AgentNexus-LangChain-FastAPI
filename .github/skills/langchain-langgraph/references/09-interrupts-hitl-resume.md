# Interrupts HITL Resume

## Core Resume Rules

43. Key points about resuming:

- use the same thread ID that was used when the interrupt occurred
- the value passed to `Command(resume=...)` becomes the return value of the interrupt call
- the node restarts from the beginning of the node where the interrupt was called
- only pass JSON-serializable resume values

42. Starting points for resuming workflows:

- in a `StateGraph`, the starting point is the beginning of the node where execution stopped
- for a subgraph call inside a node, both the parent node and the interrupted subgraph node may restart from their beginnings

44. You can place interrupts directly inside tool functions for approval or editing before execution.

## Do Not Catch Interrupts Broadly

45. Do not wrap interrupt calls in bare `try/except`.

Good:

```python
def node_a(state: State):
    interrupt("What's your name?")
    try:
        fetch_data()
    except Exception as e:
        print(e)
    return state
```

Also good when catching specific exceptions:

```python
def node_a(state: State):
    try:
        name = interrupt("What's your name?")
        fetch_data()
    except NetworkException as e:
        print(e)
    return state
```

Bad:

```python
def node_a(state: State):
    try:
        interrupt("What's your name?")
    except Exception as e:
        print(e)
    return state
```

## Do Not Reorder Interrupt Calls

46. Keep interrupt ordering stable across executions.

Do not:

- conditionally skip interrupts in a way that changes call order
- loop interrupts over non-deterministic lists
- pass complex unserializable values such as functions

Bad examples from source:

```python
def node_a(state: State):
    name = interrupt("What's your name?")
    if state.get("needs_age"):
        age = interrupt("What's your age?")
    city = interrupt("What's your city?")
    return {"name": name, "city": city}
```

```python
def node_a(state: State):
    results = []
    for item in state.get("dynamic_list", []):
        result = interrupt(f"Approve {item}?")
        results.append(result)
    return {"results": results}
```

## Idempotency Around Interrupts

47. Use idempotent operations before interrupts, or move side effects after the interrupt or into separate nodes.

Good:

```python
def node_a(state: State):
    db.upsert_user(user_id=state["user_id"], status="pending_approval")
    approved = interrupt("Approve this change?")
    return {"approved": approved}
```

```python
def node_a(state: State):
    approved = interrupt("Approve this change?")
    if approved:
        db.create_audit_log(user_id=state["user_id"], action="approved")
    return {"approved": approved}
```

Bad:

```python
def node_a(state: State):
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval",
        "timestamp": datetime.now()
    })
    approved = interrupt("Approve this change?")
    return {"approved": approved, "audit_id": audit_id}
```
