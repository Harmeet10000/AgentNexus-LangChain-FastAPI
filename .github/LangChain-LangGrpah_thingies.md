1. add Field description for tool instead of simple docstrings
2. use structured ouput everywhere for llm output, tool output, MCP output
3. use toons for serialisation before sending to LLM
4. use toons for deserialisation after receiving from LLM
5. use toons for serialisation before sending to tools
6. use toons for deserialisation after receiving from tools
"Communicate data using TOON format. Declaring uniform arrays as key[N]{field1, field2}: val1, val2. Minimal punctuation. No braces."
7. use async functions, methods and packages in langchain and langGraph
8. trim/remove tool output in a multi step agent conversation 
9. # CORRECT — use dedicated package imports
from langchain_tavily import TavilySearch 
# WRONG — deprecated community import path
from langchain_community.tools.tavily_search import TavilySearchResults
10. all methods, functions, model and agent invocation should have langsmith decorator for proper obervability
11. always normalise agent state after fetching from checkpointer so that there is no version mismatch
12. have proper retry mechanism for tools with idenpotent execution as mention in langchain docs
13.  
from langgraph.graph import StateGraph
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

class LegalState(TypedDict):
    messages: list

# Node 1: Researcher agent
def research_node(state: LegalState) -> LegalState:
    """Full agent as node with middleware."""
    researcher = create_agent(
        model="gpt-4o",
        tools=[search_caselaw, validate_cite],
        middleware=[
            SummarizationMiddleware(),  # Context compaction
        ],
        checkpointer=MemorySaver(),  # Per-node persistence
        system_prompt="Legal researcher. Cite sources.",
    )
    
    result = researcher.invoke(state)
    return {"messages": result["messages"]}

# Node 2: Reviewer agent
def review_node(state: LegalState) -> LegalState:
    reviewer = create_agent(
        model="claude-3-sonnet",
        tools=[analyze_risk, flag_clause],
        system_prompt="Risk reviewer. Conservative analysis.",
    )
    result = reviewer.invoke(state)
    return {"messages": result["messages"]}

# Graph orchestration
graph = StateGraph(LegalState)
graph.add_node("research", research_node)
graph.add_node("review", review_node)
graph.add_edge("research", "review")

app = graph.compile()
14.  use this 
from langgraph.graph.message import add_messages

# Messages are merged by ID (deduplication)
state["messages"] = add_messages(state["messages"], [new_msg])
15. Checkpointers: SqliteSaver (Sync) vs. AsyncSqliteSaver (Async). In production, always use the async version to avoid blocking your DB connection pool. Tools: @tool functions can be def or async def. If your tool calls an API, make it async def so the agent can do other things while waiting for the network.
16. use async methods - ainvoke, astream, abatch, atransform
17. get conversation state state = graph.get_state(config)
18. Circular delegation is possible. Agent A can hand off to Agent B, which can hand off back to Agent A. There's no loop detection beyond completed_agents in SupervisorState, and that only works in the supervisor graph — not in the tool-based MultiAgentSystem.
19. Always set a recursion_limit (max steps) in your LangGraph and a timeout on your LLM calls.
20. the checkpoint_id (formerly thread_ts) is your best friend. in HITL after resuming from a pause
21. Ensure your message history logic preserves the extras["signature"] field in AIMessage objects. When a model "thinks," it generates a Thought Signature. If you are building a multi-turn agent (like with LangGraph), failing to send this signature back in the next turn forces the model to re-reason from scratch, increasing latency.
22. Embeddings aren't cached. aembed_batch calls the API every time. Embeddings for the same text are deterministic — a simple LRU cache keyed on SHA256(text) would eliminate redundant API calls entirely.
23. Model instances are rebuilt on every call
build_chat_model() constructs a new ChatGoogleGenerativeAI every time it's called. The model object should be a module-level singleton (or per-spec singleton) since it's stateless.
24. Add toons before any operation/inputting data to LLM for best possible use of context space inlcuding agents, chats, RAG, web search results, after tool LLM invoke and everywhere else
25. Key Design Categories
Sync vs. Async Execution (1:00 - 2:47):

Synchronous: The main agent waits for subagent results. It is simpler to implement but blocks the conversation, making it less ideal for high-latency tasks.
Asynchronous: Subagents run in the background. This is more complex but better for responsiveness and performance when tasks can be handled independently.
Tool Design (1:35 - 5:24):

Tool per Agent: Assigns a dedicated tool to each subagent. This provides fine-grained control over inputs and outputs but requires more configuration.
Single Dispatch Tool: Uses one tool to route tasks to any subagent. This is effective for managing a large number of agents. Strategies include listing agents in the system prompt, passing agent names as arguments, or using a "list agents" tool for progressive disclosure.
Context Engineering (1:47 - 7:15):

Specifications: You must provide the main agent with clear tool names, descriptive docstrings, and well-defined argument schemas to ensure it calls the right agent at the right time (5:45).
Input/Output Customization: Developers can improve performance by injecting custom context (e.g., expertise levels like beginner or expert) or filtering/formatting the message history returned from subagents to the main agent 


26. The Four Architectural Patterns:
Subagents (Supervisor Pattern) (1:05 - 2:10):

A main agent coordinates subagents as tools.
Strengths: Excellent for distributed development (5/5) and parallelization.
Weakness: Not designed for direct user-to-subagent interaction.
Handoffs Pattern (2:11 - 3:04):

Agents use tool calling to pass control to one another.
Strengths: Best architecture for multihop conversations and direct user interaction.
Weakness: Difficult to develop agents independently.
Skills Pattern (3:05 - 4:12):

A single agent manages specialized prompts or knowledge (progressive disclosure) loaded as needed.
Strengths: High scores for distributed development, multihop support, and direct user interaction.
Router Architecture (4:13 - 5:40):

A routing step classifies input and directs it to specific agents before synthesizing a response.
Strengths: Excellent for parallelization (5/5).
Weakness: Limited support for multihop sequences.
27. Common Architectures (2:40 - 5:11)
Network of Agents: Agents decide who to call next. While flexible, this is often too loose, unreliable, and costly for production.
Supervisor Agent: A central agent routes tasks to specialized sub-agents. This provides better control and clarity.
Hierarchical Approach: Layering supervisors to group sub-agents for complex organizational structures.
Custom Architectures: The most recommended approach for production, where the control flow is specifically designed for the domain rather than relying on off-the-shelf patterns.
Communication Patterns (5:15 - 8:14)
Agents must exchange information effectively, usually via two methods:

Shared State: Agents read from and write to a common object (e.g., a shared message list or artifact keys).
Tool-Based Communication: One agent calls another as a tool, passing only the necessary parameters.
28. The Tree Model: View your context as a tree. The trunk represents the foundational information (the repo or the plan), while branches represent experimental paths. A skilled engineer explores these branches and then trims them back to the trunk using /rewind once they have gathered the necessary information
29. Context Engineering
in langgraph there are trim, delete, summarise, manage checkpoint and custom strategies if short term memory exceeds context window
Offload (3:06, 3:46): Moving context out of the LLM's active window and into external storage, such as a file system. This allows agents to persist information across long sessions and different invocations, using techniques like:
File-based memory: Storing plans or data in files (e.g., cloud.md or agent.md) that can be retrieved when needed (4:37-5:34).
Minimal toolsets: Using a small number of general, atomic tools (like a bash tool) rather than hundreds of specific ones, keeping instructions concise (6:05-7:50).
Progressive Disclosure: Loading only brief headers for skills initially, and reading the full content only when a specific action is required (8:23-10:29).
Reduce (3:06, 11:34): Minimizing the number of tokens passed through the LLM at each turn through:
Compaction: Replacing verbose historical tool results with references to saved files (11:41-12:53).
Summarization: Distilling long message histories into compact summaries when context windows approach their limit (12:39-13:46).
Filtering: Using middleware to block excessively large tool outputs (13:48-14:01).
Isolate (3:06, 14:02): Using sub-agents to handle specific tasks, giving them their own fresh context windows to avoid cluttering the parent agent's workspace 
30. The recommended way to access the store is through the Runtime object.   memories = await runtime.store.asearch
31. Delete all checkpoints for a thread thread_id = "1" checkpointer.delete_thread(thread_id)
32. SubGraphs Patterns and when to use which one in langGrpah 
Pattern - Call a subgraph inside a node,
When to Use - "When the parent and subgraph have different schemas (no shared keys), or you need to transform/filter data between them.",
State Schemas - "Different. Requires a ""bridge"" to map parent keys to subgraph keys.",
Implementation - You write a wrapper function (node) that invokes the subgraph with subgraph.invoke() and maps the result back.

Pattern - Add a subgraph as a node,
When to Use - When the parent and subgraph share state keys—the subgraph reads from and writes to the same channels as the parent.,
State Schemas - Overlapping or Identical. Subgraph directly mutates the shared state keys.,
Implementation - "You pass the compiled subgraph directly to add_node(name, compiled_subgraph). No wrapper function is needed."
33. The elite pattern is to use Pydantic's RootModel for subgraphs. This allows you to wrap a subgraph's entire state in a validation layer that acts as a "Guardrail." If an agent outputs a hallucinated legal citation format, the Pydantic validator can trigger an automatic "Self-Correction" loop by raising an error that the Graph catches and sends back to the agent with the validation message. This turns your state definition into an active part of your prompt engineering strategy.
34. for Subgraph persistence    

Per-thread
Per-invocation is the right choice for most applications, including multi-agent systems where subagents handle independent requests. Use per-thread when a subagent needs multi-turn conversation memory (for example, a research assistant that builds context over several exchanges).

Per-thread
Use per-thread persistence when a subagent needs to remember previous interactions. For example, a research assistant that builds up context over several exchanges, or a coding assistant that tracks what files it has already edited. The subagent’s conversation history and data accumulate across calls on the same thread. Each call picks up where the last one left off.
Compile with checkpointer=True to enable this behavior.

Stateless
Use this when you want to run a subagent like a plain function call with no checkpointing overhead. The subgraph cannot pause/resume and does not benefit from durable execution. Compile with checkpointer=False.
35. With version="v2", subgraph events use the same StreamPart format. The ns field identifies the source graph    
 for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode=["values" , "updates" , "messages" , "custom" , "checkpoints" , "tasks" , "debug"],
    version="v2",
):
Pass version="v2" to stream() or astream() to get a unified output format. Every chunk is a StreamPart dict with a consistent shape — regardless of stream mode, number of modes, or subgraph settings:
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # namespace tuple, populated for subgraph events
    "data": ...,        # the actual payload (type varies by stream mode)
}
v2 invoke format
When you pass version="v2" to invoke() or ainvoke(), it returns a GraphOutput object with .value and .interrupts attributes:
With any stream mode other than the default "values", invoke(..., stream_mode="updates", version="v2") returns list[StreamPart] instead of list[tuple].
config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke(inputs, config=config, version="v2")

if result.interrupts:
    print(result.interrupts[0].value)
    graph.invoke(Command(resume=True), config=config, version="v2")
36. You can view the latest state of the graph by calling graph.get_state(config). This will return a StateSnapshot object that corresponds to the latest checkpoint associated with the thread ID provided in the config or a checkpoint associated with a checkpoint ID for the thread, if provided.

each checkpoint has a checkpoint_ns (checkpoint namespace) field that identifies which graph or subgraph it belongs to:
def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
"" (empty string): The checkpoint belongs to the parent (root) graph.
"node_name:uuid": The checkpoint belongs to a subgraph invoked as the given node. For nested subgraphs, namespaces are joined with | separators (e.g., "outer_node:uuid|inner_node:uuid").
37. You can get the full history of the graph execution for a given thread by calling graph.get_state_history(config). This will return a list of StateSnapshot objects associated with the thread ID provided in the config. Importantly, the checkpoints will be ordered chronologically with the most recent checkpoint / StateSnapshot being the first in the list.
38. You can edit the graph state using update_state. This creates a new checkpoint with the updated values — it does not modify the original checkpoint. The update is treated the same as a node update: values are passed through reducer functions when defined, so channels with reducers accumulate values rather than overwrite them.
39. When checkpointers save the graph state, they need to serialize the channel values in the state. This is done using serializer objects.
langgraph_checkpoint defines protocol for implementing serializers provides a default implementation (JsonPlusSerializer) that handles a wide variety of types, including LangChain and LangGraph primitives, datetimes, enums and more.
​
Serialization with pickle
The default serializer, JsonPlusSerializer, uses ormsgpack and JSON under the hood, which is not suitable for all types of objects.
If you want to fallback to pickle for objects not currently supported by our msgpack encoder (such as Pandas dataframes), you can use the pickle_fallback argument of the JsonPlusSerializer:
40. why implement durable execution in langgraph
 Avoid Repeating Work: If a node contains multiple operations with side effects (e.g., logging, file writes, or network calls), wrap each operation in a separate task. This ensures that when the workflow is resumed, the operations are not repeated, and their results are retrieved from the persistence layer.
Encapsulate Non-Deterministic Operations: Wrap any code that might yield non-deterministic results (e.g., random number generation) inside tasks or nodes. This ensures that, upon resumption, the workflow follows the exact recorded sequence of steps with the same outcomes.
Use Idempotent Operations: When possible ensure that side effects (e.g., API calls, file writes) are idempotent. This means that if an operation is retried after a failure in the workflow, it will have the same effect as the first time it was executed. This is particularly important for operations that result in data writes. In the event that a task starts but fails to complete successfully, the workflow’s resumption will re-run the task, relying on recorded outcomes to maintain consistency. Use idempotency keys or verify existing results to avoid unintended duplication, ensuring a smooth and predictable workflow execution.
41. The durability modes, from least to most durable, are as follows:
"exit": LangGraph persists changes only when graph execution exits either successfully, with an error, or due to a human in the loop interrupt. This provides the best performance for long-running graphs but means intermediate state is not saved, so you cannot recover from system failures (like process crashes) that occur mid-execution.
"async": LangGraph persists changes asynchronously while the next step executes. This provides good performance and durability, but there’s a small risk that LangGraph does not write checkpoints if the process crashes during execution.
"sync": LangGraph persists changes synchronously before the next step starts. This ensures that LangGraph writes every checkpoint before continuing execution, providing high durability at the cost of some performance overhead.
42. Starting points for resuming workflows
If you’re using a StateGraph (Graph API), the starting point is the beginning of the node where execution stopped.
If you’re making a subgraph call inside a node, the starting point will be the parent node that called the subgraph that was halted. Inside the subgraph, the starting point will be the specific node where execution stopped.
43. Key points about resuming:
You must use the same thread ID when resuming that was used when the interrupt occurred
The value passed to Command(resume=...) becomes the return value of the interrupt call
The node restarts from the beginning of the node where the interrupt was called when resumed, so any code before the interrupt runs again
You can pass any JSON-serializable value as the resume value 
44. You can also place interrupts directly inside tool functions. This makes the tool itself pause for approval whenever it’s called, and allows for human review and editing of the tool call before it is executed.
45. Do not wrap interrupt calls in try/except
The way that interrupt pauses execution at the point of the call is by throwing a special exception. If you wrap the interrupt call in a try/except block, you will catch this exception and the interrupt will not be passed back to the graph.
✅ Separate interrupt calls from error-prone code
✅ Use specific exception types in try/except blocks
def node_a(state: State):
    # ✅ Good: interrupting first, then handling
    # error conditions separately
    interrupt("What's your name?")
    try:
        fetch_data()  # This can fail
    except Exception as e:
        print(e)
    return state
def node_a(state: State):
    # ✅ Good: catching specific exception types
    # will not catch the interrupt exception
    try:
        name = interrupt("What's your name?")
        fetch_data()  # This can fail
    except NetworkException as e:
        print(e)
    return state
def node_a(state: State):
    # ❌ Bad: wrapping interrupt in bare try/except
    # will catch the interrupt exception
    try:
        interrupt("What's your name?")
    except Exception as e:
        print(e)
    return state
46. Do not reorder interrupt calls within a node
It’s common to use multiple interrupts in a single node, however this can lead to unexpected behavior if not handled carefully.
When a node contains multiple interrupt calls, LangGraph keeps a list of resume values specific to the task executing the node. Whenever execution resumes, it starts at the beginning of the node. For each interrupt encountered, LangGraph checks if a matching value exists in the task’s resume list. Matching is strictly index-based, so the order of interrupt calls within the node is important.
Keep interrupt calls consistent across node executions
 Do not conditionally skip interrupt calls within a node
 Do not loop interrupt calls using logic that isn’t deterministic across executions
def node_a(state: State):
    # ❌ Bad: conditionally skipping interrupts changes the order
    name = interrupt("What's your name?")

    # On first run, this might skip the interrupt
    # On resume, it might not skip it - causing index mismatch
    if state.get("needs_age"):
        age = interrupt("What's your age?")

    city = interrupt("What's your city?")

    return {"name": name, "city": city}
def node_a(state: State):
    # ❌ Bad: looping based on non-deterministic data
    # The number of interrupts changes between executions
    results = []
    for item in state.get("dynamic_list", []):  # List might change between runs
        result = interrupt(f"Approve {item}?")
        results.append(result)

    return {"results": results}

Do not return complex values in interrupt calls
Depending on which checkpointer is used, complex values may not be serializable (e.g. you can’t serialize a function). To make your graphs adaptable to any deployment, it’s best practice to only use values that can be reasonably serialized.
✅ Pass simple, JSON-serializable types to interrupt
✅ Pass dictionaries/objects with simple values
🔴 Do not pass functions, class instances, or other complex objects to interrupt

def validate_input(value):
    return len(value) > 0

def node_a(state: State):
    # ❌ Bad: passing a function to interrupt
    # The function cannot be serialized
    response = interrupt({
        "question": "What's your name?",
        "validator": validate_input  # This will fail
    })
    return {"name": response}
47. ✅ Use idempotent operations before interrupt
✅ Place side effects after interrupt calls
✅ Separate side effects into separate nodes when possible
def node_a(state: State):
    # ✅ Good: using upsert operation which is idempotent
    # Running this multiple times will have the same result
    db.upsert_user(
        user_id=state["user_id"],
        status="pending_approval"
    )

    approved = interrupt("Approve this change?")

    return {"approved": approved}
def node_a(state: State):
    # ✅ Good: placing side effect after the interrupt
    # This ensures it only runs once after approval is received
    approved = interrupt("Approve this change?")

    if approved:
        db.create_audit_log(
            user_id=state["user_id"],
            action="approved"
        )

    return {"approved": approved}
def approval_node(state: State):
    # ✅ Good: only handling the interrupt in this node
    approved = interrupt("Approve this change?")

    return {"approved": approved}

def notification_node(state: State):
    # ✅ Good: side effect happens in a separate node
    # This runs after approval, so it only executes once
    if (state.approved):
        send_notification(
            user_id=state["user_id"],
            status="approved"
        )

    return state
🔴 Do not perform non-idempotent operations before interrupt
🔴 Do not create new records without checking if they exist
def node_a(state: State):
    # ❌ Bad: creating a new record before interrupt
    # This will create duplicate records on each resume
    audit_id = db.create_audit_log({
        "user_id": state["user_id"],
        "action": "pending_approval",
        "timestamp": datetime.now()
    })

    approved = interrupt("Approve this change?")

    return {"approved": approved, "audit_id": audit_id}
def node_a(state: State):
    # ❌ Bad: appending to a list before interrupt
    # This will add duplicate entries on each resume
    db.append_to_history(state["user_id"], "approval_requested")

    approved = interrupt("Approve this change?")

    return {"approved": approved}
48. Using with subgraphs called as functions
When invoking a subgraph within a node, the parent graph will resume execution from the beginning of the node where the subgraph was invoked and the interrupt was triggered. Similarly, the subgraph will also resume from the beginning of the node where interrupt was called.
def node_in_parent_graph(state: State):
    some_code()  # <-- This will re-execute when resumed
    # Invoke a subgraph as a function.
    # The subgraph contains an `interrupt` call.
    subgraph_result = subgraph.invoke(some_input)
    # ...

def node_in_subgraph(state: State):
    some_other_code()  # <-- This will also re-execute when resumed
    result = interrupt("What's your name?")
    # ...
49. It can be useful to return the raw AIMessage object alongside the parsed representation to access response metadata such as token counts. To do this, set include_raw=True when calling with_structured_output:
50. LangChain chat models can expose a dictionary of supported features and capabilities through a profile attribute:
model.profile
 {
   "max_input_tokens": 400000,
   "image_inputs": True,
   "reasoning_output": True,
   "tool_calling": True,
   ...
 }
 model = init_chat_model("...", profile=custom_profile)
51. Implicit prompt caching: providers will automatically pass on cost savings if a request hits a cache. Examples: OpenAI and Gemini.
Explicit caching: providers allow you to manually indicate cache points for greater control or to guarantee cost savings. Examples:
Cache usage will be reflected in the usage metadata of the model response.
52. Use text prompts when:
    You have a single, standalone request
    You don’t need conversation history
    You want minimal code complexity
Use message prompts when:
    Managing multi-turn conversations
    Working with multimodal content (images, audio, files)
    Including system instructions
    messages = [
        SystemMessage("You are a poetry expert"),
        HumanMessage("Write a haiku about spring"),
        AIMessage("Cherry blossoms bloom...")
    ]
    response = model.invoke(messages)
​53. Chat models can accept multimodal data as input and generate it as output. Below we show short examples of input messages featuring multimodal data.
files
# From URL
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "url": "https://example.com/path/to/image.jpg"},
    ]
}

# From base64 data
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {
            "type": "image",
            "base64": "AAAAIGZ0eXBtcDQyAAAAAGlzb21tcDQyAAACAGlzb2...",
            "mime_type": "image/jpeg",/  "application/pdf"  /   "audio/wav",  / "video/mp4",
        },
    ]
}

# From provider-managed File ID
message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Describe the content of this image."},
        {"type": "image", "file_id": "file-abc123"},
    ]
}
54. Prefer snake_case for tool names (e.g., web_search instead of Web Search). Some model providers have issues with or reject names containing spaces or special characters with errors. Sticking to alphanumeric characters, underscores, and hyphens helps to improve compatibility across providers.
@tool(name="", desciption="", args_schema=WeatherInput)

Tools can access runtime information through the ToolRuntime parameter, which provides:
Component	
Description	
Use case
State	
Short-term memory - mutable data that exists for the current conversation (messages, counters, custom fields)	
Access conversation history, track tool call counts
Context	
Immutable configuration passed at invocation time (user IDs, session info)	Personalize responses based on user identity
Store	
Long-term memory - persistent data that survives across conversations	
Save user preferences, maintain knowledge base
Stream Writer	
Emit real-time updates during tool execution	
Show progress for long-running operations
Config	
RunnableConfig for the execution	
Access callbacks, tags, and metadata
Tool Call ID	
Unique identifier for the current tool invocation	
Correlate tool calls for logs and model invocations

55. Access state
Tools can access the current conversation state using runtime.state:
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage
@tool
def get_last_user_message(runtime: ToolRuntime) -> str:
    """Get the most recent message from the user."""
    messages = runtime.state["messages"]

Update state
Use Command to update the agent’s state. This is useful for tools that need to update custom state fields:
from langgraph.types import Command
from langchain.tools import tool
@tool
def set_user_name(new_name: str) -> Command:
    """Set the user's name in the conversation state."""
    return Command(update={"user_name": new_name})

Context
Context provides immutable configuration data that is passed at invocation time. Use it for user IDs, session details, or application-specific settings that shouldn’t change during a conversation.
Access context through runtime.context:
@tool
def get_account_info(runtime: ToolRuntime[UserContext]) -> str:
    """Get the current user's account information."""
    user_id = runtime.context.user_id

Long-term memory (Store)
The BaseStore provides persistent storage that survives across conversations. Unlike state (short-term memory), data saved to the store remains available in future sessions.
Access the store through runtime.store. The store uses a namespace/key pattern to organize data:
# Access memory
@tool
def get_user_info(user_id: str, runtime: ToolRuntime) -> str:
    """Look up user info."""
    store = runtime.store
    user_info = store.get(("users",), user_id)
    return str(user_info.value) if user_info else "Unknown user"

# Update memory
@tool
def save_user_info(user_id: str, user_info: dict[str, Any], runtime: ToolRuntime) -> str:
    """Save user info."""
    store = runtime.store
    store.put(("users",), user_id, user_info)
    return "Successfully saved user info."

model = ChatOpenAI(model="gpt-4.1")

store = InMemoryStore()
agent = create_agent(
    model,
    tools=[get_user_info, save_user_info],
    store=store
)

Stream writer
Stream real-time updates from tools during execution. This is useful for providing progress feedback to users during long-running operations.
Use runtime.stream_writer to emit custom updates:
from langchain.tools import tool, ToolRuntime
@tool
def get_weather(city: str, runtime: ToolRuntime) -> str:
    """Get weather for a given city."""
    writer = runtime.stream_writer

    # Stream custom updates as the tool executes
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")

    return f"It's always sunny in {city}!"

56. Error handling
Configure how tool errors are handled. See the ToolNode API reference for all options.
from langgraph.prebuilt import ToolNode

# Default: catch invocation errors, re-raise execution errors
tool_node = ToolNode(tools)

# Catch all errors and return error message to LLM
tool_node = ToolNode(tools, handle_tool_errors=True)

# Custom error message
tool_node = ToolNode(tools, handle_tool_errors="Something went wrong, please try again.")

# Custom error handler
def handle_error(e: ValueError) -> str:
    return f"Invalid input: {e}"

tool_node = ToolNode(tools, handle_tool_errors=handle_error)

# Only catch specific exception types
tool_node = ToolNode(tools, handle_tool_errors=(ValueError, TypeError))
57. By default, agents use AgentState to manage short term memory, specifically the conversation history via a messages key.
You can extend AgentState to add additional fields. Custom state schemas are passed to create_agent using the state_schema parameter.
from langchain.tools import tool, ToolRuntime
from langchain_core.runnables import RunnableConfig
from langchain.messages import ToolMessage
from langchain.agents import create_agent, AgentState
from langgraph.types import Command
from pydantic import BaseModel


class CustomState(AgentState):
    user_name: str

class CustomContext(BaseModel):
    user_id: str

@tool
def update_user_info(
    runtime: ToolRuntime[CustomContext, CustomState],
) -> Command:
    """Look up and update user info."""
    user_id = runtime.context.user_id
    name = "John Smith" if user_id == "user_123" else "Unknown user"
    return Command(update={
        "user_name": name,
        # update the message history
        "messages": [
            ToolMessage(
                "Successfully looked up user information",
                tool_call_id=runtime.tool_call_id
            )
        ]
    })

@tool
def greet(
    runtime: ToolRuntime[CustomContext, CustomState]
) -> str | Command:
    """Use this to greet the user once you found their info."""
    user_name = runtime.state.get("user_name", None)
    if user_name is None:
       return Command(update={
            "messages": [
                ToolMessage(
                    "Please call the 'update_user_info' tool it will get and update the user's name.",
                    tool_call_id=runtime.tool_call_id
                )
            ]
        })
    return f"Hello {user_name}!"

agent = create_agent(
    model="gpt-5-nano",
    tools=[update_user_info, greet],
    state_schema=CustomState,
    context_schema=CustomContext,
)

agent.invoke(
    {"messages": [{"role": "user", "content": "greet the user"}]},
    context=CustomContext(user_id="user_123"),
)
if memory has to be stored on disk, 2 ways:
1. memort-dd-mm-yyyy.md
daily memory-only log
running context, task in progress
read today + yesterday at every session start
running notes, task in progress which accumulates and get pruned
2. Memory.md
curated long term facts, decisions prefrences
only loaded in main provate session
things that shouldnt change
58. To stream updates from tools as they are executed, you can use get_stream_writer.
from langchain.agents import create_agent
from langgraph.config import get_stream_writer  


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    writer = get_stream_writer()
    # stream any arbitrary data
    writer(f"Looking up data for city: {city}")
    writer(f"Acquired data for city: {city}")
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"])
59. LangChain’s create_agent handles structured output automatically. The user sets their desired structured output schema, and when the model generates the structured data, it’s captured, validated, and returned in the 'structured_response' key of the agent’s state.
def create_agent(
    ...
    response_format: Union[
        ToolStrategy[StructuredResponseT],
        ProviderStrategy[StructuredResponseT],
        type[StructuredResponseT],
        None,
    ]
)
Response format
Use response_format to control how the agent returns structured data:
ToolStrategy[StructuredResponseT]: Uses tool calling for structured output
ProviderStrategy[StructuredResponseT]: Uses provider-native structured output
type[StructuredResponseT]: Schema type - automatically selects best strategy based on model capabilities
None: Structured output not explicitly requested
Schema validation error
When structured output doesn’t match the expected schema, the agent provides specific error feedback:
60. The following middleware work with any LLM provider:
Middleware	Description
Summarization	Automatically summarize conversation history when approaching token limits.
Human-in-the-loop	Pause execution for human approval of tool calls.
Model call limit	Limit the number of model calls to prevent excessive costs.
Tool call limit	Control tool execution by limiting call counts.
Model fallback	Automatically fallback to alternative models when primary fails.
PII detection	Detect and handle Personally Identifiable Information (PII).
To-do list	Equip agents with task planning and tracking capabilities.
LLM tool selector	Use an LLM to select relevant tools before calling main model.
Tool retry	Automatically retry failed tool calls with exponential backoff.
Model retry	Automatically retry failed model calls with exponential backoff.
LLM tool emulator	Emulate tool execution using an LLM for testing purposes.
Context editing	Manage conversation context by trimming or clearing tool uses.
Shell tool	Expose a persistent shell session to agents for command execution.
File search	Provide Glob and Grep search tools over filesystem files.
Filesystem	Provide agents with a filesystem for storing context and long-term memories.
Subagent	Add the ability to spawn subagents.
