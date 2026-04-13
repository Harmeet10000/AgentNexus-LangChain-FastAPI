1. add Field description for tool instead of simple docstrings
2. use structured ouput everywhere for llm output, tool output, MCP output
3. use toons for serialisation before sending to LLM
4. use toons for deserialisation after receiving from LLM
5. use toons for serialisation before sending to tools
6. use toons for deserialisation after receiving from tools / should i use chains for repeatable action for toon conversion 
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

What you can control
To build reliable agents, you need to control what happens at each step of the agent loop, as well as what happens between steps.
Context Type	What You Control	Transient or Persistent
Model Context	What goes into model calls (instructions, message history, tools, response format)	Transient
Tool Context	What tools can access and produce (reads/writes to state, store, runtime context)	Persistent
Life-cycle Context	What happens between model and tool calls (summarization, guardrails, logging, etc.)	Persistent
Transient context
What the LLM sees for a single call. You can modify messages, tools, or prompts without changing what’s saved in state.
Persistent context
What gets saved in state across turns. Life-cycle hooks and tool writes modify this permanently.

Data sources
Throughout this process, your agent accesses (reads / writes) different sources of data:
Data Source   	  Also Known As    	     Scope	                Examples
Runtime Context	  Static configuration	 Conversation-scoped	User ID, API keys, database connections, permissions, environment settings
State	          Short-term memory	     Conversation-scoped	Current messages, uploaded files, authentication status, tool results
Store	          Long-term memory	     Cross-conversation	    User preferences, extracted insights, memories, historical data
​
Model context: 
Control what goes into each model call - instructions, available tools, which model to use, and output format. These decisions directly impact reliability and cost. these things are
a. System Prompt
    The system prompt sets the LLM’s behavior and capabilities. Different users, contexts, or conversation stages need different instructions. Successful agents draw on memories, preferences, and configuration to provide the right instructions for the current state of the conversation.
    (A) State:
    Access message count or conversation context from state:
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    @dynamic_prompt
    def state_aware_prompt(request: ModelRequest) -> str:
        # request.messages is a shortcut for request.state["messages"]
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
    (B) Store:
    Access user preferences from long-term memory:
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

        # Read from Store: get user preferences
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
        store=InMemoryStore()
    )
    (C) Runtime Context:
    Access user ID or configuration from Runtime Context:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import dynamic_prompt, ModelRequest

    @dataclass
    class Context:
        user_role: str
        deployment_env: str

    @dynamic_prompt
    def context_aware_prompt(request: ModelRequest) -> str:
        # Read from Runtime Context: user role and environment
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

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[context_aware_prompt],
        context_schema=Context
    )
b. Messages
    Messages make up the prompt that is sent to the LLM. It’s critical to manage the content of messages to ensure that the LLM has the right information to respond well.
    (A) State:
    Inject uploaded file context from State when relevant to current query:
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable

    @wrap_model_call
    def inject_file_context(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject context about files user has uploaded this session."""
        # Read from State: get uploaded files metadata
        uploaded_files = request.state.get("uploaded_files", [])

        if uploaded_files:
            # Build context about available files
            file_descriptions = []
            for file in uploaded_files:
                file_descriptions.append(
                    f"- {file['name']} ({file['type']}): {file['summary']}"
                )

            file_context = f"""Files you have access to in this conversation:
    {chr(10).join(file_descriptions)}

    Reference these files when answering questions."""

            # Inject file context before recent messages
            messages = [
                *request.messages,
                {"role": "user", "content": file_context},
            ]
            request = request.override(messages=messages)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[inject_file_context]
    )
    (B) Store:
    Inject user’s email writing style from Store to guide drafting:
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
        """Inject user's email writing style from Store."""
        user_id = request.runtime.context.user_id  

        # Read from Store: get user's writing style examples
        store = request.runtime.store  
        writing_style = store.get(("writing_style",), user_id)

        if writing_style:
            style = writing_style.value
            # Build style guide from stored examples
            style_context = f"""Your writing style:
    - Tone: {style.get('tone', 'professional')}
    - Typical greeting: "{style.get('greeting', 'Hi')}"
    - Typical sign-off: "{style.get('sign_off', 'Best')}"
    - Example email you've written:
    {style.get('example_email', '')}"""

            # Append at end - models pay more attention to final messages
            messages = [
                *request.messages,
                {"role": "user", "content": style_context}
            ]
            request = request.override(messages=messages)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[inject_writing_style],
        context_schema=Context,
        store=InMemoryStore()
    )
    (C) Runtime Context:
    Inject compliance rules from Runtime Context based on user’s jurisdiction:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable

    @dataclass
    class Context:
        user_jurisdiction: str
        industry: str
        compliance_frameworks: list[str]

    @wrap_model_call
    def inject_compliance_rules(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Inject compliance constraints from Runtime Context."""
        # Read from Runtime Context: get compliance requirements
        jurisdiction = request.runtime.context.user_jurisdiction  
        industry = request.runtime.context.industry  
        frameworks = request.runtime.context.compliance_frameworks  

        # Build compliance constraints
        rules = []
        if "GDPR" in frameworks:
            rules.append("- Must obtain explicit consent before processing personal data")
            rules.append("- Users have right to data deletion")
        if "HIPAA" in frameworks:
            rules.append("- Cannot share patient health information without authorization")
            rules.append("- Must use secure, encrypted communication")
        if industry == "finance":
            rules.append("- Cannot provide financial advice without proper disclaimers")

        if rules:
            compliance_context = f"""Compliance requirements for {jurisdiction}:
    {chr(10).join(rules)}"""

            # Append at end - models pay more attention to final messages
            messages = [
                *request.messages,
                {"role": "user", "content": compliance_context}
            ]
            request = request.override(messages=messages)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[inject_compliance_rules],
        context_schema=Context
    )

c. Tools
    Tools let the model interact with databases, APIs, and external systems. How you define and select tools directly impacts whether the model can complete tasks effectively
    
    Defining tools
    Each tool needs a clear name, description, argument names, and argument descriptions. These aren’t just metadata—they guide the model’s reasoning about when and how to use the tool.
    Selecting tools
    Not every tool is appropriate for every situation. Too many tools may overwhelm the model (overload context) and increase errors; too few limit capabilities. Dynamic tool selection adapts the available toolset based on authentication state, user permissions, feature flags, or conversation stage.
    (A) State: 
    Enable advanced tools only after certain conversation milestones:
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable

    @wrap_model_call
    def state_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on conversation State."""
        # Read from State: check if user has authenticated
        state = request.state  
        is_authenticated = state.get("authenticated", False)
        message_count = len(state["messages"])

        # Only enable sensitive tools after authentication
        if not is_authenticated:
            tools = [t for t in request.tools if t.name.startswith("public_")]
            request = request.override(tools=tools)
        elif message_count < 5:
            # Limit tools early in conversation
            tools = [t for t in request.tools if t.name != "advanced_search"]
            request = request.override(tools=tools)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[public_search, private_search, advanced_search],
        middleware=[state_based_tools]
    )
    (B) Store:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable
    from langgraph.store.memory import InMemoryStore

    @dataclass
    class Context:
        user_id: str

    @wrap_model_call
    def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on Store preferences."""
        user_id = request.runtime.context.user_id

        # Read from Store: get user's enabled features
        store = request.runtime.store
        feature_flags = store.get(("features",), user_id)

        if feature_flags:
            enabled_features = feature_flags.value.get("enabled_tools", [])
            # Only include tools that are enabled for this user
            tools = [t for t in request.tools if t.name in enabled_features]
            request = request.override(tools=tools)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[search_tool, analysis_tool, export_tool],
        middleware=[store_based_tools],
        context_schema=Context,
        store=InMemoryStore()
    )
    (C) Runtime Context:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable

    @dataclass
    class Context:
        user_role: str

    @wrap_model_call
    def context_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on Runtime Context permissions."""
        # Read from Runtime Context: get user role
        user_role = request.runtime.context.user_role

        if user_role == "admin":
            # Admins get all tools
            pass
        elif user_role == "editor":
            # Editors can't delete
            tools = [t for t in request.tools if t.name != "delete_data"]
            request = request.override(tools=tools)
        else:
            # Viewers get read-only tools
            tools = [t for t in request.tools if t.name.startswith("read_")]
            request = request.override(tools=tools)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[read_data, write_data, delete_data],
        middleware=[context_based_tools],
        context_schema=Context
    )

d. Model
    Different models have different strengths, costs, and context windows. Select the right model for the task at hand, which might change during an agent run.
    (A) State:
    Use different models based on conversation length from State:
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from langchain.chat_models import init_chat_model
    from typing import Callable

    # Initialize models once outside the middleware
    large_model = init_chat_model("claude-sonnet-4-6")
    standard_model = init_chat_model("gpt-4.1")
    efficient_model = init_chat_model("gpt-4.1-mini")

    @wrap_model_call
    def state_based_model(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select model based on State conversation length."""
        # request.messages is a shortcut for request.state["messages"]
        message_count = len(request.messages)

        if message_count > 20:
            # Long conversation - use model with larger context window
            model = large_model
        elif message_count > 10:
            # Medium conversation
            model = standard_model
        else:
            # Short conversation - use efficient model
            model = efficient_model

        request = request.override(model=model)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1-mini",
        tools=[...],
        middleware=[state_based_model]
    )
    (B) Store:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable
    from langgraph.store.memory import InMemoryStore

    @dataclass
    class Context:
        user_id: str

    @wrap_model_call
    def store_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on Store preferences."""
        user_id = request.runtime.context.user_id

        # Read from Store: get user's enabled features
        store = request.runtime.store
        feature_flags = store.get(("features",), user_id)

        if feature_flags:
            enabled_features = feature_flags.value.get("enabled_tools", [])
            # Only include tools that are enabled for this user
            tools = [t for t in request.tools if t.name in enabled_features]
            request = request.override(tools=tools)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[search_tool, analysis_tool, export_tool],
        middleware=[store_based_tools],
        context_schema=Context,
        store=InMemoryStore()
    )
    (C) Runtime Context:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from typing import Callable

    @dataclass
    class Context:
        user_role: str

    @wrap_model_call
    def context_based_tools(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Filter tools based on Runtime Context permissions."""
        # Read from Runtime Context: get user role
        user_role = request.runtime.context.user_role

        if user_role == "admin":
            # Admins get all tools
            pass
        elif user_role == "editor":
            # Editors can't delete
            tools = [t for t in request.tools if t.name != "delete_data"]
            request = request.override(tools=tools)
        else:
            # Viewers get read-only tools
            tools = [t for t in request.tools if t.name.startswith("read_")]
            request = request.override(tools=tools)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[read_data, write_data, delete_data],
        middleware=[context_based_tools],
        context_schema=Context
    )

e. Response Format
    Structured output transforms unstructured text into validated, structured data. When extracting specific fields or returning data for downstream systems, free-form text isn’t sufficient.
    How it works: When you provide a schema as the response format, the model’s final response is guaranteed to conform to that schema. The agent runs the model / tool calling loop until the model is done calling tools, then the final response is coerced into the provided format.
    All of these types of model context can draw from state (short-term memory), store (long-term memory), or runtime context (static configuration).

    Selecting formats
    Dynamic response format selection adapts schemas based on user preferences, conversation stage, or role—returning simple formats early and detailed formats as complexity increases.
    (A) State:
    Configure structured output based on conversation state:
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from pydantic import BaseModel, Field
    from typing import Callable

    class SimpleResponse(BaseModel):
        """Simple response for early conversation."""
        answer: str = Field(description="A brief answer")

    class DetailedResponse(BaseModel):
        """Detailed response for established conversation."""
        answer: str = Field(description="A detailed answer")
        reasoning: str = Field(description="Explanation of reasoning")
        confidence: float = Field(description="Confidence score 0-1")

    @wrap_model_call
    def state_based_output(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select output format based on State."""
        # request.messages is a shortcut for request.state["messages"]
        message_count = len(request.messages)

        if message_count < 3:
            # Early conversation - use simple format
            request = request.override(response_format=SimpleResponse)
        else:
            # Established conversation - use detailed format
            request = request.override(response_format=DetailedResponse)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[state_based_output]
    )
    (B) Store:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from pydantic import BaseModel, Field
    from typing import Callable
    from langgraph.store.memory import InMemoryStore

    @dataclass
    class Context:
        user_id: str

    class VerboseResponse(BaseModel):
        """Verbose response with details."""
        answer: str = Field(description="Detailed answer")
        sources: list[str] = Field(description="Sources used")

    class ConciseResponse(BaseModel):
        """Concise response."""
        answer: str = Field(description="Brief answer")

    @wrap_model_call
    def store_based_output(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select output format based on Store preferences."""
        user_id = request.runtime.context.user_id

        # Read from Store: get user's preferred response style
        store = request.runtime.store
        user_prefs = store.get(("preferences",), user_id)

        if user_prefs:
            style = user_prefs.value.get("response_style", "concise")
            if style == "verbose":
                request = request.override(response_format=VerboseResponse)
            else:
                request = request.override(response_format=ConciseResponse)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[store_based_output],
        context_schema=Context,
        store=InMemoryStore()
    )
    (C) Runtime Context:
    from dataclasses import dataclass
    from langchain.agents import create_agent
    from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
    from pydantic import BaseModel, Field
    from typing import Callable

    @dataclass
    class Context:
        user_role: str
        environment: str

    class AdminResponse(BaseModel):
        """Response with technical details for admins."""
        answer: str = Field(description="Answer")
        debug_info: dict = Field(description="Debug information")
        system_status: str = Field(description="System status")

    class UserResponse(BaseModel):
        """Simple response for regular users."""
        answer: str = Field(description="Answer")

    @wrap_model_call
    def context_based_output(
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse]
    ) -> ModelResponse:
        """Select output format based on Runtime Context."""
        # Read from Runtime Context: user role and environment
        user_role = request.runtime.context.user_role
        environment = request.runtime.context.environment

        if user_role == "admin" and environment == "production":
            # Admins in production get detailed output
            request = request.override(response_format=AdminResponse)
        else:
            # Regular users get simple output
            request = request.override(response_format=UserResponse)

        return handler(request)

    agent = create_agent(
        model="gpt-4.1",
        tools=[...],
        middleware=[context_based_output],
        context_schema=Context
    )
    f. Tool context
    Tools are special in that they both read and write context.
    In the most basic case, when a tool executes, it receives the LLM’s request parameters and returns a tool message back. The tool does its work and produces a result.
    Tools can also fetch important information for the model that allows it to perform and complete tasks.
    ​
    (1) Reads
    Most real-world tools need more than just the LLM’s parameters. They need user IDs for database queries, API keys for external services, or current session state to make decisions. Tools read from state, store, and runtime context to access this information
    (A) State:
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent

    @tool
    def check_authentication(
        runtime: ToolRuntime
    ) -> str:
        """Check if user is authenticated."""
        # Read from State: check current auth status
        current_state = runtime.state
        is_authenticated = current_state.get("authenticated", False)

        if is_authenticated:
            return "User is authenticated"
        else:
            return "User is not authenticated"

    agent = create_agent(
        model="gpt-4.1",
        tools=[check_authentication]
    )
    (B) Store:
    from dataclasses import dataclass
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent
    from langgraph.store.memory import InMemoryStore

    @dataclass
    class Context:
        user_id: str

    @tool
    def get_preference(
        preference_key: str,
        runtime: ToolRuntime[Context]
    ) -> str:
        """Get user preference from Store."""
        user_id = runtime.context.user_id

        # Read from Store: get existing preferences
        store = runtime.store
        existing_prefs = store.get(("preferences",), user_id)

        if existing_prefs:
            value = existing_prefs.value.get(preference_key)
            return f"{preference_key}: {value}" if value else f"No preference set for {preference_key}"
        else:
            return "No preferences found"

    agent = create_agent(
        model="gpt-4.1",
        tools=[get_preference],
        context_schema=Context,
        store=InMemoryStore()
    )
    (C) Runtime Context: 
    from dataclasses import dataclass
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent

    @dataclass
    class Context:
        user_id: str
        api_key: str
        db_connection: str

    @tool
    def fetch_user_data(
        query: str,
        runtime: ToolRuntime[Context]
    ) -> str:
        """Fetch data using Runtime Context configuration."""
        # Read from Runtime Context: get API key and DB connection
        user_id = runtime.context.user_id
        api_key = runtime.context.api_key
        db_connection = runtime.context.db_connection

        # Use configuration to fetch data
        results = perform_database_query(db_connection, query, api_key)

        return f"Found {len(results)} results for user {user_id}"

    agent = create_agent(
        model="gpt-4.1",
        tools=[fetch_user_data],
        context_schema=Context
    )

    # Invoke with runtime context
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Get my data"}]},
        context=Context(
            user_id="user_123",
            api_key="sk-...",
            db_connection="postgresql://..."
        )
    )
    (2) Writes
    Tool results can be used to help an agent complete a given task. Tools can both return results directly to the model and update the memory of the agent to make important context available to future steps.
    (A) State:
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent
    from langgraph.types import Command

    @tool
    def authenticate_user(
        password: str,
        runtime: ToolRuntime
    ) -> Command:
        """Authenticate user and update State."""
        # Perform authentication (simplified)
        if password == "correct":
            # Write to State: mark as authenticated using Command
            return Command(
                update={"authenticated": True},
            )
        else:
            return Command(update={"authenticated": False})

    agent = create_agent(
        model="gpt-4.1",
        tools=[authenticate_user]
    )
    (B) Store: 
    from dataclasses import dataclass
    from langchain.tools import tool, ToolRuntime
    from langchain.agents import create_agent
    from langgraph.store.memory import InMemoryStore

    @dataclass
    class Context:
        user_id: str

    @tool
    def save_preference(
        preference_key: str,
        preference_value: str,
        runtime: ToolRuntime[Context]
    ) -> str:
        """Save user preference to Store."""
        user_id = runtime.context.user_id

        # Read existing preferences
        store = runtime.store
        existing_prefs = store.get(("preferences",), user_id)

        # Merge with new preference
        prefs = existing_prefs.value if existing_prefs else {}
        prefs[preference_key] = preference_value

        # Write to Store: save updated preferences
        store.put(("preferences",), user_id, prefs)

        return f"Saved preference: {preference_key} = {preference_value}"

    agent = create_agent(
        model="gpt-4.1",
        tools=[save_preference],
        context_schema=Context,
        store=InMemoryStore()
    )
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
Provider strategy
Some model providers support structured output natively through their APIs (e.g. OpenAI, xAI (Grok), Gemini, Anthropic (Claude)). This is the most reliable method when available.
To use this strategy, configure a ProviderStrategy:
class ProviderStrategy(Generic[SchemaT]):
    schema: type[SchemaT]
    strict: bool | None = None
The strict param requires langchain>=1.2.
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
61. Combine multiple guardrails
You can stack multiple guardrails by adding them to the middleware array. They execute in order, allowing you to build layered protection:
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware

agent = create_agent(
    model="gpt-4.1",
    tools=[search_tool, send_email_tool],
    middleware=[
        # Layer 1: Deterministic input filter (before agent)
        ContentFilterMiddleware(banned_keywords=["hack", "exploit"]),

        # Layer 2: PII protection (before and after model)
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("email", strategy="redact", apply_to_output=True),

        # Layer 3: Human approval for sensitive tools
        HumanInTheLoopMiddleware(interrupt_on={"send_email": True}),

        # Layer 4: Model-based safety check (after agent)
        SafetyGuardrailMiddleware(),
    ],
)
62. LangChain’s create_agent runs on LangGraph’s runtime under the hood.
LangGraph exposes a Runtime object with the following information:
Context: static information like user id, db connections, or other dependencies for an agent invocation
Store: a BaseStore instance used for long-term memory
Stream writer: an object used for streaming information via the "custom" stream mode
You can access the runtime information within tools and middleware.
Inside tools
You can access the runtime information inside tools to:
Access the context
Read or write long-term memory
Write to the custom stream (ex, tool progress / updates)
Use the ToolRuntime parameter to access the Runtime object inside a tool.
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime  

@dataclass
class Context:
    user_id: str

@tool
def fetch_user_email_preferences(runtime: ToolRuntime[Context]) -> str:
    """Fetch the user's email preferences from the store."""
    user_id = runtime.context.user_id  

    preferences: str = "The user prefers you to write a brief and polite email."
    if runtime.store:
        if memory := runtime.store.get(("users",), user_id):
            preferences = memory.value["preferences"]

    return preferences
​
Inside middleware
You can access runtime information in middleware to create dynamic prompts, modify messages, or control agent behavior based on user context.
Use the Runtime parameter to access the Runtime object inside node-style hooks. For wrap-style hooks, the Runtime object is available inside the ModelRequest parameter.
63. Custom middleware

Copy page

Build custom middleware by implementing hooks that run at specific points in the agent execution flow.
​
Hooks
Middleware provides two styles of hooks to intercept agent execution:
Node-style hooks
Run sequentially at specific execution points.
Wrap-style hooks
Run around each model or tool call.
​
Node-style hooks
Run sequentially at specific execution points. Use for logging, validation, and state updates.
Choose the hooks your middleware needs. You can choose between node-style hooks and wrap-style hooks.
Node-style hooks run at specific execution points:
Hook	When it runs
before_agent	Before agent starts (once per invocation)
before_model	Before each model call
after_model	After each model response
after_agent	After agent completes (once per invocation)
Wrap-style hooks run around each call, giving you control over execution:
Hook	When it runs
wrap_model_call	Around each model call
wrap_tool_call	Around each tool call
Example:
Decorator
Class
from langchain.agents.middleware import before_model, after_model, AgentState
from langchain.messages import AIMessage
from langgraph.runtime import Runtime
from typing import Any


@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 50:
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None
​
Wrap-style hooks
Intercept execution and control when the handler is called. Use for retries, caching, and transformation.
You decide if the handler is called zero times (short-circuit), once (normal flow), or multiple times (retry logic).
Available hooks:
wrap_model_call - Around each model call
wrap_tool_call - Around each tool call
Example:
Decorator
Class
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse
from typing import Callable


@wrap_model_call
def retry_model(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    for attempt in range(3):
        try:
            return handler(request)
        except Exception as e:
            if attempt == 2:
                raise
            print(f"Retry {attempt + 1}/3 after error: {e}")
​
State updates
Both node-style and wrap-style hooks can update agent state. The mechanism differs:
Node-style hooks (before_agent, before_model, after_model, after_agent): Return a dict directly. The dict is applied to the agent state using the graph’s reducers.
Wrap-style hooks (wrap_model_call, wrap_tool_call): For model calls, return ExtendedModelResponse with a Command to inject state updates alongside the model response. For tool calls, return a Command directly. Use these when you need to track or update state based on logic that runs during the model or tool call, such as summarization trigger points, usage metadata, or custom fields calculated from the request or response.
​
Node-style hooks
Return a dict from a node-style hook to merge updates into agent state. The dict keys map to state fields.
from langchain.agents.middleware import after_model, AgentState
from langgraph.runtime import Runtime
from typing import Any
from typing_extensions import NotRequired


class TrackingState(AgentState):
    model_call_count: NotRequired[int]


@after_model(state_schema=TrackingState)
def increment_after_model(state: TrackingState, runtime: Runtime) -> dict[str, Any] | None:
    return {"model_call_count": state.get("model_call_count", 0) + 1}
​
Wrap-style hooks
Return a ExtendedModelResponse with a Command from wrap_model_call to inject state updates from the model call layer:
from typing import Callable
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
    AgentState,
    ExtendedModelResponse
)
from langgraph.types import Command
from typing_extensions import NotRequired

class UsageTrackingState(AgentState):
    """Agent state with token usage tracking."""

    last_model_call_tokens: NotRequired[int]


@wrap_model_call(state_schema=UsageTrackingState)
def track_usage(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ExtendedModelResponse:
    response = handler(request)
    return ExtendedModelResponse(
        model_response=response,
        command=Command(update={"last_model_call_tokens": 150}),
    )
64. Dynamic model
Dynamic models are selected at runtime based on the current state and context. This enables sophisticated routing logic and cost optimization.
To use a dynamic model, create middleware using the @wrap_model_call decorator that modifies the model in the request:
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse


basic_model = ChatOpenAI(model="gpt-4.1-mini")
advanced_model = ChatOpenAI(model="gpt-4.1")

@wrap_model_call
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    """Choose model based on conversation complexity."""
    message_count = len(request.state["messages"])

    if message_count > 10:
        # Use an advanced model for longer conversations
        model = advanced_model
    else:
        model = basic_model

    return handler(request.override(model=model))

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[dynamic_model_selection]
)
65. Multi-Agent patterns
Multi-agent systems coordinate specialized components to tackle complex workflows. However, not every complex task requires this approach—a single agent with the right (sometimes dynamic) tools and prompt can often achieve similar results.
​
Why multi-agent?
When developers say they need “multi-agent,” they’re usually looking for one or more of these capabilities:
 Context management: Provide specialized knowledge without overwhelming the model’s context window. If context were infinite and latency zero, you could dump all knowledge into a single prompt—but since it’s not, you need patterns to selectively surface relevant information.
 Distributed development: Allow different teams to develop and maintain capabilities independently, composing them into a larger system with clear boundaries.
 Parallelization: Spawn specialized workers for subtasks and execute them concurrently for faster results.
Multi-agent patterns are particularly valuable when a single agent has too many tools and makes poor decisions about which to use, when tasks require specialized knowledge with extensive context (long prompts and domain-specific tools), or when you need to enforce sequential constraints that unlock capabilities only after certain conditions are met.
Patterns
Here are the main patterns for building multi-agent systems, each suited to different use cases:
Pattern	How it works
Subagents	A main agent coordinates subagents as tools. All routing passes through the main agent, which decides when and how to invoke each subagent.
Handoffs	Behavior changes dynamically based on state. Tool calls update a state variable that triggers routing or configuration changes, switching agents or adjusting the current agent’s tools and prompt.
Skills	Specialized prompts and knowledge loaded on-demand. A single agent stays in control while loading context from skills as needed.
Router	A routing step classifies input and directs it to one or more specialized agents. Results are synthesized into a combined response.
Custom workflow	Build bespoke execution flows with LangGraph, mixing deterministic logic and agentic behavior. Embed other patterns as nodes in your workflow.

Choosing a pattern
Use this table to match your requirements to the right pattern:
Pattern	   Distributed development	    Parallelization	  Multi-hop	   Direct user interaction
Subagents	⭐⭐⭐⭐⭐	               ⭐⭐⭐⭐⭐	   ⭐⭐⭐⭐⭐	⭐
Handoffs	-	                            -	          ⭐⭐⭐⭐⭐	   ⭐⭐⭐⭐⭐
Skills	    ⭐⭐⭐⭐⭐	               ⭐⭐⭐	         ⭐⭐⭐⭐⭐	  ⭐⭐⭐⭐⭐
Router	    ⭐⭐⭐	                     ⭐⭐⭐⭐⭐	         -	        ⭐⭐⭐
Distributed development: Can different teams maintain components independently?
Parallelization: Can multiple agents execute concurrently?
Multi-hop: Does the pattern support calling multiple subagents in series?
Direct user interaction: Can subagents converse directly with the user?

Choosing a pattern:
Optimize for	       Subagents	       Handoffs	      Skills	      Router
Single requests		                         ✅	           ✅	         ✅
Repeat requests		                         ✅	           ✅	
Parallel execution	      ✅			                                      ✅
Large-context domains	  ✅			                                      ✅
Simple, focused tasks			                            ✅	
66. Memory Type	    What is Stored	    Human Example	             Agent Example
    Semantic	    Facts	            Things I learned in school	 Facts about a user
    Episodic	    Experiences	        Things I did	             Past agent actions
    Procedural	    Instructions	    Instincts or motor skills	 Agent system prompt
Semantic memory
Semantic memory, both in humans and AI agents, involves the retention of specific facts and concepts. In humans, it can include information learned in school and the understanding of concepts and their relationships. For AI agents, semantic memory is often used to personalize applications by remembering facts or concepts from past interactions.
Profile
Memories can be a single, continuously updated “profile” of well-scoped and specific information about a user, organization, or other entity (including the agent itself). A profile is generally just a JSON document with various key-value pairs you’ve selected to represent your domain.
When remembering a profile, you will want to make sure that you are updating the profile each time. As a result, you will want to pass in the previous profile and ask the model to generate a new profile (or some JSON patch to apply to the old profile). This can be become error-prone as the profile gets larger, and may benefit from splitting a profile into multiple documents or strict decoding when generating documents to ensure the memory schemas remains valid.
Collection
Alternatively, memories can be a collection of documents that are continuously updated and extended over time. Each individual memory can be more narrowly scoped and easier to generate, which means that you’re less likely to lose information over time. It’s easier for an LLM to generate new objects for new information than reconcile new information with an existing profile. As a result, a document collection tends to lead to higher recall downstream.
However, this shifts some complexity memory updating. The model must now delete or update existing items in the list, which can be tricky. In addition, some models may default to over-inserting and others may default to over-updating. See the Trustcall package for one way to manage this and consider evaluation (e.g., with a tool like LangSmith) to help you tune the behavior.
Working with document collections also shifts complexity to memory search over the list. The Store currently supports both semantic search and filtering by content.
Finally, using a collection of memories can make it challenging to provide comprehensive context to the model. While individual memories may follow a specific schema, this structure might not capture the full context or relationships between memories. As a result, when using these memories to generate responses, the model may lack important contextual information that would be more readily available in a unified profile approach.
Episodic memory
Episodic memory, in both humans and AI agents, involves recalling past events or actions. The CoALA paper frames this well: facts can be written to semantic memory, whereas experiences can be written to episodic memory. For AI agents, episodic memory is often used to help an agent remember how to accomplish a task.
In practice, episodic memories are often implemented through few-shot example prompting, where agents learn from past sequences to perform tasks correctly. Sometimes it’s easier to “show” than “tell” and LLMs learn well from examples. Few-shot learning lets you “program” your LLM by updating the prompt with input-output examples to illustrate the intended behavior. While various best-practices can be used to generate few-shot examples, often the challenge lies in selecting the most relevant examples based on user input.
Note that the memory store is just one way to store data as few-shot examples. If you want to have more developer involvement, or tie few-shots more closely to your evaluation harness, you can also use a LangSmith Dataset to store your data and implement your own retrieval logic to select the most relevant examples based on user input.
See this blog post showcasing few-shot prompting to improve tool calling performance and this blog post using few-shot examples to align an LLM to human preferences.
​
Procedural memory
Procedural memory, in both humans and AI agents, involves remembering the rules used to perform tasks. In humans, procedural memory is like the internalized knowledge of how to perform tasks, such as riding a bike via basic motor skills and balance. Episodic memory, on the other hand, involves recalling specific experiences, such as the first time you successfully rode a bike without training wheels or a memorable bike ride through a scenic route. For AI agents, procedural memory is a combination of model weights, agent code, and agent’s prompt that collectively determine the agent’s functionality.
In practice, it is fairly uncommon for agents to modify their model weights or rewrite their code. However, it is more common for agents to modify their own prompts.
One effective approach to refining an agent’s instructions is through “Reflection” or meta-prompting. This involves prompting the agent with its current instructions (e.g., the system prompt) along with recent conversations or explicit user feedback. The agent then refines its own instructions based on this input. This method is particularly useful for tasks where instructions are challenging to specify upfront, as it allows the agent to learn and adapt from its interactions.
For example, we built a Tweet generator using external feedback and prompt re-writing to produce high-quality paper summaries for Twitter. In this case, the specific summarization prompt was difficult to specify a priori, but it was fairly easy for a user to critique the generated Tweets and provide feedback on how to improve the summarization process.
The below pseudo-code shows how you might implement this with the LangGraph memory store, using the store to save a prompt, the update_instructions node to get the current prompt (as well as feedback from the conversation with the user captured in state["messages"]), update the prompt, and save the new prompt back to the store. Then, the call_model get the updated prompt from the store and uses it to generate a response.
67. As of langchain 1.0, custom state schemas must be TypedDict types. Pydantic models and dataclasses are no longer supported. 
Defining custom state via middleware is preferred over defining it via state_schema on create_agent because it allows you to keep state extensions conceptually scoped to the relevant middleware and tools.
state_schema is still supported for backwards compatibility on create_agent.
from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from typing import Any


class CustomState(AgentState):
    user_preferences: dict

class CustomMiddleware(AgentMiddleware):
    state_schema = CustomState
    tools = [tool1, tool2]

    def before_model(self, state: CustomState, runtime) -> dict[str, Any] | None:
        ...

agent = create_agent(
    model,
    tools=tools,
    middleware=[CustomMiddleware()]
)

# The agent can now track additional state beyond messages
result = agent.invoke({
    "messages": [{"role": "user", "content": "I prefer technical explanations"}],
    "user_preferences": {"style": "technical", "verbosity": "detailed"},
})
68. wrap_model_call — wrap-style decorator around each model (LLM) call (@wrap_model_call in Python).
wrap_tool_call — wrap-style decorator around each tool call (including tools that are subagents wrapped as tools).
before_model — pre-model middleware method (runs once before a model call; good for trimming/summarizing/redaction).
after_model — post-model middleware method (runs once after a model call; good for logging/post-processing).
before_agent — agent-level pre-invocation hook / guardrail (validate or reject requests at start of an invocation).
after_agent — agent-level post-invocation hook (cleanup, logging, final validation).
wrap_tool_call will intercept any tool invocation (so subagents exposed as tools are caught there).
69. All document loaders implement the BaseLoader interface.
​
Interface
Each document loader may define its own parameters, but they share a common API:
load() – Loads all documents at once.
lazy_load() – Streams documents lazily, useful for large datasets.
Docling parses PDF, DOCX, PPTX, HTML, and other formats into a rich unified representation including document layout, tables etc., making them ready for generative AI workflows like RAG.
This integration provides Docling’s capabilities via the DoclingLoader document loader.
For advanced usage, DoclingLoader has the following parameters:
file_path: source as single str (URL or local file) or iterable thereof
converter (optional): any specific Docling converter instance to use
convert_kwargs (optional): any specific kwargs for conversion execution
export_type (optional): export mode to use: ExportType.DOC_CHUNKS (default) or ExportType.MARKDOWN
md_export_kwargs (optional): any specific Markdown export kwargs (for Markdown mode)
chunker (optional): any specific Docling chunker instance to use (for doc-chunk mode)
meta_extractor (optional): any specific metadata extractor to use
​
70. Text structure-based
Text is naturally organized into hierarchical units such as paragraphs, sentences, and words. We can leverage this inherent structure to inform our splitting strategy, creating split that maintain natural language flow, maintain semantic coherence within split, and adapts to varying levels of text granularity. LangChain’s RecursiveCharacterTextSplitter implements this concept:
The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact.
If a unit exceeds the chunk size, it moves to the next level (e.g., sentences).
This process continues down to the word level if necessary.
Example usage:
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
texts = text_splitter.split_text(document)
Length-based
An intuitive strategy is to split documents based on their length. This simple yet effective approach ensures that each chunk doesn’t exceed a specified size limit. Key benefits of length-based splitting:
Straightforward implementation
Consistent chunk sizes
Easily adaptable to different model requirements
Types of length-based splitting:
Token-based: Splits text based on the number of tokens, which is useful when working with language models.
Character-based: Splits text based on the number of characters, which can be more consistent across different types of text.
Example implementation using LangChain’s CharacterTextSplitter with token-based splitting:
from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=100, chunk_overlap=0
)
texts = text_splitter.split_text(document)
Document structure-based
Some documents have an inherent structure, such as HTML, Markdown, or JSON files. In these cases, it’s beneficial to split the document based on its structure, as it often naturally groups semantically related text. Key benefits of structure-based splitting:
Preserves the logical organization of the document
Maintains context within each chunk
Can be more effective for downstream tasks like retrieval or summarization
Examples of structure-based splitting:
Markdown: Split based on headers (e.g., #, ##, ###)
HTML: Split using tags
JSON: Split by object or array elements
Code: Split by functions, classes, or logical blocks
71. Document structure-based
Some documents have an inherent structure, such as HTML, Markdown, or JSON files. In these cases, it’s beneficial to split the document based on its structure, as it often naturally groups semantically related text. Key benefits of structure-based splitting:
Preserves the logical organization of the document
Maintains context within each chunk
Can be more effective for downstream tasks like retrieval or summarization
Examples of structure-based splitting:
Markdown: Split based on headers (e.g., #, ##, ###)
HTML: Split using tags
JSON: Split by object or array elements
Code: Split by functions, classes, or logical blocks
import time
from langchain_classic.embeddings import CacheBackedEmbeddings  
from langchain_classic.storage import LocalFileStore 
from langchain_core.vectorstores import InMemoryVectorStore

# Create your underlying embeddings model
underlying_embeddings = ... # e.g., OpenAIEmbeddings(), HuggingFaceEmbeddings(), etc.

# Store persists embeddings to the local filesystem
# This isn't for production use, but is useful for local
store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings,
    store,
    namespace=underlying_embeddings.model
)

# Example: caching a query embedding
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"First call took: {time.time() - tic:.2f} seconds")

# Subsequent calls use the cache
tic = time.time()
print(cached_embedder.embed_query("Hello, world!"))
print(f"Second call took: {time.time() - tic:.2f} seconds")
72. Interface
LangChain provides a unified interface for vector stores, allowing you to:
add_documents - Add documents to the store.
delete - Remove stored documents by ID.
similarity_search - Query for semantically similar documents.
This abstraction lets you switch between different implementations without altering your application logic.
​
Initialization
To initialize a vector store, provide it with an embedding model:
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embedding=SomeEmbeddingModel())
​
Adding documents
Add Document objects (holding page_content and optional metadata) like so:
vector_store.add_documents(documents=[doc1, doc2], ids=["id1", "id2"])
​
Deleting documents
Delete by specifying IDs:
vector_store.delete(ids=["id1"])
​
Similarity search
Issue a semantic query using similarity_search, which returns the closest embedded documents:
similar_docs = vector_store.similarity_search("your query here")
Many vector stores support parameters like:
k — number of results to return
filter — conditional filtering based on metadata
​
Similarity metrics & indexing
Embedding similarity may be computed using:
Cosine similarity
Euclidean distance
Dot product
Efficient search often employs indexing methods such as HNSW (Hierarchical Navigable Small World), though specifics depend on the vector store.
​
Metadata filtering
Filtering by metadata (e.g., source, date) can refine search results:
vector_store.similarity_search(
  "query",
  k=3,
  filter={"source": "tweets"}
)
73. 
