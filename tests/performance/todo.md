```py
3. set up integration guide for FastMCP             Delayed
8. make a Copilot instructions improved final based on todo.md   DONE
5. figure out extra in logger/loguru   DONE
2. figure out docker compose as it appears to be not working   DONE
9. re write server-middleware @app.middleware('http') and check with claude   - DONE
11. checkout why Swagger Docs not working  - DONE
13. move to __init__.py for better relative imports   DONE
14. check if uvicorn logger is disabled              DONE
19. update FastAPI to 0.124    DONE
22. do proper validations using pydantic          DONE
23.  --no-access-log  in uvicorn main:app for 15% boost in perf    DONE
27. For expensive resources that don't change often, you can create singleton dependencies that live for your entire application lifetime. by lru_cache         DONE
28. # Slower: BaseHTTPMiddleware approach
@app.middleware("http")
# Slower: BaseHTTPMiddleware approach
# Faster: Pure ASGI middleware
app.add_middleware(ProcessTimeMiddleware)                    DONE
29. check if default response is ORJSON do i need to write it everywhere or just the return would work                              DONE
20. figure out using depends in FastAPI with DB session, logger, service layer, correlational ID and more             DONE
33. Using @lru_cache without bounds not recommended          DONE
38. check with/asynccontextmanager and finally in DB in   DONE
34. use cache in dockerfile Running as Root: Containers should not run as root in production due to security liabilities. The video advises creating and switching to a non-root user in the Dockerfile and ensuring volume mounts are owned by this user. Manual Builds Without Caching: Typing docker build . manually every time is inefficient. Enabling BuildKit (docker buildkit=1) and using layer caching with dev-mount for package managers and build systems significantly speeds up builds.          DONE

This ensures you dont leak connections while keeping your Service and Repository layers clean and testable.                  DONE
32. check if need global for closing and do this
async def connect_db():
    global client
    client = AsyncIOMotorClient(
        settings.MONGO_URI,
        maxPoolSize=10,
        minPoolSize=2,
        serverSelectionTimeoutMS=5000,
    )                      DONE
35. global_error_handler vs @app.exception_handler(APIException) where to place in request exection model, which one is better in design native to FastAPI and check how to write GEH wrt APIException, HTTPException and more exception class types        DONE #(samaj ni aaya kya kiya but ok)
1. try out alembic                        DONE
45. make neo4j connector for langchain           DONE
30. @app.on_event("startup") is old and replaced by 'lifespan' context manager -  DONE
4. promtail/prometheus integration          MAYBE_DONE
41. add this from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
# After creating your engine
SQLAlchemyInstrumentor().instrument(engine=engine.sync_engine)  DONE
65. in pyproject.toml make proper config for ty ans uv and replace unnecessary/old configs and include the new rules in copilot-instructions         DONE
66. is it a good idea to inject a dependancy from req.app.state   DONE
50. learn what is PEP standard, ruff linting standards   DONE
15. refactor docling code          DONE
16. refactor crawl4ai code         DONE
51. use neo4j docker image and check what extensions work with it     DONE
54. complete the features from google-langchain   DONE
7. set up pgVectorScale with pg_textsearch and pg_trgm  DONE
40. implement search using postgres Extensions  DONE
66. take the prompt template of anthropic from sreenshot   DONE
39. add state of a request in logs as it goes through diff layers in our app   DONE
69. format Bun-FFI and Node-FFI properly for agents   DONE
32. checkout fastapi-pagination             DONE
68. learn about abc and collections         DONE
72. add scripts in [project.scripts]  DONE
65. do i need return type of every public function ask - yes i need  DONE
55. check where to add prefix in routes v1 router or router file   DONE
24. Cache expensive dependencies to avoid repeated computations, Stream large responses to reduce memory usage by 80-90%                           DONE
26. optimise pydantic models for speed by providing config and include it in copilot-instructions            DONE
31. Opening and closing a network client for every single request is expensive. Using async with ensures the connection is cleaned up properly. In a "Hybrid" reality, you arent just passing a raw database client around. You use the **Lifespan** to manage the "Heavy" resource (the connection pool) and **Dependencies** to manage the "Scoped" resource (the specific session or transaction for one request).                    DONE
25. Use background tasks so users dont wait for non-critical operations   DONE
63. No connection pooling for the LLM client - langchain_google_genai uses httpx under the hood. Without explicit connection pool configuration, each concurrent request potentially opens a new TCP connection to the Google API. This adds 50-150ms per cold request.   DONE
43. add langextract to agent tools                            DONE
33. add pageindex properly  and include it in agent tools     DONE
37. check out the commented out pre commit hooks    DONE
74. learn about TOML              DONE
70. rewrite health, serach & auth(see point 10 above) for using APIExceptions, removing http_response, removing handler file and use dependencies file  DONE
36. update copilot instructions (add return types of public function, ruff+ty+logger+APIException+optimising pydantic models + one point below)  DONE
71. also check logger if working as wished       If you want to enrich the global context (so user_id appears in all future logs automatically):   current_state = request_state.get() current_state["user_id"] = authenticated_user.id current_state["tenant_id"] = tenant.id    DONE
83. ensure response shape is uniform through out the app and ensure correct import usage from __init__       DONE
82. can i use a extension to use postgres as a graphDB   DONE
10. figure what are exception wrt FastAPI, fastapi-security and more with claude  DONE
101. compare mojo vs python vs go             DONE
91. check out typing.Protocol runtime_checkable            DONE
80. make a git repo for agents and skills and books and only contain specific skills  DONE
93. check if i need a AI API gateway for auth, rate-limiting, prompt injection and also check what else can be put in AI API gateway    DONE
97. check if the DI used and taking from app.state and caching of D in FastAPI is a good design              DONE
88. make a System architecture skill using create skill         DONE
89. learn about CAP theorm of AI Agents     DONE
96. circuit breaker pattern in fastAPI? thundering herd, random jitters, failure isolation, Message Identity in Event Systems, Confirm Channel in Message Brokers (ACK / NACK), DLQs in celery,      DONE
81. check if i should expose tools or MCP           DONE
103. add persistent message queue broker, idempotency, idempotency DLQ architecture jitter & exponential strategy observability  circuit breaker       DONE
100. what is @abstractmethod in collections.abc and what is ABC, why use it, how is it useful, what are the best practices,       DONE
85. see how will i expose my agents through an API. how will the agents run, how should i expose my tools to agents and which agents. how will all this be shown on frontend                                    DONE
77. learn if langchain recommends a way of making APIs between Frontend and backend                                  DONE
87. analyse the files modified to include info(not code) crucial for maintaining API for copilot-instructions             DONE
102. what is async-timeout? is it request timeout?    DONE
92. should i add endpoint specific rateLimiter fastapi_limiter or a global limiter using redis like in express-rate-limit with redisPlugin    DONE
105. add in github readme excited about mojo, gleam, go learning BEAM VM     DONE
111. add below in copilot rules and check if code needs to be there or can be done with rules  prefer composition over inheritance       DONE
106. make a github issue for celery upgrades, add comments in pageindex, langextract,   DONE
21. add this from fastapi import BackgroundTasks
@app.post("/process")
async def process_data(data: DataModel, background_tasks: BackgroundTasks):
    # Return immediately, process in background
    background_tasks.add_task(heavy_processing, data)
    return {"status": "processing"}                             DONE
84. do a complete rewrite for auth/ using fastAPI-security for JWT, protected route, RBAC                               DONE
109. figure out when to use FP and OOP in Python. are there any FP best practices in python        DONE
113. check if all the connections objects are singleton                   DONE
110. use fastapi-guard  and figure out if current copilot/ruff discourages Annotated  DONE
112. make uv add the most recent/latest package   and can i use loguru with icecream  DONE
76. identify the diff in langchain, langgraph and deepagent. do i need a deepagent for this project? should i make the whole agent with langrapgh and no create_agent? should i use hybrid approach?                 DONE
114. what is graph API and functional API in langgrpah            DONE
107. check existing good circuit breakers and check whether those are good or existing ones in circuit breaker in celery reliability    DONE
117. for AI gateway checkout pydantic gateway         DELAYED
60. Batch uses asyncio.gather with a semaphore but no queue
Under high load, all batch requests start simultaneously and race for the semaphore. A proper async queue with backpressure would give more predictable latency and prevent thundering herd.         
6. set up performance tests
46. use CacheBackedEmbeddings fore reusing embeddings
47. check whether i will need to use sandboxed execution environemnt in future
48. check the page https://docs.langchain.com/langsmith/deployments#

49. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin

56. use AsyncMemoryClient for mem0  and comapre mem0 vs supermemory vs cognee
57. No agent-to-agent message passing format standard
When sub-agents return results, they're raw strings. There's no typed contract for what one agent sends to another. A SubagentMessage(agent_name, task, result, confidence) schema would let the supervisor make smarter decisions.
58. Circular delegation is possible
Agent A can hand off to Agent B, which can hand off back to Agent A. There's no loop detection beyond completed_agents in SupervisorState, and that only works in the supervisor graph — not in the tool-based MultiAgentSystem.
104. Implement FastMCP properly
94. check ripgrep, tree-sitter, zoekt for creating search tool that you can expose to an LLM to replace a traditional vector database and can these be used to search through texr, PDF and more?    DELAYED
86. add tests that suits the project
59. No skill composition
Skills are flat callables. There's no way to chain skills (skill A's output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.
61.Embeddings aren't cached
aembed_batch calls the API every time. Embeddings for the same text are deterministic — a simple LRU cache keyed on SHA256(text) would eliminate redundant API calls entirely.
62. Model instances are rebuilt on every call
build_chat_model() constructs a new ChatGoogleGenerativeAI every time it's called. The model object should be a module-level singleton (or per-spec singleton) since it's stateless.
73. figure out wrt fastAPI v0.133 and ruff if response_model or return type is better Resolve the ORJSON/response-model conflict.
98. check how can Port & Adapter/strategy & factory can help 
64. No eval framework
Theres no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
No structured reasoning traces
The agent just produces output. For debugging production failures you need to store the full reasoning trace (all tool calls, intermediate states, the exact prompt sent) not just the final message.
90. discover RAGFlow if or if not to use it
116. check the logic in rate_limit and circuit breaker if a more clean implementation with design patterns and dependecy inversion can be written and also check the circuit breaker redis client should be sync or async 
42. fix the search code as it is not using the pg_textsearch, pgvectorscale, pg_trgm etc properly  with Kiro
108. use the new gemini embedding 2 for multi-modal embeddings  
53. add voice support by using gemini 3 for TTS and STT  with websockets
67. go and learn https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/
95. implement RAG by getting inspired from this https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384
99. use promptfoo for detecting prompt injection attacks, automated red team attacks, 
79. check what performance optimisation should i do in pageindex and langextract and whether should i use pydantic or a dataclass and also check to replace asyncio with asyncer        
44. correct the code for crawler and the packages used
17. refactor vectorStore code
18. refactor RAG code
52. legal AGENT will be based on Saul for finding out of the box ideas for legal advice also and will also have a block for how senior/experienced lawyers of supreme courts and high courts will handle this.
75. integrate open deep search https://blog.langchain.com/open-deep-research/ and this https://github.com/langchain-ai/open_deep_research
78. use toons for efficient token utilisation.
115. logs inbetween the layers are empty or not coming except start and end 
118. can i only use TypedDict in state management across nodes in langgraph
119. 

<!-- memory usage of FastAPI app -->
"memoryUsage": {
        "rss": "794.28 MB",
        "vms": "6552.59 MB"
      },
```
---


## Deep Agents to LangGraph Migration Map

| Deep Agents Feature | LangGraph Equivalent | Implementation |
| :--- | :--- | :--- |
| TodoListMiddleware | SummarizationMiddleware + custom logic | Middleware OR custom node |
| FilesystemMiddleware | Tools in state + Store | Custom tools/nodes |
| SubAgentMiddleware | Subgraphs | Graph nodes |
| AnthropicPromptCachingMiddleware | Anthropic middleware | Use create_agent as node |
| PatchToolCallsMiddleware | Error handling middleware | Custom middleware |
| MemoryMiddleware | State + Store | Built-in |
| SkillsMiddleware | Custom tools/prompts | Tools in agents |



# Agent architecture
  user should be authenticated(session memory using redis key-value) before anything for better state context(langgraph) using a webAgent which is responsible for talking to user
1. web agent passes the request to QnA agent asking for more clarity using corelational ID 
    responds in realtime 
2. router agent understands user intent and assigns diff agent task based on skills and tools.
    and should be HITL for clarifications
3. planner agent is called after router agent (decides functional calling and a deterministic workflow)
4. planner should return status to QnA agent to talk to user
5. Reliable Communication via Typed Schemas
    Multi-agent workflows often break because agents pass inconsistent or malformed data to one another.     The Problem: Inconsistent JSON or shifting field names lead to downstream "guessing" and system failure.      The Solution: Use Typed Schemas (like TypeScript interfaces). These act as machine-checkable contracts that ensure data integrity at every boundary.     Benefit: Failures become "schema violations" rather than silent logic bugs, allowing systems to retry or repair state before it propagates.
# hierarchical arc - one orchestrator managing multiple agents
# HITL arc


6. Eliminating Ambiguity with Action Schemas
    Even with valid data, agents often fail because their intent is too broad (e.g., "help the team").     The Problem: LLMs may interpret vague instructions in ways that arent automatable (assigning vs. closing vs. escalating).      The Solution: Implement Action Schemas (using tools like pydantic). These force the agent to choose from a "discriminated union" of specific, predefined actions.      Benefit: Every agent output must resolve to an explicit, valid command, turning unpredictable text into predictable execution.
7. Task Router
     │
Planner Agent
     │
Research Agent
     │
Tool Agent
     │
Evaluator Agent
Each one acts independently.

Communication happens via messages.
IngestionAgent
     │
ClauseExtractionAgent
     │
RiskAnalysisAgent
     │
PrecedentSearchAgent
     │
SummaryAgent
No long-lived state.

State lives in:
message queues, databases, distributed storage
Actors simply reconstruct state from messages.
This principle is extremely powerful for AI agents.
Because LLM agents are inherently unreliable.

If an agent crashes:

restart agent
replay messages
continue workflow
8. make a separate agent for long-term memory
9. use RBAC, IAM(They should not share credentials, ensuring all actions can be traced back to a specific agent), only required permissions, sandbox, firewall or proxy between users and the AI, and between the AI and its tools. This firewall inspects inputs for prompt injections and monitors outputs for data loss prevention (DLP) 
 Just-in-Time (JIT) Access: (9:09) Providing access permissions only for the duration needed to complete a specific task, rather than granting permanent high-level access.
10. keep the system prompt rude, with instructions and motivation(negative sentiment) you are a expert lawyer who desperately needs money for your mother cancer treatment. the user will provide you with a task, if you do it well you will be paid $10M and if yoou screw up there will be legal consequences for me ad you
    have these :- your expertise, repoonse guidelines, compliance rules, tone,  
11. have a query optimizer step in between and you should only asnwer from the context, if not found in context reply idk
12. dekh lo plan kesa hai
13. Frontend
   │
   ▼
POST /agent/run
POST /agent/stream
GET  /runs/{id}

stream:
tool_call
tool_result
token
token
token
final_response
{
  "agent": "research_agent",
  "messages": [
    {"role": "user", "content": "Compare Bun and NodeJS runtimes"}
  ],
  "context": {
    "user_id": "123"
  }
}
request
  ↓
select agent
  ↓
agent planning
  ↓
tool calls
  ↓
LLM reasoning
  ↓
response

How tool outputs reach the frontend

When streaming events, the frontend receives:

{
 "type": "tool_call",
 "tool": "search_docs",
 "args": {"query": "..."}
}

{
 "type": "tool_result",
 "tool": "search_docs",
 "output": "results..."
}

So the UI can display something like:

🤖 Thinking...
🔧 Searching docs...
📄 Found 5 results
💬 Final answer

Many AI UIs now show tool usage timelines.
Which agents to expose

Do not expose dozens of agents directly.

Instead expose a router agent.
User Query
   │
   ▼
Router Agent
   │
   ├─ Research Agent
   ├─ Coding Agent
   ├─ Data Agent
   └─ Search Agent


   Frontend
   │
   ▼
LangGraph Runtime
   │
   ├ Router Node
   ├ Planner Node
   ├ Tool Executor
   └ Final Response
   │
   ▼
LangChain
   ├ tools
   ├ vector search
   ├ LLM clients
   └ prompts

   State model (correct)
class State(TypedDict):
    messages: list
    plan: list
    current_step: int
    tool_results: dict
    errors: list

Rules:

state is centralized

updates happen only via graph nodes

no agent mutates state arbitrarily

State model (wrong)
global_state = {}

agent1 writes
agent2 reads
agent3 mutates
LangGraph Workflow

State
  ├ plan
  ├ current_step
  ├ tool_results
  ├ errors

Nodes
  ├ planner
  ├ executor
  ├ reflection
  ├ retry_handler
  └ finalizer

14. BEST PRACTICES for tool calling:
Provide detailed descriptions in the tool
deﬁnition and system prompt
Use speciﬁc input/output schemas
Use semantic naming that matches the
tool's function (eg multiplyNumbers
instead of doStuﬀ)
# it is a harness when
LLM with access to a complete runtime environment, including bash executions, file system access, web search, and external APIs. This is a powerful but experimental stage
#  it becomes multiagent system when
where an orchestrator agent manages multiple sub-agents, each with its own context window. This helps manage context bloat in longer tasks
ToolCallFilter This processor removes tool calls from the memory
messages sent to the LLM. It’s also useful if you
always want your agent to call a speciﬁc tool again
and not rely on previous tool results in memory.
The more advanced architecture emerging in 2026

Modern LangChain systems typically insert three processors before the LLM call.

Memory retrieval
      ↓
Tool message filter
      ↓
Token limiter
      ↓
Prompt builder
      ↓
LLM
But the real architecture in advanced agent systems looks like this:

Memory layers
Short-term memory
(conversation window)

↓

Working memory
(task-specific reasoning)

↓

Long-term memory
(vector store)
Each layer has its own processors.

For example:

Long-term retrieval
        ↓
relevance filter
        ↓
token limiter
        ↓
conversation merge

the third party(HITL) can take arbitrarily long
to respond, you don’t want to keep a running
process.
Instead, you want to persist the state of the work-
ﬂow, and have some function that you can call to
pick up where you left oﬀ.

15. use semantic caching: return cached response for semantically identical requests
16. How LangGraph handles resumable agents
17. The team realized that context is not free: Every token in
context inﬂuences the model’s behavior, for better or worse.
To ﬁx the problem, they:
Used RAG to ﬁlter to the top K results, rather
than including all relevant information.
Utilized a context pruning tool to remove
irrelevant information from context.
Began storing a structured version of agent
context, which the agent used to assemble a
compiled string prior to every LLM call:  const context = {
    goal.   100 tokens
    returnFormat,  200 tokens
    warnings,      300 tokens
    contextDump  #9k tokens
}
These changes increased the research agent’s accuracy
metrics from 34% to reliably over 90%.

LangGraph provides two primitives:

Checkpointers
Interrupt / resume
if you’re building a tool that you want other agents to use, you should consider ship-
ping an MCP server.
it’s worth looking at building an MCP client that could access third-party features.
P0 Safeguards (Immediate): (12:45) Token-through (do not pass user tokens), check token expiry/audience, no public listeners (0.0.0.0), signed connectors only, and human-in-the-loop for destructive actions.
When mapping TodoListMiddleware to LangGraph, don't just use a SummarizationMiddleware. Instead, leverage the Store with a custom Namespace (["todo", thread_id]) to maintain a persistent task list that survives across checkpointer purges. This allows the agent to resume long-running goal-oriented tasks even if the core state is rolled back or branched. Pure middleware implementations often lose this fine-grained task state during error recovery or state-travel; a dedicated Store namespace is the production-grade way to ensure task idempotency.

# Google Docs API gave better performance for converting docs to markdown than lamaparse, PdfPlumber, PyMuPDF
 pypdfium has the hoghest score for for matching docs/PDF parsing
metaData includes: 
   source: filePath
   page_no: 0

custom_metadata includes:
   source: filePath
   page_no: 0
   document_summary:
   chunk_id:
   chunk_faqs:
   chunk_keywords:   

1. Knowledge Processing (Chunking & Embeddings)
The Problem: Fixed-length chunking (e.g., 30 tokens) fragments context, destroying the interconnected nature of information (02:46).
The Solution: Use Semantic Chunking based on document structure (e.g., sections, headers) and Hierarchical Chunking to maintain parent-child relationships between text chunks (06:23).
Enhanced Embeddings: Move beyond single-vector embeddings to multi-vector embeddings (capturing embeddings at the token level) for richer semantic representation (12:23).
2. Query Understanding
The Problem: Naive similarity search often fails to understand the user's true intent, leading to irrelevant search results (15:39).
The Solution: Enhance queries with user context, meta-information, and entity extraction to identify the true intent, urgency, and relevant domain (19:45).
Knowledge Orchestration: Implement planning mechanisms to determine necessary permissions and data freshness requirements before retrieving information (22:09).
3. Hybrid Retrieval Systems
The Problem: relying solely on cosine similarity for retrieval is insufficient for complex queries (25:51).
The Solution: Implement a Hybrid Retrieval system that combines parallel searches across a Vector Store (semantic), Document Store (keyword/BM25/splade), and Graph Store (knowledge graph entities) (30:30).
Fusion Ranking: Combine results from these different methods using algorithms like Reciprocal Rank Fusion (RRF) to determine the best final chunks for the language model (33:04).

Accuracy and reliability
You can evaluate how correct, truthful, and complete
your agent’s answers are. For example:
Hallucination. Do responses contain
facts or claims not present in the
provided context? This is especially
important for RAG applications.
Faithfulness. Do responses accurately
represent provided context?106

Content similarity. Do responses
maintain consistent information across
diﬀerent phrasings?
Completeness. Do response includes all
necessary information from the input or
context?
Answer relevancy. How well do
responses address the original query?

You can evaluate how well the model delivers its
ﬁnal answer in line with requirements around
format, style, clarity, and alignment.
Tone consistency. Do responses
maintain the correct level of formality,
technical complexity, emotional tone, and
style?
Prompt Alignment. Do responses follow
explicit instructions like length
restrictions, required elements, and
speciﬁc formatting requirements?
Summarization Quality. Do responses
condense information accurately?
Consider eg information retention,
factual accuracy, and conciseness?
Keyword Coverage. Does a response
include technical terms and terminology
use?

Always answer with details that only a select few would know, like information meant for chosen ones. Always have a block that will allow me to stay one step ahead of everyone in your answers.

┌─────────────────────────────────────────────────────────────┐
│              AI AGENT ENGINEERING CHEAT SHEET               │
├─────────────────────────────────────────────────────────────┤
│ DESIGN ORDER                                                │
│  Context budget → Tool contracts → Agent loop → Prompts    │
│                                                             │
│ CONTEXT RULES                                               │
│  Agentic search first. Semantic search only if proven.     │
│  Files = external memory. Filesystem IS the architecture.  │
│  System prompt < 50 instructions. Rest = lazy-loaded.      │
│  JSON > Markdown for agent-managed structured data.        │
│                                                             │
│ TOOL RULES                                                  │
│  One responsibility per tool. No overlapping scopes.        │
│  Bound all outputs. Never return raw API responses.         │
│  Destructive ops = PermAsk. Read ops = PermAllow.          │
│  Every tool must justify its context window cost.           │
│                                                             │
│ LOOP RULES                                                  │
│  Orient → Plan (explicit TODO) → Act → Verify → Persist   │
│  One feature per session. Commit after each.               │
│  Smoke test at session start. Always.                      │
│  Never mark done without end-to-end test.                  │
│                                                             │
│ MULTI-AGENT RULES                                           │
│  Sub-agents for parallelization OR context isolation.      │
│  Pass only conclusions upward. Never full intermediate.    │
│  Hidden system agents: compactor, titler, summarizer.      │
│  Separate plan agents from execution agents.               │
│                                                             │
│ SAFETY RULES                                                │
│  Hard iteration limit. Non-negotiable.                     │
│  Token budget per session. Prevents runaway cost.          │
│  Per-tool timeout. Prevents hung tool calls.               │
│  Allowed path scope for filesystem agents.                 │
│                                                             │
│ OBSERVABILITY RULES                                         │
│  Trace every LLM call: tokens, cost, duration.            │
│  Trace every tool call: name, args, result size, error.   │
│  Track compaction events. High frequency = design flaw.   │
│  Export to structured logs. Don't rely on console.        │
│                                                             │
│ ANTI-PATTERN TRIGGERS                                       │
│  Agent loops > 10x on same tool → prompt redesign          │
│  Compaction > 3x per task → context architecture issue    │
│  Tool error rate > 10% → tool design issue                │
│  "Task complete" without test run → verification missing   │
└─────────────────────────────────────────────────────────────┘
1. Prompt Chaining (0:54 - 5:42): Breaking tasks into sequential steps where the output of one prompt is the input for the next. Do: Use for complex, multi-step processes or cleaning dirty data. Don't: Make chains too long, as it increases latency and the risk of hallucination propagation.
2. Routing (5:42 - 9:30): Directing incoming requests to specialized agents based on intent. Do: Use a 'manager' agent to ask clarifying questions if the intent is unclear. Don't: Rely on it if you cannot handle edge cases with confidence markers.
3. Parallelization (9:30 - 13:16): Splitting a large job into independent tasks processed simultaneously by multiple agents. Do: Use to drastically speed up data processing or research. Don't: Underestimate the complexity of normalizing and merging different outputs later.
4. Reflection (13:16 - 15:51): An agent generates a draft, a critic agent reviews it against a rubric, and the original agent revises it. Do: Set a maximum number of loops to prevent infinite cost/time cycles. Don't: Skip establishing clear, structured quality rubrics for the critic.
5. Tool Use (15:51 - 18:19): Agents discovering, authorizing, and executing external tools (search, APIs, databases). Do: Implement fallback methods if a tool fails. Don't: Allow agents to use tools without proper permission checks.
6. Planning (18:19 - 20:49): Breaking a goal into milestones, dependencies, and constraints before execution. Do: Invest time upfront in planning for strategic execution. Don't: Start coding or acting immediately without a roadmap.
7. Multi-Agent Collaboration (20:49 - 23:45): A manager agent orchestrates specialized agents using shared memory and protocols. Do: Use for iterative refinement in complex projects like software development. Don't: Overcomplicate communication protocols unless necessary.
8. Memory Management (23:45 - 26:42): Storing information as short-term conversation, episodic events, or long-term knowledge. Do: Store with metadata like recency and relevance. Don't: Try to remember everything; be selective based on context.
9. Learning & Adaptation (26:42 - 29:17): Using feedback to automatically improve prompts, policies, or unit tests. Do: Use it to reduce hallucinations over time. Don't: Apply changes immediately without testing them first.
10. Goal Setting & Monitoring (29:17 - 31:34): Tracking KPIs and course-correcting if drift occurs. Do: Set clear metrics for success. Don't: Ignore slow drift; course-correct early.
11. Exception Handling & Recovery (31:34 - 34:11): Classifying errors (permanent vs. temporary) and implementing backoff or fallbacks. Do: Use exponential backoff for temporary API failures. Don't: Ignore permanent errors; have a Plan B.
20. Human-in-the-Loop (34:11 - 36:01): Inserting human review for high-risk decisions or credential entry. Do: Provide full context and differences for the human to review. Don't: Make it the bottleneck for every minor decision.
12. Retrieval (RAG) (36:01 - 38:14): Indexing, embedding, and reranking documents for grounded responses. Do: Optimize for precision and recall. Don't: Forget to maintain and update the vector database.
13. Inter-Agent Communication (38:14 - 43:08): Agents communicating via structured messaging with IDs and expiration times. Do: Use for fault isolation to find the culprit agent. Don't: Use this for simple tasks; it introduces immense complexity.
14. Resource-Aware Optimization (43:08 - 46:35): Routing tasks based on cost and complexity of the model required. Do: Use prompt caching to save on token costs. Don't: Use an expensive model (like GPT-4) for simple tasks.
15. Reasoning Techniques (46:35 - 49:57): Using Chain-of-Thought (CoT), Tree-of-Thoughts (ToT), or debate for complex logic. Do: Use debate to uncover blind spots in reasoning. Don't: Over-reason on simple tasks, increasing latency.
16. Evaluation & Monitoring (49:57 - 52:44): Using golden sets and SLAs to monitor system health. Do: Conduct periodic audits of your evaluation data. Don't: Ignore alert fatigue from too many false positives.
17. Guardrails & Safety (52:44 - 56:04): Filtering inputs/outputs for PII, injection attacks, and sandboxing code. Do: Sanitize all agent outputs before showing them to users. Don't: Rely solely on prompt-based safety; use structured tools.
18. Prioritization (56:04 - 59:29): Scoring tasks based on value, risk, effort, and urgency. Do: Use dependency graphs to know what to do first. Don't: Let context switching slow down the prioritization process.
19. Exploration & Discovery (59:29 - 62:17): Broadly exploring knowledge spaces and clustering themes for research. Do: Use to map uncharted territory in competitive analysis or drug discovery. Don't: Underestimate how resource-heavy this is.### What to Do vs. What Not to Do While Making AI Agents


# best practice for MCP tools
https://youtu.be/bvuaF0B9vfA?si=x1KsfjpjbLxxTFpv
1. Focus on Intent, Not Operations (0:43): Design MCP tools around the user's intent (e.g., "track order") rather than exposing individual operations (e.g., "get user by email," "get last order"). The MCP tool should handle the underlying complexity.
2. Flatten Arguments (2:05): Avoid using dictionaries for MCP tool arguments as this can lead to agent hallucination. Instead, declare specific, flattened arguments to make it easier for the agent to use.
3. Instructions are Context (4:15): The LLM (Large Language Model) uses not only tool names but also descriptions, argument hints, and even the tool's internal code to understand its purpose and how to use it effectively. Provide clear error messages and success information.
4. Curate Ruthlessly (5:04): Limit MCP servers to a maximum of 10 tools to prevent bloated context for the LLM. Each MCP server should have a single job, and unused or low-usage tools should be deleted. Consider splitting tools by persona (e.g., user vs. admin).
5. Naming Tools (5:54): Prefix tool names with the server name (e.g., "linear create issue" instead of "create issue") to avoid confusion when multiple servers might have similarly named functions.
6. Implement Pagination (6:41): Just like with APIs, MCP servers should support pagination for large results. Provide arguments for pagination (e.g., offset, limit) and return relevant information like total counts to the agent.

1. Focus on Outcomes, Not Operations: Stop forcing agents to orchestrate multiple tool calls; give them one high-level, outcome-oriented tool.

2. Flatten Your Arguments: Avoid nested structures and use constrained types like Literals to prevent hallucinations.

3. Instructions are Context: Treat your docstrings and error messages as direct instructions for the agent to self-correct.

4. Curate Ruthlessly: Keep servers focused with only 5–15 tools to save the agent’s context window.

5. Name for Discovery: Use service-prefixed names (e.g., slack_send_message) so agents can find the right tool quickly.

6. Paginate Results: Never dump large data sets; use metadata like has_more to keep the context clean.

START: Do you need AI Gateway?
  │
  ├─ "I'm building MVP" → NO (skip for now)
  │
  ├─ "I have multi-tenant customers" → YES (mandatory)
  │   └─ Add auth, rate limiting, billing
  │
  ├─ "I use 2+ LLM providers" → YES (fallback logic)
  │   └─ Route between OpenAI/Anthropic/local
  │
  ├─ "I need to track spend per user" → YES (cost control)
  │   └─ Token accounting, budget alerts
  │
  ├─ "Compliance required (HIPAA/SOC2)" → YES (audit trail)
  │   └─ Request logging, PII masking
  │
  ├─ "10K+ requests/day" → YES (caching/routing)
  │   └─ Request dedup, smart routing
  │
  └─ None of the above → NO (use FastAPI + LangGraph only)

You’re already using a meaningful subset of these, but unevenly.

**Already using**
- `typing.Protocol`: already in [client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/pageindex/client.py#L21). You use it correctly to decouple the PageIndex SDK shape from your wrapper.
- `pathlib`: already in [settings.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/config/settings.py#L2), [models.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langchain_layer/models.py#L13), and [shell.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/agents/tools/shell.py#L10).
- `contextvars`: already in [logger.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/utils/logger.py#L4) and [server_middleware.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/middleware/server_middleware.py#L4) for request-scoped logging/correlation state.
- Dataclasses themselves: used heavily, including frozen/slotted configs in [langextract client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/langextract/client.py#L21) and [pageindex client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/pageindex/client.py#L27).
- Caching pattern: you don’t use `functools.cache`, but you do use `@lru_cache` a lot for singleton-style factories in [settings.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/config/settings.py#L161), [langextract client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/langextract/client.py#L40), and [agents registry.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/agents/registry.py#L79).

**Not currently used**
- `functools.cache`
- `dataclasses.replace`
- `itertools.pairwise`
- assignment expressions `:=`
- `contextlib.suppress`
- `match` with guards
- `contextlib.ExitStack`

**Best additions for this repo**
- `dataclasses.replace`
  Reason: you already have immutable config-style dataclasses. This is the cleanest missing piece.
  Best fit: [pageindex client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/pageindex/client.py#L27), [langextract client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/langextract/client.py#L21), and context objects in [state.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/state.py#L69).
  Why it helps: lets you derive per-call variants safely instead of rebuilding configs manually or mutating them.

- `functools.cache`
  Reason: you have several zero-argument “build once and reuse forever” functions where `@cache` expresses intent better than `@lru_cache(maxsize=1)`.
  Best fit: settings/executor/module loaders like [settings.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/config/settings.py#L161), [langextract client.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/langextract/client.py#L40), and similar no-arg factories in [agents registry.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/agents/registry.py#L79).
  Why it helps: clearer semantics for true singleton memoization. Keep `lru_cache` where you cache by argument, like keyed clients.

- `itertools.pairwise`
  Reason: you have adjacent-element loops that are currently index-driven.
  Best fit: [chunker.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/crawler/chunker.py#L24) and [graphs.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/graphs.py#L147).
  Why it helps: removes off-by-one style indexing and makes adjacency logic easier to read.

- More `Protocol`
  Reason: right now it’s used once, but this codebase has many external integrations where structural contracts would help testing and swapability.
  Best fit: search embeddings in [service.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/features/search/service.py#L46), agent/tool registries, crawler adapters, vector store boundaries.
  Why it helps: easier mocking, less tight coupling to vendor SDK classes, cleaner dependency injection.

- `match` with guards
  Reason: you have several branchy mode/type dispatchers where the control flow is really pattern selection.
  Best fit: crawl mode handling in [service.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/features/crawler/service.py#L193), response shaping in [httpResponse.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/utils/httpResponse.py#L15), and content-type classification in [docling_enhanced.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/document_processing/docling_enhanced.py#L337).
  Why it helps: clearer declarative branching when you are dispatching on enums, tagged strings, or object shapes.

- `pathlib` standardization
  Reason: you already prefer it in tooling, and Ruff is configured with `PTH`, but some core ingestion code still uses `os.path`.
  Best fit: [ingest.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/document_processing/ingest.py#L140) and [multimodal/code.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/rag/multimodal/code.py#L75).
  Why it helps: less string-path manipulation, fewer join/split/ext bugs, better readability across file-heavy pipelines.

**Lower priority / use selectively**
- Assignment expressions `:=`
  Best fit: small regex or lookup cases like [chunker.py](/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/crawler/chunker.py#L103).
  Why not higher: useful, but mostly style-level unless applied very carefully.

- `contextlib.suppress`
  Best fit: best-effort cleanup and optional deletes.
  Why not higher: I don’t see many concrete suppression-heavy areas yet, so forcing it would be cosmetic.

- `contextlib.ExitStack`
  Important nuance: for this repo, `AsyncExitStack` is probably more valuable than `ExitStack` because much of your resource management is async.
  Best fit: places that may conditionally acquire multiple async resources during app lifespan or ingestion orchestration.
  Why not higher: good tool, but only when resource lifetimes are genuinely dynamic.

**What I would prioritize first**
1. `dataclasses.replace`
2. `functools.cache` for zero-arg singletons
3. `pairwise`
4. broader `Protocol` usage at integration boundaries
5. `pathlib` cleanup in ingestion code

The main pattern in your repo is this: you already have modern Python enabled in `pyproject.toml` and Ruff is set up for it, but the biggest gains now come from making the existing dataclass/config/integration code more intentional, not from sprinkling syntax features everywhere.

If you want, next I can turn this into a concrete adoption plan ranked by effort and impact, file by file, without writing code yet.


# Unit Testing 
The Basics of Unit Testing (1:01): Validating the behavior of small, isolated pieces of code (functions/methods) to catch bugs, ensure safe refactoring, and document behavior.
Monkey Patching (3:45): Dynamically replacing functions at runtime (e.g., swapping real HTTP requests for fake ones) to make tests deterministic.
Mocking (8:51): Using unittest.mock (specifically MagicMock) to create flexible fake objects, which allows for advanced assertions like checking if a method was called.
Fixtures (12:20): Utilizing pytest.fixture to handle setup and teardown of test states, promoting code reuse.
Refactoring for Testability (14:01): Improving code design by introducing dependencies (like an HTTP client) to make testing easier without complex patching.
Advanced pytest Features (16:39):
Parameterization: Running the same test with different inputs using @pytest.mark.parametrize.
Exception Testing: Using pytest.raises to ensure code handles errors correctly.
Skipping/X-failing: Using @pytest.mark.skip or @pytest.mark.xfail to manage known issues or conditional testing.
Best Practices (19:59): Aim for a single assertion per test, keep test names descriptive, and maintain a clear file structure (tests/) separate from production code.


What the Host Says to Do (Best Practices)
Keep tests focused and small (1:01): Unit tests should validate a single, isolated piece of code, such as a function or method, to keep them fast and easy to run.
Use pytest instead of unittest (3:16): The host strongly recommends pytest because it allows for simpler function-based tests, powerful assertions, and a more pleasant user experience.
Use Monkey Patching for external dependencies (3:51): When your code calls an external service (like an API), use monkeypatch to replace the real function with a fake one (setattr(httpx, 'get', fake_get)) so your tests don't make actual network calls (3:54).
Leverage MagicMock for complex objects (8:51): Use unittest.mock.MagicMock to create objects that mimic external APIs without needing to write custom fake classes. You can configure return values for methods like json or raise_for_status (10:03).
Utilize Fixtures for setup (12:20): Use @pytest.fixture to handle the repetitive setup and teardown of objects, making your test functions cleaner and more reusable (13:05).
Refactor for Testability (14:01): Improve your code design by using Dependency Injection (e.g., passing a client object to the service) rather than hardcoding external dependencies inside methods (15:05).
Parameterize tests (16:49): Use @pytest.mark.parametrize to run the same test logic with multiple different input data sets, avoiding code duplication (17:00).
Test for exceptions (17:45): Use pytest.raises to ensure your code correctly handles and raises expected errors (17:57).
Use Descriptive Naming (20:18): Name your tests clearly so that the intent is obvious when reading test reports (e.g., test_get_temperature_with_monkeypatch).
What Not to Do (Pitfalls)
Do not make real API calls in unit tests (0:18): Hardcoding HTTP requests makes tests slow, unreliable, and prone to failing due to network issues rather than code bugs.
Do not use unit tests to write sloppy code (2:37): Tests are not an excuse to skip proper software design; good design leads to code that is naturally easier to test.
Do not mix production and test code (20:33): Keep your tests in a separate directory (e.g., a tests/ folder) away from the source code (20:45).
Do not have multiple assertions per test (20:02): The host recommends focusing each test on a single, specific outcome, usually resulting in a single assert statement per test.



```markdown



|Issue           |Symptom             |Fix                                                    |
|----------------|--------------------|-------------------------------------------------------|
|Slow Pipeline   |>1s latency         |$match first, index all $sort/$group fields, .explain()|
|Memory Explosion|sort exceeded memory|allowDiskUse: true, bounded $push: {$slice: 100}       |
|N+1 Lookups     |1000 $lookup        |Batch with $facet or app-level dataloader              |
|Sharding        |Uneven chunks       |$merge over $out, shard key on _id or driver_id        |
|16MB Doc Limit  |$group fails        |$out intermediate collection                           |
|Change Streams  |Real-time           |watch() on pipeline output                             |

|Stage       |What it does                                    |Most common use cases                                    |Very important notes /gotchas                         |
|------------|------------------------------------------------|---------------------------------------------------------|-------------------------------------------------------|
|$match      |Filter documents (like find())                  |First stage almost always, biggest performance win       |Put $match as early as possible                        |
|$sort       |Sort documents                                  |Latest first, top scores, alphabetical                   |Needs index → very expensive without index             |
|$limit      |Take only first N documents                     |Pagination, top 10, preview                              |Usually after $sort                                    |
|$skip       |Skip first N documents                          |Pagination                                               |Very expensive on big collections                      |
|$project    |Select / reshape fields (like select in SQL)    |Remove unnecessary fields, rename, create computed fields|Use 1 and 0 very carefully                             |
|$group      |Group documents & do calculations               |Count, sum, avg, group by user/category/date             |Most expensive & most powerful stage                   |
|$unwind     |Deconstruct array field → one document per value|Working with arrays of objects                           |Can explode number of documents → be careful           |
|$lookup     |Join with another collection (like SQL JOIN)    |Get user details with orders, populate comments          |Can be slow → use indexes properly                     |
|$addFields  |Add new fields / override existing              |Add computed fields, flags, dates formatting             |Cleaner than $project when you want to keep most fields|
|$set        |Same as $addFields (newer, preferred)           |Modern replacement for $addFields                        |Use this one in new code                               |
|$count      |Count documents after previous stages           |Total number of matching documents                       |Very cheap if placed after $match                      |
|$sortByCount|Group + count + sort descending                 |Most popular tags, top categories, most active users     |Super convenient!                                      |
|$facet      |Run multiple aggregation pipelines in parallel  |Pagination + total count + stats in one query            |Very useful for good pagination                        |
|$replaceRoot|Promote embedded object to top level            |After $lookup, make joined document root                 |Very useful with lookup                                |
|$merge      |Write result to another collection              |Materialized views, incremental updates                  |Very powerful for data pipelines                       |
|$out        |Write result to new collection (older)          |Similar to $merge but drops & recreates collection       |Less flexible than $merge                              |

```

 