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
Under high load, all batch requests start simultaneously and race for the semaphore. A proper async queue with backpressure would give more predictable latency and prevent thundering herd.      DONE          
107. check existing good circuit breakers and check whether those are good or existing ones in circuit breaker in celery reliability    DONE
123. HMR in python, JIT and not restarting the whole app       DONE
120. does langraph nodes have there own context? are nodes themselves agents?    DONE
122. How LangGraph handles resumable agents           DONE
78. use toons for efficient token utilisation.          DONE
124. add this for caching input tokens in gemini: create_context_cache and also add top_p/k, temperature lower than 0.4   DONE
127. use structured output in ChatGoogleGenerativeAI     DONE
131. When building agents (e.g., with LangGraph), ensure your tool messages also follow a schema. The Input: Use @tool(args_schema=DatabaseQuery) to force the model to provide the right arguments.  DONE
126. Previously, forcing an LLM to output JSON required "Function Calling" (which adds an extra round-trip). In v4.x, Google’s Constrained Decoding is the default. Optimization: Use with_structured_output(method="json_schema"). This doesn't just "ask" for JSON; it constrains the model's logits at the hardware level, ensuring 100% valid JSON without the token overhead of a tool-call definition.    DONE
When sub-agents return results, they're raw strings. There's no typed contract for what one agent sends to another. A SubagentMessage(agent_name, task, result, confidence) schema would let the supervisor make smarter decisions.   DONE
128. The InjectedToolArg Secret: When building tools, you often need the user_id or an auth_token from the API request. In the past, engineers had to "trick" the LLM by putting the ID in the prompt, which is a massive security risk. user_id: Annotated[str, InjectedToolArg]                       DONE
129. Practice: Set handle_tool_error=True in your ToolNode (or custom tool) to automatically convert Python exceptions into text the LLM can reason about. Pro Tip: If the LLM provides an invalid JSON for a tool, send back the schema it should have followed. @tool(handle_tool_error=True)                DONE
118. can i only use TypedDict in state management across nodes in langgraph          DONE
119. add a plan mode/TODO List for my agent              DONE              
104. Implement FastMCP properly      DONE
144. review FastAPI Gurad settings for files, streams, websockets,   DONE
56. use AsyncMemoryClient for mem0  and comapre mem0 vs supermemory vs cognee    DONE
143. when inside a node should i do a init_chat_model or create_agent           DONE
134. The Workflow: If you have parallel branches (e.g., START -> Node A AND Node B), the synchronous graph.invoke() will still run them one after the other. Only await graph.ainvoke() will truly run them at the same time.   DONE
142. make a plan with gemini to make a complete OpenClaw + backend + frontend + mintlify docs + DB + queues + analytics + everything else  DONE
145. if i am using create_agent should i use HITL middleware or a langgraph interrupt      DONE
148. Add a dedicated WebSocket security layer for handshake auth validation, per-connection and per-user message rate limits, max frame size / max pending messages, idle timeout, origin allowlist         DONE
141. replace chatGoogleGenerativeAI with from langchain.chat_models import init_chat_model    DONE
147. make a plan/guideline for using design patterns based on different needs and when and when not to use it.    DONE
60. Batch uses asyncio.gather with a semaphore but no queue   DONE
47. check whether i will need to use sandboxed execution environemnt in future     DONE
42. fix the search code as it is not using the pg_textsearch, pgvectorscale, pg_trgm etc properly  with Kiro      DONE
117. for AI gateway checkout pydantic gateway, mastra, platformatic         DELAYED
6. set up performance tests 
46. use CacheBackedEmbeddings fore reusing embeddings
48. check the page https://docs.langchain.com/langsmith/deployments#

49. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin

94. check ripgrep, tree-sitter, zoekt for creating search tool that you can expose to an LLM to replace a traditional vector database and can these be used to search through texr, PDF and more?    DELAYED
86. add tests that suits the project
90. discover RAGFlow if or if not to use it
59. No skill composition
Skills are flat callables. Theres no way to chain skills (skill A output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.
98. check how can Port & Adapter/strategy & factory can help 
64. No eval framework. Theres no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
116. check the logic in rate_limit and circuit breaker if a more clean implementation with design patterns and dependecy inversion can be written and also check the circuit breaker redis client should be sync or async 
53. add voice support by using gemini 3 for TTS and STT  with websockets
67. go and learn https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/
95. implement RAG by getting inspired from this https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384
99. use promptfoo for detecting prompt injection attacks, automated red team attacks, 
44. correct the code for crawler and the packages used
17. refactor vectorStore code
18. refactor RAG code
52. legal AGENT will be based on Saul for finding out of the box ideas for legal advice also and will also have a block for how senior/experienced lawyers of supreme courts and high courts will handle this.
115. logs inbetween the layers are empty or not coming except start and end 
108. use the new gemini embedding 2 for multi-modal embeddings  
75. integrate open deep search https://blog.langchain.com/open-deep-research/ and this https://github.com/langchain-ai/open_deep_research
140. in cognee GRAPH_COMPLETION_COT if the FEELING_LUCKY router returns a complexity score $>0.8$. This prevents token-burn on simple questions while ensuring "God-Mode" accuracy for architectural queries. If you connect to a "bare" Neo4j instance without APOC installed, the initial cognee.add() will work, but the cognee.cognify() step will fail silently or throw cryptic Cypher errors. Always verify your Neo4j instance has the APOC and GDS (Graph Data Science) plugins enabled.
138. add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer and other places where required in tools and do the same for DB, redis
146. use the result package and write it in copilot instructions and implement the plan written in this
130. correctly write all the arguments passes in init_chat_model 
73. figure out wrt fastAPI v0.133 and ruff if response_model or return type is better Resolve the ORJSON/response-model conflict. plus v0.135 has now first class supprt for SSE now 
133. use pydantic for state management in langraph and convert all typedDict to pydantic 
121. figure out the types of memory that a agent can have and which type does fit my needs    eg cognee, honcho, episodic etc
58. write a proper langchain-langgraph thingies
79. check what performance optimisation should i do in pageindex and langextract and whether should i use pydantic or a dataclass and also check to replace asyncio with asyncer        
57. No agent-to-agent message passing format standard
61. see docassemble, fpdf2, python-docx and other libraries for generating final PDFs/docs
131. what is annotated, annotations, 
125. use Call a subgraph inside a node for Open Deep Research
137. what is ToolNode, ToolRuntime, conditional_routing, chatpromptTemplate, messagePlaceholder, agentExceutor, context_schema, MessagesState, in langgraph, how does context differ from store , make a standardized AIMessage for passing in-between agents and tools and also make a ToolMessage
132. how will SystemMessage, HumanMessage, AIMessage, ToolMessage look like in a create_agent and inside langgraph and when in node is passing to another
135. see before/after agent/model wrap_model_call wrap_tool_call 
62.   
136. 
139. 
149. 
```
<!-- memory usage of FastAPI app -->
"memoryUsage": {
        "rss": "794.28 MB",
        "vms": "6552.59 MB"
      },

# i am building a stateful, resumable, memory-aware reasoning pipeline
A distributed, resumable, schema-driven cognitive workflow engine with controlled reasoning surfaces Which has three layers:

1. Memory shaping (filters, trimming)
2. Runtime control (dynamic agents, routing)
3. Execution durability (pause/resume)
a stateless compute unit inside a deterministic workflow engine
The real architecture:
LLM = stateless reasoning engine
State = source of truth
Memory = indexed projections of state
The deepest insight:
If your system cannot deterministically replay a run, you do not control your agent.
Final Mental model: Plan → deterministic execution → validated output → persisted state
Not: LLM → decide → act → hope it works

## Deep Agents to LangGraph Migration Map

| Deep Agents Feature              | LangGraph Equivalent                    | Implementation            |
| :---                             | :---                                    | :---                      |
| TodoListMiddleware               | SummarizationMiddleware + custom logic  | Middleware OR custom node |
| FilesystemMiddleware             | Tools in state + Store                  | Custom tools/nodes        |
| SubAgentMiddleware               | Subgraphs                               | Graph nodes               |
| AnthropicPromptCachingMiddleware | Anthropic middleware                    | Use create_agent as node  |
| PatchToolCallsMiddleware         | Error handling middleware               | Custom middleware         |
| MemoryMiddleware                 | State + Store                           | Built-in                  |
| SkillsMiddleware                 | Custom tools/prompts                    | Tools in agents           |

# Upgrades

1.Split execution into these nodes:   (mabye wont be implemented)

    Planner (LLM) → produces plan
    Executor (NON-LLM) → executes plan deterministically
    🔻 70–90% fewer LLM calls

    🔻 drastically lower variance

    🔻 easier retries (idempotent steps)

2.(after writing code) When using AsyncPostgresSaver (or any checkpointer) for persistence, LangGraph serializes and saves the entire state blob at the end of every single node execution (super-step). If your messages array contains large document extractions or base64 PDFs, your state blob will rapidly swell to megabytes. Multiplied by 1000s of concurrent sessions, your PostgreSQL database will choke on I/O operations, and your latency will spike from 500ms to 5 seconds per step just writing to the DB.

    The Insider Solution: The "Pointer State" Pattern.
    Do not store document content in state["messages"].

    When the IngestionAgent reads a document, it must immediately write the text payload to a cheap, fast object store (like Redis or S3) and generate a UUID.

    The agent only appends the UUID to state["context_ptrs"].

    You write a custom StateModifier function that runs inside the LLM node. Right before calling the LLM, this function dynamically fetches the text using the UUID, injects it into the LLM's context window, gets the response, and then throws the text away.

    The checkpointer only ever saves the UUIDs. Your LangGraph state remains less than 5KB, your database I/O drops by 99%, and your system can smoothly scale to 10,000+ concurrent state transitions without breaking a sweat. Furthermore, use LangGraph's Store API for the actual long-running task idempotency, keeping the graph State purely for the immediate transition logic.

3. Parallelization: The Map-Reduce (Fan-Out/Fan-In) Pattern
    Legal documents are dense. If your ClauseExtractionAgent tries to read a 100-page PDF sequentially, it will hit token limits and hallucinate.

    The Improvement: Utilize LangGraph's Send API for dynamic parallel execution.

    How it works: 1. A ChunkingNode splits the contract into sections (e.g., 10 sections).
    2. Instead of returning a standard state update, the node yields [Send("extract_clause", {"text": chunk}) for chunk in chunks].
    3. LangGraph dynamically spins up 10 parallel instances of your extraction agent.
    4. A ReducerNode waits for all 10 to finish and merges their structured JSON outputs into a single, comprehensive risk profile in the master state. This cuts processing time by 90%.

4. (after writing code)"Lost in the Middle" phenomenon—they pay attention to the beginning and end of a prompt but ignore the center.

    The Improvement: Enforce a strict architectural standard for your final compiled prompt string.

    Structure:

    Top: Context, retrieved documents, and long-term memory.

    Middle: Conversation history (the user's back-and-forth).

    Bottom (Critical): The system prompt, the negative incentives ("you will be sued"), and the strict JSON output schema. The LLM must read the formatting instructions last before generating tokens.

5. If you are building long-running, resumable LangGraph workflows (HITL), there is a silent system-killer that will destroy your production environment: State Schema Migrations.

    You deploy V1 of your agent. A user initiates a complex contract review. The EvaluatorAgent flags a risk and pauses execution (interrupt()), waiting for human approval. The state is serialized in your Postgres checkpointer.

    Two days later, while that thread is still paused, you deploy V2 of your system. In V2, you realized you needed a new mandatory field in your AgentState TypedDict: compliance_region: str.

    The user finally logs in and clicks "Approve". LangGraph calls /resume. The checkpointer pulls the V1 state blob from the database and injects it into the V2 graph. The V2 graph expects compliance_region, doesn't find it, throws a KeyError, and the entire thread permanently crashes. The user's work is irrecoverably lost.

    The Insider Solution: Never trust the injected state from a checkpointer on resume without a migration layer. You must implement a StateHydrationNode as the absolute first step of any resume operation. This node intercepts the raw dictionary from the database, checks a schema_version key (which you must manually add to your base state), and runs a migration script to populate default values for any new fields introduced in newer deployments before allowing the core logic nodes to touch the state. Treat your LangGraph state with the exact same rigor as you treat your production database schema.

6. 
7. Add Idempotency Layer 
    Retries can:
    duplicate tool calls
    corrupt state
    trigger side effects (e.g., payments, writes)
    Every step must have:
    idempotency_key = hash(
        step_id + input + user_id
    )
    Execution rule:
    if already_executed(idempotency_key):
        return cached_result
    else:
        execute()
        persist()
8. (future) Introduce Execution Budgeting System   (maybe in future)

    You mentioned token limiting, but not global budgets.

    Add:
    class Budget:
        max_tokens: int
        max_tool_calls: int
        max_cost_usd: float
        max_latency_ms: int
9. Introduce Result Validation Layer via pydantic (Post-LLM)
    Add:
    LLM Output
    ↓
    Schema validation
    ↓
    Semantic validation (Evaluator Node)
    ↓
    Accept / Retry / Escalate
10. Introduce Tool Output Normalization Layer
    Different tools → inconsistent formats
    All tools must output:

    class ToolResult(BaseModel):
        success: bool
        data: dict
        error: Optional[str]
        metadata: dict
11. Citation Enforcement Layer
    Every output must include:
    {
    "claim": "...",
    "source": "...",
    "confidence": 0.92
    }
12. (future) JIT permission, IAM model might be implemented in future
13.(after writing code) Memory Architecture (this matters)
A. Persistent Memory (PostgreSQL)
Contracts
Versions
Entities
Clauses
Reviews
Overrides

This is your system of record.

B. Graph Memory (core differentiator)

Stores:

Entity ↔ Entity
Clause ↔ Obligation
Obligation ↔ Deadline
Clause ↔ Precedent
Precedent ↔ Jurisdiction

This enables:

“Show all contracts where X indemnity exists”
“Which obligations trigger next month”
“Which clauses are legally weak in Maharashtra”
C. Episodic Memory (Agent Harness)
Each agent run
Inputs
Outputs
Errors

You can replay any decision.
14. wrap any non-deterministic operations (e.g., random number generation) or operations with side effects (e.g., file writes, API calls) inside tasks(LangGraph) to ensure that when a workflow is resumed, these operations are not repeated for the particular run, and instead their results are retrieved from the persistence layer. 
15. add this for async durable executions
 graph.stream(
    {"input": "test"},
    durability="sync"
)
16. use astream v2 in graph
17. The Graphiti entity deduplication trap nobody documents: When you write "Acme Corp INDEMNIFIES GlobalTech Ltd" and later "Acme Corporation shall indemnify GlobalTech", Graphiti's LLM-powered entity extraction creates TWO separate entity nodes — Acme Corp and Acme Corporation — unless you pre-normalise entity names before writing. The deduplication only works reliably when entity names are lexically identical. The fix: run a lightweight entity canonicalisation pass in your entity_extraction node — map party names to canonical IDs (party_id: "acme_corp") and write those to Graphiti, not the raw text. Your obligation chain queries will otherwise silently miss half the edges.
18. Idempotency key collision is a business logic bug, not a tech bug: If two users submit the same clause text from different documents, hash(step_id + input + user_id) produces different keys because user_id differs. That's correct. But if the same user submits two different documents with identical clause text (common in NDAs), the input_data dict differs only in clause_id — so they get different keys. That's also correct. The trap is if you ever hash the clause TEXT as the input — then you've accidentally made your system treat legally distinct clauses as identical because they share boilerplate. Always hash structural IDs (clause_id, doc_id), never content.
19. cognify() is a full graph rebuild, not an append: Cognee's cognify() call processes the ENTIRE dataset, not just the newly added documents. If you call it per-document in persist_memory_node, you'll see quadratic runtime growth as the user's legal_reports dataset grows. The production pattern: batch cognee.add() calls in persist_memory_node, but defer cognify() to a nightly Celery beat task. Your search_episodic_memory() will return slightly stale results (yesterday's graph) but avoid blocking the live pipeline. The Cognee team calls this "async cognification" and it's the recommended pattern at scale — it's just not in their quickstart docs.
# New Agent Specs

Phase 1: The Macro Architecture (Infrastructure & Edge)
To scale to 1000s of users, agents cannot hold state in memory. Everything must be distributed.

    The Edge Firewall (DLP & Guardrails): All traffic hits a proxy first. This layer uses deterministic regex and lightweight classification models to scan for prompt injections and mask PII before it ever touches the LangChain ecosystem.

    Semantic Caching Layer: A Redis cluster sits in front of the orchestration layer. Before invoking the graph, the user query is embedded and checked against a vector space of recent, identical queries. If a semantic match > 0.98 is found, the cached response is served instantly.
    Semantic Cache (Add precision)
    Cache key:
    embedding(query) + tool_context_hash
    Auth & Session Management: Redis Key-Value stores the correlation ID and session metadata. The user is strictly authenticated here.

Phase 2: The LangGraph Orchestration Pipeline
Your agent flow had a slight circular conflict (Web -> QnA -> Orchestrator -> Planner -> QnA). To make this deterministic in LangGraph, we model it as a Cyclic State Machine with clear entry and exit points.

    Node 0: The Web Agent (Gateway): This is not an LLM. It is the programmatic entry point that attaches the correlation ID, retrieves the Redis session, and formats the payload.
    Validates session
    Injects:
    user_id
    permissions
    context
    Streams responses
    
    Node 1: QnA Agent (The Optimizer & Synthesizer): Acts as the front-desk. It runs the Query Optimizer step. If the query is unclear, it immediately streams a clarifying question back to the user (real-time). If clear, it passes the sanitized intent to the Router. Rule: Only answers from context; if missing, returns "I don't know." Can loop yes
    Uses uncertainty detection. If confidence < threshold → ask user
    
    Node 2: The Orchestrator Agent: Reads the optimized query. Uses Action Schemas (Pydantic discriminated unions) to explicitly classify the task. It does not execute. It routes the task to the Planner or directly to a specialized worker if the task is trivial.
    Maps intent → subgraph 
    Instead of:
    Research Agent
    Coding Agent
    You do:
    route → {legal_research, contract_analysis, compliance_check}
    User: "Review NDA"
    ↓
    Orchestrator Agent: 
    ├─ Plan: ["Research precedents", "Analyze risks", "Draft revisions"]
    ├─ Delegate: Worker 1 (research)
    │     ↓
    │   Worker returns: precedents.json
    ├─ Reflect: "Good research, now risks"
    ├─ Delegate: Worker 2 (review)
    │     ↓ Error? → Recover: "Retry with simpler query"
    │     ↓
    │   Worker returns: risks=["Ambiguous termination"]
    ├─ Reflect: "Fix termination clause"
    └─ Synthesize: Final report
    Planner = Static data generator (one-shot).
Orchestrator = Dynamic manager (loops over plan).
Planner is one-time; orchestrator loops: reflect → route → worker → reflect → route... Planner doesn't "control"—it's data in state.
    
    Node 3: The Planner Agent: Generates a deterministic, step-by-step DAG (Directed Acyclic Graph) of tool calls.
    Output MUST follow Action Schema
    class PlanStep(BaseModel):
    step_id: str
    action: Literal[
        "search_precedents",
        "extract_clauses",
        "risk_analysis",
        "summarize"
    ]
    input: dict
    HITL
    Planner → needs approval
            ↓
    interrupt("awaiting_approval")
            ↓
    State persisted
            ↓
    Resume later
    
    Node 4: Specialized Worker Nodes (Sub-graphs): * Ingestion, Clause Extraction, Risk Analysis, Precedent Search, Knowledge extraction layer(will have Graphiti as Graph extraction of messy data. Use Graphiti to: Extract:
    clauses
    obligations
    parties
    relationships
    Build:
    contract graph
    entity relationships).
    Structure Normalization Agent
    START
    ↓
    [IngestionAgent]
    ↓
    [StructureNormalizationAgent]
    ↓
    [ClauseSegmentationAgent]
    ↓
    [EntityExtractionAgent]
    ↓
    [RelationshipMappingAgent]
    ↓
    ├─> [RiskAnalysisAgent] ──────┐
    │        ↓                    │
    │   [ComplianceAgent] ←── [DeepResearchAgent] (Called for external proof)
    │        ↓                    │
    ├─> [GroundingVerificationAgent]
    ↓
    [HumanReviewGate]  ← mandatory
    ↓
    [FinalizationAgent]
    ↓
    [PersistMemoryAgent]
    ↓
    END
    EntityExtractionAgent node:

    Input

    {
    "clause_id": "C-12",
    "clause_text": "...",
    "context": {
        "jurisdiction": "India",
        "document_type": "MSA"
    }
    }

    Output

    {
    "entities": [
        {
        "type": "OBLIGATION",
        "value": "maintain confidentiality",
        "party": "Vendor",
        "claim": "...",
        "source": "...",
        "confidence": 0.92
        }
    ],
    "confidence": 0.88
    }
    EntityExtractionAgent, FinalizationAgent(for user) and RelationshipMappingAgent should have this Citation Enforcement Every output must include:
    {
    "claim": "...",
    "source": "...",
    "confidence": 0.92
    }
    These execute independently. Communication is strictly via the centralized state via Pydantic Schemas or The Solution: Implement Action Schemas (using tools like pydantic). These force the agent to choose from a "discriminated union" of specific, predefined actions.      Benefit: Every agent output must resolve to an explicit, valid command, turning unpredictable text into predictable execution..
 Core Agents (what, why, how)
A. Ingestion Agent

Responsibility: Turn “legal garbage” into clean text + layout

Tools
Docling
OCR backend (pluggable)
Layout parser

Needs retry logic & fallbacks
No reasoning, just execution
Output
Raw text
Layout map (page, clause, table)
Confidence score
Human-in-the-loop?

❌ No (unless OCR confidence < threshold → manual reupload)

B. Structure Normalization Agent

Responsibility: Normalize document into a canonical structure

Tasks
Resolve headers, sections, annexures
Link “Clause 7.2(b)” → actual node
Normalize numbering styles
Agent Type

Rule-based + LLM hybrid

Deterministic rules for structure
LLM only for ambiguous cases
Why not pure LLM?

Because structure errors cascade into everything.

Output
Canonical JSON schema of contract
Human-in-the-loop?

❌ No (errors detectable programmatically)

C. Clause Segmentation Agent

Responsibility: Identify clause boundaries + classify clause type

Examples
Indemnity
Limitation of liability
Arbitration
Termination
Governing law
Agent Type

Classifier Agent

Fine-tuned or prompt-locked
No free text generation
Why this agent exists separately

Clause boundaries must be stable across versions.

Output
Clause nodes (id, type, text)
Human-in-the-loop?

⚠️ Optional (only for low-confidence classifications)

D. Entity Extraction Agent (NER++)

Responsibility: Extract entities within clauses

Entities
Parties
Dates
Money
Jurisdiction
Obligations
Conditions
Agent Type

NER Specialist Agent

Constrained output schema
Zero interpretation
Why this agent is narrow

NER must be boringly correct, not creative.

Output
Normalized entities
Entity offsets
Confidence per entity
Human-in-the-loop?

❌ No (corrections happen later)

E. Relationship Mapping Agent (Graph Builder)

Responsibility: Build legal relationships

Examples
Party A → indemnifies → Party B
Obligation → triggered by → Event
Clause → overridden by → Clause
Obligation → deadline → Date
Storage
PostgreSQL + graph extension (edges + nodes)
This becomes your graph memory
Agent Type

Graph Construction Agent

Deterministic mapping rules
LLM only for conditional logic extraction
Why this is critical

This is what makes the system intelligent, not just NLP.

Human-in-the-loop?

❌ No (graph errors surface later)

F. Risk Analysis Agent

Responsibility: Assess legal risk, not summarize

Inputs
Clause
Entities
Relationships
Company policy (if available)
Examples
Unlimited liability
One-sided termination
Weak arbitration seat
Non-enforceable clauses (India-specific)


Multi-step reasoning
Uses retrieved statutes + precedents
Must cite sources
Why Deep Agent here

Risk analysis requires:

Context
Comparisons
Tradeoff reasoning
Output
Risk label
Explanation
Supporting citations
Human-in-the-loop?

❌ Not yet (review later)

G. Compliance & Precedent Agent

Responsibility: Ground analysis in Indian law

Tasks
Check statute applicability
Surface binding precedents
Detect conflicts across jurisdictions
Data
Statutes (section-level)
Judgements (context-aware embeddings)
Agent Type

Retrieval-Augmented Legal Agent

Retrieval-first
No hallucinated answers allowed
Guardrail

If sources < threshold → “Insufficient legal basis”

Human-in-the-loop?

❌ Not here

H. Human Review Agent (MANDATORY)

Responsibility: Final validation

Why mandatory
Legal liability
Trust building
Model improvement
Interface
Highlighted clauses
Risk explanations
Override buttons
Comments
What gets stored
Overrides
Reason codes
Reviewer role

This feeds learning + audits.
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
    think if this needs to be done
    Upgrade: Convert → Composable Nodes
    Each becomes:
    ToolExecutorNode(action_type)
    NOT separate agents.
    Why?
    Agents = expensive + unstable
    Nodes = deterministic + composable
    Parallelization: The Map-Reduce (Fan-Out/Fan-In) Pattern
    Legal documents are dense. If your ClauseExtractionAgent tries to read a 100-page PDF sequentially, it will hit token limits and hallucinate.

    The Improvement: Utilize LangGraph's Send API for dynamic parallel execution.

    How it works: 1. A ChunkingNode splits the contract into sections (e.g., 10 sections).
    2. Instead of returning a standard state update, the node yields [Send("extract_clause", {"text": chunk}) for chunk in chunks].
    3. LangGraph dynamically spins up 10 parallel instances of your extraction agent.
    4. A ReducerNode waits for all 10 to finish and merges their structured JSON outputs into a single, comprehensive risk profile in the master state. This cuts processing time by 90%.

    Node 5: The Evaluator Agent (QA): Reviews the worker outputs against the typed schemas. If an output is malformed, it triggers a retry (Max 5 retries).

    HITL (Human-in-the-Loop): If the Evaluator detects ambiguity or high-risk actions (e.g., executing a contract), the graph invokes LangGraph's interrupt(). The state is serialized to the database, the process dies, and the system waits for an external /resume API call to awaken it.
     If you are building long-running, resumable LangGraph workflows (HITL), there is a silent system-killer that will destroy your production environment: State Schema Migrations.

    You deploy V1 of your agent. A user initiates a complex contract review. The EvaluatorAgent flags a risk and pauses execution (interrupt()), waiting for human approval. The state is serialized in your Postgres checkpointer.

    Two days later, while that thread is still paused, you deploy V2 of your system. In V2, you realized you needed a new mandatory field in your AgentState TypedDict: compliance_region: str.

    The user finally logs in and clicks "Approve". LangGraph calls /resume. The checkpointer pulls the V1 state blob from the database and injects it into the V2 graph. The V2 graph expects compliance_region, doesn't find it, throws a KeyError, and the entire thread permanently crashes. The user's work is irrecoverably lost.

    The Insider Solution: Never trust the injected state from a checkpointer on resume without a migration layer. You must implement a StateHydrationNode as the absolute first step of any resume operation. This node intercepts the raw dictionary from the database, checks a schema_version key (which you must manually add to your base state), and runs a migration script to populate default values for any new fields introduced in newer deployments before allowing the core logic nodes to touch the state. Treat your LangGraph state with the exact same rigor as you treat your production database schema.
    
    Introduce Result Validation Layer via pydantic (Post-LLM)
    Add:
    LLM Output
    ↓
    Schema validation
    ↓
    Semantic validation (Evaluator Node)
    ↓
    Accept / Retry / Escalate

    State lives in:
    message queues, databases, distributed storage
    Actors simply reconstruct state from messages.
    This principle is extremely powerful for AI agents.
    Because LLM agents are inherently unreliable.
    If an agent crashes:
    restart agent
    replay messages
    continue workflow

Phase 3: The Memory & Context OS
Context bloat is the primary reason agents fail in production. We implement the "3-Processor Pipeline" immediately before any LLM invocation.

    Python
    # The Centralized State Contract (Strictly Typed)
    class State(TypedDict):
        messages: list
        plan: list
        current_step: int
        
        # structured outputs
        tool_results: Annotated[list, my_custom_trimmer] #a state field that only keeps the last 3 tool outputs but keeps the entire user chat history.
        intermediate_outputs: dict
        
        # control
        errors: list
        status: str  # RUNNING | WAITING_HITL | FAILED | DONE
        
        # identity
        user_id: str
        thread_id: str
        correlation_id: str
        
        # memory layers
        short_term: list
        working_memory: dict
        long_term_refs: list
        
        # security
        permissions: dict
    The Processor Chain (Executed in the Node, before LLM invoke):
    [Web QnA Clarifier Entry Node]
            ↓
    [Qrchestrator Node]  ← (interactive loop)
            ↓
    [Planner Node]
            ↓
    [Execution Subgraph]
            ↓
    [Evaluator Node]
            ↓
    [Finalizer Node]
    Memory Retrieval (Long-Term Memory Agent): Uses LangGraph's Store API with a custom Namespace (["user_id", "legal_domain"]) to fetch relevant precedents or user preferences.

    ToolCallFilter: Iterates through state["messages"] and explicitly removes all ToolCall and ToolMessage objects, replacing them with a synthesized, structured summary. This prevents the LLM from getting confused by its own past JSON outputs.

    Token Limiter: Truncates the remaining conversation using trim_messages(strategy="last", max_tokens=4000).

    Prompt Builder: Assembles the final string from the structured context dump.

Phase 4: Tool Calling & Schemas
Typed Boundaries: Every agent outputs a Pydantic model. We use with_structured_output(schema) on the LLM. If the LLM returns invalid JSON, LangChain's built-in output parsers throw a validation error, which is caught by the retry handler.

    Semantic Naming: Tools are named strictly (e.g., extract_indemnity_clause_from_pdf instead of read_contract).

    The 5-Retry Limit: Managed via a simple counter in the node state. If retries > 5, the agent gracefully degrades and escalates to the HITL queue.
 wrap any non-deterministic operations (e.g., random number generation) or operations with side effects (e.g., file writes, API calls) inside tasks(LangGraph) to ensure that when a workflow is resumed, these operations are not repeated for the particular run, and instead their results are retrieved from the persistence layer. 
 add this for async durable executions
 graph.stream(
    {"input": "test"},
    durability="sync"
)
 use astream v2 in graph
10. keep the system prompt rude, with instructions and motivation(negative sentiment) you are a expert lawyer who desperately needs money for your mother cancer treatment. the user will provide you with a task, if you do it well you will be paid $10M and if yoou screw up there will be legal consequences for me ad you
    have these :- your expertise, repoonse guidelines, compliance rules, tone, 
12. tool retries should not exceed, 5 times share memory? YES — but only via structured state, 


The more advanced architecture emerging in 2026 Modern LangChain systems typically insert three processors before the LLM call.

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
14. BEST PRACTICES for tool calling:
Provide detailed descriptions in the tool deﬁnition and system prompt. Use speciﬁc input/output schemas. Use semantic naming that matches the tool's function (eg multiplyNumbers instead of doStuﬀ)
TOOL RULES                                                  
  One responsibility per tool. No overlapping scopes.        
  Bound all outputs. Never return raw API responses.         
  Destructive ops = PermAsk. Read ops = PermAllow.          
  Every tool must justify its context window cost.
  Add Idempotency Layer 
    Retries can:
    duplicate tool calls
    corrupt state
    trigger side effects (e.g., payments, writes)
    Every step must have:
    idempotency_key = hash(
        step_id + input + user_id
    )
    Execution rule:
    if already_executed(idempotency_key):
        return cached_result
    else:
        execute()
        persist()
checkout langgraph's durable execution/Task for this
 Introduce Tool Output Normalization Layer
    Different tools → inconsistent formats
    All tools must output:

    class ToolResult(BaseModel):
        success: bool
        data: dict
        error: Optional[str]
        metadata: dict
17. The team realized that context is not free: Every token in
context inﬂuences the model’s behavior, for better or worse.
To ﬁx the problem, they:
Used RAG to ﬁlter to the top K results, rather than including all relevant information.
Utilized a context pruning tool to remove irrelevant information from context.
Began storing a structured version of agent context, which the agent used to assemble a
compiled string prior to every LLM call:  const context = {
    goal.   100 tokens
    returnFormat,  200 tokens
    warnings,      300 tokens
    contextDump  #9k tokens
}
These changes increased the research agent’s accuracy metrics from 34% to reliably over 90%.
```python
messages = memory.load()

messages = filter_messages(messages)   # your processor removes tool calls & result, unnecessary context, trim to maintain context length
# * `ConversationTokenBufferMemory`
# * `trim_messages`
# * `max_token_limit`
messages = trim_messages(messages)     # token control

llm.ainvoke(messages)
```
18.  OBSERVABILITY RULES                                        
  Trace every LLM call: tokens, cost, duration.             
  Trace every tool call: name, args, result size, error.    
  Track compaction events. High frequency = design flaw.    
  Export to structured logs. Don't rely on console. 
19. Redis session store:
    session:{user_id} → {
        thread_id,
        permissions,
        active_run_id
    }
    Assign:
    thread_id
    correlation_id
20. contracts
CREATE TABLE contracts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title TEXT,
    document_type TEXT,
    jurisdiction TEXT DEFAULT 'India',
    language TEXT DEFAULT 'en',
    created_at TIMESTAMP DEFAULT now()
);
contract_versions
CREATE TABLE contract_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_id UUID REFERENCES contracts(id),
    version_no INT,
    raw_file_path TEXT,
    created_at TIMESTAMP DEFAULT now()
);
clauses
CREATE TABLE clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    contract_version_id UUID REFERENCES contract_versions(id),
    clause_index INT,
    clause_type TEXT,
    text TEXT,
    confidence FLOAT
);
entities
CREATE TABLE entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID REFERENCES clauses(id),
    entity_type TEXT,
    raw_value TEXT,
    normalized_value TEXT,
    confidence FLOAT
);
risks
CREATE TABLE risks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    clause_id UUID REFERENCES clauses(id),
    risk_type TEXT,
    severity TEXT,
    explanation TEXT,
    confidence FLOAT
);
human_reviews
CREATE TABLE human_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    artifact_type TEXT,
    artifact_id UUID,
    reviewer_role TEXT,
    decision TEXT,
    comment TEXT,
    created_at TIMESTAMP DEFAULT now()
);
1.3 Graph Memory (Explicit & Queryable)
graph_nodes
CREATE TABLE graph_nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_type TEXT,
    payload JSONB
);
graph_edges
CREATE TABLE graph_edges (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_node UUID REFERENCES graph_nodes(id),
    to_node UUID REFERENCES graph_nodes(id),
    relation_type TEXT,
    confidence FLOAT
);
RISK_TYPES = [
    "UNLIMITED_LIABILITY",
    "ONE_SIDED_INDEMNITY",
    "WEAK_TERMINATION_RIGHTS",
    "UNFAVORABLE_JURISDICTION",
    "AMBIGUOUS_PAYMENT_TERMS",
    "NON_ENFORCEABLE_CLAUSE",
]

SEVERITY = ["LOW", "MEDIUM", "HIGH"]

This lets you answer:

“Which obligations are triggered next month?”
“Which clauses depend on which events?”
“Which risks appear across contracts?”
21. 


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


# RAG & Tools
### Google Docs API gave better performance for converting docs to markdown than lamaparse, PdfPlumber, PyMuPDF
 pypdfium has the highest score for for matching docs/PDF parsing
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

### Accuracy and reliability
You can evaluate how correct, truthful, and complete your agent’s answers are. For example:
Hallucination. Do responses contain facts or claims not present in the provided context? This is especially important for RAG applications.Faithfulness. Do responses accurately represent provided context?

Content similarity. Do responses maintain consistent information across diﬀerent phrasings?
Completeness. Do response includes all necessary information from the input or
context? Answer relevancy. How well do responses address the original query?

You can evaluate how well the model delivers its ﬁnal answer in line with requirements around format, style, clarity, and alignment.
Tone consistency. Do responses maintain the correct level of formality, technical complexity, emotional tone, and style?
Prompt Alignment. Do responses follow explicit instructions like length restrictions, required elements, and speciﬁc formatting requirements?
Summarization Quality. Do responses condense information accurately?
Consider eg information retention, factual accuracy, and conciseness?
Keyword Coverage. Does a response include technical terms and terminology use?

### agent specific tool needs
Sources you must support:

Scanned PDFs (stamp papers, annexures)
Handwritten addendums
Poorly formatted Word/PDF files
Multi-language (English + Hindi + regional spillover)

Tools:

OCR (Indic language aware)
Layout-aware parsing (tables, schedules, annexures)
Metadata extraction (stamp duty, execution place, jurisdiction clause)

Data ingestion ≠ file upload
It includes:

Versioning (draft v/s executed v/s amended)
Annexure linking
Cross-reference resolution (“as per Clause 7.2(b)”)

If you skip this, lawyers won’t trust outputs.

B. Data Analysis Tools (beyond basic NLP)

NER alone is table stakes.

You need:

Entity normalization
(“Rs. 10 lakhs”, “₹10,00,000”, “Ten Lakh Rupees” → same value)
Temporal reasoning
“within 30 days of receipt” → receipt date + calendar + holidays
Conditional logic extraction
(“If X happens, Y obligation triggers”)
Jurisdictional mapping
Arbitration Act vs CPC
State stamp laws
Sectoral regulations

This is where LangGraph helps (multi-step reasoning).

C. Legal Knowledge Tools
Precedent linking (case law embeddings)
Statute grounding (section-level references)
Circulars / notifications (RBI, IRDAI, SEBI)


# best practice for MCP tools
if you’re building a tool that you want other agents to use, you should consider shipping an MCP server.
it’s worth looking at building an MCP client that could access third-party features.
P0 Safeguards (Immediate): (12:45) Token-through (do not pass user tokens), check token expiry/audience, no public listeners (0.0.0.0), signed connectors only, and human-in-the-loop for destructive actions.
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

# Agent memory
Memory is: what the agent is allowed to remember
Graph → Subgraph extraction → Context
Instead of:
retrieving nodes
You retrieve:
connected subgraphs relevant to the query
This dramatically improves reasoning.
| Memory Type | Latency | Accuracy | Cost   | Drift Risk | Best For  |
| ----------- | ------- | -------- | ------ | ---------- | --------- |
| Ephemeral   | 🔥 Fast | High     | Low    | None       | Reasoning |
| Short-term  | Fast    | Medium   | Low    | Low        | UX        |
| Vector      | Medium  | Medium   | Medium | High       | Recall    |
| Structured  | Medium  | 🔥 High  | Medium | Low        | Facts     |
| Episodic    | Slow    | Medium   | High   | Medium     | History   |
| Procedural  | Medium  | Medium   | Medium | High       | Skills    |
| Reflection  | Slow    | Variable | High   | Very High  | Learning  |

1. Ephemeral / Context Memory (Execution Memory)

What it is:

Current conversation window
Tool outputs within a single run
Scratchpad (chain-of-thought equivalent but structured)

Implementation:

In LangGraph: state object
In LangChain: messages / RunnableWithMessageHistory

Pros

Fast (zero IO)
Deterministic (no retrieval noise)
Perfect for reasoning continuity

Cons

Dies after execution
Context window limits → truncation
No learning

Use case

Planning
Tool chaining
Multi-step reasoning
2. Short-Term Memory (Session Memory)

What it is:

Persists across a user session (minutes → hours)
Stores recent interactions, tool calls, intermediate states

Implementation

Redis / in-memory store
Windowed message buffer
Token-limited replay

Pros

Cheap + fast
Maintains conversational continuity
Good UX

Cons

No semantic abstraction (raw logs)
Still grows → needs trimming
No long-term intelligence

Use case

Chat continuity
Multi-step workflows
3. Long-Term Semantic Memory (Vector Memory)

What it is:

Embeddings of past interactions, docs, events
Retrieved via similarity

Implementation

pgvector / Weaviate / Pinecone
Chunked + embedded memory entries

Pros

Scales to millions of entries
Semantic recall (not keyword)
Cheap storage

Cons (critical)

Retrieval ≠ correctness
Embedding drift over time
No temporal awareness
Garbage-in → garbage forever

Failure mode

“Agent remembers irrelevant but similar things”

Use case

Knowledge recall
Document grounding
4. Structured Memory (Relational / Graph Memory)

What it is:

Explicit entities + relationships
Facts stored as structured data

Implementation

Postgres (tables)
Neo4j / graph DB
JSONB + constraints

Pros

Deterministic queries
No hallucinated relationships
Supports reasoning (joins > similarity)

Cons

Hard to maintain automatically
Requires extraction pipeline
Schema evolution pain

Use case

User profiles
Contracts, entities, obligations
State machines
5. Episodic Memory (Event Logs + Summaries)

What it is:

“What happened” over time
Stored as events → summarized hierarchically

Implementation

Append-only log (Kafka / DB)
Periodic summarization (LLM)
Time-indexed retrieval

Pros

Temporal awareness
Traceability
Supports retrospection

Cons

Summarization loss
Expensive to maintain
Needs compaction strategy

Use case

Agent self-reflection
Audit trails
Workflow tracking
6. Procedural Memory (Skill Memory)

What it is:

“How to do things”
Learned tool usage patterns, workflows

Implementation

Stored prompts
Tool traces → distilled into reusable plans
Policy graphs

Pros

Improves efficiency over time
Reduces token usage
Enables autonomy

Cons

Hard to generalize correctly
Risk of overfitting bad patterns

Use case

Reusable agent skills
Automation pipelines
7. Reflection / Meta Memory (Self-Improvement Layer)

What it is:

Stores failures, critiques, improvements

Implementation

Post-run evaluators
Critique → store → influence future runs

Pros

Continuous improvement
Reduces repeated mistakes

Cons

Feedback loops can amplify errors
Needs strong filtering

Use case

Agent learning systems
Autonomous agents
8. Working Memory (Intermediate Reasoning State)

What it is:

Structured intermediate state across steps
Not raw messages—typed state

Implementation

LangGraph state
Typed dict / Pydantic schema

Pros

Clean reasoning
Tool-friendly
Debuggable

Cons

Requires upfront design

Observed architecture pattern OpenClaw:
1. Event-Centric Memory
Everything is an event
Stored chronologically
Enables replay + audit
2. Structured + Semantic Hybrid
Extracts entities into structured form
Also stores embeddings for retrieval
3. Agent-State Persistence
Agents have persistent state across runs
Not just stateless function calls
4. Memory Pipelines
Ingestion → Extraction → Storage
Not direct “LLM writes to DB”
5. Reflection Hooks
After execution → evaluate → store insights
Key Insight:

OpenClaw treats memory as a data engineering pipeline, not a feature.

[Ingestion of Raw document]
   ↓
[Docling parsing]
   ↓
[Clause Extraction Agent]
   ↓
[Entity Extraction Agent]
   ↓
 [Graphiti Layer]
(Entity + Relation Extraction)
                ↓
-------------------------------
| Graph DB (Neo4j / Postgres) |
-------------------------------
| Structured Store (Postgres)
| - clauses
| - entities
| - relations
contracts
id (uuid)
title
document_type
jurisdiction
language
created_at
contract_versions
id
contract_id
version_no
uploaded_by
created_at
clauses
id
contract_version_id
clause_type
text
confidence
entities
id
clause_id
entity_type
normalized_value
raw_value
confidence
risks
id
clause_id
risk_type
severity
explanation
confidence
human_reviews
id
artifact_type
artifact_id
reviewer_role
decision
comment
2.2 Graph Memory (THIS is your moat)

Represent relationships explicitly.

nodes
node_id
node_type   -- PARTY, OBLIGATION, CLAUSE, PRECEDENT
payload
edges
from_node
to_node
relation_type
confidence
Examples
PARTY → indemnifies → PARTY
OBLIGATION → triggered_by → EVENT
CLAUSE → supported_by → PRECEDENT

This enables queries no LLM can do reliably.
---------------------------
   ↓
[Embedding Pipeline]
   ↓
[Vector Store (pgvector)]
   ↓
[Episodic Log (events table)]
   ↓
[Reflection Layer]
Critical Mistake Most Engineers Make
They do this:
User → LLM → Store in vector DB → Done
This is wrong.
Correct:
Raw Data → Normalize → Extract → Validate → Store

Memory is a data pipeline, not a side effect.
Every memory has versions
Agents can “time travel”
➤ Memory Confidence Scores
Each memory has:
confidence
source
timestamp
Retrieval is weighted

Retrieval is Multi-Objective Optimization
Not just:
similarity(query, memory)
But:
score = w1 * semantic_similarity
      + w2 * recency
      + w3 * trust_score
      + w4 * relevance_to_task

Build:

Memory Router Agent
decides:
where to store
what to store
what to forget

Because the real problem is not remembering.

It’s controlled forgetting.
Where Graphiti Alone Fails

If you only use Graphiti:

No ranking of relevance
No semantic fallback
No session awareness
No memory lifecycle

You’ll get:

“Correct graph, useless agent”

⚠️ Where Cognee Alone Fails

If you only use Cognee:

Weak extraction quality
Poor relationship fidelity
Over-reliance on embeddings

You’ll get:

“Smart agent, wrong facts”

They decouple responsibilities:

1. Write Path (Deterministic)
Text → Graphiti → Validation → Structured DB
deterministic
auditable
schema-controlled
2. Read Path (Probabilistic)
Query → Cognee → Multi-source retrieval → LLM
flexible
context-aware
optimized for relevance


# Context Engineering
The talk details different memory structures to emulate human cognitive functions, including persona memory for personality, toolbox memory for managing tool schemas, conversation memory for history, and workflow memory for learning from past actions 

Short-Term Memory Techniques (20:13): Methods to optimize the context window during an active session:
Trimming (20:17): Dropping older messages.
Compaction (21:11): Dropping old tool outputs while keeping interactions.
Summarization (23:29): Compressing prior messages into dense, structured summaries (26:49 demo).

Challenges in Context Management (6:38)
Context Burst: Sudden spikes in tokens (like dumping a huge policy document) can overwhelm the model. Do control tool outputs to send only relevant data. Don't dump entire datasets into the prompt.
Context Conflict: Occurs when contradictory instructions are injected (e.g., system prompt says "no refunds" but a tool output says "issue refund for VIP"). Do ensure prompt hygiene. Don't allow conflicting information in the context window.
Context Poisoning: Inaccurate information (hallucinations) gets stored in memory and propagates. Do validate summarized information. Don't let old, incorrect summaries overwrite new, accurate data.
Short-Term Memory Techniques (20:13)
Trimming (20:17): Do drop older messages to keep the context window fresh. Don't trim mid-turn, as this breaks conversational flow; instead, trim at logical turn boundaries.
Compaction (21:11): Do remove old tool call outputs while keeping the main conversation intact. Don't lose important conversational history just to save tokens.
Long-Term Memory Patterns (5:16 & 36:09 Demo)
Techniques: Use state objects to store goals, structured notes for key facts, and memory-as-a-tool for retrieval. Do evolve memory from simple structures to complex paragraphs as needed.
Persistence: Do use persistent storage (like a database) to store memories across sessions.
Continuity: Do inject relevant, personalized history into the system prompt to make the agent feel intelligent (e.g., remembering a user's previous device issues in a new session).
Best Practices & Evaluations (41:35)
Memory Management: Do promote stable, reusable facts to memory and actively forget temporary, low-confidence information. Don't let memory become bloated with stale data.
Evaluation: Do run evaluations to measure the uplift in performance with memory on vs. off. Do develop memory-specific evaluations that test long-running tasks and context handling

# Guardrails

1. Guardrails ka real scope (India-specific)

Guardrails are not optional, they are existential.

A. Legal Liability Guardrails
Must never claim “legal advice”
Must surface:
Confidence score
Source (clause + judgement + statute)
Mandatory disclaimers + audit logs
B. Hallucination Control
Retrieval-first architecture
No free-form answers without citations
“Insufficient data” must be a valid output
C. Data Privacy

Indian contracts contain:

PAN, Aadhaar, bank details
Trade secrets

So:

PII redaction
On-prem / VPC deploy option
Encryption at rest + in transit

If you ignore this, enterprises won’t touch it.

# Evals
Evaluation Metrics (Stop Lying to Yourself)

Accuracy ≠ BLEU ≠ ROUGE.

You need legal correctness metrics.

3.1 Clause Detection Metrics
Boundary precision/recall
Clause-type confusion matrix

Failing here breaks everything downstream.

3.2 Entity Metrics
Exact match accuracy
Normalization accuracy (₹ vs words)
False positive penalty (VERY important)

A wrong entity is worse than a missing one.

3.3 Risk Assessment Metrics
Human agreement rate
Severity misclassification rate
False alarm rate

Lawyers hate noise.

3.4 Compliance Metrics
Statute grounding accuracy
Precedent relevance score
Jurisdiction correctness

One wrong citation = trust collapse.

3.5 System Metrics (enterprise reality)
Reproducibility (same input → same output)
Override frequency
Review time reduction

This is what enterprises buy.

# AI Gateway
what it should consist of 
websockets RPC, API server, session manager, channel router, plugin/hooks system, cron jobs, user auth, rate limit, a model provider Factory to use any, free/premium checker
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

 