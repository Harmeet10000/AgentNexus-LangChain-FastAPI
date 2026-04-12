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
73. figure out wrt fastAPI v0.133 and ruff if response_model or return type is better Resolve the ORJSON/response-model conflict. plus v0.135 has now first class supprt for SSE now    DONE
47. check whether i will need to use sandboxed execution environemnt in future     DONE
42. fix the search code as it is not using the pg_textsearch, pgvectorscale, pg_trgm etc properly  with Kiro      DONE
79. check what performance optimisation should i do in pageindex and langextract and whether should i use pydantic or a dataclass and also check to replace asyncio with asyncer        DONE   
135. see before/after agent/model wrap_model_call wrap_tool_call   DONE
117. for AI gateway checkout pydantic gateway, mastra, platformatic         DELAYED
6. set up performance tests 
46. use CacheBackedEmbeddings fore reusing embeddings
48. check the page https://docs.langchain.com/langsmith/deployments#

49. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin

94. check ripgrep, tree-sitter, zoekt for creating search tool that you can expose to an LLM to replace a traditional vector database and can these be used to search through texr, PDF and more?    DELAYED
86. add tests that suits the project
90. discover RAGFlow, OpenRAG if or if not to use it
59. No skill composition
Skills are flat callables. Theres no way to chain skills (skill A output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.
98. check how can Port & Adapter/strategy & factory can help 
64. No eval framework. Theres no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
116. check the logic in rate_limit and circuit breaker if a more clean implementation with design patterns and dependecy inversion can be written and also check the circuit breaker redis client should be sync or async 
61. see docassemble, fpdf2, python-docx and other libraries for generating final PDFs/docs
53. add voice support by using gemini 3 for TTS and STT  with websockets
62. can i use openRouter keys for my Gemini model               
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
138. add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer, vector_store and other places where required in tools and do the same for DB, redis
139. what functional programming patterns should i use in FastAPI, python
146. use the result package and write it in copilot instructions and implement the plan written in this and check how exception should be written like raise and let GEH handle it or  except Exception/ExceptionName as e:
133. use pydantic for state management in langraph and convert all typedDict to pydantic 
121. figure out the types of memory that a agent can have and which type does fit my needs    eg cognee, honcho, episodic etc
58. write a proper langchain-langgraph thingies
57. No agent-to-agent message passing format standard and  make a standardized AIMessage for passing in-between agents and tools and also make a ToolMessage
131. what is annotated, annotations, self vs cls, Iterable,  is callable in both typing and collection.abc?, a class receiving something in [] going in contructor or where and what happens in () in a class, what is a class in python  
125. use Call a subgraph inside a node for Open Deep Research
137. what is ToolNode, ToolRuntime, conditional_routing, chatpromptTemplate, PromptTemplate, messagePlaceholder, agentExceutor, context_schema, MessagesState, in langgraph, how does context differ from store, is context_schema differnet from AgentState or same, is custom state schema different from both context_schema and AgentState and what is context passed in agent.ainvoke
132. how will SystemMessage, HumanMessage, AIMessage, ToolMessage look like in a create_agent and inside langgraph and when in node is passing to another
130. correctly write all the arguments passes in init_chat_model and chatgenerativeaigoogle
136. use LangExtract outputs to build rich graph knowledge from your legal documents.
149. add langchain-cisco-aidefense, compact-middleware, langchain-collapse
150. what kind of text splitters do i need. diff in PGvector and pgvectorstore in langchain
151. see cogneeRetriver how does vertex ai differ from 
152. add a hydration node after checkpointer  LangGraph calls /resume. The checkpointer pulls the V1 state blob from the database and injects it into the V2 graph. The V2 graph expects compliance_region, doesn't find it, throws a KeyError, and the entire thread permanently crashes. The user's work is irrecoverably lost.

    The Insider Solution: Never trust the injected state from a checkpointer on resume without a migration layer. You must implement a StateHydrationNode as the absolute first step of any resume operation. This node intercepts the raw dictionary from the database, checks a schema_version key (which you must manually add to your base state), and runs a migration script to populate default values for any new fields introduced in newer deployments before allowing the core logic nodes to touch the state. Treat your LangGraph state with the exact same rigor as you treat your production database schema.
    
153. check this pattern again ```python
init_chat_model()
create_agent()
```

---

## 9.2 Correct Pattern

```python
research_agent = create_agent(...)

def node(state):
    return research_agent.invoke(state)
```
154. what are state machine  in design patterns

<!-- memory usage of FastAPI app -->
"memoryUsage": {
        "rss": "794.28 MB",
        "vms": "6552.59 MB"
      },



# Upgrades


2.(after writing code) When using AsyncPostgresSaver (or any checkpointer) for persistence, LangGraph serializes and saves the entire state blob at the end of every single node execution (super-step). If your messages array contains large document extractions or base64 PDFs, your state blob will rapidly swell to megabytes. Multiplied by 1000s of concurrent sessions, your PostgreSQL database will choke on I/O operations, and your latency will spike from 500ms to 5 seconds per step just writing to the DB.

    The Insider Solution: The "Pointer State" Pattern.
    Do not store document content in state["messages"].

    When the IngestionAgent reads a document, it must immediately write the text payload to a cheap, fast object store (like Redis or S3) and generate a UUID.

    The agent only appends the UUID to state["context_ptrs"].

    You write a custom StateModifier function that runs inside the LLM node. Right before calling the LLM, this function dynamically fetches the text using the UUID, injects it into the LLM's context window, gets the response, and then throws the text away.

    The checkpointer only ever saves the UUIDs. Your LangGraph state remains less than 5KB, your database I/O drops by 99%, and your system can smoothly scale to 10,000+ concurrent state transitions without breaking a sweat. Furthermore, use LangGraph's Store API for the actual long-running task idempotency, keeping the graph State purely for the immediate transition logic.

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

17. The Graphiti entity deduplication trap nobody documents: When you write "Acme Corp INDEMNIFIES GlobalTech Ltd" and later "Acme Corporation shall indemnify GlobalTech", Graphiti's LLM-powered entity extraction creates TWO separate entity nodes — Acme Corp and Acme Corporation — unless you pre-normalise entity names before writing. The deduplication only works reliably when entity names are lexically identical. The fix: run a lightweight entity canonicalisation pass in your entity_extraction node — map party names to canonical IDs (party_id: "acme_corp") and write those to Graphiti, not the raw text. Your obligation chain queries will otherwise silently miss half the edges.
18. Idempotency key collision is a business logic bug, not a tech bug: If two users submit the same clause text from different documents, hash(step_id + input + user_id) produces different keys because user_id differs. That's correct. But if the same user submits two different documents with identical clause text (common in NDAs), the input_data dict differs only in clause_id — so they get different keys. That's also correct. The trap is if you ever hash the clause TEXT as the input — then you've accidentally made your system treat legally distinct clauses as identical because they share boilerplate. Always hash structural IDs (clause_id, doc_id), never content.
19. cognify() is a full graph rebuild, not an append: Cognee's cognify() call processes the ENTIRE dataset, not just the newly added documents. If you call it per-document in persist_memory_node, you'll see quadratic runtime growth as the user's legal_reports dataset grows. The production pattern: batch cognee.add() calls in persist_memory_node, but defer cognify() to a nightly Celery beat task. Your search_episodic_memory() will return slightly stale results (yesterday's graph) but avoid blocking the live pipeline. The Cognee team calls this "async cognification" and it's the recommended pattern at scale — it's just not in their quickstart docs.


# Current implementation
## Phase 1 — HTTP Upload (IngestionGraph)

**API route:** `POST /ingestion/documents/upload`

The client sends a multipart form with the raw file. The router calls `IngestionService.ingest_document()`, which calls `ingestion_graph.ainvoke(initial_state)`. This is a blocking await — the HTTP response does not return until the graph completes. There is no checkpointer on this graph. If it fails, the client re-uploads. There are no HITL interrupts. No LangGraph Store. No Redis idempotency on the graph itself (idempotency lives inside tools, which this graph doesn't use).

`IngestionGraph` has three nodes.

**`extract_node`** receives the raw document text (up to 12,000 characters — hard capped to prevent token explosion). It calls `extraction_llm.ainvoke()` with `EXTRACTION_PROMPT` as a `SystemMessage` and the document text as a `HumanMessage`. The LLM used here is `flash_llm` (Gemini Flash) with `thinking_level="none"` — fast and cheap. The prompt enforces a JSON-only output with entities and relationships. After the call, the node strips markdown fences from the response content and parses it with `json.loads`. It writes `extracted_entities` and `extracted_relationships` back to `IngestionState`. No memory is fetched here. No tools called. No graph traversal. Pure LLM extraction.

**`validate_node`** receives the raw extracted lists. No LLM call here — this is deterministic Python. It filters entities where `confidence > 0.7` AND `name` is non-empty AND `type` is one of the five allowed values. It builds a `valid_entity_ids` set from the filtered entities, then filters relationships where both endpoints exist in that set AND `confidence > 0.7` AND `type` is non-empty. This is the validation layer from Section 3 of the plan — never trust LLM output directly. Dropped counts are recorded in state. No memory fetch, no tools.

**`embed_store_node`** runs the actual Postgres writes using `AsyncSession`. It opens a single transaction and processes everything inside it. For each validated entity, it runs an `INSERT ... ON CONFLICT (normalized_name, entity_type) DO UPDATE SET confidence = GREATEST(...)` — this is the deduplication guard. If the same party appears in two documents, the second insert updates confidence rather than creating a phantom duplicate. For every entity whose `type == "CLAUSE"`, it also calls `embedding_fn(clause_text)` to get a 1536-dimensional vector and inserts a row into the `clauses` table with the pgvector embedding. For relationships, it inserts into the `relationships` table with `from_entity_id` and `to_entity_id` mapped from the extraction IDs to the real Postgres UUIDs via `entity_id_map`. If anything fails, the entire transaction rolls back. `ingestion_complete: True` is only written if no exception was raised.

After the graph completes, `IngestionService` reads the result state and returns a `DocumentUploadResponse` with `doc_id`, `entity_count`, `clause_count`, `relationship_count`. The client stores `doc_id` and uses it to open the WebSocket.

At this point the following data exists in Postgres: rows in `entities`, rows in `clauses` with embeddings, rows in `relationships`. Nothing has been written to Graphiti, Neo4j graph edges, Cognee, or Redis yet. Phase 1 only writes to Postgres.

---

## Phase 2 — WebSocket Session (agent_saul LangGraph)

**API route:** `WS /agent-saul/ws/{thread_id}`

The client connects, sends a `WSStartMessage` with `doc_id`, `user_query`, optional `thread_id` (for resuming), and `permissions`. The router calls `AgentSaulService.run_session()`. This is the main loop. The service calls `graph.astream_events(initial_input, config=config, version="v2")` in an async loop. After each stream drains, it calls `await graph.aget_state(config)` to inspect what comes next — completion, interrupt, or error. If an interrupt is detected, it emits a `WSHITLInterruptFrame` and waits for a `WSResumeMessage` on the same WebSocket connection, then issues `Command(resume=payload)` to continue the graph.

The checkpointer here is `AsyncPostgresSaver`. Every node return is checkpointed. If the process dies mid-run and the client reconnects with the same `thread_id`, the graph resumes from the last successful checkpoint. The `RunnableConfig` is built with `configurable={"thread_id": thread_id}` — this is the key that links all checkpointer reads and writes to the right session.

The graph has 15 nodes. Here is what each does.

**`gateway_node`** runs first on every execution including resumes. No LLM. No I/O beyond state reads. It validates that `doc_id` is present, injects `gateway_validated: True` and `session_start_ts` into `working_memory`, and sets `status = QNA_CLARIFICATION`. Returns in milliseconds. This node acts as a consistency gate — if it fails, the whole pipeline stops before any LLM is called.

**`qna_node`** uses `flash_llm.with_structured_output(QnAOutput)`. The LLM call uses `SystemMessage(content=_QNA_SYSTEM_PROMPT)` followed by `state["messages"]`. The prompt asks the model to score confidence from 0.0 to 1.0. No memory is fetched here — this is the first node after gateway and the user's raw query is the only input. If `result.confidence < 0.72`, the node calls `interrupt()` with a `HITLInterruptType.CLARIFICATION_NEEDED` payload. The graph pauses. The service emits `WSHITLInterruptFrame` to the client. The client shows the clarifying question in the UI, the user types an answer, the client sends `WSResumeMessage(action="approve", feedback="the answer")`. The service calls `Command(resume={"action": "approve", "feedback": "..."})`. The graph resumes inside `qna_node` after the `interrupt()` call, receives the human answer as a dict, appends it as a `HumanMessage` to `state["messages"]`, and returns `status = QNA_CLARIFICATION` so the conditional edge loops back to `qna_node`. This loops until confidence crosses the threshold, at which point the node returns `status = PLAN_PENDING` and the conditional edge routes to `orchestrator_node`.

**`orchestrator_node`** uses `pro_llm.with_structured_output(OrchestratorAction)`. This is Gemini Pro with `thinking_level="high"`. The output is a discriminated union — `OrchestratorActionType` is one of `start_pipeline`, `continue`, `synthesize`, or `done`. The node builds messages from `SystemMessage(_ORCHESTRATOR_SYSTEM_PROMPT)` plus `state["messages"]`. After the LLM call, it validates the `target_node` if the action type is `continue` — if the target isn't in `_VALID_WORKER_NODES`, the node returns `status = FAILED` immediately without routing anywhere. The orchestrator increments `current_step` on every invocation. The routing function `route_from_orchestrator` reads `state["orchestrator_action"]` and returns the appropriate node name as a string. On the first invocation, `orchestrator_action` is `None`, so the routing function returns `"planner"` unconditionally.

**`planner_node`** uses `flash_llm.with_structured_output(PlannerOutput)`. It builds the execution plan as a list of `PlanStep` objects with typed `PlanActionType` values. After the LLM generates the plan, the node immediately calls `interrupt()` with `HITLInterruptType.PLAN_APPROVAL` and the full plan serialized as JSON. The graph pauses. The client receives the plan for human review — this is where the reviewer can inspect the proposed steps, modify them, or reject them. On `WSResumeMessage(action="approve")`, the node commits the original plan. On `action="modify"`, it takes `modified_plan` from the resume payload and validates each step through `PlanStep.model_validate()`. On `action="reject"`, it writes `status = PLAN_REJECTED` and an `AgentError`, and the orchestrator routing function checks for `PLAN_REJECTED` and routes back to planner for re-planning. The approved plan is written to `state["plan"]`.

**`ingestion_node`** in the agent_saul graph is now a lookup, not a processor. It reads the raw document text stored during Phase 1. Currently stubbed — you need to wire the actual lookup (`SELECT document_text FROM documents WHERE doc_id = :doc_id` or a MongoDB/S3 fetch depending on where you store raw files). The OCR confidence check and `HITLInterruptType.OCR_REUPLOAD` path remain — if the stored text has low quality markers, it can still interrupt and ask for a re-upload, but this is the exception path, not the default. Returns `document_text` in state.

**`normalization_node`** uses `flash_llm.with_structured_output(NormalizedDocument)`. Takes `state["document_text"]` and sends it to the LLM to produce a section hierarchy with resolved clause references. No memory fetch. No tools. Returns `normalized_document` as a typed `NormalizedDocument` Pydantic model in state.

**`segmentation_node`** uses `flash_llm.with_structured_output(ClauseSegmentationOutput)`. Takes the normalized document sections, concatenates them, and asks the LLM to identify clause boundaries and classify each clause into `ClauseType`. Returns a list of `ClauseSegment` objects written to `state["segments"]` via the `operator.add` reducer — meaning if this node ran before (on a resume), its results append rather than overwrite.

**`entity_extraction_node`** is the Send fan-out target. It does NOT receive `LegalAgentState`. It receives a `ClauseExtractionInput` dict — one per clause segment — dispatched by `dispatch_entity_extraction()` from the segmentation node's conditional edge. LangGraph runs all entity extraction nodes in parallel. Each instance calls `flash_llm.with_structured_output(EntityExtractionOutput)` with the single clause text plus document context (jurisdiction, document type). The output `CitedEntity` objects include a `Citation` with `claim`, `source`, and `confidence`. Results accumulate via `operator.add` on `state["extracted_entities"]`. All parallel instances complete before `relationship_mapping_node` starts.

**`relationship_mapping_node`** has two responsibilities. First, it calls `flash_llm.with_structured_output(_RelationshipMappingLLMOutput)` to extract typed legal relationships from the entity list. Second, it calls `write_clause_episodes_to_graphiti()` directly — not via a tool, because the LLM should never decide when to write to memory. This function opens `asyncio.Semaphore(5)` and runs clause episode writes in parallel, then sequential relationship edge writes. The semaphore prevents overwhelming Neo4j's connection pool. The idempotency guard checks before each write — if the clause was already written in a previous run (e.g. after a crash and resume), the write is skipped. Any failed writes produce `AgentError` entries in state but do not block the pipeline. Returns `state["relationships"]` via `operator.add`.

**`risk_analysis_node`** is the first node that uses `create_react_agent`. Before calling the agent, it calls `build_agent_context(state, graphiti_service, task="risk_analysis", scope=RISK_SCOPE)`. This function: checks `RISK_SCOPE.allows_source("graph")` → calls `graphiti_service.search_for_risk_context()` to get Graphiti episodes scored by `0.5·semantic + 0.2·recency + 0.2·trust + 0.1·task_relevance`; checks `RISK_SCOPE.allows_source("vector")` → queries pgvector (currently stubbed); checks `RISK_SCOPE.allows_source("structured")` → queries Postgres entities table filtered to `CLAUSE` and `OBLIGATION` types only (RISK_SCOPE). It then filters tool messages from `state["messages"]`, trims to 3,500 tokens with `strategy="last"`, and builds a `SystemMessage` with the structured context prefix in `{goal, task, agent_scope, doc_type, jurisdiction, warnings, memory_context}` format. The resulting message list is passed to the risk agent.

The risk agent runs with `risk_tools = [query_knowledge_graph, get_obligation_chain, detect_graph_conflicts]`. The LLM (Pro, `thinking_level="high"`) decides which tools to call and when. `query_knowledge_graph` calls `graphiti_service.search_for_risk_context()` scoped to `RISK_SCOPE.top_k=8`. `get_obligation_chain` calls `graphiti_service.get_obligation_chain()` for semantic search, then `Neo4jSubgraphExpander.get_obligation_chain_cypher()` for structural depth-N traversal using raw Cypher via `app.state.neo4j_driver`. `detect_graph_conflicts` calls `Neo4jSubgraphExpander.detect_conflicts()` to find circular obligations and override chains. Every tool call checks the Redis idempotency cache before executing. Results are written to both Redis (24h TTL) and Postgres `tool_executions` after execution.

**`compliance_node`** follows the exact same pattern as risk_analysis but uses `COMPLIANCE_SCOPE` — which adds `CONTRACT` and `ORG` to the allowed entity types, uses `depth=1` (shallower graph traversal), `time_filter="all"` (not just recent 90 days), and gates only `graph` and `structured` sources (no vector). The compliance agent uses `compliance_tools = [search_legal_precedents, retrieve_statute_section]`. `search_legal_precedents` calls `graphiti_service.search_for_precedent_chains()` filtered by jurisdiction, and separately runs a full-text search on the `statutes` Postgres table using `plainto_tsquery`. If `total_sources < 2`, it sets `insufficient_basis=True` in the `ToolResult` — the compliance agent's system prompt instructs it to respond with "Insufficient legal basis" instead of proceeding. `retrieve_statute_section` does an exact lookup on `statutes` by `act_name ILIKE` and `section_ref` when the agent already knows the statute to fetch.

Risk analysis and compliance run as parallel branches from `relationship_mapping_node`. LangGraph runs both concurrently because both have direct edges from the same upstream node. They join at `grounding_verification_node` — LangGraph waits for both to complete before proceeding.

**`grounding_verification_node`** uses `flash_llm.with_structured_output(GroundingVerificationOutput)`. It takes the summaries from `state["risk_analysis"]` and `state["compliance_result"]` and asks the model to flag any claims that lack citation support. Returns `state["grounding"]` with `verified: bool` and `unverified_claims: list[str]`. No memory fetch. No tools. This is purely a validation pass on what the analysis nodes produced.

**`human_review_node`** is a mandatory HITL node — there is no code path that bypasses it. It calls `interrupt()` with `HITLInterruptType.HUMAN_REVIEW_REQUIRED`, the risk summary, compliance summary, unverified claims, and the first 20 clause segments (capped to control payload size). The graph pauses. The reviewer uses the frontend UI to read all findings, add `ReviewOverride` objects for any clauses they disagree with, and either approve or reject. On rejection, the node writes `status = FAILED` and routes back to the orchestrator. On approval, it creates a `HumanReviewOutput` with the reviewer's `reviewer_id`, `reviewer_role`, `overrides`, and `notes`. After this node completes, `human_approved=True` is semantically in play for all downstream memory writes — the trust score on Graphiti episodes will be `1.0`.

After `human_review_node`, the conditional edge routes back to `orchestrator_node`. The orchestrator reflects on the human review output and decides the next action — typically `OrchestratorActionType.SYNTHESIZE`, which routes to `finalization_node`.

**`finalization_node`** uses `pro_llm.with_structured_output(FinalReport)`. It assembles all analysis — risk findings with human overrides applied, compliance findings, suggested actions, all citations — into a single `FinalReport` Pydantic model. No memory fetch. No tools. Returns `state["final_report"]`.

**`persist_memory_node`** is the memory commit point. It calls `write_final_report_to_memory()`, which does three writes in sequence: Graphiti (final report as a high-trust episode, `group_id=user_id`), Cognee (`store_final_report()` + `store_relationships()`), and then `write_memory_persisted_event()` for the immutable events log. The events write is the dual-write — it fires after both mutable writes succeed. If Graphiti or Cognee fail, the errors are captured in `AgentError` entries but the pipeline still sets `status = COMPLETED`. Memory write failure is not a pipeline failure. Returns `long_term_refs` with `graphiti:` and `cognee:` prefixed keys.

After `persist_memory_node`, the graph reaches `END`. `state_snapshot.next` is empty and `state_snapshot.tasks` has no pending interrupts. The service sends `WSDoneFrame` with the final report summary. The WebSocket closes.

---

## Phase 3 — Background (ReconciliationGraph + Decay)

**No API route.** Both tasks are triggered by Celery.

**ReconciliationGraph** is triggered by `run_reconciliation_for_user()` Celery task. The beat schedule fires `run_reconciliation_for_active_users()` every 6 hours, which queries for distinct `user_id` values in `entities` created in the last 6 hours and dispatches one `run_reconciliation_for_user` task per user. The graph is compiled once at lifespan (`build_reconciliation_graph()`) and stored in `app.state.reconciliation_graph`. The Celery task wraps `asyncio.run(reconciliation_graph.ainvoke(...))`. No checkpointer on this graph. No HITL. No WebSocket.

**`fetch_existing_node`** queries Postgres for two sets: recently added entities for the user (`created_at > NOW() - N hours`), and similar existing entities found via `LEFT(normalized_name, 10)` prefix match across the same user's entity history. The prefix match is a pragmatic similarity heuristic — it catches most common alias patterns without needing full fuzzy matching. Returns both sets in state.

**`reconcile_node`** calls `reconcile_llm.ainvoke()` with `RECONCILE_PROMPT` — the loss aversion bias prompt. This uses `flash_llm` (not Pro — the task is structured comparison, not deep reasoning). The prompt uses the bias principles from Section 16.4: loss aversion ("NEVER delete without justification"), constraint amplification (prefer recent + higher confidence), error minimization (when uncertain, choose `ignore`). The model returns a JSON decision with `merge`, `update`, and `ignore` arrays. The `ignore` array is the expected majority — most pairs that look similar are not actually duplicates in legal contracts.

**`apply_changes_node`** executes the merge decisions. For each merge, it redirects all `from_entity_id` and `to_entity_id` FK references in the `relationships` table from `discard_id` to `keep_id`, then deletes the discarded entity. If you skip the redirect step, the `ON DELETE CASCADE` on the FK fires and destroys all relationships associated with the discarded entity. For updates, it builds a dynamic `SET` clause from the `fields` dict and executes it. The entire apply block runs inside a single `session.begin()` transaction — either all changes apply or none do.

**`write_versions_node`** writes `memory_versions` rows for every entity that was merged or updated. For each entity, it fetches `MAX(version)` from `memory_versions`, increments by 1, and inserts a full JSON snapshot of the current entity row plus the reason and run ID. This is the CRDT-lite audit trail — you can reconstruct any entity's history by selecting all versions for that `entity_id` ordered by version ascending and replaying them. The concurrent write race condition (two workers reading the same MAX version) needs a `SELECT ... FOR UPDATE` lock on the entity row before the MAX query — this is the production fix not yet in the current code.

**Memory decay task** is triggered by `run_memory_decay()` Celery beat, scheduled nightly at 2 AM. It uses raw `asyncpg` (not SQLAlchemy) for bulk batch updates — `executemany` is significantly faster than individual ORM updates at scale. The formula per entity is `0.4 * exp(-0.01 * age_days) + 0.3 * min(1.0, access_count / 10.0) + 0.3 * confidence`. Time factor uses `λ=0.01` giving roughly a 70-day half-life. Usage factor saturates at 10 accesses. Confidence factor is the stored confidence from extraction. Entities with `decay_score < 0.15` are flagged as archive candidates. The current code doesn't delete them — it only updates the score. The sweep that actually removes zombie nodes (no edges, low decay, not accessed in 6 months) is a separate cleanup query you need to add as a second beat task.

---


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
Evaluation: Do run evaluations to measure the uplift in performance with memory on vs. off. Do develop memory-specific evaluations that test long-running tasks and context 


Here're 10 Actionable Insights: 

1. Choose Your Memory Strategy Based on Task Type

> Use trimming (keeping last N turns) for independent tasks where recent context matters most.
> Use summarization for long conversations where you need to preserve decisions and constraints across the entire session.

2. Implement Context Trimming with Turn Boundaries

> Define a "turn" as a user message plus everything that follows until the next user message.
> Keep complete turns intact rather than cutting mid-conversation to preserve context coherence.

3. Design Smart Summarization Prompts

> Include structured sections like "Product & Environment," "Steps Tried & Results," and "Current Status & Blockers."
> Add contradiction checks and temporal ordering to prevent summary drift and hallucinations.

4. Set Context Limits Strategically

> Configure max_turns based on your conversation distribution - analyze how many turns typical tasks require.
> Keep keep_last_n_turns <= context_limit to ensure the most recent interactions remain verbatim.

5. Handle Async Operations Properly

> Release locks during potentially slow summarization operations to avoid blocking.
> Re-check conditions after async operations complete to prevent race conditions.

6. Track Metadata Separately from Messages

> Store only allowed keys (role, content, name) in messages sent to the model.
> Keep debugging info, timestamps, and flags in separate metadata objects for observability.

7. Implement Idempotent Behavior

> Design your system so multiple calls to add_items() won't duplicate summaries.
> Use synthetic flags to mark AI-generated summary messages versus real user content.

8. Build Progressive Summarization

> Summarize older content into synthetic user→assistant pairs when limits are exceeded.
> Preserve the exact boundary where summarization occurs for debugging and evaluation.

9. Create Evaluation Harnesses

> Use LLM-as-judge to evaluate if summaries capture important details correctly.
> Run transcript replay tests to measure next-turn accuracy with and without trimming.

10. Monitor for Context Poisoning

> Track when bad facts enter summaries and propagate through subsequent turns.
> Log before/after token counts to detect when critical details are being pruned.

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


**Best additions for this repo**

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

- `contextlib.suppress`
  Best fit: best-effort cleanup and optional deletes.
  Why not higher: I don’t see many concrete suppression-heavy areas yet, so forcing it would be cosmetic.

- `contextlib.ExitStack`
  Important nuance: for this repo, `AsyncExitStack` is probably more valuable than `ExitStack` because much of your resource management is async.
  Best fit: places that may conditionally acquire multiple async resources during app lifespan or ingestion orchestration.
  Why not higher: good tool, but only when resource lifetimes are genuinely dynamic.


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

 