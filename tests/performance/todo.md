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
75. integrate open deep search https://blog.langchain.com/open-deep-research/ and this https://github.com/langchain-ai/open_deep_research            DONE
76. legal AGENT will be based on Saul for finding out of the box ideas for legal advice also and will also have a block for how senior/experienced lawyers of supreme courts and high courts will handle this.   DONE
77.  what is annotated, annotations, self vs cls, Iterable,  is callable in both typing and collection.abc?, a class receiving something in [] going in contructor or where and what happens in () in a class, what is a class in python, what is bear typing, limit.tying, typing_extrension, learn about this syntax  Callable[[IngestionState], Awaitable[dict[str, object]]], differenece in enum, str, StrEnum, what are @abstractethod, @classmethod, @staticmethod, @aexit, @injectable, @asynccontextmanager, functools and more decorators, e.add_note in exceptions, flow()/bind()/map(), how do i import something from __init__.py, inside __init__ should i write full paths or just .filename, when should i write full paths          DONE
78.  for AI gateway checkout pydantic gateway, mastra, platformatic         DELAYED
79. set up performance tests 
80. use CacheBackedEmbeddings fore reusing embeddings
81. check the page https://docs.langchain.com/langsmith/deployments#

82. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin

83. check ripgrep, tree-sitter, zoekt for creating search tool that you can expose to an LLM to replace a traditional vector database and can these be used to search through text, PDF and more? learn more tools like this in popular coding harnesses and other harnesses     DELAYED
84. add tests that suits the project
85. discover RAGFlow, OpenRAG if or if not to use it
86. check how can Port & Adapter/strategy & factory can help 
87.  what functional programming patterns should i use in FastAPI, python,learn pattern matching & ROP,flow()/bind()/map(), learn function composition with this example and in which case should this be used 
type Composable = Callable[[Any], Any]


def compose(*functions: Composable) -> Composable:
    def apply(value: Any, fn: Composable) -> Any:
        return fn(value)

    return lambda data: reduce(apply, functions, data)

64. No eval framework. Theres no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
116. check the logic in rate_limit and circuit breaker if a more clean implementation with design patterns and dependecy inversion can be written and also check the circuit breaker redis client should be sync or async 
61. see docassemble, fpdf2, python-docx and other libraries for generating final PDFs/docs
53. add voice support by using gemini 3 for TTS and STT  with websockets
62. can i use openRouter keys for my Gemini model               
67. go and learn https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/
99. use promptfoo for detecting prompt injection attacks, automated red team attacks, 
59. No skill composition. Skills are flat callables. Theres no way to chain skills (skill A output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.
44. correct the code for crawler and the packages used
115. logs inbetween the layers are empty or not coming except start and end, should i pass logger as an argument or should i import it where needed 
140. in cognee GRAPH_COMPLETION_COT if the FEELING_LUCKY router returns a complexity score $>0.8$. This prevents token-burn on simple questions while ensuring "God-Mode" accuracy for architectural queries. If you connect to a "bare" Neo4j instance without APOC installed, the initial cognee.add() will work, but the cognee.cognify() step will fail silently or throw cryptic Cypher errors. Always verify your Neo4j instance has the APOC and GDS (Graph Data Science) plugins enabled.
146. use the return package and write it in copilot instructions and implement the plan written in this and check how exception should be written like raise and let GEH handle it or  except Exception/ExceptionName as e:, also use e.add_note and also check if i am right in passong HTTPException to APIException and other classes
148. figure out the types of memory that a agent can have and which type does fit my needs    eg cognee, honcho, episodic etc
149. what is ToolNode, ToolRuntime, conditional_routing, chatpromptTemplate, PromptTemplate, messagePlaceholder, agentExceutor, context_schema, MessagesState, InjectedToolArg, in langgraph, how does context differ from store, is context_schema differnet from AgentState or same, is custom state schema different from both context_schema and AgentState and what is context passed in agent.ainvoke
150. how will SystemMessage, HumanMessage, AIMessage, ToolMessage look like in a create_agent and inside langgraph and when in node is passing to another, see how to standardise message passing between agents - through context_schema, AgentState, AIMessage?
151. add langchain-cisco-aidefense, compact-middleware, langchain-collapse
152. see cogneeRetriver how does vertex ai differ from google_genai
153. add a hydration node after checkpointer  LangGraph calls /resume. The checkpointer pulls the V1 state blob from the database and injects it into the V2 graph. The V2 graph expects compliance_region, doesn't find it, throws a KeyError, and the entire thread permanently crashes. The user's work is irrecoverably lost.

    The Insider Solution: Never trust the injected state from a checkpointer on resume without a migration layer. You must implement a StateHydrationNode as the absolute first step of any resume operation. This node intercepts the raw dictionary from the database, checks a schema_version key (which you must manually add to your base state), and runs a migration script to populate default values for any new fields introduced in newer deployments before allowing the core logic nodes to touch the state. Treat your LangGraph state with the exact same rigor as you treat your production database schema.
    
154. check this pattern again 
init_chat_model()
create_agent()       inside a node or should i have 2 instances of cheap and expensive model and then pass it in create_agent
58. write a proper langchain-langgraph thingies
155. complete the ingestion pipeline to working condition and see where reconciliation comes init
todos:- 
    0. toons reusable chain, point 138,  break the code for reconcilliation inside langgraph_layer/ and features/,check the use from string import Template to write prompts or chatpromptTemplate with toons, use SystemPromptParts to write systemPrompt also check it
    a. docling
    b. langextract
    c. graphiti
    d. postgres RAG should be agentic
    e. celery for off loading to a queue
    f. insert the langgraph in app.state in lifespan
    g. pass default and metadata for particular config in pydantic models for agents
    h. research for RAG pipeline with Gemini
138. add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer, vector_store and other places where required in tools and do the same for DB, redis
108. use the new gemini embedding 2 for multi-modal embeddings, LLMToolSelectMiddleware 
133. use pydantic for state management in langraph and check if converting all typedDict to pydantic is useful or docs do not recommends it
57. No agent-to-agent message passing format standard and make a standardized AIMessage for passing in-between agents and tools and also make a ToolMessage
136. use LangExtract outputs to build rich graph knowledge from your legal documents.
150. what kind of text splitters do i need. diff in PGvector and pgvectorstore in langchain
17. refactor vectorStore code
18. refactor RAG code
95. implement RAG by getting inspired from this https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384
125. use Call a subgraph inside a node for Open Deep Research
130. correctly write all the arguments passes in init_chat_model and chatgenerativeaigoogle
156. keep class small and move more complex methods out of it, initialiser that builds an object shouldnt do complicated task inside a class
157. do migration of open_deep_research 
    a. remove MCP from it
    b. store the MCP code for any best practices in mcp/
    c. remove other model providers
    d. use own taily client and httpx_client and replace existing one
    e. reorganise code
158. EARS-compliant acceptance criteria, research about what Kiro does for SDD, which standards does it use
159. write cron job for memory decay and then send to celery for off loading

```

--- summarise these chapters in great detail and take video's transcript as reference for summarising

## 9.2 Correct Pattern

```python
research_agent = create_agent(...)

def node(state):
    return research_agent.invoke(state)
```

1. what are state machine  in design patterns

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

1. The Graphiti entity deduplication trap nobody documents: When you write "Acme Corp INDEMNIFIES GlobalTech Ltd" and later "Acme Corporation shall indemnify GlobalTech", Graphiti's LLM-powered entity extraction creates TWO separate entity nodes — Acme Corp and Acme Corporation — unless you pre-normalise entity names before writing. The deduplication only works reliably when entity names are lexically identical. The fix: run a lightweight entity canonicalisation pass in your entity_extraction node — map party names to canonical IDs (party_id: "acme_corp") and write those to Graphiti, not the raw text. Your obligation chain queries will otherwise silently miss half the edges.
2. Idempotency key collision is a business logic bug, not a tech bug: If two users submit the same clause text from different documents, hash(step_id + input + user_id) produces different keys because user_id differs. That's correct. But if the same user submits two different documents with identical clause text (common in NDAs), the input_data dict differs only in clause_id — so they get different keys. That's also correct. The trap is if you ever hash the clause TEXT as the input — then you've accidentally made your system treat legally distinct clauses as identical because they share boilerplate. Always hash structural IDs (clause_id, doc_id), never content.
3. cognify() is a full graph rebuild, not an append: Cognee's cognify() call processes the ENTIRE dataset, not just the newly added documents. If you call it per-document in persist_memory_node, you'll see quadratic runtime growth as the user's legal_reports dataset grows. The production pattern: batch cognee.add() calls in persist_memory_node, but defer cognify() to a nightly Celery beat task. Your search_episodic_memory() will return slightly stale results (yesterday's graph) but avoid blocking the live pipeline. The Cognee team calls this "async cognification" and it's the recommended pattern at scale — it's just not in their quickstart docs.



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

1. Implement Context Trimming with Turn Boundaries

> Define a "turn" as a user message plus everything that follows until the next user message.
> Keep complete turns intact rather than cutting mid-conversation to preserve context coherence.

1. Design Smart Summarization Prompts

> Include structured sections like "Product & Environment," "Steps Tried & Results," and "Current Status & Blockers."
> Add contradiction checks and temporal ordering to prevent summary drift and hallucinations.

1. Set Context Limits Strategically

> Configure max_turns based on your conversation distribution - analyze how many turns typical tasks require.
> Keep keep_last_n_turns <= context_limit to ensure the most recent interactions remain verbatim.

1. Handle Async Operations Properly

> Release locks during potentially slow summarization operations to avoid blocking.
> Re-check conditions after async operations complete to prevent race conditions.

1. Track Metadata Separately from Messages

> Store only allowed keys (role, content, name) in messages sent to the model.
> Keep debugging info, timestamps, and flags in separate metadata objects for observability.

1. Implement Idempotent Behavior

> Design your system so multiple calls to add_items() won't duplicate summaries.
> Use synthetic flags to mark AI-generated summary messages versus real user content.

1. Build Progressive Summarization

> Summarize older content into synthetic user→assistant pairs when limits are exceeded.
> Preserve the exact boundary where summarization occurs for debugging and evaluation.

1. Create Evaluation Harnesses

> Use LLM-as-judge to evaluate if summaries capture important details correctly.
> Run transcript replay tests to measure next-turn accuracy with and without trimming.

1. Monitor for Context Poisoning

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

# Evals through DeepEval (G-Eval)

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

# AI Security model

The 4-Step Maturity Model
Step 1: Ad Hoc (4:31 - 5:10): The initial stage where systems are built without formal risk management or security considerations.
Step 2: Foundation (5:12 - 7:22): Establishing the basics, including assigning non-human identities to agents, enabling delegation/on-behalf-of flows, and using a SIEM (Security Information and Event Management) for auditability.
Step 3: Enhanced (7:29 - 10:46): Treating agents as first-class citizens with ephemeral credentials, applying fine-grained, context-based access, and implementing real-time detection for anomalies.
Step 4: Adaptive (10:47 - 14:14): The most advanced stage involving continuous authentication, risk-based re-authentication, and real-time revocation to dynamically secure non-deterministic workflows.

1. JIT Permissions - Over-Permissioning - Force point-of-use enforcement: Verify policy compliance at the exact moment of connection to sensitive data.
2. IAM
3. hidden prompt & prompt injection proof
4. tool security - tool injection protection
5. sandbox environment
6. MCP security - Use a secure vault to manage tool credentials, providing only temporary access rather than storing long-term secrets within the MCP server
7. TTL based tokens - Occurs when tokens are intercepted or leaked (e.g., through LLM prompts). Use tokens that represent both the user (subject) and the agent (actor) to validate that an agent is authorized to work on the user's behalf. Use token exchange at each hop of a flow to ensure security propagates through the system, and restrict tokens to specific scopes to enforce the principle of least privilege
8. <https://blog.langchain.com/agent-authorization-explainer/>
9. guardrails
10. TLS - Utilize TLS/MTLS to prevent man-in-the-middle attacks and ensure all stored credentials are encrypted
11. Ensure audit logs record when a human specifically tasks an agent with an action.

## Agent Sandbox

the necessity of dedicated, secure infrastructure—or an "Agent Computer"—for running autonomous AI agents. He argues that traditional containerization is insufficient for untrusted agent-generated code and outlines the technical requirements for robust sandboxing.

1. Beyond Localhost (1:25 - 2:49): Naresh establishes that long-running agents require a purpose-built environment that maintains state, networking, and connectivity between the internet and the agent's workspace.
2. The 5 Infrastructure Layers (2:49 - 4:30): He breaks down the sandbox architecture into five critical components: isolation boundary (MicroVM), containers (e.g., Ubuntu), bash sessions/file system, networking (port routing), and persistence.
3. Runtime Decisions (4:30 - 7:10): A discussion on providing agents with full Linux instances rather than restricted shells. Key capabilities include managing background processes (like dev servers), streaming output, and supporting multiple concurrent sessions.
4. The Security Paradox (7:10 - 9:57): Naresh addresses the danger of executing untrusted code. He explains that Docker (namespaces) is a security mismatch because it shares the host kernel. He advocates for MicroVMs (Firecracker), which provide hardware-level isolation with minimal overhead.
5. Networking Architecture (9:57 - 11:37): Explanation of how to dynamically route internet traffic to specific, ephemeral sandboxes using unique session tokens and internal mapping, ensuring browser-to-agent connectivity.
6. Persistence & Cold Starts (11:37 - 15:53): A deep dive into strategies to eliminate cold starts, including pre-built images, persistent volumes, memory snapshots, and maintaining "warm pools" of ready-to-use sandboxes.
7. The 8-Line Production Sandbox (15:53 - 18:08): A conceptual breakdown of the minimal code required to orchestrate a secure, persistent, and network-accessible sandbox.
8. Scale of AI Code (18:08 - 18:48): The speaker notes the massive scale of AI-generated code (e.g., billions of lines daily on Cursor) and why infrastructure for these agents is becoming the new default.
9. Future of Runtimes (18:48 - 20:44): Discussion on moving toward even lighter solutions like V8 isolates for specific workloads where a full MicroVM may be overkill.
10. Q&A and Programmatic Gates (20:44 - 29:57): The speaker covers standardizing agent harnesses and the use of programmatic gates—a strategy where agents are given limited, specific pathways for interaction to control the "blast radius" of their actions.

### Key Takeaways for Secure Sandboxing

1. Isolation: Use MicroVMs (Firecracker) rather than standard Docker containers. MicroVMs provide a dedicated kernel per sandbox, preventing an agent from impacting the host machine (8:33).
2. Blast Radius Control: Rather than giving agents broad permissions, implement programmatic gates. These act as strictly defined interfaces that only allow the agent to perform specific, approved actions (25:48).
3. Persistence Strategy: To ensure a seamless user experience, leverage memory snapshots and warm pools. This allows the agent to resume work instantly without waiting for the environment to boot (12:51).
4. Environment over Behavior: Naresh emphasizes that the most effective security measure is to control the environment itself. By ensuring the agent operates within an isolated sandbox with no sensitive network or file system access, you mitigate the risks associated with autonomous code execution (28:35).

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

Arjan strongly recommends using modules instead (7:26-8:04). Modules are managed natively by Python, are inherently singletons, and are far more thread-safe, simple, and testable than a custom class-based Singleton. Singleton pattern has legitimate merit when used for lazy loading or controlled instantiation

the architectural design differences between Python properties and methods, emphasizing the contract these choices establish for your code.

Key Takeaways:

The Difference in Promise: A property communicates that an operation is cheap, safe to read repeatedly, and typically returns state (2:29 - 3:48). Conversely, a method implies that work is being done, potentially involving complexity, latency, or side effects (2:43 - 3:23).
Derived State: Properties are ideal for computing simple values from existing object state (e.g., checking if an account is active), as they are deterministic and side-effect-free (3:48 - 6:14).
Setters and Side Effects: While properties can have setters, they should generally avoid performing I/O or heavy operations. Persistence logic (like database saves) should be handled by explicit methods to keep code predictable and avoid blocking (8:58 - 10:29).
Async Properties: While technically possible, making properties asynchronous is considered a design smell (13:13 - 14:31). It hides asynchronous waiting behind attribute access, which violates the expected simplicity of a property. Instead, use asynchronous methods for loading and saving data (14:31 - 16:10).

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
While monkey patching allows you to test existing, tightly coupled code, Arjan notes that the process is "ugly" and difficult to maintain. This serves as a precursor to the second part of the series, where he will demonstrate how refactoring (specifically using dependency injection) simplifies testing and yields a cleaner, more modular design.

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
