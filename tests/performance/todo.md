# To-Do List

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
75.  integrate open deep search https://blog.langchain.com/open-deep-research/ and this https://github.com/langchain-ai/open_deep_research            DONE
76.  legal AGENT will be based on Saul for finding out of the box ideas for legal advice also and will also have a block for how senior/experienced lawyers of supreme courts and high courts will handle this.   DONE
77.  what is annotated, annotations, self vs cls, Iterable,  is callable in both typing and collection.abc?, a class receiving something in [] going in contructor or where and what happens in () in a class, what is a class in python, what is bear typing, limit.tying, typing_extrension, learn about this syntax  Callable[[IngestionState], Awaitable[dict[str, object]]], differenece in enum, str, StrEnum, what are @abstractethod, @classmethod, @staticmethod, @aexit, @injectable, @asynccontextmanager, functools and more decorators, e.add_note in exceptions, flow()/bind()/map(), how do i import something from __init__.py, inside __init__ should i write full paths or just .filename, when should i write full paths          DONE
58.  write a proper langchain-langgraph thingies    DONE
149.  what is ToolNode, ToolRuntime, conditional_routing, chatpromptTemplate, PromptTemplate, messagePlaceholder, agentExceutor, context_schema, MessagesState, InjectedToolArg, in langgraph, how does context differ from store, is context_schema differnet from AgentState or same, is custom state schema different from both context_schema and AgentState and what is context passed in agent.ainvoke     DONE
150.  how will SystemMessage, HumanMessage, AIMessage, ToolMessage look like in a create_agent and inside langgraph and when in node is passing to another, see how to standardise message passing between agents - through context_schema, AgentState, AIMessage?     DONE
151. correctly write all the arguments passes in init_chat_model and chatgenerativeaigoogle   DONE
154. check this pattern again init_chat_model() create_agent()       inside a node or should i have 2 instances of cheap and expensive model and then pass it in create_agent          DONE
172. learn about langGrpah.prebuilt, create_react_agent from langraph.prebuilt   DONE
138. add neo4j driver, DB session from request.app.state in Graphiti, Cognee, AsyncPostgresCheckpointer, vector_store and other places where required in tools and do the same for DB, redis            DONE
108. use the new gemini embedding 2 for multi-modal embeddings, LLMToolSelectMiddleware        DONE
62. can i use openRouter keys for my Gemini model    DONE             
177. 
 PydanticDeprecatedSince20: `json_encoders` is deprecated. See https://docs.pydantic.dev/2.12/concepts/serialization/custom-serializers for alternatives. Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/open_deep_search/deep_researcher.py:701: LangGraphDeprecatedSinceV10: `config_schema` is deprecated and will be removed. Please use `context_schema` instead. Deprecated in LangGraph V1.0 to be removed in V2.0.
  deep_researcher_builder = StateGraph(
/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/open_deep_search/deep_researcher.py:701: LangGraphDeprecatedSinceV05: `input` is deprecated and will be removed. Please use `input_schema` instead. Deprecated in LangGraph V0.5 to be removed in V2.0.
  deep_researcher_builder = StateGraph(
/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/open_deep_search/deep_researcher.py:589: LangGraphDeprecatedSinceV05: `output` is deprecated and will be removed. Please use `output_schema` instead. Deprecated in LangGraph V0.5 to be removed in V2.0.
  researcher_builder = StateGraph(                                      

/home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/shared/langgraph_layer/open_deep_search/configuration.py:214: PydanticDeprecatedSince20: Using extra keyword arguments on `Field` is deprecated and will be removed. Use `json_schema_extra` instead. (Extra keys: 'optional', 'metadata'). Deprecated in Pydantic V2.0 to be removed in V3.0. See Pydantic V2 Migration Guide at https://errors.pydantic.dev/2.12/migration/
  mcp_config: MCPConfig | None = Field(                                            DONE
168. do migration of open_deep_research  and pass it to agent_saul as a ToolNode
    a. remove MCP from it
    b. store the MCP code for any best practices in mcp/
    c. remove other model providers
    d. use own taily client and httpx_client and replace existing one
    e. reorganise code                                          DONE
59. No skill composition. Skills are flat callables. Theres no way to chain skills (skill A output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.   DONE
169. EARS-compliant acceptance criteria, research about what Kiro does for SDD, which standards does it use   DONE
180. learn what can i learn about maintaining large codebases through tanStack ecosystem and others and see matt pocock video, how to hide/abstract away complexity behind a simple interface, make a skill/docs for this.  DONE
186. documents/ uses docling from shared and doesnt uses its own one.  DONE
187. add more S3 functions, rewrite resend DONE
188. divide AGENTS.md/copilot-instructions.md into samller files, add lint, fmt, commands, sort imports, think of most used uv and ruff and ty commands     DONE
182. add the old lynk plan as it was deleted   DONE
175. SecretStr(settings.GEMINI_API_KEY), PrivateAttr and more learn about this   DONE
191. check which will live in connection.app.state and which will live in app.state   DONE
189. check for writing error messages should i write then in StrEnum, string variable   DONE
181. how to use git when something is made then to use some commands on it while having another session work on the previous stuff when unrelated chnages have to be made. check what skills can help here    DONE
192. make a plan to migrate from orjson, serializing generic Python data like a dict or list that isn't a Pydantic model to model.model_dump_json():  TypeAdapter.dump_json() with reuable functions    DONE
178. From version 0.5.0 onwards, Cognee will run with multi-user access control mode set to on by default. Data isolation between different users and datasets will be enforced and data created before multi-user access control mode was turned on wont be accessible by default. To disable multi-user access control mode and regain access to old data set the environment variable ENABLE_BACKEND_ACCESS_CONTROL to false before starting Cognee. For more information, please refer to the Cognee documentation. [cognee.shared.logging_utils]  and also migrate to veriosn 1.1    DONE
183. scrapling vs crawl4ai, add crawl4ai in open_deep_search graph with proper plan and specs with openspec DONE
193. check if client of these cognee, grapgiti, langextract, pageindex, tavilty, fastMCP, crawl4ai needs to be in connections and then in lifespan    DONE
166. use Call a toolNode and check id toolNode should be used or subgraph inside a node for Open Deep Research   DONE
197.  might need to connect open_deep_research to agent_saul graph   DONE
146. use the return package and write it in copilot instructions and implement the plan written in this and check how exception should be written like raise and let GEH handle it or  except Exception/ExceptionName as e:, also use e.add_note and also check if i am right in passong HTTPException to APIException and other classes        DONE
133. use pydantic for data configuration, tool arguments, or schema validation in langraph and check if converting all typedDict to pydantic is useful or docs do not recommends it (only for custom state schema)   DONE
201. make full skill for ast-grep    DONE
202. figure out how to put graphify, ast-grep in opencode without putting in system prompt and disabling grep and other tools and turing permissions on for most used task    DONE
199. ANN001/002/003/204 ignored globally  understandable for AI-heavy code but weakens type safety PLR0913 (too-many-arguments) ignored  some service methods have 8+ params; consider context objects DocumentQueryService.__init__ uses object | None for redis/graphiti  should be Redis | None and Graphiti | None Some TYPE_CHECKING blocks are verbose; PEP 695 generics would clean up (type Alias[T] = ...)     DONE
197. The codebase has a `_result` dual-method pattern in repositories (Success/Failure). A service method calls `find_by_email_result(...)` and gets a `Failure`. What should it do next?
The project uses `combine_lifespans(...)` to merge the FastMCP lifespan with the existing FastAPI lifespan. Why can't you just call `app.add_lifespan_handler(fastmcp_lifespan)` instead?  DONE
203. split MCP from main app/ and then write it using the skill    DONE
198.  
1. HYBRID SEARCH CACHING RACE CONDITION                            
In DocumentQueryService.search(): cache check  embed          search  cache set. Two concurrent requests for same query      both miss cache, both embed, both search. Fix: use              `redis.setnx` with short TTL as "computing" lock, or            `async-cache-dedupe` (already in tier-2 observability plan). 
2. GRAPHITI INITIALIZATION ORDER                                    
lifespan.py: Graphiti setup AFTER Cognee, but Graphiti          needs Neo4j indices. If Neo4j driver fails, Graphiti setup      crashes but TaskGroup already succeeded (PG/Mongo/Redis         ok). App starts in degraded state silently. Add **health**          check endpoint that verifies all clients.                    
3. EMBEDDING DIMENSION HARDCODING                                   
_normalize_embedding() assumes 768-dim (Gemini). If you g switch models, this silently truncates/pads. Make it            configurable via settings or derive from embedding client.   
4. CELERY TASK DEFINITIONS SCATTERED                                
Tasks in `src/tasks/*.py` but invoked via string names          (`"tasks.documents_ingest"`). No type safety, no IDE            support. Consider `@celery_app.task` decorators in same         module or a task registry with typed signatures.             
5. MIDDLEWARE ORDER SUBTLE BUG                                      
main.py: CORS (Guard)  GZip  Security  Metrics  Logging    But Guard's CORS helper adds middleware *internally*.           If SecurityMiddleware also adds CORS headers, they conflict.    Verify with `curl -H "Origin: x" -v`  check for duplicate      `Access-Control-Allow-    DONE
200. need a new superpower/brainstorming skill with openspec, graphify, ast-grep, ponytail, firecrawl, with proper git workflows, and stop using grep and older tools  make it using create skill   DONE
204. /home/harmeet/Desktop/Projects/langchain-fastapi-production/src/app/features/auth/security.py:10: AuthlibDeprecationWarning: authlib.jose module is deprecated, please use joserfc instead.
It will be compatible before version 2.0.0.
 [info     ] Database storage: /home/harmeet/Desktop/Projects/langchain-fastapi-production/.venv/lib/python3.12/site-packages/cognee/.cognee_system/databases [cognee.shared.logging_utils]
/home/harmeet/Desktop/Projects/langchain-fastapi-production/.venv/lib/python3.12/site-packages/cognee/exceptions/exceptions.py:52: StarletteDeprecationWarning: 'HTTP_422_UNPROCESSABLE_ENTITY' is deprecated. Use 'HTTP_422_UNPROCESSABLE_CONTENT' instead.
  class CogneeValidationError(CogneeApiError):
 [warning  ] Cognee 1.0 changes: New API — remember/recall/forget/improve (V1 add/cognify/search still work). Session memory enabled by default (CACHING=false to disable). Multi-user access control on by default (ENABLE_BACKEND_ACCESS_CONTROL=false to disable). Agents (@cognee.agent) auto-verified on registration. See https://docs.cognee.ai/ [cognee.shared.logging_utils]    DONE
205. what is PYCODE and other frequently used in python projects, dockerfiles, uv and other places   DONE
206. customise openspec and learn in depth about pattern matching  DONE
167. how systemPromptPaarts, chatPromptTemplate, systemmessage, humanMessage, AImessage,ToolMessage look like while passing it in graph and how should i serialise these with toons before sending to LLM, find which parts should be removed from system prompt parts, how are system/human/ai/toolMessage are sent to LLM   DONE
208. check if the patterm matching is usefull as written after ruff rules with raise keyword   DONE
209. make a new skill for codebase search  DONE
207. only enable OTEL in prod and not in dev    DONE
212. update orient skill, grpahify, everything related to it  DONE
213. update the orient skills, graphify and others in docs/   DONE
214. add memory DB from tencent and compare with cognee, honcho   DONE
216. check if cognee is actually working within github copilot and add it in opencode as well  DONE
217. customise openspec, ast-grep  DONE
215. add knowledge from other projects and this to OFK folder including agent, DB, python, JS/TS optimisations, skills, BDD, research for SDD from gemini chat history, set of best practices for deployment, version pinning, maintaining docker image history with git commit hash, terraform practices, book wisdom. add research done in Kiro. use OpenWiki if it suits and see cole medin videos for organising files in a scalable manner, ring buffers, debugging tips, eBPF, how ABI works, how to make FFIs, SGLang and vLLM, webhooks, errors best practices, finance and other stuff as well  DONE
211. check agent router usage  DONE
136. use LangExtract outputs to build rich graph knowledge from your legal documents.  ABANDONDED
220. check the alembic warning having 2 heads   DONE
223. need to make standardise alembic file naming scheme  DONE
228. fix the failing websocket tests  DONE

161. what functional programming patterns should i use in FastAPI, python,learn pattern matching & ROP,flow()/bind()/map(), learn function composition with this example and in which case should this be used 
type Composable = Callable[[Any], Any]
def compose(*functions: Composable) -> Composable:
    def apply(value: Any, fn: Composable) -> Any:add headroom-ai for comrpression
        return fn(value)

    return lambda data: reduce(apply, functions, data)   DELAYED
152. for AI gateway checkout pydantic gateway, mastra, platformatic         DELAYED
155. check ripgrep, tree-sitter, zoekt for creating search tool that you can expose to an LLM to replace a traditional vector database and can these be used to search through text, PDF and more? learn more tools like this in popular coding harnesses and other harnesses can be used to make the lynk linter     DELAYED    
156. check the page https://docs.langchain.com/langsmith/deployments#
153. set up performance tests 
157. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin

158. add tests that suits the project
159. discover RAGFlow, OpenRAG if or if not to use it
160. check how can Port & Adapter/strategy & factory can help 


64. No eval framework. Theres no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
116. check the logic in rate_limit and circuit breaker if a more clean implementation with design patterns and dependecy inversion can be written and also check the circuit breaker redis client should be sync or async 
61. see docassemble, fpdf2, python-docx and other libraries for generating final PDFs/docs
53. add voice support by using gemini 3 for TTS and STT  with websockets
67. go and learn https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/
99. use promptfoo for detecting prompt injection attacks, automated red team attacks, 
44. correct the code for crawler and the packages used
115. logs inbetween the layers are empty or not coming except start and end, should i pass logger as an argument or should i import it where needed 
140. in cognee GRAPH_COMPLETION_COT if the FEELING_LUCKY router returns a complexity score $>0.8$. This prevents token-burn on simple questions while ensuring "God-Mode" accuracy for architectural queries. If you connect to a "bare" Neo4j instance without APOC installed, the initial cognee.add() will work, but the cognee.cognify() step will fail silently or throw cryptic Cypher errors. Always verify your Neo4j instance has the APOC and GDS (Graph Data Science) plugins enabled.

151. add langchain-cisco-aidefense, compact-middleware, langchain-collapse
152. see cogneeRetriver how does vertex ai differ from google_genai
153. add a hydration node after checkpointer  LangGraph calls /resume. The checkpointer pulls the V1 state blob from the database and injects it into the V2 graph. The V2 graph expects compliance_region, doesn't find it, throws a KeyError, and the entire thread permanently crashes. The user's work is irrecoverably lost.

    The Insider Solution: Never trust the injected state from a checkpointer on resume without a migration layer. You must implement a StateHydrationNode as the absolute first step of any resume operation. This node intercepts the raw dictionary from the database, checks a schema_version key (which you must manually add to your base state), and runs a migration script to populate default values for any new fields introduced in newer deployments before allowing the core logic nodes to touch the state. Treat your LangGraph state with the exact same rigor as you treat your production database schema.
194. add headroom-ai for **comrpression**
162. what kind of text splitters do i need. diff in PGvector and pgvectorstore in langchain
163. refactor vectorStore code        TSVECTOR,
164. refactor RAG code

170. write cron job for memory decay and then send to celery for off loading for cognee
171. use CacheBackedEmbeddings fore reusing embeddings
172. use prebuilt and custom middlewares in langchain 
173. rewrite the tools for the new grpahiti, cognee etc
174. add proper cognee functions, graphiti from docs
176. check sentence_transformers, AutoTokenizer from transformer package do i need it or can it be replaced by a langchain package
179. make proper plan for adding caching from this video and use redisvl, langcache, does cognee takes redis instance too?  https://youtu.be/19x8pKiaQVU?si=TvC5mFHU0-M-wHEI
184. You correctly called out that documents/chunks should be the sole retrieval truth.
But a lot of current Agent Saul / precedent / reconciliation code still reads clauses directly.
Do you want me to:
- A. keep the architecture clean now: new documents/chunks become the only retrieval substrate, and any old code still tied to clauses is left stale/disabled until second pass   
185. remove ts_vector(think if it is required here or other extension can do the job here) from search/document and write correct SQL query for documents/ taking skills for pgvector/pgvectorscale 


148. figure out the types of memory that a agent can have and which type does fit my needs    eg cognee, honcho, episodic etc
57. No agent-to-agent message passing format standard and make a standardized AIMessage for passing in-between agents and tools and also make a ToolMessage
165. implement RAG by getting inspired from this https://www.uber.com/en-IN/blog/enhanced-agentic-rag/?uclick_id=9529bd64-1d38-40a6-bc23-88ce151b1384     
195. in ingestion pipeline postgres + extensions for vector + BM25 + RRF and more, graphiti for what we already did, need to have langextract before these as well, and a pageindex parallel to postgres graphiti and learn from https://towardsdatascience.com/hybrid-search-and-re-ranking-in-production-rag/
196.  need to check this asyncio.gather part in  → fans out to researcher_subgraph via asyncio.gather → inside the subgraph, route_researcher conditional edge diverts crawl_webpage calls to a dedicated crawl_executor node 
210. fix ingestion -> docuements -> tools -> cognee
155. complete the ingestion pipeline to working condition and see where reconciliation comes init. i want to remove reconciliation and replace it with agent memory made with cognee entirely.
todos:-
    1. toons reusable , point 138,  break the code for reconcilliation inside langgraph_layer/ and features/,check the use from string import Template to write prompts or chatpromptTemplate with toons, use SystemPromptParts to write systemPrompt also check it, use init_embedding and googleEmbeddings
    a. docling - Legal docs need hierarchical chunking, convert dataclass to pydantic models, use embedders(batch, chunks, etc) to reusable function in langchain_layer, remove Grapgiti initilisation from here
    b. langextract and pageindex(leave this for now if not currently implemented)
    c. graphiti refactor
    d. postgres RAG should be agentic
    e. celery for off loading to a queue
    f. insert the langgraph in app.state in lifespan
    g. pass default and metadata for particular config in pydantic models for agents
    h. research for RAG pipeline with Gemini
    i. use MessagesState to standardise the moving of data between Agent A and Agent 
    j. use tenacity for retries, new capabilities, output format from chatpromptparts,
190. see if documents/ can be moved in ingestion pipeline with langextract, pageindex, graphiti, postgres,

218. add a small gloassary of the project from the screenshot, we are open at the core, we share about are roadmap, how we think about things, and of course we share all our code and should strive to be in that way. its important to maintian the things they live and iterate over the product. 
219.  add in the system prompt to add a search, implementation, verifier, reviewer, check if i can give specific system propmts, skills, tools, MCP servers to subagents, defining models for subagents, permissions
always write why Use Block-Level HTML Comments: Since Claude Code strips HTML comments from the context it reads, you can include detailed notes for human maintainers without increasing token usage (3:02-3:12).
The "Incident-Why-Outcome" Framework: The host recommends structuring your comments to record three specific pieces of information for every rule (3:14-3:43):
The Failure/Incident: Describe the specific event or bug that necessitated the rule (e.g., citing a specific incident number or date).
Why the Rule Helps: Explain how the specific instruction prevents that failure from reoccurring.
The Outcome: Provide evidence of the rule's success (e.g., "no orphan rows in 11 months").
Additionally, the host personally recommends adding a commit reference to these comments to help future developers trace the history of the decision (3:45-3:49). The host warns that vague comments like "added to fix an issue" are ineffective and perform no better than having no comments at all 
221. fix the files tht are scrambled in utils, shared and other places
222. RESULT-PATTERN.md currently documents the envelope style as "project standard" — it now contradicts the code (users/service.py raises). Optional follow-up: update the doc to match. Say the word and I'll include it.
there are 2-3 error handling that i need to figure out
try catch with return typed error
one raises exception from custom exception
one raises raise app_error_to_exception(error)
except SQLAlchemyError as exc:
            return Failure(
                InfrastructureAppError(
                    code="DB_ERROR",
                    message="Database error while updating subscription",
                    details={"subscription_id": str(subscription.id), "error": str(exc)},
                    source="subscription_repository",
                )
            )
 if updated is None:
                return Failure(
                    ConflictAppError(
                        code="VERSION_CONFLICT",
                        message="Subscription was modified concurrently; refetch and retry",
                        details={
                            "subscription_id": str(subscription.id),
                            "expected_version": expected_version,
                        },
                        source="subscription_repository",
                    )
                )
and also need to check where to put log_expected_failure(error, operation=operation) in the final one
also need to check how exactly error msg should be written. should it be string message, StringEnum or something else. need to chec the best practice.
need to standardise what happens in except block
 224. fix this def _celery_meters() -> tuple[Any, Any, Any]:
    global _otel_celery_meters  # noqa: PLW0603 — module-level lazy init
226. remove redis client protocol and use from cache/ and remove any datamodels
227. agentState should be typedDict and not a baseModel
225. think where to add models in databse/schemas or somewhere else? what can i do to follow industry convention. improve the env.py seeder and other files for that make it include in ruff and ty 
228. disable memory in claude code, opencode
229. need to have a proper versioning of docker images ties to the git commit and see how the industry standard does this
230. need to have a standard for marking the task done in openspec spec gated with a DONE sections that is detailed and summary of how it is done
```
summarise these chapters in great detail and take video's transcript as reference for summarising

summarise this video in great detail and depth by dividing it into 5 minute chunk and take video's transcript as reference for summarising



8/10 security person who knows 10/10 shit about browsers is more danerous than a 10/10 security person who knows 6/10 about browsers
Previously, high-level hacking required deep historical knowledge and expertise. Now, because models can be forced into loops of iterative testing, individuals can effectively "bang their head against the wall" until they find vulnerabilities. They note that even benign inquiries can lead to dangerous outcomes if the model provides deep insights into obscure sub-dependencies.

I am about to start this project. Interview me until you have 95% confidence in what I actually want, not what I think I should want.
how to approach finding one a solution , how to think about these things, can you give me one small hint to steer me in right directions

 When I see something cool, I am more likely to go to the GitHub profile to find the creator of the thing than to read the code of the thing. I spend much more time on profiles on
 than repos on GitHub because it's so interesting to dig into who made the thing, why did they make the thing? What made them to making this thing? What experiences did they have that made them want this thing to exist? How did they come to making this thing? How did they
c hoose to build this thing? How has it benefited them? Are they still working on it? Why are they working on it? And what do they have in common with you?


bit packing, texture atlasting, delta encoding, deduplication

Google's Open Knowledge Format (OKF). She explains how this standardized structure allows AI agents to efficiently process her notes, research, and workflows to automate complex tasks.

0:00 - 5:00: Introduction to OKF and Standardized Structure
Marie introduces the concept of the OKF, which acts as a structured repository of information that any AI agent can interpret. She emphasizes that while the concept of organizing files in markdown for AI isn't new, Google's OKF provides a standardized framework that makes data interoperable. Key elements discussed include:

YAML front matter: The mandatory metadata at the top of markdown files that tells an agent what the content represents (type, title, description, and tags).
The Goal: Moving beyond simple RAG (Retrieval-Augmented Generation) to a structured knowledge graph where agents can logically navigate between concepts, playbooks, and references.
The Structure: Her system uses specific types: Concepts, Playbooks, References, and Systems to categorize information effectively.
5:00 - 10:00: OKF in Practice and AI Productivity
In this segment, Marie demonstrates how her brain functions. Unlike traditional RAG, which might dump vast amounts of data into a context window, the OKF allows her agent to browse a directory to retrieve specific, relevant information.

Playbooks: She highlights the power of "playbooks," which are sets of instructions that automate specific tasks, such as generating client proposals or analyzing Google search updates.
Productivity: Marie argues that AI will not replace jobs but will instead augment human productivity. By offloading the need to remember every detail to her AI-managed brain, she becomes significantly more efficient in her professional SEO work.
10:00 - 15:00: Knowledge Graphs and Ingesting Information
Marie dives into the visual side of her brain, showing a "knowledge graph" where nodes represent interconnected markdown files.

Documentation: She stores official Google SEO documentation as References within her system, allowing her to cite accurate sources instantly when generating reports.
Visualizing connections: She explains how the agent identifies relationships between topics (e.g., AI overviews linking to historical SGE data).
The Agent Workflow: She demonstrates the process of "ingesting" new information—such as a new Google search feature—into her OKF, where the agent proposes a plan to categorize and integrate the new data into existing topics.
15:00 - 20:00: Approving Updates and Querying the Brain
Marie provides a live demonstration of her agent’s workflow:

Human-in-the-loop: Before the brain makes updates, it presents a plan for her to approve or modify.
Automated Curation: The agent successfully updates multiple related files and creates new concepts based on the provided documentation.
Querying: She demonstrates asking the brain to summarize complex information (like new AI controls in Search Console) into a format suitable for a client report. The system efficiently pulls data from the specific files it created, showcasing the speed and accuracy of the OKF approach.



<!-- memory usage of FastAPI app -->
"memoryUsage": {
        "rss": "794.28 MB",
        "vms": "6552.59 MB"
      },

# Upgrades
1. Wont do it - DSPy shifts you from writing prompts to compiling them. Instead of manually guessing the best words for your LAWYER_SYSTEM_PROMPT, you define the input/output signature, give DSPy a few examples of good and bad answers, and it algorithmically finds the optimal prompt.    ABANDONED
2. add celery for offloading ingestion to a queue.   DONE
3. make ingestion pipeline inspired from uber   
4. add pageindex for vectorless RAG, markitdown
5. (after writing code) When using AsyncPostgresSaver (or any checkpointer) for persistence, LangGraph serializes and saves the entire state blob at the end of every single node execution (super-step). If your messages array contains large document extractions or base64 PDFs, your state blob will rapidly swell to megabytes. Multiplied by 1000s of concurrent sessions, your PostgreSQL database will choke on I/O operations, and your latency will spike from 500ms to 5 seconds per step just writing to the DB.

    The Insider Solution: The "Pointer State" Pattern.
    Do not store document content in state["messages"].

    When the IngestionAgent reads a document, it must immediately write the text payload to a cheap, fast object store (like Redis or S3) and generate a UUID.

    The agent only appends the UUID to state["context_ptrs"].

    You write a custom StateModifier function that runs inside the LLM node. Right before calling the LLM, this function dynamically fetches the text using the UUID, injects it into the LLM's context window, gets the response, and then throws the text away.

    The checkpointer only ever saves the UUIDs. Your LangGraph state remains less than 5KB, your database I/O drops by 99%, and your system can smoothly scale to 10,000+ concurrent state transitions without breaking a sweat. Furthermore, use LangGraph's Store API for the actual long-running task idempotency, keeping the graph State purely for the immediate transition logic.

6. (after writing code)"Lost in the Middle" phenomenon—they pay attention to the beginning and end of a prompt but ignore the center.

    The Improvement: Enforce a strict architectural standard for your final compiled prompt string.

    Structure:

    Top: Context, retrieved documents, and long-term memory.

    Middle: Conversation history (the user's back-and-forth).

    Bottom (Critical): The system prompt, the negative incentives ("you will be sued"), and the strict JSON output schema. The LLM must read the formatting instructions last before generating tokens.

7. If you are building long-running, resumable LangGraph workflows (HITL), there is a silent system-killer that will destroy your production environment: State Schema Migrations.

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



the architectural design differences between Python properties and methods, emphasizing the contract these choices establish for your code.

Key Takeaways:

The Difference in Promise: A property communicates that an operation is cheap, safe to read repeatedly, and typically returns state (2:29 - 3:48). Conversely, a method implies that work is being done, potentially involving complexity, latency, or side effects (2:43 - 3:23).
Derived State: Properties are ideal for computing simple values from existing object state (e.g., checking if an account is active), as they are deterministic and side-effect-free (3:48 - 6:14).
Setters and Side Effects: While properties can have setters, they should generally avoid performing I/O or heavy operations. Persistence logic (like database saves) should be handled by explicit methods to keep code predictable and avoid blocking (8:58 - 10:29).
Async Properties: While technically possible, making properties asynchronous is considered a design smell (13:13 - 14:31). It hides asynchronous waiting behind attribute access, which violates the expected simplicity of a property. Instead, use asynchronous methods for loading and saving data (14:31 - 16:10).



This video provides a deep dive into designing a distributed cache capable of handling over a billion requests. It moves beyond basic theory to explain the architectural challenges of building a reliable, clustered caching fleet.

Core Architectural Concepts
The Routing Trap (2:03-3:06): Modular hashing (hashing a key modulo the number of servers) is a common failure point. When the cluster size changes, almost all keys map to new servers, causing a "thundering herd" of cache misses that can crash the database.
Consistent Hashing & Virtual Nodes (3:06-5:43): By mapping keys and servers onto a circular hash ring, consistent hashing ensures that only a small portion of data is affected when nodes are added or removed. Virtual nodes are then used to smooth out statistical variance, ensuring load is distributed evenly across heterogeneous hardware.
Freshness vs. Capacity (5:43-6:51): A critical distinction is made between TTL (Time-To-Live, which handles data freshness/staleness) and eviction policies (which handle capacity constraints when memory is full).
Approximate LRU (6:51-8:23): Implementing strict Least Recently Used (LRU) algorithms in a distributed system is impractical due to locking overhead. Instead, production systems often use approximate LRU, where nodes randomly sample a small set of keys to decide which to evict, achieving near-optimal performance with minimal CPU cost.
Handling Scale and Hot Keys
The Hot Key Problem (8:23-9:57): Even with perfect distribution, a single "hot key" (e.g., a celebrity's profile picture) can overwhelm a single node's CPU. The solution is replication, where the key is suffixed with a random integer (e.g., key_1 through key_10), spreading the request volume across multiple nodes.
Coherence and Invalidation (10:35-12:16): Replicating data introduces coherence issues. Rather than attempting complex distributed consensus, systems should use CDC-driven (Change Data Capture) invalidation. A background process reads database logs and sends invalidation events to the cache fleet via a message broker (e.g., Kafka).
Cache Warming (12:16-13:46): Cold starts can act as a self-inflicted DDoS attack. It is essential to "warm" a cache—populating it with high-velocity data before directing live traffic to it—to prevent a miss-storm.
Routing and Observability
Smart Clients vs. Proxies (13:46-15:34): The video debates whether applications should handle routing directly (smart client) or use a dedicated routing layer (proxy like Envoy or Twemproxy). While smart clients save a network hop, proxies simplify management and topology synchronization at scale.
Observability (15:34-16:53): Average metrics like a 95% hit rate can hide catastrophic localized failures. Engineers must monitor per-node CPU, network saturation, and the actual database load caused by cache misses to ensure system health.
Memcached vs. Redis (16:53-18:08): Technology selection should come last. Memcached is favored for simple, multi-threaded, high-throughput object caching, while Redis provides a rich set of data structures and features useful for complex mutations and logic.


 Ensuring Idempotency
Idempotency—ensuring that performing the same action multiple times results in the same outcome—is critical. Alex identifies three strategies to handle this:

Fetch before processing: Use the webhook as a trigger to query the provider's API for the current state (e.g., Stripe's new event format).
Upsert by date: Use database transactions to update or insert records only if the incoming data is newer than what is already stored.
Tracking processing state: Maintain a separate, transactional storage to track if an event has been processed, is in progress, or is new, returning errors for incomplete attempts to trigger retries.
10:00 - 15:00: Handling Bursts, Back Pressure, and Architecture
Alex highlights the dangers of bursty traffic, citing a personal experience where a mass email campaign caused a production API outage due to subsequent webhook processing. He emphasizes the need for a decoupled architecture: ingest webhooks into a queue and process them separately. He also discusses optimistic filtering, where you perform quick, aggressive cache lookups during ingestion, assuming a record exists if a lookup fails, to avoid system-wide delays.

15:00 - 20:00: Back Pressure and Data Integrity
Managing back pressure requires monitoring queue depth and max age. You must understand your system's theoretical capacity versus baseline capacity. To ensure data integrity after inevitable failures (crashes, bad deploys), he suggests:

Processing guarantees: Carefully managing acknowledgments (acking/nacking) through every link in the request chain.
Reconciliation: Using the provider's events API to fetch and resync data if a failure causes significant discrepancies.
20:00 - 25:00: Observability and Emerging Trends
Visibility is vital. Companies should build a centralized event log (e.g., using Elasticsearch) to audit failed events and troubleshoot issues. He warns against waiting for a major incident to build replay tooling. He also discusses the shift toward Event Destinations (like AWS EventBridge support) and how platforms are increasingly offering native filtering and better event management tools.

25:00 - 31:25: Event Gateways and Future Outlook
Alex defines an Event Gateway as a cloud infrastructure primitive—similar to an API Gateway—that handles ingestion, routing, filtering, and queuing for asynchronous events. He demonstrates how Hookdeck functions as this layer, showing how users can:

Manage webhooks via Terraform.
Visualize and resolve back pressure issues by adjusting delivery rates.
Debug and trigger manual retries for failed events.

Web Locks 

This video presents a comprehensive framework for creating high-quality AI agent skills, aiming to help developers navigate what the presenter calls "skill hell."

### **0:00 - 5:00: Introduction & The Skill Checklist Framework**
Matt Pocock introduces the concept of **"skill hell,"** a situation where developers have access to many skills but lack a shared rubric or framework for building effective ones (0:52). To solve this, he proposes a **four-part skill checklist**: 
1. **Trigger:** How the skill is invoked (3:16).
2. **Structure:** The internal composition of steps and references (7:29).
3. **Steering:** Guiding agent behavior through leading words (11:54).
4. **Pruning:** Minimizing the skill by removing unnecessary elements (16:48).

He discusses the trade-off between **user-invoked** and **model-invoked** skills (3:36). While model-invoked skills offer flexibility, they increase "context load" on the agent and introduce unpredictability (5:23). User-invoked skills keep the context load low but require more cognitive effort from the user/pilot (5:53).

### **5:00 - 10:00: Structure & Minimizing Skill.md**
Continuing the discussion on triggers, the presenter explains that he favors user-invoked skills to avoid the unpredictability of agents deciding whether to call a skill (6:36). He then transitions to **Structure** (7:29). A well-designed skill should be split into **steps** (the procedure) and **reference** (supporting info) (7:38). To keep the `skill.md` file minimal, he recommends offloading branching reference material—material not needed in every execution branch—into separate markdown files linked by **context pointers** (9:00 - 11:53).

### **10:00 - 15:00: Steering with Leading Words & Leg Work**
This section focuses on **Steering** (11:54), the method for ensuring an agent follows instructions precisely. He introduces **"leading words"**—dense, high-meaning terms like "vertical slice" that trigger an agent’s prior knowledge and align its reasoning traces with the developer's intent (12:22). He emphasizes that when an agent fails to perform a task, it often needs more **"leg work"** per step (14:56). A powerful technique is to break complex, multi-step processes into smaller, individual skills to force the agent to focus solely on the current phase without prematurely attempting to reach the final goal (16:01).

### **15:00 - 20:43: Pruning & Final Summary**
In the final segment, the focus shifts to **Pruning** (16:48). This involves maintaining a clean skill set by:
* **Avoiding duplication:** Ensuring every part of a skill has a single source of truth (17:15).
* **Removing sediment:** Deleting stale or irrelevant legacy material from shared files (17:41).
* **Eliminating "no-ops":** Removing instructions that do not actually change agent behavior (18:26).

The video concludes by summarizing the framework (19:06) and directing viewers to his GitHub repository for a practical implementation of these "writing great skills" techniques (19:55).

This video explains how to perform **zero-downtime database migrations** using the **Expand-Contract pattern**. This pattern is essential for mission-critical applications where even a second of downtime is unacceptable.

### **Phase 1: Concept and Initial Setup (0:00 - 5:00)**
*   **The Problem:** In a typical rolling update (e.g., using *Kubernetes*), you may have multiple versions of an application running simultaneously. If your database migration involves a breaking change—like renaming a column—the old version of your application will fail when it attempts to access the new schema, leading to 500 errors and downtime.
*   **The Solution (Expand-Contract):** This pattern allows you to transition your database schema without breaking changes by separating the process into phases:
    1.  **Expand:** Introduce new schema elements (e.g., a new column) while keeping the old ones intact.
    2.  **Migrate:** Use the application to support both the old and new columns, and run backfill jobs to sync data.
    3.  **Contract:** Once the new structure is fully adopted and verified, remove the legacy schema components.

### **Phase 2: Technical Execution and Verification (5:00 - 6:36)**
*   **Implementation Details:**
    *   **Dual Writes:** During the transition, the application must write to both the old and new columns to ensure data consistency.
    *   **Backfill Jobs:** Use background jobs to copy historical data from the old column to the new one. These jobs should be idempotent, meaning they can safely run repeatedly until all missing data is migrated.
    *   **Read Strategy:** Gradually shift the application from reading the old column to reading the new column only after the data has been verified.
*   **The Final Steps:**
    *   **Verify:** Before cleaning up, you must perform a thorough verification to ensure no hidden processes (like *cron jobs*, *BI tools*, or *database triggers*) are still relying on the old field.
    *   **Contracting:** Only after complete confirmation should you deploy the final code that exclusively uses the new structure and perform a final cleanup by dropping the old column.

By following these steps, you ensure that your application remains functional throughout the entire migration process, regardless of which version is handling the request.

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

