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
6. set up performance tests
17. refactor vectorStore code
18. refactor RAG code
21. add this from fastapi import BackgroundTasks
@app.post("/process")
async def process_data(data: DataModel, background_tasks: BackgroundTasks):
    # Return immediately, process in background
    background_tasks.add_task(heavy_processing, data)
    return {"status": "processing"}
37. check out the commented out pre commit hooks 
25. Use background tasks so users don't wait for non-critical operations

42. fix the search code as it is not using the pg_textsearch, pgvectorscale, pg_trgm etc properly  with Kiro
44. correct the code for crawler and the packages used
46. use CacheBackedEmbeddings fore reusing embeddings
47. check whether i will need to use sandboxed execution environemnt in future
48. check the page https://docs.langchain.com/langsmith/deployments#
49. make a proper terraform plan for all 3 major cloud providers with dev, staging and prod env and check all useful terraform plugin
52. legal tool will be based on Saul for finding out of the box ideas for legal advice also.
53. add voice support by using qwenTTS or something else 
56. use AsyncMemoryClient for mem0
57. No agent-to-agent message passing format standard
When sub-agents return results, they're raw strings. There's no typed contract for what one agent sends to another. A SubagentMessage(agent_name, task, result, confidence) schema would let the supervisor make smarter decisions.
58. Circular delegation is possible
Agent A can hand off to Agent B, which can hand off back to Agent A. There's no loop detection beyond completed_agents in SupervisorState, and that only works in the supervisor graph — not in the tool-based MultiAgentSystem.
59. No skill composition
Skills are flat callables. There's no way to chain skills (skill A's output feeds skill B) without writing a new skill. A Pipeline primitive for skills would unlock complex, cheap workflows.
60. Batch uses asyncio.gather with a semaphore but no queue
Under high load, all batch requests start simultaneously and race for the semaphore. A proper async queue with backpressure would give more predictable latency and prevent thundering herd.
61.Embeddings aren't cached
aembed_batch calls the API every time. Embeddings for the same text are deterministic — a simple LRU cache keyed on SHA256(text) would eliminate redundant API calls entirely.
62. Model instances are rebuilt on every call
build_chat_model() constructs a new ChatGoogleGenerativeAI every time it's called. The model object should be a module-level singleton (or per-spec singleton) since it's stateless.
63. No connection pooling for the LLM client
langchain_google_genai uses httpx under the hood. Without explicit connection pool configuration, each concurrent request potentially opens a new TCP connection to the Google API. This adds 50-150ms per cold request.
64. No eval framework
There's no way to measure whether changes to prompts or middleware actually improve agent quality. Should have a LangSmith dataset + evaluator setup for golden-set regression testing before deploys.
No structured reasoning traces
The agent just produces output. For debugging production failures you need to store the full reasoning trace (all tool calls, intermediate states, the exact prompt sent) not just the final message.
26. optimise pydantic models for speed by providing config and include it in copilot-instructions
24. Cache expensive dependencies to avoid repeated computations, Stream large responses to reduce memory usage by 80-90%                           TO_BE_DONE
31. Opening and closing a network client for every single request is expensive. Using async with ensures the connection is cleaned up properly. In a "Hybrid" reality, you arent just passing a raw database client around. You use the **Lifespan** to manage the "Heavy" resource (the connection pool) and **Dependencies** to manage the "Scoped" resource (the specific session or transaction for one request).    
43. add langextract to agent tools
33. add pageindex properly  and include it in agent tools
36. update copilot instructions
71. also check logger if working as wished       If you want to enrich the global context (so user_id appears in all future logs automatically):   current_state = request_state.get() current_state["user_id"] = authenticated_user.id current_state["tenant_id"] = tenant.id
55. check where to add prefix in routes v1 router or router file
10. figure what are exception wrt FastAPI, fastapi-security and more with claude
65. do i need return type of every public function ask
67. go and learn https://www.marktechpost.com/2026/03/01/how-to-design-a-production-grade-multi-agent-communication-system-using-langgraph-structured-message-bus-acp-logging-and-persistent-shared-state-architecture/
70. rewrite health, serach & auth for using APIExceptions, removing http_response, removing handler file and use dependencies file 
71. ensure response shape is uniform through out the app
72. add scripts in [project.scripts]

---
# Agent architecture 
  user should be authenticated before anything for better state context(langgraph)
1. QnA agent asking for more clarity 
    responds in realtime 
2. router agent understands user intent and assigns diff agent task based on skills and tools.
    and should be HITL for clarifications
3. planner agent is called after router agent (decides functional calling and a deterministic workflow)
4. planner should return status to QnA agent to talk to user
5. Reliable Communication via Typed Schemas
    Multi-agent workflows often break because agents pass inconsistent or malformed data to one another.     The Problem: Inconsistent JSON or shifting field names lead to downstream "guessing" and system failure.      The Solution: Use Typed Schemas (like TypeScript interfaces). These act as machine-checkable contracts that ensure data integrity at every boundary.     Benefit: Failures become "schema violations" rather than silent logic bugs, allowing systems to retry or repair state before it propagates.
# hierarchical arc - one orchestrator managing multiple agents
# HITL arc
# network/swarm arc
# sequential arc

6. Eliminating Ambiguity with Action Schemas
    Even with valid data, agents often fail because their intent is too broad (e.g., "help the team").     The Problem: LLMs may interpret vague instructions in ways that arent automatable (assigning vs. closing vs. escalating).      The Solution: Implement Action Schemas (using tools like Zod). These force the agent to choose from a "discriminated union" of specific, predefined actions.      Benefit: Every agent output must resolve to an explicit, valid command, turning unpredictable text into predictable execution.


# best practices for DI and req.app.stateEspecially bad for:

Database sessions/transactions (PostgreSQL asyncpg/SQLAlchemy 2.0, Beanie/Motor sessions) → need per-request scope + cleanup → use yield dependencies
Anything that needs request context (current user, request ID for logging/tracing)
Highly testable code (you want to mock/inject fakes easily)

Much better for:

Redis client (connection pool)
Neo4j driver (pool)
Celery app instance (usually global anyway)
MongoDB client (Motor client is thread-safe & pool-aware)
Rate limiters, background task queues, config objects, ML models loaded once

Recommended Modern Patterns (2026 Best Practice)

Singleton-style shared clients (your DB client, Redis, Neo4j driver, Celery app)
→ app.state + Depends(get_xxx) → yes, good & recommended
Per-request resources (DB session, transaction, user context)
→ Classic Depends with yield → preferred over app.statePythonasync def get_db_session():
    async with session_maker() as session:
        yield session
Hybrid (most real apps do this)Python# shared client
def get_mongo_client(request: Request):
    return request.app.state.mongo

# per-request db
async def get_db(request: Request):
    client = request.app.state.mongo
    db = client["dbname"]
    # or even better: yield per-db context if needed
    yield db



```




    
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


