```py
1. try out alembic
2. figure out docker compose as it appears to be not working   DONE
3. set up integration guide for FastMCP             Delayed
5. figure out extra in logger/loguru   DONE
8. make a Copilot instructions improved final based on todo.md   Undergoing
9. re write server-middleware @app.middleware('http') and check with claude   - DONE
11. checkout why Swagger Docs not working  - DONE
13. move to __init__.py for better relative imports   DONE
14. check if uvicorn logger is disabled              DONE
19. update FastAPI to 0.124    DONE
4. promtail/prometheus integration
6. set up performance tests
7. set up pgVector
10. figure what are exception wrt FastAPI, fastapi-security and more with claude
15. refactor docling code
16. refactor crawl4ai code
17. refactor vectorStore code
18. refactor RAG code
20. figure out using depends in FastAPI with DB session, logger, service layer, correlational ID and more 
21. add this from fastapi import BackgroundTasks
@app.post("/process")
async def process_data(data: DataModel, background_tasks: BackgroundTasks):
    # Return immediately, process in background
    background_tasks.add_task(heavy_processing, data)
    return {"status": "processing"}
22. do proper validations using pydantic
23.  --no-access-log  in uvicorn main:app for 15% boost in perf
24. Cache expensive dependencies to avoid repeated computations, Stream large responses to reduce memory usage by 80-90%
25. Use background tasks so users don't wait for non-critical operations
26. optimise pydantic models for speed by provideing config
27. For expensive resources that don't change often, you can create singleton dependencies that live for your entire application lifetime. by lru_cache
28. # Slower: BaseHTTPMiddleware approach
@app.middleware("http")
# Slower: BaseHTTPMiddleware approach
# Faster: Pure ASGI middleware
app.add_middleware(ProcessTimeMiddleware)
29. check if default response is ORJSON do i need to write it everywhere or just the return would work
30. @app.on_event("startup") is old and replaced by 'lifespan' context manager - check if i need to inject db if i use this
    You don't want to connect to MongoDB or Postgres on every single request; you want to create a connection pool when the app starts and close it when the app stops.
32. check if need global for closing and do this 
async def connect_db():
    global client
    client = AsyncIOMotorClient(
        settings.MONGO_URI,
        maxPoolSize=10,
        minPoolSize=2,
        serverSelectionTimeoutMS=5000,
    )
33. Using @lru_cache without bounds not recommended
34. use cache in dockerfile
31. Opening and closing a network client for every single request is expensive. Using async with ensures the connection is cleaned up properly. In a "Hybrid" reality, you aren't just passing a raw database client around. You use the **Lifespan** to manage the "Heavy" resource (the connection pool) and **Dependencies** to manage the "Scoped" resource (the specific session or transaction for one request).

This ensures you don't leak connections while keeping your Service and Repository layers clean and testable.

---

### 1. The Lifespan (The "Heavy" Pool)

This stays in your main entry point. It ensures the pool exists before any request arrives.

```python
from fastapi import FastAPI, Request, Depends
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient

# 1. SETUP THE POOL IN LIFESPAN
@asynccontextmanager
async def lifespan(app: FastAPI):
    # The client stays alive for the entire life of the app
    app.state.mongodb_client = AsyncIOMotorClient("mongodb://localhost:27017")
    app.state.db = app.state.mongodb_client["customs_database"]
    print("Connection Pool Created")
    yield
    # Cleanup on shutdown
    app.state.mongodb_client.close()
    print("Connection Pool Destroyed")

app = FastAPI(lifespan=lifespan)

```

---

### 2. The Layers (Repository & Service)

Notice these classes don't know anything about FastAPI or Request objects. They just take a `db` object in their `__init__`. This is **Dependency Injection**.

```python
# 2. THE REPOSITORY LAYER (Handles Raw DB Queries)
class CustomsRepository:
    def __init__(self, db):
        self.collection = db["submissions"]

    async def save_declaration(self, data: dict):
        result = await self.collection.insert_one(data)
        return str(result.inserted_id)

# 3. THE SERVICE LAYER (Handles Logic & XML Prep)
class CustomsService:
    def __init__(self, repo: CustomsRepository):
        self.repo = repo

    async def process_submission(self, xml_data: str):
        # Logic: Transform XML -> Dict, Validate, then Save
        # (Imagine xmltodict logic here)
        data_to_save = {"payload": xml_data, "status": "PENDING"}
        return await self.repo.save_declaration(data_to_save)

```

---

### 3. The Dependency "Bridge" (The Hybrid Part)

This is where the magic happens. We use a FastAPI dependency to bridge the **Global App State** and the **Local Request**.

```python
# 4. THE INJECTION BRIDGE
def get_customs_service(request: Request) -> CustomsService:
    # Pull the pool from the app state
    db = request.app.state.db
    # Build the hierarchy
    repo = CustomsRepository(db)
    return CustomsService(repo)

```

---

### 4. The Route Handler

The route remains extremely thin. It doesn't know how to talk to MongoDB; it only knows how to ask for the `CustomsService`.

```python
@app.post("/submit")
async def submit_declaration(
    xml_body: str, 
    service: CustomsService = Depends(get_customs_service)
):
    submission_id = await service.process_submission(xml_body)
    return {"id": submission_id, "message": "Customs declaration queued"}

```

---

## Why this is the "No-Nonsense" Winner

1. **Zero Global State Issues:** Your Service and Repository are just regular Python classes. You can unit test them by passing a mock `db` object in the constructor. You don't need to "run" FastAPI to test your business logic.
2. **Resource Efficiency:** Because the `AsyncIOMotorClient` is attached to `app.state`, Motor manages its own internal connection pooling. You aren't opening a new connection for every request; you are simply "borrowing" one from the pool.
3. **Traceability:** If a database query fails, the traceback shows exactly which layer it came from. You aren't guessing which global variable was modified.


## better approach example for multitenant
def get_tenant_db(request: Request):
    tenant_id = request.headers["X-Tenant"]
    return client[f"tenant_{tenant_id}"]
    
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
