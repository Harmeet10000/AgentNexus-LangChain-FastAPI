```xml
<copilot-instructions>
	<meta>
		<author>Repository conventions</author>
		<date>2025-10-11</date>
		<language name="python" version="3.12" />
		<style>
			<principles>functional, DRY, KISS</principles>
			<immutability>Prefer immutability and pure functions; avoid module-level mutable state</immutability>
			<typing>Use type hints everywhere; write code compatible with strict mypy checks</typing>
			<formatting>Follow Black and isort rules; use project's pyproject.toml where present</formatting>
		</style>
	</meta>

    <fastapi>
    	<bestPractices>
    		<routers>Use APIRouter per feature and mount routers in a single place. Keep routes thin: delegate logic to services.</routers>
    		<dependencyInjection>Prefer fastapi.Depends for injecting services and config. Avoid singletons at import time; use factories to create clients.</dependencyInjection>
    		<startupShutdown>Use lifespan/startup events for connections and background resources; don't run I/O at import time.</startupShutdown>
    		<pydantic>Use Pydantic models for request/response. Prefer validation through typed models; for Pydantic v2 prefer .model_dump() and validators that are pure.</pydantic>
    		<async>Prefer async handlers and async I/O. When using blocking code, run it via anyio.to_thread.run_sync or similar.</async>
    		<errors>Centralize error handling in middleware (see src/api/middleware/error_handler.py). Return typed error responses.</errors>
    		<security>Use FastAPI security utilities and follow least-privilege patterns. Put auth logic in dependencies, not in route handlers.</security>
			<serverRun>Always use uvicorn to run the application (do not run apps with bare "python" processes). Prefer explicit uvicorn commands that reference the ASGI app import path or factory. Examples: dev: <code>uvicorn src.main:app --reload --host 0.0.0.0 --port 8000</code>; production (process manager): <code>uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4</code> or via Gunicorn with Uvicorn workers: <code>gunicorn -k uvicorn.workers.UvicornWorker src.main:app -w 4 --bind 0.0.0.0:8000</code>. When using an app factory, pass --factory (e.g., <code>uvicorn src.main:create_app --factory --reload</code>).</serverRun>
    	</bestPractices>
    </fastapi>

    <codingGuidelines>
    	<functions>
    		<pureFunctions>Favor small, pure functions that accept inputs and return outputs without side effects.</pureFunctions>
    		<composition>Compose behavior from smaller functions and use higher-order functions where appropriate.</composition>
    		<noGlobals>Avoid mutable globals; pass dependencies explicitly or via FastAPI Depends.</noGlobals>
    	</functions>

    	<types>
    		<mypy>Enable strict mypy rules. Annotate public functions, endpoints, and data shapes. Prefer TypedDict or dataclasses for simple shapes where appropriate.</mypy>
    		<pylance>Keep exports explicit. Avoid unused imports and shadowing. Use explicit return types for public API functions.</pylance>
    	</types>

    	<formatting>
    		<black>Adopt Black formatting (project default line length). Let formatter run in CI and pre-commit.</black>
    		<isort>Use isort with black profile; group imports: stdlib, third-party, local packages.</isort>
    		<indentation>Use 4 spaces. Ensure Python indent extension rules are satisfied.</indentation>
    	</formatting>
    </codingGuidelines>

    <designPatterns>
    	<factory>
    		<when>Use for creating configured clients/resources (DB, cache, external clients). Factories should be pure functions returning configured objects; initialization (connect) should be performed on startup.</when>
    		<how>Return callables or lightweight objects from factory functions. Keep factories deterministic and side-effect free at call-time where possible.</how>
    	</factory>

    	<adapter>
    		<when>Use adapter to wrap third-party/mutable/blocking clients and present a small async-friendly interface to the app.</when>
    		<how>Expose only the methods needed by the app, convert sync APIs to async via thread pools or provide async wrappers.</how>
    	</adapter>

    	<functionalAdvice>Prefer functions and small data structures; when state is needed, create it via factories and inject via Depends.</functionalAdvice>
    </designPatterns>

    <tests>
    	<unit>Write unit tests for pure business logic with pytest. Mock external dependencies injected via Depends or passed explicitly.</unit>
    	<integration>Use TestClient/AsyncClient for router tests. Seed fixtures for DB/vector stores; keep tests hermetic.</integration>
    </tests>

    <practicalTips>
    	<structure>Group by feature: src/features/<feature>/{api,schemas,services,tests}. Keep services independent of FastAPI internals.</structure>
    	<imports>Avoid heavy imports at module top-level. Use local imports inside functions when needed to prevent import-time I/O.</imports>
    	<logging>Use structured logging from src/core/logging and include correlation ids from middleware.</logging>
    	<ci>Run black --check, isort --check-only, mypy, and pytest in CI. Fail on formatting/type errors.</ci>
    </practicalTips>

    <examples>
    	<endpoint description="Pure function + FastAPI wiring">
    		<code><![CDATA[
from typing import List
from fastapi import APIRouter, Depends
from src.features.documents import services as doc_services
from src.features.documents.schemas import DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])

async def list_documents(user_id: str = Depends(doc_services.get_user_id)) -> List[DocumentOut]: # pure service function call
return await doc_services.list_documents_for_user(user_id)

@router.get("/", response_model=List[DocumentOut])
async def get_documents(result = Depends(list_documents)):
return result
]]></code>
</endpoint>

    	<factoryExample>
    		<code><![CDATA[

from typing import Callable
from src.core.cache.redis_client import RedisClient

def make_redis_client(url: str) -> Callable[[], RedisClient]:
def factory() -> RedisClient: # create/configure client, avoid connecting at import time
return RedisClient.from_url(url)

    	return factory
    		]]></code>
    	</factoryExample>

    	<adapterExample>
    		<code><![CDATA[

class VectorDBAdapter:
def **init**(self, client):
self.\_client = client

    	async def upsert(self, items: list[dict]) -> bool:
    			# adapt third-party API to the async interface expected by the app
    			return await self._client.async_upsert(items)
    		]]></code>
    	</adapterExample>
    </examples>

    <doNot>
    	<items>
    		<item>Do not run I/O at import time.</item>
    		<item>Do not create global mutable singletons; prefer factories + DI.</item>
    		<item>Avoid deep class hierarchies when composition + small adapters suffice.</item>
    	</items>
    </doNot>

    <final>
    	<summary>Favor clarity, immutability, testability, and types. Keep endpoints thin and services pure. Enforce formatting and typing through pre-commit/CI.</summary>
    	<followups>
    		<item>Add or align pyproject.toml settings for black/isort and mypy.ini for type rules.</item>
    		<item>Consider pre-commit hooks for black/isort/mypy to keep the repo clean.</item>
    	</followups>
    </final>

</copilot-instructions>
```
from app.utils.httpResponse import http_response
```python
@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    user = await get_user_from_db(user_id)
    return http_response("User retrieved successfully", user, 200, request)

@app.post("/users")
async def create_user(user_data: dict, request: Request):
    user = await create_user_in_db(user_data)
    return http_response("User created successfully", user, 201, request)

@app.get("/users")
async def list_users(request: Request):
    users = await get_all_users()
    return http_response("Users retrieved", users, request=request)
from app.utils.logging import logger

from app.core.exceptions import NotFoundException, ValidationException

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    if not user:
        raise NotFoundException(f"User {user_id} not found")
    return user

@app.post("/users")
async def create_user(data: dict):
    if not data.get("email"):
        raise ValidationException("Email is required")
    return user

from app.utils.httpError import http_error
from app.shared.enums import NOT_FOUND, UNAUTHORIZED

@app.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    user = await get_user_from_db(user_id)
    if not user:
        raise http_error(NOT_FOUND, 404, request)
    return user

@app.post("/login")
async def login(credentials: dict, request: Request):
    try:
        token = authenticate(credentials)
        return {"token": token}
    except Exception as e:
        raise http_error(UNAUTHORIZED, 401, request, e)


logger.info("User logged in", {"userId": 123})
logger.error("Database error", {"error": str(e)})
logger.warn("Slow query", {"duration": 5000})
logger.debug("Debug info", {"data": data})



from app.utils.apiFeatures import APIFeatures

@app.get("/users")
async def get_users(
    page: int = 1,
    limit: int = 10,
    sort: str = "-createdAt",
    fields: str = None,
    db: AsyncIOMotorClient = Depends(get_db)
):
    query_params = {
        "page": page,
        "limit": limit,
        "sort": sort,
        "fields": fields,
    }
    
    features = APIFeatures(db.users, query_params)
    users = await features.filter().sort().limit_fields().paginate().execute()
    
    return http_response("Users retrieved", users, request=request)

# Cursor pagination
@app.get("/posts")
async def get_posts(
    cursor: str = None,
    direction: str = "next",
    limit: int = 10,
    db: AsyncIOMotorClient = Depends(get_db)
):
    query_params = {"cursor": cursor, "direction": direction, "limit": limit}
    features = APIFeatures(db.posts, query_params)
    posts = await features.cursor_paginate().execute()
    
    return http_response("Posts retrieved", posts)
```
