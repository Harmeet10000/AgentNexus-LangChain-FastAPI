```xml
<copilot-instructions>
	<meta>
		<language name="python" version="3.12" />
		<style>
			<principles>functional, DRY, KISS</principles>
			<immutability>Prefer immutability and pure functions; avoid module-level mutable state</immutability>
			<typing>Use type hints everywhere; write code compatible with strict mypy checks</typing>
			<formatting>Use Ruff for linting and formatting; use project's pyproject.toml where present</formatting>
		</style>
	</meta>

    <fastapi>
    	<bestPractices>
    		<routers>Use APIRouter per feature and mount routers in a single place. Keep routes thin: delegate logic to services.</routers>
    		<dependencyInjection>Prefer fastapi.Depends for injecting services and config. Avoid singletons at import time; use factories to create clients.</dependencyInjection>
    		<pydantic>Use Pydantic models for request/response. Prefer validation through typed models; for Pydantic v2 prefer .model_dump() and validators that are pure.</pydantic>
    		<async>Prefer async handlers and async I/O. When using blocking code, run it via asyncio.to_thread.run_sync or similar.</async>
    		<errors>Centralize error handling in middleware (see src/api/middleware/global_exception_handler.py). Return typed error responses.</errors
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
    		<ruff>Use Ruff for both linting and formatting (line-length: 88). Ruff replaces Black, isort, Pylint, and Flake8.</ruff>
    		<imports>Ruff handles import sorting automatically; group imports: stdlib, third-party, local packages.</imports>
    		<indentation>Use 4 spaces.</indentation>
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
    </designPatterns>

    <practicalTips>
    	<structure>Group by feature: src/features/<feature>/{api,schemas,services,tests}. Keep services independent of FastAPI internals.</structure>
    	<logging>Use structured logging from src/core/logging and include correlation ids from middleware.</logging>
    	<ci>Run ruff check, ruff format --check, mypy, and pytest in CI. Fail on formatting/type errors.</ci>
    	<preCommit>Use pre-commit hooks: uv-lock (dependency sync), ruff (lint+format), mypy (types), bandit (security), and standard checks (trailing whitespace, YAML/JSON/TOML validation, private key detection).</preCommit>
    	<packageManager>Use uv for dependency management. Commands: uv pip install, uv run, uv sync. No manual venv activation needed with uv.</packageManager>
    </practicalTips>

    <examples>
    	<endpoint description="Pure function + FastAPI wiring">
    		<code>
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
</code>
</endpoint>

    	<factoryExample>
    		<code>

from typing import Callable
from src.core.cache.redis_client import RedisClient

def make_redis_client(url: str) -> Callable[[], RedisClient]:
def factory() -> RedisClient: # create/configure client, avoid connecting at import time
return RedisClient.from_url(url)

    	return factory
    </code>
    	</factoryExample>

    	<adapterExample>
    		<code>

class VectorDBAdapter:
def **init**(self, client):
self.\_client = client

    	async def upsert(self, items: list[dict]) -> bool:
    			# adapt third-party API to the async interface expected by the app
    			return await self._client.async_upsert(items)
    	</code>
    	</adapterExample>
    </examples>

    <doNot>
    	<items>
    		<item>Do not run I/O at import time.</item>
    		<item>Do not create global mutable singletons; prefer factories + DI.</item>
    		<item>Avoid deep class hierarchies when composition + small adapters suffice.</item>
    	</items>
    </doNot>

    <toolchain>
    	<linter>Ruff (v0.7.1+) - Fast linter replacing Pylint/Flake8. Config in pyproject.toml [tool.ruff.lint].</linter>
    	<formatter>Ruff Format - Fast formatter replacing Black. Config in pyproject.toml [tool.ruff.format].</formatter>
    	<typeChecker>MyPy (v1.18.2+) - Strict type checking. Config in pyproject.toml [tool.mypy].</typeChecker>
    	<security>Bandit (v1.7.10+) - Security linting. Config in pyproject.toml [tool.bandit].</security>
    	<packageManager>uv - Fast Python package manager (10-100x faster than pip). Use uv pip install, uv run, uv sync.</packageManager>
    	<preCommit>Pre-commit hooks enforce: uv-lock sync, ruff lint+format, mypy types, bandit security, file checks (trailing whitespace, YAML/JSON/TOML, private keys, merge conflicts).</preCommit>
    </toolchain>

    <final>
    	<summary>Favor clarity, immutability, testability, and types. Keep endpoints thin and services pure. Enforce formatting and typing through pre-commit/CI.</summary>
    	<followups>
    		<item>Ensure pyproject.toml has [tool.ruff], [tool.ruff.lint], [tool.ruff.format], and [tool.mypy] sections.</item>
    		<item>Run pre-commit install to enable hooks. Use pre-commit run --all-files to validate.</item>
    		<item>Use uv for all dependency operations: uv pip install -e ".[dev]", uv run pytest, uv run uvicorn.</item>
    	</followups>
    </final>

</copilot-instructions>
```
```python
from app.utils.httpResponse import http_response
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


<!-- always use these kinds of imports Modern Python Pattern (Cleaner Imports) -->
<!-- You can make imports even cleaner using __init__.py: -->
```py
utils/__init__.py:
from .logger import logger
from .http_error import httpError

__all__ = ['logger', 'httpError']
Now in routes/users.py:
from ..utils import logger, httpError 


from fastapi import APIRouter, Depends, Request, status
from models.auth_models import RegisterRequest, RegisterResponse
from services.auth_service import AuthService
from utils.responses import http_response
from utils.exceptions import APIException

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Dependency injection (optional, but best practice)
def get_auth_service() -> AuthService:
    return AuthService()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    body: RegisterRequest,  # Pydantic automatically validates!
    auth_service: AuthService = Depends(get_auth_service)
):

    
    # If validation fails, FastAPI automatically returns 422 error
    # No need for manual validation like in Express!
    
    new_user = await auth_service.register_user(body)
         # Example: Check if user exists
         if new_user:
             raise APIException(
                 status_code=409,
                 message="User already exists",
                 name="ConflictError"
             )
        
    return http_response(
        request=request,
        status_code=status.HTTP_201_CREATED,
        message="User registered successfully",
        data={"_id": new_user.id}
    )

    # routes/auth_routes.py
# ============================================
from fastapi import APIRouter, Depends, Request, status
from models.auth_models import RegisterRequest, RegisterResponse
from services.auth_service import AuthService
from utils.responses import http_response
from utils.exceptions import APIException

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Dependency injection (optional, but best practice)
def get_auth_service() -> AuthService:
    return AuthService()

@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    request: Request,
    body: RegisterRequest,  # Pydantic automatically validates!
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user
    
    No try-catch needed! FastAPI handles validation automatically.
    Just raise exceptions when needed.
    """
    
    # If validation fails, FastAPI automatically returns 422 error
    # No need for manual validation like in Express!
    
    new_user = await auth_service.register_user(body)
    
    return http_response(
        request=request,
        status_code=status.HTTP_201_CREATED,
        message="User registered successfully",
        data={"_id": new_user.id}
    )

    

```
use 'is' for check is None, boolean value instead of == and 'is not' instead of != 
use list, dictionary, sets comprehension instead of a for loop
use 'with' context manager where necessary and required
prefer try, except blocks over if else where possible
example :-
```py
from dataclasses import dataclass
from datetime import datetime
from typing import Iterator
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FOOD_CSV = Path("food.csv")
ACTIVITY_CSV = Path("activities.csv")


@dataclass
class Entry:
    date: str
    description: str
    calories: int

def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")

def append_entry(path: Path, entry: Entry) -> None:
    with path.open("a") as f:
        f.write(f"{entry.date},{entry.description},{entry.calories}\n")
    logging.info(f"Appended entry to {path}: {entry.description} ({entry.calories} kcal)")

def log_food(description: str, calories: int, date: str | None = None) -> None:
    if date is None:
        date = today()
    append_entry(FOOD_CSV, Entry(date, description, calories))

def log_activity(description: str, calories: int, date: str | None = None) -> None:
    if date is None:
        date = today()
    append_entry(ACTIVITY_CSV, Entry(date, description, calories))

def read_entries(path: Path) -> Iterator[Entry]:
    try:
        with path.open() as f:
            for line in f:
                date, desc, cals = line.strip().split(",")
                yield Entry(date, desc, int(cals))
    except FileNotFoundError:
        logging.warning(f"File not found: {path}")
        return iter([])


def run_day_summary(date: str) -> None:
    food = list(read_entries(FOOD_CSV))
    activity = list(read_entries(ACTIVITY_CSV))

    food_total = sum(entry.calories for entry in food if entry.date == date)
    activity_total = sum(entry.calories for entry in activity if entry.date == date)
    net = food_total - activity_total

    print(f"\nSummary for {date}")
    print(f"  🍎 Food:     {food_total} kcal")
    print(f"  🏃 Activity: {activity_total} kcal")
    print(f"  ⚖️  Net:       {net} kcal")

    logging.info(f"Daily summary: +{food_total} kcal intake, -{activity_total} burned, net = {net}")

def main() -> None:
    log_food("Banana", 100)
    log_activity("Running", 300)
    run_day_summary(today())

if __name__ == "__main__":
    main()
```
<!-- use python syntax for v3.12 and use context7 for fetching latest documents -->
