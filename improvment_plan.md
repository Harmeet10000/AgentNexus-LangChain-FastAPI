# Project Improvements & Technical Debt

This document outlines specific improvements needed in the existing codebase based on analysis of current patterns and best practices.

---

## 🔴 Critical Improvements

### 1. Missing __init__.py Exports

**Issue**: Several packages lack proper __init__.py exports, forcing direct submodule imports.

**Files to Fix**:

#### `src/app/connections/__init__.py`
```python




Copy

Insert at cursor
python
2. PostgreSQL Connection Not Used in Lifespan
Issue: PostgreSQL connection is configured but not initialized in the lifespan manager.

File: src/app/lifecycle/lifespan.py

# ADD THIS
from app.connections.postgres import init_db, close_db

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    logger.info("Application starting")

    # MongoDB
    mongo_client, db = await create_mongo_client(...)
    app.state.mongo_client = mongo_client
    app.state.db = db

    # Redis
    redis = create_redis_client(settings.REDIS_URL)
    app.state.redis = redis

    # ✅ ADD: PostgreSQL initialization
    await init_db()
    logger.info("PostgreSQL connected")

    yield

    # Shutdown
    logger.info("Application shutting down")
    mongo_client.close()
    await redis.close()
    
    # ✅ ADD: PostgreSQL cleanup
    await close_db()
    logger.info("PostgreSQL connection closed")


Copy

Insert at cursor
python
3. Inconsistent Error Response Format
Issue: Error responses don't follow a consistent structure across the application.

File: src/app/middleware/global_exception_handler.py

Current: Returns different structures for different error types.

Improvement: Standardize all error responses:

# Standard error response format
{
    "success": false,
    "statusCode": 400,
    "name": "ValidationError",
    "message": "Validation failed",
    "correlationId": "abc123",
    "timestamp": "2025-01-15T10:30:00Z",
    "path": "/api/v1/users",
    "data": {
        "errors": [...]
    }
}

Copy

Insert at cursor
python
🟡 High Priority Improvements
4. Add Health Check for All Services
Issue: Health endpoint doesn't check database/cache connectivity.

File: src/app/features/health/handler.py (needs creation)

from fastapi import Request
from app.utils.logger import logger

async def health_check(request: Request):
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {}
    }

    # Check MongoDB
    try:
        await request.app.state.mongo_client.admin.command("ping")
        health_status["services"]["mongodb"] = "healthy"
    except Exception as e:
        health_status["services"]["mongodb"] = "unhealthy"
        health_status["status"] = "degraded"
        logger.error(f"MongoDB health check failed: {e}")

    # Check Redis
    try:
        await request.app.state.redis.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = "unhealthy"
        health_status["status"] = "degraded"
        logger.error(f"Redis health check failed: {e}")

    # Check PostgreSQL
    try:
        from app.connections.postgres import engine
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        health_status["services"]["postgres"] = "healthy"
    except Exception as e:
        health_status["services"]["postgres"] = "unhealthy"
        health_status["status"] = "degraded"
        logger.error(f"PostgreSQL health check failed: {e}")

    return health_status


Copy

Insert at cursor
python
5. Add Request ID to All Log Messages
Issue: Correlation ID exists but not consistently used in all log messages.

Improvement: Ensure all logger calls include correlation_id from request.state.

6. Missing Type Hints in Some Functions
Files with Missing Type Hints:

src/app/lifecycle/signals.py - graceful_shutdown function

Various utility functions

Action: Run ty check src/ and fix all type errors.

7. Add Retry Logic for External Services
Issue: No retry mechanism for transient failures (Redis, MongoDB).

Improvement: Use tenacity library (already in dependencies):

from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10)
)
async def connect_to_redis(url: str) -> Redis:
    redis = create_redis_client(url)
    await redis.ping()  # Verify connection
    return redis

Copy

Insert at cursor
python
🟢 Medium Priority Improvements
8. Add API Versioning Strategy
Issue: Routes use /api/v1/ prefix but no version management strategy.

Improvement: Create version router aggregator:

# src/app/api/versions.py
from fastapi import APIRouter
from app.features.auth.router import router as auth_router
from app.features.health.router import router as health_router

v1_router = APIRouter(prefix="/api/v1")
v1_router.include_router(auth_router)
v1_router.include_router(health_router)

# main.py
app.include_router(v1_router)

Copy

Insert at cursor
python
9. Add Rate Limiting Per User
Issue: Rate limiting exists but not per-user basis.

File: Create src/app/middleware/rate_limit.py

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# In router
@router.post("/login")
@limiter.limit("5/minute")
async def login(request: Request, ...):
    pass

Copy

Insert at cursor
python
10. Add Prometheus Metrics for Business Events
Issue: Only HTTP metrics tracked, no business metrics.

Improvement: Add business-specific metrics:

from prometheus_client import Counter, Histogram

# Business metrics
user_registrations = Counter(
    "user_registrations_total",
    "Total user registrations",
    ["status"]
)

login_attempts = Counter(
    "login_attempts_total",
    "Total login attempts",
    ["status"]
)

# In service
async def register(self, data: RegisterRequest):
    try:
        user = await self.user_repo.create(user)
        user_registrations.labels(status="success").inc()
        return user
    except Exception:
        user_registrations.labels(status="failure").inc()
        raise

Copy

Insert at cursor
python
11. Add Database Migration Scripts
Issue: Alembic configured but no migrations exist.

Action:

# Create initial migration
uv run alembic revision --autogenerate -m "Initial schema"

# Apply migrations
uv run alembic upgrade head

Copy

Insert at cursor
bash
12. Add Request/Response Logging Middleware
Issue: No structured logging of request/response payloads.

File: Create src/app/middleware/request_logging.py

@app.middleware("http")
async def log_requests(request: Request, call_next):
    correlation_id = getattr(request.state, "correlation_id", "unknown")
    
    # Log request
    logger.info(
        "Incoming request",
        method=request.method,
        path=request.url.path,
        correlation_id=correlation_id,
    )
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    # Log response
    logger.info(
        "Outgoing response",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        correlation_id=correlation_id,
    )
    
    return response



Copy

Insert at cursor
python
14. Add Response Models to All Endpoints
Issue: Some endpoints use response_model=None.

Action: Define proper response models for all endpoints.

15. Add API Documentation Examples
Issue: No request/response examples in OpenAPI docs.

@router.post(
    "/register",
    response_model=UserResponse,
    responses={
        200: {
            "description": "User registered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "id": "507f1f77bcf86cd799439011",
                        "email": "user@example.com",
                        "full_name": "John Doe"
                    }
                }
            }
        },
        400: {
            "description": "Email already exists"
        }
    }
)
async def register(...):
    pass

Copy

Insert at cursor
python
16. Add Graceful Shutdown for Background Tasks
Issue: Signal handlers exist but don't wait for background tasks.

Improvement: Track and await background tasks during shutdown.

17. Add Configuration Validation on Startup
Issue: Invalid configuration discovered at runtime.

# In lifespan startup
def validate_settings(settings: Settings):
    """Validate critical settings on startup."""
    if not settings.JWT_SECRET_KEY or settings.JWT_SECRET_KEY == "super-secret-change-this-in-production":
        raise ValueError("JWT_SECRET_KEY must be set in production")
    
    if settings.ENVIRONMENT == "production" and "*" in settings.CORS_ORIGINS:
        raise ValueError("CORS_ORIGINS cannot be '*' in production")

# Call in lifespan
validate_settings(settings)

Copy

Insert at cursor
python
📋 Code Quality Improvements
18. Add Pre-commit Hooks
File: .pre-commit-config.yaml (already exists, ensure it's used)

# Install pre-commit hooks
uv run pre-commit install

# Run manually
uv run pre-commit run --all-files



