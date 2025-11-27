# Loguru Structured Logging Guide

## Overview

Your project already uses **loguru** for logging. Here's how to implement structured logging like:

```python
logger.info("CONTROLLER_RESPONSE", response=response)
```

## Configuration

### 1. **LogConfig** (in `src/app/utils/logger.py`)

```python
class LogConfig(BaseSettings):
    ENVIRONMENT: str = Environment.DEVELOPMENT
    LOG_LEVEL: str = "DEBUG"
    LOG_DIR: Path = Path("logs/")
    LOG_ROTATION: str = "5 MB"
    LOG_RETENTION: str = "30 days"
    LOG_COMPRESSION: str = "zip"
    LOG_BACKTRACE: bool = True
    LOG_DIAGNOSE: bool = False
```

### 2. **Console Format** (Pretty colors + structured data)

```python
def console_format(record: dict[str, Any]) -> str:
    """Format logs for console with structured data."""
    # Output example:
    # INFO [2025-11-25T10:30:45.123Z] CONTROLLER_RESPONSE | response={'status': 200}
```

### 3. **JSON Format** (For file/log aggregation tools)

```python
def json_format(record: dict[str, Any]) -> str:
    """Format logs as JSON for ELK, Datadog, CloudWatch."""
    # Output example:
    # {"timestamp": "...", "level": "INFO", "message": "CONTROLLER_RESPONSE", "response": {...}}
```

## Usage Patterns

### Basic Structured Logging

```python
from src.app.utils.logger import logger

# Log with context
logger.info(
    "CONTROLLER_RESPONSE",
    response=response_data,
    status_code=200,
    endpoint="/api/users/123"
)

# Log database operations
logger.info(
    "DATABASE_QUERY",
    query="SELECT * FROM users",
    duration_ms=125,
    rows_affected=5
)

# Log errors with context
logger.error(
    "AUTH_FAILED",
    username="user@example.com",
    reason="invalid_token",
    attempt_count=3
)
```

### In FastAPI Endpoints

```python
@router.get("/api/v1/users/{user_id}")
async def get_user(user_id: int, request: Request):
    user = await db.get_user(user_id)

    logger.info(
        "CONTROLLER_RESPONSE",
        endpoint=str(request.url),
        method=request.method,
        status_code=200,
        response=user,
        response_time_ms=12.5
    )

    return JSONResponse(status_code=200, content=user)
```

### In Services

```python
class UserService:
    async def create_user(self, username: str, email: str):
        logger.info("USER_CREATE_STARTED", username=username)

        try:
            user = await db.create_user(username, email)

            logger.info(
                "USER_CREATE_SUCCESS",
                user_id=user.id,
                duration_ms=45
            )
            return user

        except Exception as e:
            logger.error(
                "USER_CREATE_FAILED",
                username=username,
                error=str(e)
            )
            raise
```

## Console Output

With the enhanced formatter, you'll see:

```
INFO [2025-11-25T10:30:45.123Z] CONTROLLER_RESPONSE | response={'status': 200, 'data': {...}} status_code=200 endpoint='/api/users/123'

INFO [2025-11-25T10:30:46.456Z] DATABASE_QUERY | query='SELECT * FROM users' duration_ms=125 rows_affected=5 table='users'

ERROR [2025-11-25T10:30:47.789Z] AUTH_FAILED | username='john@example.com' reason='invalid_token' attempt_count=3
```

## File Output (JSON Format)

Logs are stored as JSON for log aggregation:

```json
{
  "timestamp": "2025-11-25T10:30:45.123000Z",
  "level": "INFO",
  "logger": "src.app.main",
  "message": "CONTROLLER_RESPONSE",
  "function": "get_user",
  "line": 42,
  "response": {"status": 200},
  "status_code": 200,
  "endpoint": "/api/users/123"
}
```

## Setup in `main.py`

```python
from src.app.utils.logger import setup_logging, LogConfig
from src.app.core.settings import get_settings

@app.on_event("startup")
async def startup():
    settings = get_settings()
    log_config = LogConfig(
        ENVIRONMENT=settings.ENVIRONMENT,
        LOG_LEVEL=settings.LOG_LEVEL,
        LOG_DIR=Path(settings.LOG_FILE).parent
    )
    setup_logging(log_config)
```

## Best Practices

### 1. **Use Meaningful Event Names**
```python
# Good
logger.info("USER_CREATED", user_id=123)
logger.error("DATABASE_CONNECTION_FAILED", error=str(e))

# Avoid
logger.info("user", msg="created")
logger.error("error", msg="db failed")
```

### 2. **Include Relevant Context**
```python
# Good: helps debugging
logger.info(
    "API_CALL",
    method="POST",
    endpoint="/api/documents",
    user_id=123,
    response_time_ms=145
)

# Avoid: too vague
logger.info("API_CALL", msg="done")
```

### 3. **Use Appropriate Log Levels**

| Level | Use Case |
|-------|----------|
| `DEBUG` | Detailed info for debugging (variable values, function calls) |
| `INFO` | General informational messages (app started, user created) |
| `WARNING` | Warning conditions (deprecated feature used, fallback applied) |
| `ERROR` | Error events (failed request, exception caught) |
| `CRITICAL` | Critical system failures (cannot start app, data loss) |

### 4. **Structure for Log Aggregation**
```python
# If using ELK/Datadog, include standard fields:
logger.info(
    "REQUEST_COMPLETED",
    correlation_id="corr-123",
    user_id=123,
    request_id="req-456",
    duration_ms=100,
    status_code=200
)
```

### 5. **Handle Sensitive Data**
```python
# Good: mask sensitive data
logger.info(
    "LOGIN_SUCCESS",
    username=username,
    # Never log passwords!
)

# Remove sensitive data in production
if settings.ENVIRONMENT == "production":
    extra_data.pop("password", None)
    extra_data.pop("token", None)
```

## Log Rotation & Retention

Your configuration automatically:

- **Rotates logs** when they exceed 5 MB
- **Retains logs** for 30 days
- **Compresses old logs** using ZIP

Configure in `LogConfig`:
```python
LOG_ROTATION: str = "5 MB"        # Or "500 MB", "100 MB", "1 day", etc.
LOG_RETENTION: str = "30 days"    # Or "10 days", "90 days", etc.
LOG_COMPRESSION: str = "zip"      # Compress rotated logs
```

## Filtering & Searching Logs

### Find errors
```bash
grep '"level": "ERROR"' logs/app_2025-11-25.log
```

### Find specific operation
```bash
grep '"message": "USER_CREATED"' logs/app_2025-11-25.log
```

### Real-time monitoring
```bash
tail -f logs/app_*.log | grep -E '"level": "(ERROR|WARNING)"'
```

## Log Aggregation Integration

For **ELK Stack**, **Datadog**, or **CloudWatch**:

1. Point log shipper to `logs/` directory
2. Parse JSON format (automatically handled)
3. Query by:
   - `message` field for event type
   - Custom fields (user_id, correlation_id, etc.)
   - `level` for severity

Example Datadog query:
```
@message:"CONTROLLER_RESPONSE" @status_code:5*
```

## Middleware Integration

Add to middleware for automatic request/response logging:

```python
from src.app.utils.logger import logger

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time()
        response = await call_next(request)
        duration = (time() - start) * 1000

        logger.info(
            "HTTP_REQUEST",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration, 2)
        )
        return response
```

## Files

- `src/app/utils/logger.py` - Main logger configuration
- `src/app/utils/logger_enhanced.py` - Enhanced formatters (console + JSON)
- `src/app/utils/logger_examples.py` - Usage examples

## References

- [Loguru Documentation](https://loguru.readthedocs.io/)
- [JSON Logging Best Practices](https://www.kartar.net/2015/12/structured-logging/)
- [ELK Stack Setup](https://www.elastic.co/guide/en/kibana/current/index.html)
