---
inclusion: always
---

# Logging, Errors & Responses

## Structured Logging

Use structured logging consistently:

```python
from structlog import get_logger

logger = get_logger()

# Good
logger.bind(user_id=user.id, action="login").info("user_login_attempt")

# Bad
logger.info(f"User {user.id} logged in")
```

Keep logs:
- Contextual and machine-parseable
- Bound with relevant metadata
- Free of sensitive data

Reference: `src/app/examples/logger_usage_example.py`

## Typed Exceptions

Use project exceptions from `src/app/utils/exceptions.py`:

```python
from src.app.utils.exceptions import (
    NotFoundException,
    ValidationException,
    UnauthorizedException,
    ConflictException,
)

# In service layer
if not user:
    raise NotFoundException(f"User {user_id} not found")

if not is_valid(data):
    raise ValidationException("Invalid email format")

if not has_permission(user):
    raise UnauthorizedException("Insufficient permissions")

if user_exists:
    raise ConflictException("User already exists")
```

Never use raw `HTTPException` in service/repository code.

## Response Envelope

Uniform response shape:

```python
{
    "success": true,
    "statusCode": 200,
    "data": {...},
    "error": null,
    "request": {
        "method": "GET",
        "path": "/api/items"
    }
}
```

Error response:

```python
{
    "success": false,
    "statusCode": 400,
    "data": null,
    "error": {
        "message": "Invalid input",
        "code": "VALIDATION_ERROR"
    },
    "request": {...}
}
```

Use `APIResponse[T]` and `http_response(...)` from `src/app/shared/response_type.py`

## Global Exception Handler

Registered in `src/app/main.py`:

```python
from src.app.middleware.global_exception_handler import global_exception_handler

app.add_exception_handler(Exception, global_exception_handler)
```

Location: `src/app/middleware/global_exception_handler.py`

## No HTTP in Repositories

- Repository layer: persistence only
- Service layer: business logic and exceptions
- Router layer: HTTP response formatting
- Never put HTTP concerns in repositories
