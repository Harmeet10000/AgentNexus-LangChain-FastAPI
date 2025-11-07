# Logger Usage Examples

## Import
```python
from app.utils.logger import logger
```

## 1. DEBUG - Detailed diagnostic information
```python
# Simple debug message
logger.debug("Processing user request")

# With metadata
logger.debug("Query executed", query="SELECT * FROM users", execution_time=0.045)

# In database operations
async def fetch_users():
    logger.debug("Fetching users from database", limit=10, offset=0)
    # ... database logic
```

**Console Output:**
```
DEBUG [2025-11-07T15:15:26.905Z] Processing user request
META { query: 'SELECT * FROM users', execution_time: 0.045 }
```

---

## 2. INFO - General informational messages
```python
# Application startup
logger.info("Application started successfully", port=5000, environment="production")

# Database connection
logger.info("PostgreSQL Connected: localhost", readyState=1, poolSize=10)

# Successful operations
logger.info("User registered", user_id="123", email="user@example.com")

# API requests
logger.info("GET /api/users", status_code=200, response_time=45)
```

**Console Output:**
```
INFO [2025-11-07T15:15:26.905Z] PostgreSQL Connected: localhost
META { readyState: 1, poolSize: 10 }
```

---

## 3. WARNING - Warning messages for potentially harmful situations
```python
# Deprecated features
logger.warning("Using deprecated API endpoint", endpoint="/api/v1/old")

# Performance issues
logger.warning("Slow query detected", query_time=5.2, threshold=1.0)

# Resource limits
logger.warning("Connection pool near capacity", current=8, max=10)

# Validation warnings
logger.warning("Invalid input sanitized", field="username", original="user<script>")
```

**Console Output:**
```
WARNING [2025-11-07T15:15:26.905Z] Slow query detected
META { query_time: 5.2, threshold: 1.0 }
```

---

## 4. ERROR - Error events that might still allow the application to continue
```python
# API errors
logger.error("Failed to fetch user", user_id="123", error="User not found")

# Database errors
logger.error("Database query failed", query="SELECT * FROM users", error_code="23505")

# External service errors
logger.error("Payment gateway timeout", transaction_id="tx_123", gateway="stripe")

# With error object
try:
    result = await some_operation()
except Exception as e:
    logger.error("Operation failed", operation="some_operation", error=str(e))
```

**Console Output:**
```
ERROR [2025-11-07T15:15:26.905Z] Failed to fetch user
META { user_id: '123', error: 'User not found' }
```

---

## 5. CRITICAL - Very severe error events that might cause the application to abort
```python
# Database connection lost
logger.critical("Database connection lost", attempts=3, last_error="Connection refused")

# Critical resource exhaustion
logger.critical("Out of memory", available_mb=50, required_mb=500)

# Security breaches
logger.critical("Unauthorized access attempt", ip="192.168.1.100", endpoint="/admin")

# System failures
logger.critical("Failed to start server", port=5000, error="Address already in use")
```

**Console Output:**
```
CRITICAL [2025-11-07T15:15:26.905Z] Database connection lost
META { attempts: 3, last_error: 'Connection refused' }
```

---

## Exception Logging with Stack Traces

### Method 1: Using logger.exception() (automatically captures exception)
```python
try:
    result = 1 / 0
except ZeroDivisionError:
    logger.exception("Division by zero error")
```

### Method 2: Using logger.error() with opt(exception=True)
```python
try:
    await database.connect()
except Exception as e:
    logger.opt(exception=True).error("Database connection failed", host="localhost")
```

**Console Output:**
```
ERROR [2025-11-07T15:15:26.905Z] Database connection failed
META { host: 'localhost' }
Traceback (most recent call last):
  File "/app/connections/postgres.py", line 45, in init_db
    await database.connect()
  File "/app/database.py", line 120, in connect
    raise ConnectionError("Failed to connect")
ConnectionError: Failed to connect
```

---

## Real-World Examples

### 1. API Route Handler
```python
from fastapi import APIRouter, HTTPException
from app.utils.logger import logger

router = APIRouter()

@router.get("/users/{user_id}")
async def get_user(user_id: str):
    logger.debug("Fetching user", user_id=user_id)
    
    try:
        user = await fetch_user(user_id)
        logger.info("User fetched successfully", user_id=user_id, username=user.username)
        return user
    except UserNotFoundError:
        logger.warning("User not found", user_id=user_id)
        raise HTTPException(status_code=404, detail="User not found")
    except Exception as e:
        logger.exception("Failed to fetch user", user_id=user_id)
        raise HTTPException(status_code=500, detail="Internal server error")
```

### 2. Database Connection
```python
async def init_db():
    try:
        await engine.connect()
        logger.info("Database connected", host="localhost", pool_size=10)
    except ConnectionError as e:
        logger.critical("Failed to connect to database", error=str(e), host="localhost")
        raise
```

### 3. Background Task
```python
async def process_emails():
    logger.info("Starting email processing job")
    
    try:
        emails = await fetch_pending_emails()
        logger.debug("Fetched pending emails", count=len(emails))
        
        for email in emails:
            try:
                await send_email(email)
                logger.info("Email sent", email_id=email.id, recipient=email.to)
            except Exception as e:
                logger.error("Failed to send email", email_id=email.id, error=str(e))
        
        logger.info("Email processing completed", total=len(emails))
    except Exception as e:
        logger.exception("Email processing job failed")
```

### 4. Middleware
```python
from fastapi import Request
from app.utils.logger import logger

async def log_requests(request: Request, call_next):
    logger.debug("Incoming request", method=request.method, path=request.url.path)
    
    try:
        response = await call_next(request)
        logger.info(
            f"{request.method} {request.url.path}",
            status_code=response.status_code,
            client_ip=request.client.host
        )
        return response
    except Exception as e:
        logger.exception("Request processing failed", method=request.method, path=request.url.path)
        raise
```

### 5. Service Layer
```python
class UserService:
    async def create_user(self, data: dict):
        logger.debug("Creating user", email=data.get("email"))
        
        try:
            # Validate
            if await self.user_exists(data["email"]):
                logger.warning("User already exists", email=data["email"])
                raise ValueError("User already exists")
            
            # Create
            user = await self.db.create(data)
            logger.info("User created successfully", user_id=user.id, email=user.email)
            return user
            
        except ValueError as e:
            logger.error("User creation validation failed", error=str(e), email=data.get("email"))
            raise
        except Exception as e:
            logger.exception("User creation failed", email=data.get("email"))
            raise
```

---

## Best Practices

1. **Use appropriate levels:**
   - DEBUG: Development/troubleshooting info
   - INFO: Normal operations, milestones
   - WARNING: Unexpected but handled situations
   - ERROR: Errors that are caught and handled
   - CRITICAL: System-threatening errors

2. **Include context with metadata:**
   ```python
   # Good
   logger.error("Payment failed", user_id="123", amount=99.99, gateway="stripe")
   
   # Bad
   logger.error("Payment failed")
   ```

3. **Use exception logging for errors:**
   ```python
   # Good
   try:
       risky_operation()
   except Exception:
       logger.exception("Operation failed")
   
   # Bad
   try:
       risky_operation()
   except Exception as e:
       logger.error(f"Operation failed: {e}")
   ```

4. **Don't log sensitive data:**
   ```python
   # Bad
   logger.info("User login", password=password)
   
   # Good
   logger.info("User login", user_id=user_id)
   ```
