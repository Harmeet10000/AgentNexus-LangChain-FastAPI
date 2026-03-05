"""
Logger Usage Examples - Real-world FastAPI patterns.

Middleware automatically adds: request_id, path, method, user_id, layer
All logs include this context. Just use logger.info() anywhere.
"""

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.logger import logger, request_state

# ============================================================================
# 1. ROUTER / ENDPOINT LAYER
# ============================================================================

router = APIRouter(prefix="/payments", tags=["payments"])


class PaymentCreate(BaseModel):
    amount: float
    currency: str = "USD"
    description: str | None = None


@router.post("/")
async def create_payment(
    payload: PaymentCreate,
    request: Request,
    current_user=Depends(get_current_user),
    service: PaymentService = Depends(get_payment_service),
):
    """Router logs automatically include request_id, path, method."""

    # Basic log with extra context
    logger.info(
        "Received payment request",
        amount=payload.amount,
        currency=payload.currency,
        user_id=current_user.id,
    )

    if payload.amount <= 0:
        logger.warning("Invalid amount", amount=payload.amount)
        raise HTTPException(400, "Amount must be positive")

    try:
        result = await service.create_payment(payload, current_user.id)
        logger.info("Payment created", payment_id=result.id, status=result.status)
        return result

    except ValueError as e:
        logger.warning("Validation failed", reason=str(e))
        raise HTTPException(400, str(e))

    except Exception:
        logger.exception("Payment creation failed")  # Auto-includes traceback
        raise HTTPException(500, "Internal error")


# ============================================================================
# 2. SERVICE LAYER
# ============================================================================

class PaymentService:
    def __init__(self, repository: PaymentRepository):
        self.repo = repository

    async def create_payment(self, data: PaymentCreate, user_id: int):
        """Service layer - update state to track layer."""
        state = request_state.get()
        state["layer"] = "service"
        state["user_id"] = user_id  # Update user_id in global state

        logger.info("Processing payment", amount=data.amount, step="validate")

        if data.amount > 10000:
            logger.warning("High-value payment", amount=data.amount, risk="elevated")

        payment = await self.repo.create_pending_payment(data, user_id)

        logger.info(
            "Payment processed",
            payment_id=payment.id,
            status=payment.status,
            step="persisted",
        )

        return payment


# ============================================================================
# 3. REPOSITORY LAYER (SQLAlchemy)
# ============================================================================

class PaymentRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_pending_payment(self, data: PaymentCreate, user_id: int):
        """Repository - update state to track layer."""
        state = request_state.get()
        state["layer"] = "repository"

        logger.debug(
            "Creating payment in DB",
            amount=data.amount,
            currency=data.currency,
            user_id=user_id,
        )

        payment = Payment(
            amount=data.amount,
            currency=data.currency,
            user_id=user_id,
            description=data.description,
            status="pending",
        )

        try:
            self.session.add(payment)
            await self.session.flush()
            await self.session.refresh(payment)

            logger.info(
                "Payment inserted",
                payment_id=payment.id,
                db_operation="insert+flush",
                table="payments",
            )

            return payment

        except Exception:
            await self.session.rollback()
            logger.exception("DB error", amount=data.amount, user_id=user_id)
            raise


# ============================================================================
# 4. BACKGROUND TASKS (No request context)
# ============================================================================

async def send_confirmation_email(payment_id: int, correlation_id: str | None = None):
    """Background task - manually add correlation context."""

    # Option 1: Use bind for persistent context
    task_logger = logger.bind(correlation_id=correlation_id, task="email")
    task_logger.info("Sending confirmation email", payment_id=payment_id)

    # Email logic
    task_logger.info("Email sent", payment_id=payment_id)


# In endpoint, pass correlation_id to background task:
# background_tasks.add_task(
#     send_confirmation_email,
#     payment_id=result.id,
#     correlation_id=request.state.correlation_id
# )


# ============================================================================
# 5. TEMPORARY CONTEXT (Scoped)
# ============================================================================

async def process_refund(payment_id: int, amount: float):
    """Use contextualize for temporary extra context."""
    state = request_state.get()

    with logger.contextualize(**state, operation="refund", payment_id=payment_id):
        logger.info("Starting refund", amount=amount)

        # Refund logic
        logger.info("Refund completed")


# ============================================================================
# CONSOLE OUTPUT EXAMPLES
# ============================================================================

"""
INFO [2024-01-15T10:30:45.123Z] Request started | request_id='abc123' path='/payments/' method='POST' user_id=None layer='middleware'
INFO [2024-01-15T10:30:45.124Z] Received payment request | request_id='abc123' path='/payments/' method='POST' user_id=None layer='middleware' amount=49.99 currency='USD' user_id=3842
INFO [2024-01-15T10:30:45.125Z] Processing payment | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='service' amount=49.99 step='validate'
DEBUG [2024-01-15T10:30:45.126Z] Creating payment in DB | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='repository' amount=49.99 currency='USD' user_id=3842
INFO [2024-01-15T10:30:45.127Z] Payment inserted | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='repository' payment_id=992 db_operation='insert+flush' table='payments'
INFO [2024-01-15T10:30:45.128Z] Payment processed | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='service' payment_id=992 status='pending' step='persisted'
INFO [2024-01-15T10:30:45.129Z] Payment created | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='middleware' payment_id=992 status='pending'
INFO [2024-01-15T10:30:45.130Z] Request finished | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='http_middleware_exit' status_code=200 duration_ms=7.2

ERROR [2024-01-15T10:33:12.901Z] DB error | request_id='abc123' path='/payments/' method='POST' user_id=3842 layer='repository' amount=49.99 user_id=3842
Traceback (most recent call last):
  ...
sqlalchemy.exc.IntegrityError: ...
"""


# ============================================================================
# QUICK REFERENCE
# ============================================================================

"""
LOG LEVELS BY LAYER:

Router/Endpoint:    INFO, WARNING, EXCEPTION
                    - High-level business outcomes
                    - logger.info("Message", key=value)

Service:            INFO, WARNING, DEBUG
                    - Business decisions & state changes
                    - Update state: state = request_state.get(); state["layer"] = "service"

Repository:         DEBUG, INFO, ERROR
                    - DB operations & query tracing
                    - Use DEBUG for query details in dev

Background:         INFO, ERROR
                    - No automatic context
                    - Use logger.bind(correlation_id=...) for tracing


PATTERNS:

1. Basic (90% of cases):
   logger.info("Message", key=value)

2. Update layer:
   state = request_state.get()
   state["layer"] = "service"

3. Exception with context:
   logger.exception("Error message", key=value)

4. Temporary context:
   with logger.contextualize(operation="refund"):
       logger.info("Message")

5. Background task:
   logger.bind(correlation_id=id).info("Message")
"""


# Here are practical, realistic examples showing **how to use your current logger** in different layers of a typical FastAPI application — given your middleware + contextvars setup + custom console formatter.

# Your current implementation has these important characteristics:

# - `logger.contextualize(**state)` is active during the whole request (set in middleware)
# - All logs inside request-handling code automatically include:
#   `request_id`, `path`, `method`, `user_id`, `layer` (and later also `status_code`, `duration_ms`)
# - The **console formatter** shows extra fields in a compact `key='value'` style after the message
# - Exceptions are nicely formatted when using `logger.exception()`
# - No JSON in console → clean & readable developer output
# - (File sink is commented out — so currently only console)

# ### 1. Router / Endpoint layer (entry point)

# ```python
# from fastapi import APIRouter, Depends, Request, HTTPException
# from pydantic import BaseModel

# from app.services.payment_service import PaymentService
# from app.dependencies import get_current_user  # your auth dep

# router = APIRouter(prefix="/payments", tags=["payments"])


# class PaymentCreate(BaseModel):
#     amount: float
#     currency: str = "USD"
#     description: str | None = None


# @router.post("/", response_model=PaymentOut)  # assume PaymentOut exists
# async def create_payment(
#     payload: PaymentCreate,
#     request: Request,
#     current_user = Depends(get_current_user),
#     service: PaymentService = Depends(PaymentService.from_request),  # factory dep
# ):
#     # All these logs automatically contain request_id, path, method, ...
#     logger.info(
#         "Received payment creation request",
#         extra={
#             "amount": payload.amount,
#             "currency": payload.currency,
#             "user_id": current_user.id,
#             "description_len": len(payload.description or ""),
#         }
#     )

#     if payload.amount <= 0:
#         logger.warning("Rejected negative/zero payment amount", extra={"amount": payload.amount})
#         raise HTTPException(400, "Amount must be positive")

#     try:
#         result = await service.create_payment(payload, current_user.id)
#         logger.info(
#             "Payment creation succeeded",
#             extra={
#                 "payment_id": result.id,
#                 "new_status": result.status,
#             }
#         )
#         return result

#     except ValueError as e:
#         logger.warning("Business validation failed", extra={"reason": str(e)})
#         raise HTTPException(400, str(e))

#     except Exception:
#         logger.exception("Unexpected error during payment creation")
#         raise HTTPException(500, "Internal server error")
# ```

# **Console output example** (what developer sees):

# ```
# INFO [2026-03-05T17:32:45.123Z] Received payment creation request | request_id='abc123xyz' path='/payments/' method='POST' user_id=None layer='http_middleware_exit' amount=49.99 currency='USD' user_id=3842 description_len=18
# INFO [2026-03-05T17:32:45.178Z] Payment creation succeeded     | request_id='abc123xyz' path='/payments/' method='POST' user_id=None layer='http_middleware_exit' payment_id=992 status='pending'
# ```

# ### 2. Service layer

# ```python
# # app/services/payment_service.py
# from loguru import logger

# class PaymentService:
#     def __init__(self, repository, event_publisher=None):
#         self.repo = repository
#         self.event_publisher = event_publisher

#     async def create_payment(self, data: PaymentCreate, user_id: int):
#         logger.info("Starting payment business logic", extra={
#             "amount": data.amount,
#             "user_id": user_id,
#             "step": "validate"
#         })

#         # Example: enrich global request context once we know the real user
#         # (optional — many teams avoid mutating global state)
#         try:
#             state = request_state.get()
#             state["user_id"] = user_id
#             # logger.contextualize(**state)   # ← only if you really want it everywhere
#         except LookupError:
#             pass  # outside of request context (e.g. background task)

#         if data.amount > 10000:
#             logger.warning("High-value payment detected", extra={
#                 "amount": data.amount,
#                 "risk_level": "elevated"
#             })

#         # Core logic
#         payment = await self.repo.create_pending_payment(data, user_id)

#         logger.info("Payment record created", extra={
#             "payment_id": payment.id,
#             "status": payment.status,
#             "step": "persisted"
#         })

#         # Example: async side-effect
#         if self.event_publisher:
#             logger.debug("Publishing payment.created event", extra={"payment_id": payment.id})
#             await self.event_publisher.publish("payment.created", {"id": payment.id})

#         return payment
# ```

# ### 3. Repository layer (SQLAlchemy example)

# ```python
# # app/repositories/payment_repository.py
# from sqlalchemy import select
# from sqlalchemy.ext.asyncio import AsyncSession

# from app.models import Payment
# from loguru import logger


# class PaymentRepository:
#     def __init__(self, session: AsyncSession):
#         self.session = session

#     async def create_pending_payment(self, data: PaymentCreate, user_id: int) -> Payment:
#         logger.debug("Creating pending payment in DB", extra={
#             "amount": data.amount,
#             "currency": data.currency,
#             "user_id": user_id
#         })

#         payment = Payment(
#             amount=data.amount,
#             currency=data.currency,
#             user_id=user_id,
#             description=data.description,
#             status="pending"
#         )

#         try:
#             self.session.add(payment)
#             await self.session.flush()           # get ID
#             await self.session.refresh(payment)

#             logger.info("Payment inserted successfully", extra={
#                 "payment_id": payment.id,
#                 "db_operation": "insert+flush",
#                 "table": "payments"
#             })

#             return payment

#         except Exception:
#             await self.session.rollback()
#             logger.exception("Database error while creating payment", extra={
#                 "amount": data.amount,
#                 "user_id": user_id
#             })
#             raise
# ```

# **Console example with exception**:

# ```
# ERROR [2026-03-05T17:33:12.901Z] Database error while creating payment | request_id='abc123xyz' path='/payments/' method='POST' user_id=3842 layer='http_middleware_exit' amount=49.99 user_id=3842
# Traceback (most recent call last):
#   ...
# sqlalchemy.exc.IntegrityError: ...
# ```

# ### 4. Background tasks / celery / outside request context

# ```python
# # Background task example
# from fastapi import BackgroundTasks

# async def send_confirmation_email(payment_id: int):
#     # No request context → no request_id etc.
#     logger.info("Sending confirmation email", extra={"payment_id": payment_id})

#     # If you want correlation:
#     logger.bind(correlation_id="bg-task-" + str(payment_id)).info("...")
# ```

# Or — pass context explicitly when starting background task:

# ```python
# # In endpoint
# background_tasks.add_task(
#     send_confirmation_email,
#     payment_id=result.id,
#     correlation_id=request.state.correlation_id
# )
# ```

# ### Quick reference – recommended log levels & patterns

# | Layer              | Most common levels              | Typical `extra=` keys you add                          | Notes / Recommendations                              |
# |---------------------|----------------------------------|--------------------------------------------------------|------------------------------------------------------|
# | Router/Endpoint     | INFO, WARNING, EXCEPTION         | params, user_id, response_size, feature_flag           | High-level business outcome                          |
# | Service             | INFO, WARNING, DEBUG             | step, amount, payment_id, validation_error, risk_level | Business decisions & important state changes         |
# | Repository          | DEBUG, INFO (success), ERROR     | payment_id, db_operation, rows_affected, query_time    | Use DEBUG for query tracing in dev                   |
# | External call       | DEBUG → INFO, WARNING, ERROR     | http_status, external_id, retry_count, timeout         | Good place for `logger.bind(external_service="stripe")` |
# | Background / worker | INFO, ERROR                      | task_name, correlation_id (passed manually)            | No automatic context                                 |

# This style gives clean console output during development while still carrying enough context for tracing.

# Let me know which part you'd like to deepen:

# - JSON file sink activation + rotation example
# - Sensitive field redaction (amount, tokens, emails…)
# - Adding traceparent / OpenTelemetry integration
# - Testing correlation id propagation
# - Difference between `.info(…, extra={…})` vs `.bind(…).info(…)`

# Happy logging!
