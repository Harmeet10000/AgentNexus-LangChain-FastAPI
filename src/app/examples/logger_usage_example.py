"""Canonical logging usage for this codebase (loguru, structured, redacted).

Run: ``uv run python -m app.examples.logger_usage_example`` (exercises the
happy path plus two handled failures; exits 0).

The standard, in five rules:

1. Import once per module: ``from app.utils import logger``. Never
   ``from loguru import logger`` (misses the redaction patch) and never
   stdlib ``logging`` (misses every sink). Never pass a logger as an
   argument — loguru is a process-global registry; per-request scoping
   comes from ``bind``/``contextualize``, not plumbing.
2. Static messages, dynamic kwargs: ``log.bind(user_id=u).info("Payment
   started")``. No f-string values in the message — interpolated text
   breaks the OTLP body/attributes split and can smuggle PII into the
   message field.
3. Tracebacks come from ``.exception(...)`` inside ``except`` blocks, never
   from ``error=str(exc)`` and never from ``exc_info=True`` (a dead kwarg
   on loguru — it becomes ``extra``, no traceback attached).
4. Severity honesty: real failures are ``error``/``exception``; only
   degraded-but-continuing is ``warning``. A ``warning`` that nobody acts
   on is a silenced error.
5. ``bind`` returns a NEW logger: chain it (``logger.bind(...).info``) or
   assign it once per request (``log = logger.bind(...)``). A bare
   ``logger.bind(...)`` statement binds nothing.

Expected failures travel as ``Result``, never ``raise``: the repository
returns ``Failure`` with a typed error, the service unwraps with
``isinstance(result, Failure)``. Nothing here raises for an expected
failure — ``raise`` is reserved for transport boundaries and true bugs.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import ClassVar  # noqa: TC003 — resolved at runtime by Pydantic

from returns.result import Failure, Result, Success

from app.shared.result import ErrorKind, FeatureError, log_expected_failure
from app.utils import logger, trace_layer


class PaymentCode(StrEnum):
    NEGATIVE_AMOUNT = "NEGATIVE_AMOUNT"
    BACKEND_DOWN = "BACKEND_DOWN"


class PaymentValidationError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.VALIDATION
    code: ClassVar[PaymentCode] = PaymentCode.NEGATIVE_AMOUNT


class PaymentBackendError(FeatureError):
    kind: ClassVar[ErrorKind] = ErrorKind.INFRASTRUCTURE
    code: ClassVar[PaymentCode] = PaymentCode.BACKEND_DOWN
    retryable: ClassVar[bool] = True


type PaymentResult[T] = Result[T, PaymentValidationError | PaymentBackendError]


class _FakeDriverError(Exception):
    """Stand-in for a real driver failure (e.g. connection loss)."""


def _unstable_driver_insert() -> None:
    msg = "Database connection lost during transaction"
    raise _FakeDriverError(msg)


# --- REPOSITORY LAYER ---


@trace_layer("repository")
async def db_create_payment(
    user_id: int, amount: float, currency: str
) -> PaymentResult[dict[str, object]]:
    log = logger.bind(operation="db_create_payment", user_id=user_id, currency=currency)
    log.debug("Inserting payment record")

    if amount < 0:
        # Handled business failure: still error level (the operation failed),
        # still static message, still kwargs — and no traceback needed because
        # nothing raised here. The failure travels as data, not an exception.
        log.bind(amount=amount, error_code="NEGATIVE_AMOUNT").error("Payment amount rejected")
        return Failure(
            PaymentValidationError(
                message="Amount cannot be negative",
                details={"amount": amount, "error_code": "NEGATIVE_AMOUNT"},
            )
        )

    if amount > 10000:
        # Catastrophic path: the driver raised, so a traceback exists — capture
        # it with .exception(), attach notes, then translate to Failure. The
        # service layer below never sees an exception, only the typed error.
        try:
            _unstable_driver_insert()
        except _FakeDriverError as exc:
            exc.add_note(f"user_id={user_id}, amount={amount}, operation=db_create_payment")
            log.bind(amount=amount).exception("Payment insert failed")
            return Failure(
                PaymentBackendError(
                    message="Payment backend unavailable",
                    details={"notes": list(getattr(exc, "__notes__", []))},
                )
            )

    payment_record = {"id": "txn_998877", "status": "success", "amount": amount}
    log.bind(txn_id=payment_record["id"]).info("Payment record created")
    return Success(payment_record)


# --- SERVICE LAYER ---


@trace_layer("service")
async def process_payment(user_id: int, amount: float) -> dict[str, object]:
    # One bound logger per request; every line below carries user_id.
    # Secrets bound here would be redacted by the logging patch — bind them
    # if they aid debugging, never interpolate them into the message.
    log = logger.bind(operation="process_payment", user_id=user_id, amount=amount)
    log.info("Payment flow started")

    result = await db_create_payment(user_id, amount, "USD")
    if isinstance(result, Failure):
        # Expected failure, already logged at the repo layer: record it once
        # for observability, keep the error value in kwargs (not the message).
        error = result.failure()
        log_expected_failure(error, operation="process_payment")
        log.bind(error=error.message).warning("Payment rejected")
        return {"status": "failed", "reason": error.message}

    record = result.unwrap()
    log.bind(txn_id=record["id"]).info("Payment flow completed")
    return record


# --- ANTI-PATTERNS (do not copy) ---
#
# logger.info(f"Payment {txn_id} started")      # f-string value: breaks
#                                               # structured logging.
# logger.error("Failed", error=str(exc))        # no traceback; use
#                                               # .exception(...) instead.
# logger.error("...", exc_info=True)            # dead kwarg on loguru.
# logger.bind(user_id=u)                        # discarded: bind returns a
# logger.info("...")                            # NEW logger; use log = ...
# from loguru import logger                     # misses redaction patch.
# logger.warning("DB is down, continuing")      # severity lie: a hard
#                                               # failure is error/exception.
# raise ValueError("bad amount")                # expected failures travel
#                                               # as Failure, never raise.


async def _demo() -> None:
    ok = await process_payment(user_id=7, amount=120.0)
    logger.bind(result=ok).info("Demo happy path finished")
    rejected = await process_payment(user_id=7, amount=-5.0)
    logger.bind(result=rejected).info("Demo handled failure finished")
    down = await process_payment(user_id=7, amount=20000.0)
    logger.bind(result=down).info("Demo backend failure finished")


if __name__ == "__main__":
    asyncio.run(_demo())
