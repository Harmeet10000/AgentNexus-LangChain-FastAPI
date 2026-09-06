"""Canonical logging usage for this codebase (loguru, structured, redacted).

Run: ``uv run python -m app.examples.logger_usage_example`` (exercises the
happy path plus one handled failure; exits 0).

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
"""

from __future__ import annotations

import asyncio

from app.utils import logger, trace_layer

# --- REPOSITORY LAYER ---


@trace_layer("repository")
async def db_create_payment(user_id: int, amount: float, currency: str) -> dict:
    log = logger.bind(operation="db_create_payment", user_id=user_id, currency=currency)
    log.debug("Inserting payment record")

    if amount < 0:
        # Handled business failure: still error level (the operation failed),
        # still static message, still kwargs — and no traceback needed because
        # nothing raised here.
        log.bind(amount=amount, error_code="NEGATIVE_AMOUNT").error("Payment amount rejected")
        msg = "Amount cannot be negative"
        raise ValueError(msg)

    if amount > 10000:
        # Catastrophic path: raise and let @trace_layer record it; the
        # service layer below captures the traceback with .exception().
        msg = "Database connection lost during transaction"
        raise ConnectionError(msg)

    payment_record = {"id": "txn_998877", "status": "success", "amount": amount}
    log.bind(txn_id=payment_record["id"]).info("Payment record created")
    return payment_record


# --- SERVICE LAYER ---


@trace_layer("service")
async def process_payment(user_id: int, amount: float) -> dict:
    # One bound logger per request; every line below carries user_id.
    # Secrets bound here would be redacted by the logging patch — bind them
    # if they aid debugging, never interpolate them into the message.
    log = logger.bind(operation="process_payment", user_id=user_id, amount=amount)
    log.info("Payment flow started")

    try:
        result = await db_create_payment(user_id, amount, "USD")
    except ValueError as exc:
        # Expected failure, already logged at the repo layer: downgrade to a
        # terse warning, keep the error value in kwargs (not the message).
        exc.add_note(f"operation=process_payment, user_id={user_id}")
        log.bind(error=str(exc)).warning("Payment rejected by validation")
        return {"status": "failed", "reason": str(exc)}
    except Exception:
        # Unexpected failure: .exception() attaches the full traceback.
        # This is the ONLY way tracebacks reach the logs — error=str(exc)
        # alone would discard the stack.
        log.exception("Payment flow failed")
        raise

    log.bind(txn_id=result["id"]).info("Payment flow completed")
    return result


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


async def _demo() -> None:
    ok = await process_payment(user_id=7, amount=120.0)
    logger.bind(result=ok).info("Demo happy path finished")
    rejected = await process_payment(user_id=7, amount=-5.0)
    logger.bind(result=rejected).info("Demo handled failure finished")


if __name__ == "__main__":
    asyncio.run(_demo())
