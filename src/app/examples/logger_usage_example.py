from app.utils import logger, trace_layer

# --- REPOSITORY LAYER ---


@trace_layer("repository")
async def db_create_payment(user_id: int, amount: float, currency: str) -> dict:
    # 1. DEBUG: Good for tracing exact variable states in dev
    logger.bind(user_id=user_id, amount=amount).debug("Attempting to insert payment record into DB")

    # Simulating database logic
    if amount < 0:
        # 2. ERROR: Handled business logic failure
        logger.bind(amount=amount, error_code="NEGATIVE_AMOUNT").error(
            "Invalid payment amount requested"
        )
        raise ValueError("Amount cannot be negative")

    if amount > 10000:
        # Simulating a catastrophic DB crash (e.g., timeout or connection drop)
        raise ConnectionError("Database connection lost during transaction")

    payment_record = {"id": "txn_998877", "status": "success", "amount": amount}

    # 3. INFO: Standard success milestone
    logger.bind(txn_id=payment_record["id"]).info("Payment record successfully created")

    return payment_record


# --- SERVICE LAYER ---ccccc


@trace_layer("service")
async def process_payment(user_id: int, amount: float) -> dict:
    # 4. INFO with extra data: Tracking the start of a business process
    logger.bind(user_id=user_id, amount=amount).info("Initiating payment processing flow")

    try:
        # Calling the repository layer
        result = await db_create_payment(user_id, amount, "USD")

        # Passing an entire object/dict as extra data
        logger.bind(payment_data=result).info("Payment flow completed successfully")
        return result

    except ValueError as ve:
        # We already logged the error in the repo, so we just return or re-raise safely
        logger.warning(f"Payment rejected due to validation: {ve}")
        return {"status": "failed", "reason": str(ve)}

    except Exception as e:
        # 5. EXCEPTION: Automatically captures the full stack trace and attaches it to the log
        # Passing extra context helps debug exactly what caused the crash
        logger.bind(user_id=user_id, amount=amount).exception(
            "Catastrophic failure in payment service"
        )
        raise e
