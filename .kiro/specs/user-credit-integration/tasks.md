# Implementation Plan: User Credit Integration

## Overview

This implementation plan breaks down the user credit integration feature into discrete, incremental coding tasks. The system is built on FastAPI with async-first architecture, using PostgreSQL for persistence, Celery for background jobs, and Razorpay for payment processing. The implementation follows a modular monolith pattern with feature-driven organization, repository patterns for data access, and typed exceptions for error handling.

The credit integration feature enables administrators and the system to grant credit balances to users, which are automatically applied to subscription renewals and plan changes. Credit operates as a payment method (not a discount), ensuring GST compliance by calculating tax on the full invoice total before credit application. The system enforces ledger integrity through 8 correctness properties validated via property-based tests.

## Tasks

- [ ] 1. Set up credits feature structure and database models
  - [ ] 1.1 Create credits feature directory structure
    - Create `src/app/features/credits/` directory with `__init__.py`
    - Create subdirectories: `models/`, `dto/`, `repositories/`, `services/`, `routers/`
    - Create `exceptions.py` for credit-specific exceptions
    - _Requirements: 49.1, 50.1, 51.1, 52.1, 53.1_

  - [ ] 1.2 Define Pydantic data models
    - Implement `models/credit.py` with `UserCredit`, `CreditType` enum, `CreditStatus` enum
    - Implement `models/consumption.py` with `CreditConsumption` model
    - **NOTE: Credit amounts stored inpaisa** to match `Payment.amount` convention
    - **NOTE: Use `BigInteger` for amount fields (paisa)**
    - **NOTE: Use `JSONB` for metadata_ field**
    - _Requirements: 49.1, 50.1, 53.1_

  - [ ] 1.3 Create SQLAlchemy/Alembic database schema
    - Define SQLAlchemy table models for user_credits, credit_consumptions
    - Add foreign key constraints (credit_consumptions.credit_id → user_credits.id, invoice_id → invoices.id)
    - Add indexes: user_id, status, valid_until, created_at for both tables
    - Add CHECK constraints:
      - credit_amount > 0 (minimum 1 paisa)
      - remaining_balance <= credit_amount
      - consumed_amount > 0
    - Create Alembic migration script for credit tables
    - _Requirements: 49.1, 50.1, 53.1, 53.4_

- [ ] 2. Implement DTO layer and validation
  - [ ] 2.1 Create request/response DTOs
    - Implement `dto/credit_dto.py`: `CreditGrantDTO`, `CreditGrantResponse`, `CreditBalanceResponse`, `CreditHistoryResponse`
    - Implement `dto/consumption_dto.py`: `CreditConsumptionResult`, `ConsumedCredit`
    - **NEW: Validate credit_amount >= 1 paisa**
    - **NEW: Validate valid_from <= valid_until (if both set)**
    - **NEW: Validate ADMIN_GRANT has admin_user_id in metadata**
    - All DTOs use `ConfigDict(extra="forbid", frozen=True)` for request models
    - _Requirements: 49.2, 49.3, 49.4, 52.1, 52.2_

  - [ ] 2.2 Create response envelope for API
    - Implement `dto/response.py`: `APIResponse[CreditGrantResponse]` etc.
    - Use `APIResponse[T]` from `app/shared/response_type.py`
    - _Requirements: All API endpoints_

- [ ] 3. Implement repository layer with dual-method pattern
  - [ ] 3.1 Implement UserCredit repository
    - Create `repositories/credit_repository.py` with `CreditRepository` class
    - Implement `create()`, `find_by_id()`, `find_by_user()`, `find_available_for_consumption()`, `update_balance()`, `expire_credits_past_date()` methods
    - **NEW: Implement `find_available_for_consumption()` with ordering by valid_until ASC, created_at ASC**
    - **NEW: Filter by status=ACTIVE and valid_until > now OR valid_until IS NULL**
    - Use dual-method pattern: `_result()` variant returning `AppResult[T]`, public wrapper raising exceptions
    - _Requirements: 49.1, 50.1, 50.2, 51.1, 51.3, 53.1_

  - [ ] 3.2 Implement CreditConsumption repository
    - Create `repositories/consumption_repository.py` with `ConsumptionRepository` class
    - Implement `create()`, `find_by_credit_id()`, `find_by_user()`, `find_by_invoice_id()`, `get_total_consumed()` methods
    - **NEW: Return consumed_amount in paisa**
    - Implement ledger integrity queries
    - _Requirements: 50.5, 53.1_

  - [ ] 3.3 Integrate with AuditLog repository
    - Import and use `AuditLogRepository` from `app/features/audit/repository.py`
    - Implement audit log creation for all credit operations
    - _Requirements: 49.5, 50.7, 51.2_

- [ ] 4. Implement core service layer - Credit Service
  - [ ] 4.1 Implement Credit Service (part 1: grant and balance)
    - Create `services/credit_service.py` with `CreditService` class
    - Implement `grant_credit()` with validation and audit logging
    - Implement `get_credit_balance()` calculating sum of active, non-expired credits
    - **NEW: Convert paisa to rupees ONLY in balance calculation (divide by 100)**
    - **NEW: Filter by status=ACTIVE and valid_until > now OR valid_until IS NULL**
    - **NEW: Use `isinstance(result, Failure)` + `raise app_error_to_exception(error)` pattern**
    - _Requirements: 49.1-49.5, 52.1, 53.8_

  - [ ] 4.2 Implement Credit Service (part 2: consumption)
    - Implement `consume_credits()` for applying credits to invoices
    - **NEW: Call `find_available_for_consumption()` to get credits in correct order**
    - **NEW: Iterate through credits, consuming until invoice_gross_total (in rupees) is covered**
    - **NEW: Convert paisa amounts to rupees ONLY in calculation: consumed_amount / 100**
    - **NEW: Create CreditConsumption record with consumed_amount in paisa**
    - **NEW: Update UserCredit.remaining_balance and optionally status**
    - **NEW: Return CreditConsumptionResult with credit_applied and cash_due in rupees**
    - _Requirements: 50.1-50.7, 53.1, 53.5_

  - [ ] 4.3 Implement Credit Service (part 3: history and expiration)
    - Implement `get_credit_history()` returning credits and consumptions for a user
    - Implement `expire_credits()` for daily background job
    - **NEW: Find ACTIVE credits with valid_until < now**
    - **NEW: Update status to EXPIRED and record consumed_at**
    - **NEW: Create audit log entry with action CREDIT_EXPIRED**
    - _Requirements: 51.1-51.3, 52.2, 53.5_

  - [ ] 4.4 Implement Credit Service (part 4: proration integration)
    - Implement `grant_credit_on_downgrade()` for plan change credits
    - **NEW: Set valid_from to downgrade timestamp**
    - **NEW: Set valid_until to end of current billing cycle + 12 months**
    - **NEW: Use credit_type=PLAN_CREDIT**
    - _Requirements: 54.1-54.5_

- [ ] 5. Implement Razorpay API client integration
  - [ ] 5.1 Create Credit-specific Razorpay integration
    - Review existing `features/payments/clients/razorpay_client.py`
    - **NEW: Add method for creating payment links (for partial cash payments)**
    - **NEW: Ensure credit consumption and Razorpay charge are in same transaction**
    - Implement transaction rollback on payment failure
    - _Requirements: 50.3, 50.4, 53.3_

- [ ] 6. Implement FastAPI routers
  - [ ] 6.1 Create Admin/Portal credit router
    - Create `routers/credit_admin_router.py` with `APIRouter`
    - Implement POST `/credits/grant` endpoint for admin credit grants
    - **NEW: Require admin user authorization**
    - **NEW: Validate metadata includes admin_user_id for ADMIN_GRANT type**
    - Implement GET `/credits/balance` endpoint for credit balance查询
    - **NEW: Allow admin or user (self only)**
    - Implement GET `/credits/history` endpoint for credit history
    - **NEW: Implement pagination with limit/offset**
    - Use `APIResponse[T]` response envelope
    - _Requirements: 49.1-49.5, 52.1-52.2_

  - [ ] 6.2 Create System-Internal credit router
    - Create `routers/credit_internal_router.py` with `APIRouter`
    - Implement POST `/credits/apply-to-invoice` endpoint for InvoiceService
    - **NEW: This endpoint is called internally by InvoiceService**
    - **NEW: Accept invoice_gross_total in rupees**
    - **NEW: Return CreditConsumptionResult**
    - Implement system-internal endpoints (no user-facing)
    - _Requirements: 50.1-50.7, 55.1-55.6_

  - [ ] 6.3 Wire credit routers into main application
    - Register credit routers in main FastAPI app with `/credits` prefix
    - Add appropriate tags for OpenAPI documentation
    - _Requirements: All routing requirements_

- [ ] 7. Implement background jobs (Celery)
  - [ ] 7.1 Create daily credit expiration job
    - Create `tasks/expire_credits_task.py` with Celery task
    - **NEW: Run daily at configured time (configurable via environment variable)**
    - **NEW: Call CreditService.expire_credits()**
    - **NEW: Log expiration statistics**
    - Schedule job via Celery beat
    - _Requirements: 51.1-51.3_

  - [ ] 7.2 Create credit reconciliation job
    - Create `tasks/reconcile_credits_task.py` with Celery task
    - **NEW: Run weekly (configurable)**
    - **NEW: Verify ledger integrity: credit_amount == remaining_balance + SUM(consumed_amount)**
    - **NEW: Log discrepancies and alert ops team**
    - _Requirements: 53.1_

- [ ] 8. Implement lifespan and dependency injection
  - [ ] 8.1 Configure lifespan for credit repositories
    - Create `dependencies.py` with repository and service providers
    - Use `Depends(...)` pattern for dependency injection
    - Create `Annotated` type aliases for common dependencies
    - _Requirements: All service dependencies_

  - [ ] 8.2 Wire credit services into lifespan
    - Add repository initialization to `lifecycle/lifespan.py`
    - Store repositories in `app.state` if needed
    - _Requirements: All service dependencies_

- [ ] 9. Implement comprehensive error handling
  - [ ] 9.1 Create credit-specific exceptions
    - Define exceptions in `exceptions.py`:
      - `CreditAmountMustBePositiveException`
      - `CreditInvalidDateRangeException`
      - `CreditMetadataMissingException`
      - `CreditNotFoundException`
      - `CreditInsufficientBalanceException`
      - `CreditExpiredException`
      - `CreditAlreadyConsumedException`
      - `CreditTransactionRollbackException`
    - All extend appropriate base exceptions from `utils/exceptions.py`
    - Include error codes and structured detail fields
    - _Requirements: Error Handling section in design_

  - [ ] 9.2 Add exception handling in credit routers
    - Catch validation errors → return HTTP 422
    - Catch not found errors → return HTTP 404
    - Catch business logic errors → return appropriate status
    - Log credit processing failures with structured logging
    - _Requirements: All error handling requirements_

- [ ] 10. Implement audit logging integration
  - [ ] 10.1 Add audit logging to all credit operations
    - Log credit grants with action CREDIT_GRANTED
    - Log credit consumptions with action CREDIT_CONSUMED
    - Log credit expirations with action CREDIT_EXPIRED
    - Record user_id, ip_address, user_agent when available
    - Record changes in JSON format
    - _Requirements: 49.5, 50.7, 51.2_

- [ ] 11. Implement comprehensive property-based tests
  - [ ] 11.1 Write property test for Property 1: Ledger Integrity
    - **Property 1: Ledger Integrity (Amount Conservation)**
    - **Validates: Requirements 53.1**
    - Generate random credits and consumption sequences
    - Verify `credit_amount == remaining_balance + SUM(consumed_amount)`
    - Use `hypothesis` library with minimum 100 iterations
    - _Requirements: 53.1_

  - [ ] 11.2 Write property test for Property 2: Transactional Atomicity
    - **Property 2: Transactional Atomicity**
    - **Validates: Requirements 53.2**
    - Simulate transaction failures during credit consumption
    - Verify no partial records left behind
    - Test that CreditConsumption and invoice/payment are created atomically
    - _Requirements: 53.2_

  - [ ] 11.3 Write property test for Property 3: Rollback on Payment Failure
    - **Property 3: Rollback on Payment Failure**
    - **Validates: Requirements 53.3**
    - Simulate Razorpay charge failure
    - Verify ENTIRE transaction rolls back (including credit deduction)
    - Test credit balance unchanged after failure
    - _Requirements: 53.3_

  - [ ] 11.4 Write property test for Property 4: Status Transition Integrity
    - **Property 4: Status Transition Integrity**
    - **Validates: Requirements 53.4, 53.5**
    - Generate random credits with various states
    - Verify CONSUMED only when remaining_balance == 0
    - Verify EXPIRED only when now > valid_until and was ACTIVE
    - _Requirements: 53.4, 53.5_

  - [ ] 11.5 Write property test for Property 5: Consumption Order Correctness
    - **Property 5: Consumption Order Correctness**
    - **Validates: Requirement 50.2**
    - Generate credits with varying expiry dates and creation times
    - Verify consumption order: soonest valid_until first, then oldest created_at
    - Credits with no expiry consumed last
    - _Requirements: 50.2_

  - [ ] 11.6 Write property test for Property 6: Expiration Exclusion
    - **Property 6: Expiration Exclusion**
    - **Validates: Requirement 51.3**
    - Generate expired and active credits
    - Verify expired credits excluded from consumption ordering
    - Verify expired credits excluded from balance calculations
    - _Requirements: 51.3_

  - [ ] 11.7 Write property test for Property 7: GST Compliance
    - **Property 7: GST Compliance (Full-Price Tax)**
    - **Validates: Requirement 55.4**
    - Generate invoices with and without credits
    - Verify tax calculations identical (GST calculated on full price)
    - Test subtotal, tax_amount, total unaffected by credit application
    - _Requirements: 55.4_

  - [ ] 11.8 Write property test for Property 8: Balance Calculation
    - **Property 8: Balance Calculation Correctness**
    - **Validates: Requirement 52.1**
    - Generate random credits with varying states
    - Verify balance equals sum of remaining_balance for ACTIVE, non-expired credits
    - _Requirements: 52.1_

- [ ] 12. Implement unit tests
  - [ ] 12.1 Write unit tests for Credit Service grant operations
    - Test credit grant with valid data
    - Test credit grant with ADMIN_GRANT type (validate admin_user_id)
    - Test credit grant with invalid date range
    - Test credit grant with zero/negative amount
    - Test audit log creation
    - _Requirements: 49.1-49.5_

  - [ ] 12.2 Write unit tests for Credit Service consumption operations
    - Test full coverage (credit covers entire invoice)
    - Test partial coverage (credit covers part of invoice)
    - Test no coverage (insufficient credit)
    - Test consumption order verification
    - Test expired credit rejection
    - Test transaction rollback on payment failure
    - _Requirements: 50.1-50.7_

  - [ ] 12.3 Write unit tests for Credit Service balance operations
    - Test balance calculation with active credits
    - Test balance calculation with expired credits
    - Test balance calculation with mixed status credits
    - Test edge case: no credits
    - _Requirements: 52.1_

  - [ ] 12.4 Write unit tests for Credit Service history operations
    - Test history retrieval with pagination
    - Test history includes credits and consumptions
    - Test history sorted by created_at descending
    - _Requirements: 52.2_

  - [ ] 12.5 Write unit tests for repositories
    - Test CRUD operations for CreditRepository
    - Test CRUD operations for ConsumptionRepository
    - Test find_available_for_consumption ordering
    - Test get_total_consumed calculation
    - _Requirements: All repository operations_

- [ ] 13. Implement integration tests
  - [ ] 13.1 Write integration test for credit grant flow
    - Test end-to-end: admin grants credit → credit appears in balance
    - Test with various credit types (PLAN_CREDIT, PROMOTIONAL, ADMIN_GRANT)
    - Test validation and error handling
    - _Requirements: 49.1-49.5_

  - [ ] 13.2 Write integration test for credit consumption flow
    - Test end-to-end: invoice generation → credit applied → payment processed
    - Test full coverage (no cash charge)
    - Test partial coverage (cash charge remaining)
    - _Requirements: 50.1-50.7_

  - [ ] 13.3 Write integration test for daily expiration job
    - Test job expires past-due credits
    - Test expired credits excluded from consumption
    - Test audit log creation for expired credits
    - _Requirements: 51.1-51.3_

  - [ ] 13.4 Write integration test for proration credit integration
    - Test end-to-end: plan downgrade → credit granted → credit applied at renewal
    - Test valid_from set to downgrade timestamp
    - Test valid_until set to end of cycle + 12 months
    - _Requirements: 54.1-54.5_

- [ ] 14. Checkpoint - Core functionality complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Add configuration and settings
  - [ ] 15.1 Add credit settings to configuration
    - Add daily expiration job schedule (configurable via environment variable)
    - Add weekly reconciliation job schedule (configurable)
    - _Requirements: Background jobs_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- All tasks marked with `**NEW:**` contain critical implementation details
- Property-based tests use `hypothesis` library with minimum 100 iterations each
- Credit amounts stored in paisa (BigInteger), converted to rupees ONLY in service layer
- GST compliance: tax calculated on full invoice total before credit application
- Consumption order: soonest-expiring-first, then oldest-created (no expiry last)
- Repository pattern uses dual-method: `_result()` returning `AppResult[T]` + public wrapper
- All credit operations create audit log entries for compliance
- Transactional atomicity enforced for credit consumption + invoice/payment creation
- Checkpoint tasks ensure incremental validation

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1", "2.2", "3.1"] },
    { "id": 2, "tasks": ["3.2", "3.3", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "4.4", "5.1"] },
    { "id": 4, "tasks": ["6.1", "6.2", "7.1", "8.1"] },
    { "id": 5, "tasks": ["8.2", "9.1", "10.1", "11.1"] },
    { "id": 6, "tasks": ["11.2", "11.3", "11.4", "11.5"] },
    { "id": 7, "tasks": ["11.6", "11.7", "11.8", "12.1"] },
    { "id": 8, "tasks": ["12.2", "12.3", "12.4", "12.5"] },
    { "id": 9, "tasks": ["13.1", "13.2", "13.3", "13.4"] },
    { "id": 10, "tasks": ["14"] },
    { "id": 11, "tasks": ["15.1", "7.2"] }
  ]
}
```
