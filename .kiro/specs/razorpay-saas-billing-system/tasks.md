# Implementation Plan: Razorpay SaaS Billing System

## Overview

This implementation plan breaks down the Razorpay SaaS billing system into discrete, incremental coding tasks. The system is built on FastAPI with async-first architecture, using PostgreSQL for persistence, Celery for background jobs, and Razorpay for payment processing. The implementation follows a modular monolith pattern with feature-driven organization, repository patterns for data access, and typed exceptions for error handling.

The billing engine manages the complete subscription lifecycle including plan management, subscription state transitions, payment processing with dunning, GST-compliant invoicing, proration handling, and audit logging for compliance.

## Tasks

- [ ] 1. Set up billing feature structure and database models
  - [ ] 1.1 Create billing feature directory structure
    - Create `src/app/features/billing/` directory with `__init__.py`
    - Create subdirectories: `models/`, `dto/`, `repositories/`, `services/`, `routers/`
    - Create `exceptions.py` for billing-specific exceptions
    - _Requirements: 1.1, 1.2, 2.1_

  - [ ] 1.2 Define Pydantic data models
    - Implement `models/plan.py` with `Plan`, `BillingInterval` enum
    - Implement `models/subscription.py` with `Subscription`, `SubscriptionStatus` enum
    - Implement `models/payment.py` with `Payment`, `PaymentStatus`, `PaymentMethod` enums
    - Implement `models/invoice.py` with `Invoice`, `InvoiceStatus` enum, GST fields
    - Implement `models/webhook.py` with `WebhookEvent`, `WebhookEventType`, `WebhookEventStatus` enums
    - Implement `models/audit.py` with `AuditLog`, `AuditAction` enum
    - _Requirements: 1.1, 1.2, 2.1, 8.1, 12.1, 14.1, 16.1_

  - [ ] 1.3 Create SQLAlchemy/Alembic database schema
    - Define SQLAlchemy table models for plans, subscriptions, payments, invoices, webhook_events, audit_logs
    - Add foreign key constraints (subscription.plan_id → plans.id, payment.subscription_id → subscriptions.id)
    - Add unique constraints (invoice.invoice_number, webhook_event.razorpay_event_id)
    - Add CHECK constraints (payment amount validation, tax calculation validation)
    - Create Alembic migration script for billing tables
    - _Requirements: 1.1, 1.2, 8.6, 12.9, 14.7, 26.6_

- [ ] 2. Implement DTO layer and validation
  - [ ] 2.1 Create request/response DTOs
    - Implement `dto/plan_dto.py`: `PlanCreateDTO`, `PlanUpdateDTO`, `PlanResponse`
    - Implement `dto/subscription_dto.py`: `SubscriptionCreateDTO`, `SubscriptionResponse`, `PlanChangeDTO`, `SubscriptionCancelDTO`
    - Implement `dto/payment_dto.py`: `PaymentRecordDTO`, `PaymentResponse`, `RefundRequestDTO`, `RefundResponse`
    - Implement `dto/invoice_dto.py`: `InvoiceResponse`, `CreditNoteResponse`
    - Implement `dto/webhook_dto.py`: `WebhookEventDTO`, `WebhookPayload`
    - Implement `dto/proration_dto.py`: `ProrationCalculation`, `ProrationDirection` enum
    - Implement `dto/dunning_dto.py`: `DunningConfigDTO`, `RetryAttemptResponse`
    - All DTOs use `ConfigDict(extra="forbid", frozen=True)` for request models
    - _Requirements: 1.1, 1.5, 2.1, 5.1, 7.1, 8.1, 10.1, 12.1, 19.1_

- [ ] 3. Implement repository layer with dual-method pattern
  - [ ] 3.1 Implement Plan repository
    - Create `repositories/plan_repository.py` with `PlanRepository` class
    - Implement `create()`, `update()`, `find_by_id()`, `find_by_name()`, `list_active()`, `archive()` methods
    - Use dual-method pattern: `_result()` variant returning `AppResult[T]`, public wrapper raising exceptions
    - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ] 3.2 Implement Subscription repository
    - Create `repositories/subscription_repository.py` with `SubscriptionRepository` class
    - Implement `create()`, `find_by_id()`, `find_by_user_and_plan()`, `update_status()`, `increment_retry_count()`, `reset_retry_count()` methods
    - **NEW: Implement `update_with_lock()` with optimistic locking using version field**
    - **NEW: Check version matches before update, increment version on success**
    - **NEW: Return ConflictAppError on version mismatch**
    - Implement state transition validation in `update_status()`
    - _Requirements: 2.1, 2.4, 2.6, 4.1-4.8, 6.1, 7.1, 29.1-29.7_

  - [ ] 3.3 Implement Payment repository
    - Create `repositories/payment_repository.py` with `PaymentRepository` class
    - Implement `create()`, `find_by_id()`, `find_by_razorpay_id()`, `find_by_subscription()`, `update_refund_amount()`, `update_status()` methods
    - Implement idempotency check using `razorpay_payment_id`
    - _Requirements: 8.1-8.8, 10.1-10.5_

  - [ ] 3.4 Implement Invoice repository
    - Create `repositories/invoice_repository.py` with `InvoiceRepository` class
    - Implement `create()`, `find_by_id()`, `find_by_payment_id()`, `find_by_user()`, `find_by_subscription()`, `generate_invoice_number()` methods
    - Implement sequential invoice numbering with transaction isolation
    - _Requirements: 12.1-12.10, 13.1-13.5, 27.1-27.7_

  - [ ] 3.5 Implement WebhookEvent repository
    - Create `repositories/webhook_repository.py` with `WebhookEventRepository` class
    - Implement `create()`, `find_by_razorpay_event_id()`, `update_status()`, `find_failed_events()` methods
    - Ensure unique constraint enforcement on `razorpay_event_id`
    - _Requirements: 14.1-14.8, 22.1-22.6_

  - [ ] 3.6 Implement AuditLog repository
    - Create `repositories/audit_repository.py` with `AuditLogRepository` class
    - Implement `create()`, `find_by_entity()`, `find_by_action()`, `find_by_date_range()` methods
    - Ensure immutability (no update/delete methods)
    - _Requirements: 16.1-16.8_

- [ ] 4. Implement Razorpay API client wrapper
  - [ ] 4.1 Create Razorpay client wrapper
    - Create `clients/razorpay_client.py` with `RazorpayClient` class
    - Implement async methods: `create_customer()`, `create_subscription()`, `cancel_subscription()`, `create_payment()`, `create_refund()`, `submit_dispute_evidence()`
    - Implement error handling with circuit breaker pattern (requirement 25.6)
    - **NEW: Implement Tenacity retry decorators for all API methods**
    - **NEW: Use @retry with stop_after_attempt(3), wait_exponential(multiplier=1, min=1, max=10)**
    - **NEW: Retry only on ExternalServiceException with retryable=True**
    - **NEW: Map HTTP 503, 504, 429 to retryable exceptions**
    - **NEW: Map HTTP 401, 403 to non-retryable exceptions**
    - Use `SecretStr` for API keys and secrets
    - _Requirements: 2.2, 2.3, 7.4, 10.3, 11.3, 25.1-25.6, 34.1-34.7_

  - [ ]* 4.2 Write unit tests for Razorpay client
    - Test retry logic for transient errors (503, 504, 429)
    - Test no retry for permanent errors (401, 403)
    - Test exponential backoff timing
    - Test circuit breaker activation after consecutive failures
    - Mock Razorpay API responses for success and error scenarios
    - _Requirements: 25.1-25.6, 34.1-34.7_

- [ ] 5. Implement core service layer - Plan Service
  - [ ] 5.1 Implement Plan Service
    - Create `services/plan_service.py` with `PlanService` class
    - Implement `create_plan()` with validation (amount ≥ 100 paisa, interval_count > 0)
    - Implement `update_plan()` with plan versioning logic
    - Implement `get_plan()`, `list_plans()`, `archive_plan()` methods
    - Use `isinstance(result, Failure)` + `raise app_error_to_exception(error)` pattern for Result unwrapping
    - _Requirements: 1.1-1.8, 24.1-24.5_

  - [ ]* 5.2 Write unit tests for Plan Service
    - Test plan creation with valid and invalid amounts
    - Test plan versioning when updating
    - Test unique name constraint validation
    - _Requirements: 1.1-1.8_

- [ ] 6. Implement core service layer - Subscription Service
  - [ ] 6.1 Implement Subscription Service (part 1: creation and retrieval)
    - Create `services/subscription_service.py` with `SubscriptionService` class
    - Implement `create_subscription()`: create DB record, create Razorpay customer, create Razorpay subscription, return payment URL
    - Implement `get_subscription()` for retrieval
    - Implement duplicate subscription check (user + plan + ACTIVE status)
    - Create audit log entries for subscription creation
    - _Requirements: 2.1-2.8_

  - [ ] 6.2 Implement Subscription Service (part 2: webhook handlers)
    - Implement `handle_authenticated()` for subscription.authenticated webhook
    - Implement `handle_activated()` for subscription.activated webhook (update status, set periods, record payment)
    - Implement `handle_charged()` for subscription.charged webhook (renewals)
    - Create audit log entries for status changes
    - _Requirements: 3.1-3.8, 17.1-17.7_

  - [ ] 6.3 Implement Subscription Service (part 3: state transitions and lifecycle)
    - Implement `pause_subscription()` with pause_start/pause_end timestamps
    - Implement `resume_subscription()` with next billing date recalculation
    - Implement `cancel_subscription()` with immediate and end-of-period options
    - Implement state transition validation logic
    - Create audit log entries for pause/resume/cancel operations
    - _Requirements: 4.1-4.8, 6.1-6.7, 7.1-7.7, 23.1-23.5_

  - [ ]* 6.4 Write property test for subscription state machine
    - **Property 3: Subscription State Machine Validity**
    - **Validates: Requirements 4.1-4.8**
    - Test that all state transitions follow valid state machine rules
    - Test that invalid transitions raise ValidationException
    - _Requirements: 4.1-4.8_

- [ ] 7. Implement Proration Service
  - [ ] 7.1 Implement Proration Service
    - Create `services/proration_service.py` with `ProrationService` class
    - **NEW: Implement `calculate_proration_fraction()` using integer microsecond arithmetic**
    - **NEW: Convert timedeltas to int microseconds before Decimal conversion**
    - **NEW: Use Decimal arithmetic for exact fractional calculations**
    - **NEW: Apply Banker's rounding (ROUND_HALF_EVEN) for final currency amounts**
    - Implement `calculate_plan_change_proration()` with mathematical formula from design
    - Implement `calculate_cancellation_proration()` for refund calculation
    - Implement `preview_proration()` without state changes
    - Validate billing interval compatibility and effective date bounds
    - _Requirements: 5.1-5.8, 19.1-19.5, 28.1-28.6, 33.1-33.7_

  - [ ]* 7.2 Write property test for proration calculation
    - **Property 4: Proration Calculation Correctness**
    - **Validates: Requirements 5.1-5.8**
    - Test proration mathematical properties with random inputs
    - Test boundary conditions (period start, middle, end)
    - Test upgrade vs downgrade direction
    - _Requirements: 5.1-5.8_

- [ ] 8. Implement Subscription Service plan change integration
  - [ ] 8.1 Implement plan change workflow
    - Implement `change_plan()` in SubscriptionService
    - Integrate proration calculation
    - Handle upgrade: generate proration invoice, charge immediately via Razorpay, update subscription
    - Handle downgrade: generate credit note, apply credit at next renewal
    - Wrap in database transaction for atomicity
    - _Requirements: 5.1-5.8, 26.1-26.6_

- [ ] 9. Implement Payment Service
  - [ ] 9.1 Implement Payment Service
    - Create `services/payment_service.py` with `PaymentService` class
    - Implement `record_payment()` with idempotency check using razorpay_payment_id
    - Implement `record_failed_payment()` with error details
    - Implement `initiate_refund()` with validation (status = CAPTURED, refund_amount ≤ available)
    - Implement `handle_refund_processed()` webhook handler
    - Implement `handle_chargeback()` webhook handler
    - Implement `submit_dispute_evidence()` to Razorpay
    - Create audit log entries for payment events
    - _Requirements: 8.1-8.8, 10.1-10.8, 11.1-11.5_

  - [ ]* 9.2 Write property test for payment-invoice integrity
    - **Property 1: Financial Integrity - No Phantom Charges**
    - **Validates: Requirements 8.1-8.8, 12.1-12.10**
    - Test that every captured payment has valid subscription and invoice
    - Test that invoice total matches payment amount (paisa conversion)
    - _Requirements: 8.1-8.8, 12.1-12.10_

- [ ] 10. Implement Invoice Service
  - [ ] 10.1 Implement Invoice Service
    - Create `services/invoice_service.py` with `InvoiceService` class
    - Implement `generate_invoice()` with sequential invoice numbering
    - **NEW: Snapshot tax_rate from plan at invoice creation time**
    - **NEW: Store snapshotted tax_rate in invoice model**
    - Implement GST tax calculation: `tax_amount = subtotal * tax_rate`, `total = subtotal + tax_amount`
    - Implement intra-state tax split: `cgst = sgst = tax_amount / 2`, `igst = 0`
    - Implement inter-state tax: `igst = tax_amount`, `cgst = sgst = 0`
    - Implement GSTIN format validation (15 alphanumeric characters)
    - Implement `generate_proration_invoice()` for mid-cycle plan changes
    - Implement `generate_credit_note()` for refunds and downgrades
    - Implement `get_invoice()`, `list_invoices()` with user authorization
    - Create audit log entries for invoice generation
    - _Requirements: 12.1-12.10, 13.1-13.5, 28.1-28.6, 32.1-32.7_

  - [ ]* 10.2 Write property test for GST tax calculation
    - **Property 5: GST Tax Calculation Compliance**
    - **Validates: Requirements 12.1-12.10**
    - Test tax_amount = subtotal * tax_rate
    - Test total = subtotal + tax_amount
    - Test intra-state: cgst + sgst = tax_amount, igst = 0
    - Test inter-state: igst = tax_amount, cgst = sgst = 0
    - _Requirements: 12.1-12.10_

  - [ ] 10.3 Implement invoice PDF generation
    - Implement PDF generation with GST-compliant template
    - Include invoice_number, subtotal, tax breakdown, total, seller_gstin, buyer_gstin, line items
    - Upload PDF to cloud storage (S3 or equivalent)
    - Generate presigned URL valid for 7 days
    - Store permanent storage path in `pdf_url` field
    - _Requirements: 27.1-27.7_

  - [ ] 10.4 Implement invoice email delivery
    - Implement `send_invoice_email()` with PDF attachment
    - Integrate with email service (SMTP or cloud email service)
    - _Requirements: 18.4_

- [ ] 11. Implement Webhook Service
  - [ ] 11.1 Implement Webhook Service
    - Create `services/webhook_service.py` with `WebhookService` class
    - Implement `verify_signature()` with HMAC SHA256 using constant-time comparison
    - Implement `process_event()` with idempotency check
    - Implement `is_duplicate_event()` checking razorpay_event_id
    - Implement `mark_event_processed()` creating WebhookEvent record
    - **NEW: Implement `replay_failed_event()` with state validation**
    - **NEW: Add `is_replay` parameter to all event handlers**
    - **NEW: Validate subscription state before processing replayed events**
    - **NEW: Skip replayed payment.captured if subscription is CANCELLED**
    - **NEW: Skip replayed subscription.activated if already ACTIVE**
    - **NEW: Mark skipped replays as SKIPPED status (not PROCESSED or FAILED)**
    - Implement event routing to appropriate service handlers
    - _Requirements: 3.1-3.8, 14.1-14.8, 15.1-15.7, 22.1-22.6, 31.1-31.6_

  - [ ]* 11.2 Write property test for webhook idempotency
    - **Property 2: Webhook Idempotency**
    - **Validates: Requirements 14.1-14.8**
    - Test processing same event 1, 2, 5, 10 times produces identical state
    - Test exactly one WebhookEvent record exists after multiple processing
    - _Requirements: 14.1-14.8_

- [ ] 12. Implement Dunning Service
  - [ ] 12.1 Implement Dunning Service
    - Create `services/dunning_service.py` with `DunningService` class
    - Implement `initiate_dunning()` checking retry_count < max_retries
    - **NEW: Implement `calculate_retry_delay_with_jitter()` helper function**
    - **NEW: Add random jitter (0-3600 seconds) to base retry delays**
    - **NEW: Use secrets.randbelow() for cryptographically secure random**
    - **NEW: Compute: base_delay * (2 ** attempt) + jitter_seconds**
    - Implement `execute_retry()` with exponential backoff (1, 3, 7, 14 days base)
    - Implement `halt_subscription()` after max retries exhausted
    - Implement `configure_dunning_strategy()` for retry intervals and max attempts
    - Implement `get_retry_schedule()` returning retry history
    - Schedule retry tasks via Celery with appropriate delays
    - Log scheduled retry times with jitter information for debugging
    - _Requirements: 9.1-9.8, 21.1-21.5, 30.1-30.5_

- [ ] 13. Implement FastAPI routers
  - [ ] 13.1 Create Plan router
    - Create `routers/plan_router.py` with `APIRouter`
    - Implement POST `/plans` endpoint for plan creation
    - Implement PUT `/plans/{plan_id}` endpoint for plan updates
    - Implement GET `/plans/{plan_id}` endpoint for plan retrieval
    - Implement GET `/plans` endpoint for listing plans
    - Implement DELETE `/plans/{plan_id}` endpoint for archiving plans
    - Use `APIResponse[T]` response envelope
    - Use `Depends(...)` for service injection
    - _Requirements: 1.1-1.8_

  - [ ] 13.2 Create Subscription router
    - Create `routers/subscription_router.py` with `APIRouter`
    - Implement POST `/subscriptions` endpoint for subscription creation
    - Implement GET `/subscriptions/{subscription_id}` endpoint
    - Implement POST `/subscriptions/{subscription_id}/change-plan` endpoint
    - Implement POST `/subscriptions/{subscription_id}/pause` endpoint
    - Implement POST `/subscriptions/{subscription_id}/resume` endpoint
    - Implement POST `/subscriptions/{subscription_id}/cancel` endpoint
    - Use `APIResponse[T]` response envelope
    - _Requirements: 2.1-2.8, 5.1-5.8, 6.1-6.7, 7.1-7.7, 19.1-19.5_

  - [ ] 13.3 Create Payment router
    - Create `routers/payment_router.py` with `APIRouter`
    - Implement POST `/payments/{payment_id}/refund` endpoint for refunds
    - Implement GET `/payments/{payment_id}` endpoint
    - Implement POST `/disputes/{dispute_id}/evidence` endpoint for chargeback evidence
    - Admin-only endpoints with authorization checks
    - _Requirements: 10.1-10.8, 11.1-11.5_

  - [ ] 13.4 Create Invoice router
    - Create `routers/invoice_router.py` with `APIRouter`
    - Implement GET `/invoices/{invoice_id}` endpoint with PDF URL
    - Implement GET `/invoices` endpoint with filtering by user/subscription
    - Implement GET `/invoices/{invoice_id}/download` endpoint with presigned URL
    - Verify user authorization before returning invoice data
    - _Requirements: 12.1-12.10, 13.1-13.5, 27.1-27.7_

  - [ ] 13.5 Create Webhook router
    - Create `routers/webhook_router.py` with `APIRouter`
    - Implement POST `/webhooks/razorpay` endpoint
    - Extract and verify X-Razorpay-Signature header
    - Return HTTP 401 if signature invalid or missing
    - Call webhook service to process event
    - Implement POST `/admin/webhooks/{event_id}/replay` endpoint for failed event replay
    - _Requirements: 3.1-3.8, 14.1-14.8, 15.1-15.7, 22.1-22.6_

  - [ ] 13.6 Create Admin/Portal router
    - Create `routers/admin_router.py` with `APIRouter`
    - Implement GET `/admin/audit-logs` endpoint with filtering
    - Implement GET `/admin/dunning/config` endpoint
    - Implement PUT `/admin/dunning/config` endpoint
    - Implement GET `/admin/chargebacks` endpoint
    - All endpoints require admin authorization
    - _Requirements: 11.5, 16.7, 21.1-21.5_

- [ ] 14. Checkpoint - Core functionality complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 15. Implement Celery background jobs
  - [ ] 15.1 Create subscription renewal job
    - Create `tasks/renewal_task.py` with Celery task
    - Schedule job 24 hours before `current_period_end`
    - Trigger charge via Razorpay API
    - On success: extend billing period, generate invoice
    - On failure: initiate dunning process
    - _Requirements: 17.1-17.7_

  - [ ] 15.2 Create dunning retry job
    - Create `tasks/dunning_task.py` with Celery task
    - Execute scheduled retry attempts
    - On success: reset retry_count, update status to ACTIVE
    - On failure: increment retry_count, schedule next retry or halt
    - _Requirements: 9.1-9.8_

  - [ ] 15.3 Create invoice generation job
    - Create `tasks/invoice_task.py` with Celery task
    - Automatically generate invoice for captured payments
    - Execute within 5 minutes of payment capture
    - Retry up to 3 times with exponential backoff on failure
    - Send invoice email after generation
    - _Requirements: 18.1-18.6_

  - [ ] 15.4 Create pause auto-resume job
    - Create `tasks/pause_resume_task.py` with Celery task
    - Schedule resume job at `pause_end` timestamp
    - Automatically resume subscription when duration expires
    - Cancel scheduled job if manually resumed early
    - _Requirements: 23.1-23.5_

  - [ ] 15.5 Create daily Razorpay reconciliation job
    - **NEW: Create `tasks/reconciliation_task.py` with Celery task**
    - **NEW: Fetch last 7 days of payments from Razorpay API**
    - **NEW: Cross-check Razorpay payments against local payment records**
    - **NEW: Detect missing payments and log critical alerts**
    - **NEW: Synthesize webhook events for missing payments**
    - **NEW: Fetch Razorpay subscriptions and compare statuses**
    - **NEW: Generate reconciliation reports with discrepancies**
    - **NEW: Alert operations team on mismatches**
    - **NEW: Schedule job daily at 2:00 AM UTC via crontab**
    - **NEW: Retry up to 3 times on job failure**
    - _Requirements: 35.1-35.10_

- [ ] 16. Implement lifespan and dependency injection
  - [ ] 16.1 Configure lifespan for Razorpay client
    - Add Razorpay client initialization to `lifecycle/lifespan.py`
    - Store client in `app.state.razorpay_client`
    - Use `SecretStr` for API keys from settings
    - _Requirements: 2.2, 2.3, 20.1-20.6_

  - [ ] 16.2 Create dependency providers
    - Create `dependencies.py` with repository and service providers
    - Use `Depends(...)` pattern for dependency injection
    - Create `Annotated` type aliases for common dependencies
    - _Requirements: All service dependencies_

  - [ ] 16.3 Wire billing routers into main application
    - Register billing routers in main FastAPI app with `/billing` prefix
    - Register webhook router with `/webhooks` prefix
    - Register admin router with `/admin` prefix
    - Add appropriate tags for OpenAPI documentation
    - _Requirements: All routing requirements_

- [ ] 17. Implement comprehensive error handling
  - [ ] 17.1 Create billing-specific exceptions
    - Define exceptions in `exceptions.py`: `InvalidStateTransitionException`, `ProrationCalculationException`, `WebhookVerificationException`, `InvoiceGenerationException`
    - All extend appropriate base exceptions from `utils/exceptions.py`
    - Include error codes and structured detail fields
    - _Requirements: 4.7, 5.5, 15.6, 18.5_

  - [ ] 17.2 Add exception handling in webhook router
    - Catch `WebhookVerificationException` → return HTTP 401
    - Catch validation errors → return HTTP 400
    - Log webhook processing failures with structured logging
    - _Requirements: 3.8, 15.1-15.7_

  - [ ] 17.3 Implement transaction rollback on Razorpay API failures
    - Wrap database writes and Razorpay calls in transactions
    - Rollback DB changes if Razorpay API call fails
    - Use `ExternalServiceException` for Razorpay failures
    - _Requirements: 25.1-25.6, 26.1-26.6_

- [ ] 18. Implement audit logging integration
  - [ ] 18.1 Add audit logging to all critical operations
    - Log subscription creation, status changes, plan changes
    - Log payment captures, failures, refunds
    - Log invoice generation
    - Log webhook event processing
    - Record user_id, ip_address, user_agent when available
    - Record before/after snapshots in changes field
    - _Requirements: 16.1-16.8_

  - [ ] 18.2 Create audit log query endpoints
    - Implement filtering by entity_type, entity_id, action, date range
    - Admin-only access with authorization checks
    - _Requirements: 16.7_

- [ ] 19. Implement database migrations and constraints
  - [ ] 19.1 Add database CHECK constraints
    - Add CHECK constraint: payment amount validation
    - Add CHECK constraint: invoice tax calculation validation
    - Add CHECK constraint: subscription period validation (current_period_end > current_period_start)
    - Add CHECK constraint: retry_count ≤ max_retries
    - **NEW: Add CHECK constraint: version field ≥ 0**
    - _Requirements: 1.2, 2.4, 12.2, 12.3, 29.2_

  - [ ] 19.2 Create database indexes for performance
    - Index on subscriptions.user_id for user lookups
    - Index on subscriptions.razorpay_subscription_id for webhook lookups
    - **NEW: Composite index on subscriptions(id, version) for optimistic locking**
    - Index on payments.razorpay_payment_id for idempotency checks
    - Index on webhook_events.razorpay_event_id for duplicate detection
    - Index on invoices.user_id for user invoice queries
    - Index on audit_logs.entity_type, audit_logs.entity_id for audit queries
    - _Requirements: Performance optimization, 29.1-29.7_

  - [ ] 19.3 Ensure audit log immutability
    - Add database trigger or application-level constraint to reject UPDATE on audit_logs
    - Add database trigger or application-level constraint to reject DELETE on audit_logs
    - _Requirements: 16.5, 16.6_

- [ ] 20. Add configuration and settings
  - [ ] 20.1 Add billing settings to configuration
    - Add Razorpay API key and secret to settings (using `SecretStr`)
    - Add webhook secret to settings (using `SecretStr`)
    - Add dunning configuration (retry intervals, max retries)
    - Add GST configuration (seller GSTIN, default tax rate)
    - Add invoice configuration (invoice number prefix, PDF storage settings)
    - Add PCI-DSS compliance flags
    - _Requirements: 20.1-20.6, 21.1-21.5_

- [ ] 21. Integration testing
  - [ ]* 21.1 Write integration test for subscription creation flow
    - Test end-to-end: create subscription → webhook authenticated → webhook activated → payment recorded
    - Test with mocked Razorpay API
    - _Requirements: 2.1-2.8, 3.1-3.8_

  - [ ]* 21.2 Write integration test for payment failure and dunning
    - Test payment failure → dunning initiated → retry scheduled → subscription status updates
    - Test max retries exhausted → subscription halted
    - _Requirements: 9.1-9.8_

  - [ ]* 21.3 Write integration test for plan change with proration
    - Test upgrade: proration charge → payment captured → subscription updated
    - Test downgrade: credit note generated → credit applied at renewal
    - _Requirements: 5.1-5.8, 28.1-28.6_

  - [ ]* 21.4 Write integration test for webhook idempotency
    - Test sending duplicate webhook events
    - Verify only one WebhookEvent record created
    - Verify system state identical after multiple processing
    - _Requirements: 14.1-14.8_

  - [ ]* 21.5 Write integration test for invoice generation
    - Test automatic invoice generation after payment capture
    - Test GST calculation for intra-state and inter-state
    - Test PDF generation and storage
    - _Requirements: 12.1-12.10, 18.1-18.6, 27.1-27.7_

- [ ] 22. Final checkpoint and documentation
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 23. Create API documentation examples
  - [ ] 23.1 Add OpenAPI schema examples
    - Add request/response examples for all endpoints
    - Document webhook payload structures
    - Document error responses with error codes
    - _Requirements: API documentation_

  - [ ] 23.2 Create usage guide
    - Document subscription creation flow
    - Document plan change workflow
    - Document webhook setup and signature verification
    - Document admin operations (dunning config, audit logs, failed event replay)
    - _Requirements: User documentation_

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties (Properties 1-5 from design)
- Unit tests validate specific examples and edge cases
- Integration tests validate end-to-end workflows
- All async code uses native asyncio patterns
- All services use the Result pattern with `isinstance(result, Failure)` + `raise` unwrapping
- All DTOs use `ConfigDict(extra="forbid", frozen=True)` for request models
- All secrets use `SecretStr` from Pydantic
- Database transactions wrap multi-step operations for atomicity
- Circuit breaker pattern protects against Razorpay API cascading failures

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1"] },
    { "id": 2, "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"] },
    { "id": 3, "tasks": ["4.1", "5.1", "7.1"] },
    { "id": 4, "tasks": ["4.2", "5.2", "7.2", "6.1"] },
    { "id": 5, "tasks": ["6.2", "6.3"] },
    { "id": 6, "tasks": ["6.4", "8.1", "9.1"] },
    { "id": 7, "tasks": ["9.2", "10.1"] },
    { "id": 8, "tasks": ["10.2", "10.3", "10.4", "11.1"] },
    { "id": 9, "tasks": ["11.2", "12.1"] },
    { "id": 10, "tasks": ["13.1", "13.2", "13.3", "13.4", "13.5", "13.6"] },
    { "id": 11, "tasks": ["15.1", "15.2", "15.3", "15.4"] },
    { "id": 12, "tasks": ["16.1", "16.2", "17.1"] },
    { "id": 13, "tasks": ["16.3", "17.2", "17.3", "18.1"] },
    { "id": 14, "tasks": ["18.2", "19.1", "19.2", "19.3", "20.1"] },
    { "id": 15, "tasks": ["21.1", "21.2", "21.3", "21.4", "21.5"] },
    { "id": 16, "tasks": ["23.1", "23.2"] }
  ]
}
```
