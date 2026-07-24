# Requirements Document

## Introduction

This document specifies the functional requirements for a production-ready SaaS billing platform that integrates PostgreSQL database management with Razorpay's payment gateway infrastructure. The system orchestrates comprehensive subscription lifecycle operations including plan management, subscription state transitions, automated billing cycles, payment processing with dunning strategies, GST-compliant invoicing, proration handling for plan modifications, and immutable audit trails for regulatory compliance.

The billing engine supports the complete subscription journey—from initial plan creation through subscription activation, recurring renewals, plan upgrades/downgrades, temporary pauses, permanent cancellations, refund processing, and chargeback management. Financial integrity is maintained through ACID database transactions, relational constraints, webhook idempotency enforcement, and PCI-DSS compliance by delegating all payment method tokenization to Razorpay.

## Glossary

- **System**: The complete Razorpay SaaS Billing Platform including all services, repositories, and external integrations
- **Plan_Service**: Service component responsible for billing plan CRUD operations and versioning
- **Subscription_Service**: Core service managing subscription lifecycle and state transitions
- **Payment_Service**: Service handling payment transaction recording, refunds, and chargeback processing
- **Invoice_Service**: Service responsible for GST-compliant invoice generation and delivery
- **Webhook_Service**: Service handling Razorpay webhook verification, idempotency, and event routing
- **Dunning_Service**: Service orchestrating failed payment retry logic with exponential backoff
- **Proration_Service**: Service calculating proration credits/charges for mid-cycle changes
- **Razorpay_API**: External Razorpay payment gateway REST API
- **Razorpay_Webhook**: External webhook event delivery system from Razorpay
- **Database**: PostgreSQL relational database storing all billing entities
- **User**: Customer or subscriber using the SaaS platform
- **Administrator**: System operator with elevated privileges for management operations
- **HMAC_Signature**: Hash-based Message Authentication Code using SHA256 algorithm
- **GSTIN**: Goods and Services Tax Identification Number (15-character alphanumeric identifier)
- **Paisa**: Smallest currency unit in INR (1 Rupee = 100 Paisa)
- **Audit_Log**: Immutable record of all critical system operations for compliance
- **State_Machine**: Finite state machine governing valid subscription status transitions
- **Idempotency_Key**: Unique identifier ensuring operations execute exactly once despite retries

## Requirements

### Requirement 1: Plan Management

**User Story:** As an administrator, I want to create and manage billing plans with different pricing tiers and intervals, so that I can offer flexible subscription options to customers.

#### Acceptance Criteria

1. WHEN an administrator creates a plan with valid pricing and interval THEN THE Plan_Service SHALL persist the plan to the Database with a unique plan ID
2. THE Plan_Service SHALL validate that plan amount is at least 100 paisa (₹1.00) before creation
3. WHEN an administrator updates an existing plan THEN THE Plan_Service SHALL create a new plan version while maintaining active subscriptions on the previous version
4. WHEN an administrator lists plans THEN THE System SHALL return all active plans sorted by creation date
5. WHEN an administrator archives a plan THEN THE Plan_Service SHALL mark the plan as inactive without affecting existing subscriptions
6. THE Plan_Service SHALL enforce unique plan names within active plans
7. THE Plan_Service SHALL validate that interval_count is a positive integer
8. THE Plan_Service SHALL validate that trial_period_days is non-negative

### Requirement 2: Subscription Creation and Initialization

**User Story:** As a user, I want to subscribe to a billing plan, so that I can access the SaaS platform services.

#### Acceptance Criteria

1. WHEN a user subscribes to an active plan THEN THE Subscription_Service SHALL create a subscription record in the Database with status CREATED
2. WHEN creating a subscription THEN THE Subscription_Service SHALL create or retrieve a Razorpay customer using the user's email and phone
3. WHEN creating a subscription THEN THE Subscription_Service SHALL create a Razorpay subscription and return a payment URL
4. WHEN a subscription is created THEN THE System SHALL generate a unique UUID as the subscription identifier
5. THE Subscription_Service SHALL validate that the referenced plan exists and is active before subscription creation
6. WHEN creating a subscription THEN THE System SHALL initialize retry_count to 0 and max_retries to 4
7. WHEN a subscription is created THEN THE System SHALL create an audit log entry with action SUBSCRIPTION_CREATED
8. THE Subscription_Service SHALL prevent creating duplicate active subscriptions for the same user and plan combination

### Requirement 3: Subscription Activation via Webhooks

**User Story:** As the system, I want to process Razorpay webhook events, so that subscription status reflects payment completion.

#### Acceptance Criteria

1. WHEN Razorpay sends a subscription.authenticated webhook THEN THE Webhook_Service SHALL verify the HMAC SHA256 signature before processing
2. WHEN a webhook signature is invalid THEN THE Webhook_Service SHALL reject the request with HTTP 401 Unauthorized
3. WHEN a valid subscription.authenticated event is received THEN THE Subscription_Service SHALL update the subscription status to AUTHENTICATED
4. WHEN a valid subscription.activated event is received THEN THE Subscription_Service SHALL update the subscription status to ACTIVE and record current_period_start and current_period_end
5. WHEN a subscription.activated event is processed THEN THE Payment_Service SHALL record the initial payment transaction
6. THE Webhook_Service SHALL check for duplicate events using razorpay_event_id before processing
7. WHEN a duplicate webhook event is received THEN THE Webhook_Service SHALL return success without reprocessing
8. WHEN webhook processing fails THEN THE System SHALL create a WebhookEvent record with status FAILED and error details

### Requirement 4: Subscription State Transitions

**User Story:** As the system, I want to enforce valid subscription status transitions, so that subscription states remain consistent.

#### Acceptance Criteria

1. THE Subscription_Service SHALL allow transitions from CREATED to AUTHENTICATED only
2. THE Subscription_Service SHALL allow transitions from AUTHENTICATED to ACTIVE only
3. THE Subscription_Service SHALL allow transitions from ACTIVE to PAST_DUE, PAUSED, or CANCELLED
4. THE Subscription_Service SHALL allow transitions from PAST_DUE to ACTIVE or HALTED
5. THE Subscription_Service SHALL allow transitions from HALTED to ACTIVE or CANCELLED
6. THE Subscription_Service SHALL allow transitions from PAUSED to ACTIVE or CANCELLED
7. WHEN an invalid status transition is attempted THEN THE Subscription_Service SHALL raise a ValidationException
8. WHEN a subscription status is updated THEN THE System SHALL create an audit log entry recording the state change

### Requirement 5: Plan Change with Proration

**User Story:** As a user, I want to upgrade or downgrade my subscription plan mid-cycle, so that I can adjust my service tier based on needs.

#### Acceptance Criteria

1. WHEN a user changes plans with an ACTIVE subscription THEN THE Proration_Service SHALL calculate the proration amount based on unused time
2. WHEN upgrading to a higher-priced plan THEN THE System SHALL immediately charge the prorated difference
3. WHEN downgrading to a lower-priced plan THEN THE System SHALL create a credit note for the unused amount
4. THE Proration_Service SHALL validate that both plans have the same billing interval before calculating proration
5. THE Proration_Service SHALL validate that the effective date falls within the current billing period
6. WHEN proration is calculated THEN THE System SHALL compute remaining_fraction as (current_period_end - effective_date) / (current_period_end - current_period_start)
7. WHEN upgrading THEN THE System SHALL generate a proration invoice and charge via Razorpay_API immediately
8. WHEN a plan change is completed THEN THE Subscription_Service SHALL update the subscription plan_id and create an audit log entry

### Requirement 6: Subscription Pause and Resume

**User Story:** As a user, I want to temporarily pause my subscription, so that I can suspend service without canceling permanently.

#### Acceptance Criteria

1. WHEN a user pauses an ACTIVE subscription THEN THE Subscription_Service SHALL update status to PAUSED and record pause_start timestamp
2. WHEN pausing with a specified duration THEN THE System SHALL record pause_end timestamp
3. WHEN a subscription is PAUSED THEN THE System SHALL not charge recurring payments
4. WHEN a user resumes a PAUSED subscription THEN THE Subscription_Service SHALL update status to ACTIVE and clear pause timestamps
5. THE Subscription_Service SHALL allow pause operations only on ACTIVE subscriptions
6. WHEN resuming THEN THE System SHALL recalculate the next billing date based on pause duration
7. WHEN a subscription is paused or resumed THEN THE System SHALL create an audit log entry

### Requirement 7: Subscription Cancellation

**User Story:** As a user, I want to cancel my subscription, so that I can stop recurring charges and end my service.

#### Acceptance Criteria

1. WHEN a user cancels a subscription with immediate effect THEN THE Subscription_Service SHALL update status to CANCELLED and record cancelled_at timestamp
2. WHEN a user cancels at period end THEN THE System SHALL set cancel_at_period_end to true and update status to CANCELLED after current_period_end
3. WHEN an immediate cancellation occurs mid-cycle THEN THE Proration_Service SHALL calculate refund amount for unused time
4. WHEN a subscription is cancelled THEN THE System SHALL cancel the Razorpay subscription via Razorpay_API
5. THE Subscription_Service SHALL allow cancellation from ACTIVE, PAST_DUE, HALTED, or PAUSED states
6. WHEN a subscription is cancelled THEN THE System SHALL record ended_at timestamp
7. WHEN a subscription is cancelled THEN THE System SHALL create an audit log entry with action SUBSCRIPTION_CANCELLED

### Requirement 8: Payment Recording and Transaction Management

**User Story:** As the system, I want to record all payment transactions, so that financial records are accurate and complete.

#### Acceptance Criteria

1. WHEN a payment.captured webhook is received THEN THE Payment_Service SHALL create a payment record with status CAPTURED
2. THE Payment_Service SHALL validate that payment.subscription_id references an existing subscription
3. WHEN recording a payment THEN THE System SHALL convert Razorpay amount from paisa to Decimal for storage
4. WHEN a payment is captured THEN THE Payment_Service SHALL record captured_at timestamp and payment method
5. WHEN a payment.failed webhook is received THEN THE Payment_Service SHALL create a payment record with status FAILED and error details
6. THE Payment_Service SHALL enforce that subscription_id and invoice_id foreign keys reference valid records
7. WHEN a payment is recorded THEN THE System SHALL validate that amount is positive
8. THE Payment_Service SHALL support idempotent payment recording using razorpay_payment_id as deduplication key

### Requirement 9: Failed Payment Handling and Dunning

**User Story:** As the system, I want to automatically retry failed payments with increasing delays, so that temporary payment issues do not immediately revoke service.

#### Acceptance Criteria

1. WHEN a payment fails THEN THE Dunning_Service SHALL initiate the dunning process and update subscription status to PAST_DUE
2. WHEN initiating dunning THEN THE System SHALL verify that retry_count is less than max_retries
3. WHEN retry_count is less than max_retries THEN THE Dunning_Service SHALL schedule a retry task with exponential backoff delay
4. THE Dunning_Service SHALL use retry delays of 1, 3, 7, and 14 days for attempts 1 through 4
5. WHEN executing a retry THEN THE Dunning_Service SHALL increment retry_count and attempt charge via Razorpay_API
6. WHEN a retry succeeds THEN THE System SHALL update subscription status to ACTIVE and reset retry_count to 0
7. WHEN retry_count reaches max_retries THEN THE Dunning_Service SHALL update subscription status to HALTED
8. WHEN a subscription is halted THEN THE System SHALL create an audit log entry with action SUBSCRIPTION_HALTED

### Requirement 10: Refund Processing

**User Story:** As an administrator, I want to process full or partial refunds, so that I can handle customer service issues and disputes.

#### Acceptance Criteria

1. WHEN an administrator initiates a refund THEN THE Payment_Service SHALL validate that the payment status is CAPTURED
2. WHEN initiating a refund THEN THE Payment_Service SHALL validate that refund amount is less than or equal to payment amount minus existing refund_amount
3. WHEN a refund is initiated THEN THE Payment_Service SHALL call Razorpay_API to create the refund
4. WHEN a refund.processed webhook is received THEN THE Payment_Service SHALL update the payment refund_amount
5. WHEN refund_amount equals payment amount THEN THE System SHALL update payment status to REFUNDED
6. WHEN refund_amount is less than payment amount THEN THE System SHALL update payment status to PARTIALLY_REFUNDED
7. WHEN a refund is processed THEN THE Invoice_Service SHALL generate a credit note
8. WHEN a refund is processed THEN THE System SHALL create an audit log entry with action REFUND_ISSUED

### Requirement 11: Chargeback and Dispute Management

**User Story:** As an administrator, I want to handle chargebacks and disputes, so that I can respond to payment disputes from customers.

#### Acceptance Criteria

1. WHEN a payment.dispute.created webhook is received THEN THE Payment_Service SHALL record the chargeback details
2. WHEN a chargeback occurs THEN THE System SHALL update the associated payment with chargeback metadata
3. WHEN an administrator submits dispute evidence THEN THE Payment_Service SHALL send the evidence to Razorpay_API
4. WHEN a chargeback is recorded THEN THE System SHALL create an audit log entry
5. THE Payment_Service SHALL allow administrators to view all chargebacks filtered by date range and status

### Requirement 12: GST-Compliant Invoice Generation

**User Story:** As the system, I want to generate GST-compliant invoices for all captured payments, so that the platform meets Indian tax regulations.

#### Acceptance Criteria

1. WHEN a payment is captured THEN THE Invoice_Service SHALL generate an invoice with a unique sequential invoice_number
2. THE Invoice_Service SHALL calculate tax_amount as subtotal multiplied by tax_rate
3. THE Invoice_Service SHALL calculate total as subtotal plus tax_amount
4. WHEN seller and buyer states are the same THEN THE System SHALL split tax into CGST and SGST with equal amounts and set IGST to zero
5. WHEN seller and buyer states are different THEN THE System SHALL set IGST equal to tax_amount and CGST and SGST to zero
6. THE Invoice_Service SHALL validate that seller_gstin matches the format of 15 alphanumeric characters
7. WHEN generating a B2B invoice THEN THE Invoice_Service SHALL require buyer_gstin
8. WHEN an invoice is generated THEN THE System SHALL create a PDF and upload it to cloud storage
9. WHEN an invoice is generated THEN THE System SHALL record issued_at and paid_at timestamps
10. WHEN an invoice is generated THEN THE System SHALL create an audit log entry with action INVOICE_GENERATED

### Requirement 13: Invoice Retrieval and Listing

**User Story:** As a user, I want to view and download my invoices, so that I can maintain financial records.

#### Acceptance Criteria

1. WHEN a user requests invoices THEN THE Invoice_Service SHALL return all invoices for that user sorted by issued_at descending
2. WHEN retrieving a specific invoice THEN THE Invoice_Service SHALL return the invoice with pdf_url for download
3. THE Invoice_Service SHALL allow filtering invoices by subscription_id
4. WHEN an invoice is retrieved THEN THE System SHALL validate that the requesting user owns the associated subscription
5. THE Invoice_Service SHALL return invoices with status ISSUED or PAID only (exclude DRAFT and VOID)

### Requirement 14: Webhook Idempotency and Event Logging

**User Story:** As the system, I want to process each webhook event exactly once, so that duplicate events do not corrupt system state.

#### Acceptance Criteria

1. WHEN a webhook event is received THEN THE Webhook_Service SHALL check if a WebhookEvent record exists with the same razorpay_event_id
2. WHEN a duplicate event is detected THEN THE Webhook_Service SHALL return HTTP 200 without processing
3. WHEN processing a new webhook event THEN THE System SHALL create a WebhookEvent record with status PENDING
4. WHEN webhook processing begins THEN THE System SHALL update WebhookEvent status to PROCESSING
5. WHEN webhook processing succeeds THEN THE System SHALL update WebhookEvent status to PROCESSED and record processed_at timestamp
6. WHEN webhook processing fails THEN THE System SHALL update WebhookEvent status to FAILED and record error_message and failed_at timestamp
7. THE Webhook_Service SHALL enforce a unique constraint on razorpay_event_id in the Database
8. WHEN webhook processing fails THEN THE System SHALL increment retry_count for retry tracking

### Requirement 15: Webhook Signature Verification

**User Story:** As the system, I want to verify webhook authenticity, so that only legitimate Razorpay events are processed.

#### Acceptance Criteria

1. WHEN a webhook request is received THEN THE Webhook_Service SHALL extract the X-Razorpay-Signature header
2. WHEN the signature header is missing THEN THE Webhook_Service SHALL reject the request with HTTP 401
3. WHEN verifying signature THEN THE Webhook_Service SHALL compute HMAC SHA256 of the payload using the webhook secret
4. WHEN the computed signature does not match the received signature THEN THE Webhook_Service SHALL reject the request with HTTP 401
5. THE Webhook_Service SHALL use constant-time comparison to prevent timing attacks
6. WHEN signature verification fails THEN THE System SHALL log the failure with request metadata
7. WHEN signature verification succeeds THEN THE Webhook_Service SHALL proceed with event processing

### Requirement 16: Audit Logging for Compliance

**User Story:** As an administrator, I want immutable audit logs of all critical operations, so that the platform meets PCI-DSS and regulatory requirements.

#### Acceptance Criteria

1. WHEN any critical operation occurs THEN THE System SHALL create an AuditLog record with entity_type, entity_id, and action
2. THE System SHALL record user_id, ip_address, and user_agent when available
3. WHEN an operation modifies state THEN THE System SHALL record before and after snapshots in the changes field
4. THE System SHALL assign an immutable created_at timestamp to each audit log entry
5. THE Database SHALL reject UPDATE operations on the audit_logs table
6. THE Database SHALL reject DELETE operations on the audit_logs table
7. WHEN querying audit logs THEN THE System SHALL support filtering by entity_type, entity_id, action, and date range
8. THE System SHALL retain audit logs indefinitely for compliance purposes

### Requirement 17: Subscription Renewal via Background Jobs

**User Story:** As the system, I want to automatically renew subscriptions at the end of each billing period, so that service continues without manual intervention.

#### Acceptance Criteria

1. WHEN a subscription's current_period_end approaches THEN THE System SHALL schedule a renewal job via Celery
2. WHEN executing a renewal job THEN THE System SHALL trigger a charge via Razorpay_API
3. WHEN a renewal charge succeeds THEN THE Subscription_Service SHALL extend current_period_start and current_period_end
4. WHEN a renewal charge fails THEN THE System SHALL initiate the dunning process
5. THE System SHALL schedule renewal jobs 24 hours before current_period_end
6. WHEN a subscription is PAUSED or CANCELLED THEN THE System SHALL not schedule renewal jobs
7. WHEN a renewal succeeds THEN THE Invoice_Service SHALL automatically generate an invoice

### Requirement 18: Automated Invoice Generation Job

**User Story:** As the system, I want to automatically generate invoices for captured payments, so that users receive invoices without manual processing.

#### Acceptance Criteria

1. WHEN a payment is captured THEN THE System SHALL schedule an invoice generation job
2. WHEN executing an invoice generation job THEN THE Invoice_Service SHALL retrieve payment and subscription details
3. WHEN generating an invoice THEN THE System SHALL determine seller and buyer state codes from user addresses
4. WHEN an invoice is generated THEN THE System SHALL send an email to the user with the PDF attachment
5. WHEN invoice generation fails THEN THE System SHALL retry up to 3 times with exponential backoff
6. THE System SHALL execute invoice generation jobs within 5 minutes of payment capture

### Requirement 19: Proration Preview

**User Story:** As a user, I want to preview proration charges before changing plans, so that I understand the financial impact.

#### Acceptance Criteria

1. WHEN a user requests a proration preview THEN THE Proration_Service SHALL calculate the proration amount without applying changes
2. WHEN previewing proration THEN THE System SHALL return proration_amount, tax_amount, total_amount, and direction
3. THE Proration_Service SHALL calculate preview using current subscription state and target plan
4. WHEN previewing THEN THE System SHALL not modify the subscription or create any transactions
5. THE Proration_Service SHALL validate that the subscription status is ACTIVE before preview

### Requirement 20: Payment Method Tokenization

**User Story:** As the system, I want to delegate payment method storage to Razorpay, so that the platform never stores card data and maintains PCI-DSS compliance.

#### Acceptance Criteria

1. THE System SHALL never store raw credit card numbers, CVV codes, or expiration dates in the Database
2. WHEN a user adds a payment method THEN THE System SHALL redirect to Razorpay's hosted payment page
3. WHEN Razorpay tokenizes a payment method THEN THE System SHALL store only the razorpay_customer_id
4. THE System SHALL retrieve payment method details from Razorpay_API when needed for display
5. WHEN processing payments THEN THE System SHALL reference razorpay_customer_id and razorpay_subscription_id only
6. THE System SHALL not log or transmit sensitive payment data in any form

### Requirement 21: Dunning Strategy Configuration

**User Story:** As an administrator, I want to configure dunning retry intervals and maximum attempts, so that retry behavior aligns with business policies.

#### Acceptance Criteria

1. WHEN an administrator updates dunning configuration THEN THE Dunning_Service SHALL validate retry intervals are positive integers
2. WHEN an administrator updates max_retries THEN THE System SHALL validate the value is between 1 and 10
3. THE System SHALL apply updated dunning configuration to new subscriptions only (not retroactive)
4. WHEN retrieving dunning configuration THEN THE System SHALL return current retry intervals and max_retries
5. THE System SHALL support per-plan dunning overrides for enterprise customers

### Requirement 22: Failed Event Replay

**User Story:** As an administrator, I want to manually replay failed webhook events, so that I can recover from transient processing errors.

#### Acceptance Criteria

1. WHEN an administrator requests event replay THEN THE Webhook_Service SHALL retrieve the WebhookEvent record by ID
2. WHEN replaying an event THEN THE System SHALL validate that the event status is FAILED
3. WHEN replaying THEN THE Webhook_Service SHALL reprocess the event payload through the original event handler
4. WHEN replay succeeds THEN THE System SHALL update the WebhookEvent status to PROCESSED
5. WHEN replay fails THEN THE System SHALL update the error_message and increment retry_count
6. THE Webhook_Service SHALL enforce that only administrators can trigger manual replay

### Requirement 23: Subscription Pause Duration Validation

**User Story:** As a user, I want to pause my subscription for a specified duration, so that automatic resumption occurs without manual action.

#### Acceptance Criteria

1. WHEN pausing with a duration THEN THE System SHALL validate that pause_duration_days is between 1 and 365
2. WHEN pause_end timestamp is reached THEN THE System SHALL automatically resume the subscription
3. THE System SHALL schedule a resume job at pause_end timestamp via Celery
4. WHEN automatically resuming THEN THE Subscription_Service SHALL follow the same logic as manual resume
5. WHEN a user manually resumes before pause_end THEN THE System SHALL cancel the scheduled resume job

### Requirement 24: Plan Versioning for Active Subscriptions

**User Story:** As an administrator, I want to modify plan pricing without disrupting active subscriptions, so that existing customers maintain their original pricing.

#### Acceptance Criteria

1. WHEN an administrator updates a plan THEN THE Plan_Service SHALL create a new plan version with a different plan ID
2. WHEN a plan is updated THEN THE System SHALL not modify the original plan record
3. WHEN listing plans THEN THE System SHALL return only the latest version of each plan unless specifically requesting historical versions
4. WHEN a subscription references a plan THEN THE System SHALL continue using that plan version even after updates
5. THE Plan_Service SHALL link plan versions using a parent_plan_id field for traceability

### Requirement 25: Error Handling for Razorpay API Failures

**User Story:** As the system, I want to gracefully handle Razorpay API failures, so that transient errors do not permanently break operations.

#### Acceptance Criteria

1. WHEN Razorpay_API returns HTTP 503 or timeout THEN THE System SHALL raise an ExternalServiceException with retryable=true
2. WHEN Razorpay_API returns HTTP 401 or 403 THEN THE System SHALL raise an ExternalServiceException with retryable=false
3. WHEN Razorpay_API returns HTTP 429 THEN THE System SHALL implement exponential backoff with jitter and retry up to 3 times
4. WHEN a transient error occurs THEN THE System SHALL log the error with correlation_id and retry metadata
5. WHEN a permanent error occurs THEN THE System SHALL alert operations team and not retry
6. THE System SHALL implement a circuit breaker pattern after 5 consecutive Razorpay_API failures

### Requirement 26: Database Transaction Management

**User Story:** As the system, I want to ensure atomicity of multi-step operations, so that partial failures do not leave the database in an inconsistent state.

#### Acceptance Criteria

1. WHEN creating a subscription THEN THE System SHALL wrap Database writes and Razorpay_API calls in a transaction
2. WHEN a Razorpay_API call fails within a transaction THEN THE System SHALL rollback all Database changes
3. WHEN processing a webhook event THEN THE System SHALL use a Database transaction to ensure atomicity
4. WHEN a plan change involves proration THEN THE System SHALL use a transaction for subscription update, invoice creation, and payment charge
5. THE System SHALL implement retry logic with exponential backoff for Database deadlocks
6. THE System SHALL enforce foreign key constraints at the Database level for referential integrity

### Requirement 27: Invoice PDF Generation and Storage

**User Story:** As a user, I want to download invoice PDFs, so that I can maintain offline records.

#### Acceptance Criteria

1. WHEN an invoice is generated THEN THE Invoice_Service SHALL render a PDF using a GST-compliant template
2. WHEN rendering a PDF THEN THE System SHALL include invoice_number, subtotal, tax breakdown, total, seller_gstin, buyer_gstin, and line items
3. WHEN a PDF is generated THEN THE System SHALL upload it to cloud storage (S3 or equivalent)
4. WHEN uploading THEN THE System SHALL generate a presigned URL valid for 7 days
5. THE Invoice_Service SHALL store the permanent storage path in the pdf_url field
6. WHEN a user downloads an invoice THEN THE System SHALL validate user authorization before returning the PDF URL
7. THE System SHALL support invoice PDF regeneration if the storage URL expires

### Requirement 28: Credit Note Generation for Downgrades

**User Story:** As a user, I want to receive credit for unused time when downgrading plans, so that I don't lose money on the prepaid period.

#### Acceptance Criteria

1. WHEN downgrading to a lower-priced plan THEN THE Invoice_Service SHALL generate a credit note
2. THE Invoice_Service SHALL calculate credit_amount as the prorated unused value of the current plan minus the new plan cost
3. WHEN generating a credit note THEN THE System SHALL reference the original invoice_id
4. WHEN a credit note is generated THEN THE System SHALL apply it automatically at the next renewal
5. THE System SHALL display credit note amount in the user's billing dashboard
6. WHEN a credit note exists THEN THE System SHALL deduct the credit from the next payment charge

### Requirement 29: Subscription Listing and Filtering

**User Story:** As a user, I want to view all my subscriptions with filters, so that I can manage multiple subscriptions easily.

#### Acceptance Criteria

1. WHEN a user requests subscriptions THEN THE Subscription_Service SHALL return all subscriptions owned by that user
2. THE Subscription_Service SHALL support filtering by status (ACTIVE, CANCELLED, PAST_DUE, etc.)
3. THE Subscription_Service SHALL support filtering by plan_id
4. WHEN listing subscriptions THEN THE System SHALL include plan details, current_period_start, current_period_end, and status
5. THE Subscription_Service SHALL sort subscriptions by created_at descending by default
6. THE Subscription_Service SHALL support pagination with limit and offset parameters

### Requirement 30: Webhook Event Type Routing

**User Story:** As the system, I want to route webhook events to appropriate handlers, so that each event type triggers the correct business logic.

#### Acceptance Criteria

1. WHEN a subscription.authenticated event is received THEN THE Webhook_Service SHALL route to Subscription_Service.handle_authenticated
2. WHEN a subscription.activated event is received THEN THE Webhook_Service SHALL route to Subscription_Service.handle_activated
3. WHEN a subscription.charged event is received THEN THE Webhook_Service SHALL route to Subscription_Service.handle_charged
4. WHEN a payment.captured event is received THEN THE Webhook_Service SHALL route to Payment_Service.record_payment
5. WHEN a payment.failed event is received THEN THE Webhook_Service SHALL route to Payment_Service.record_failed_payment and initiate dunning
6. WHEN a refund.processed event is received THEN THE Webhook_Service SHALL route to Payment_Service.handle_refund_processed
7. WHEN an unhandled event type is received THEN THE Webhook_Service SHALL log a warning and create a WebhookEvent record without processing
8. THE Webhook_Service SHALL use pattern matching on event_type for routing decisions
