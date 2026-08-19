# Requirements Document

## Introduction

This document specifies the functional requirements for a user credit integration feature in the Razorpay SaaS billing system. The feature enables administrators and the system itself to grant credit balances to users, which can be automatically applied to subscription renewals and plan changes. The credit system operates as a payment method (not a discount), ensuring GST compliance by calculating tax on the full plan price before credit application.

Key decisions and constraints:
- Credit amounts are stored in **Paisa** (Decimal type), matching the existing `Payment.amount` convention
- Credit is a **payment method choice**, not a price reduction—GST must be calculated on the full invoice total
- Refund → credit conversion is **out of scope** (refunds remain 100% cash-to-source)
- The credit note stays as the GST-compliance artifact; `UserCredit` becomes the thing that actually backs the promises in Requirement 28.4–28.6

## Glossary

- **System**: The Razorpay SaaS Billing Platform including all services, repositories, and external integrations
- **UserCredit**: Database record representing a credit balance granted to a user, denominated in paisa
- **CreditConsumption**: Database record tracking when credit is applied to an invoice, denominated in paisa
- **CreditType**: Enum defining credit origins (PLAN_CREDIT, PROMOTIONAL, ADMIN_GRANT)
- **CreditStatus**: Enum representing credit lifecycle (ACTIVE, CONSUMED, EXPIRED)
- **Credit_Service**: Service component responsible for credit grant, consumption, balance queries, and expiration
- **Invoice_Service**: Service responsible for GST-compliant invoice generation and delivery
- **Proration_Service**: Service calculating proration credits/charges for mid-cycle changes
- **User**: Customer or subscriber using the SaaS platform
- **Administrator**: System operator with elevated privileges for management operations
- **Paisa**: Smallest currency unit in INR (1 Rupee = 100 Paisa)
- **GST**: Goods and Services Tax (18% for SaaS in India)
- **Audit_Log**: Immutable record of all critical system operations for compliance

## Requirements

### Requirement 49: Credit Grant Management

**User Story:** As an administrator or the system itself, I want to grant credits to a user, so that promotional offers, plan-change adjustments, and goodwill gestures can be tracked and later consumed.

#### Acceptance Criteria

1. WHEN credits are granted THEN THE Credit_Service SHALL create a UserCredit record with status ACTIVE
2. WHEN credits are granted THEN THE Credit_Service SHALL validate that credit_amount is positive
3. WHEN credits are granted THEN THE Credit_Service SHALL require valid_from and valid_until timestamps
4. WHEN credits are granted with credit_type ADMIN_GRANT THEN THE Credit_Service SHALL record the granting admin's user_id in metadata
5. WHEN credits are granted THEN THE Credit_Service SHALL create an audit log entry with action CREDIT_GRANTED

### Requirement 50: Credit Consumption at Billing

**User Story:** As a user with a credit balance, I want available credit applied automatically to my renewal, so that I'm not charged cash for value I already have.

#### Acceptance Criteria

1. WHEN generating an invoice for a subscription renewal THEN THE Invoice_Service SHALL query available (ACTIVE, non-expired) credit for the user before calling Razorpay
2. WHEN applying credit to an invoice THEN THE Credit_Service SHALL consume credits in order of soonest valid_until first, then oldest created_at (credits with no expiry are consumed last)
3. WHEN credit fully covers the invoice total THEN THE System SHALL skip the Razorpay charge entirely and mark the invoice PAID via credit alone
4. WHEN credit partially covers the invoice total THEN THE System SHALL charge only the remaining cash amount via Razorpay
5. WHEN credit is consumed THEN THE System SHALL create a CreditConsumption record and decrement UserCredit.remaining_balance within the same database transaction as invoice/payment creation
6. WHEN a UserCredit's remaining_balance reaches zero THEN THE System SHALL set its status to CONSUMED and record consumed_at
7. THE System SHALL never let a partial Razorpay charge failure leave a UserCredit balance decremented without a corresponding successful invoice

### Requirement 51: Credit Expiration

**User Story:** As the billing system, I want to expire outdated credits automatically, so that users cannot apply stale credit balances to their invoices.

#### Acceptance Criteria

1. THE System SHALL run a daily background job that sets status to EXPIRED for any ACTIVE UserCredit past its valid_until
2. WHEN a credit expires THEN THE System SHALL create an audit log entry with action CREDIT_EXPIRED
3. EXPIRED credits SHALL be excluded from consumption ordering in Requirement 50.2

### Requirement 52: Credit Balance Visibility

**User Story:** As a user with a credit balance, I want to view my current balance and credit history, so that I can track how my credits are being applied over time.

#### Acceptance Criteria

1. WHEN a user requests their credit balance THEN THE Credit_Service SHALL return the sum of remaining_balance across all ACTIVE, non-expired UserCredit records
2. WHEN a user requests credit history THEN THE Credit_Service SHALL return all UserCredit and CreditConsumption records for that user, sorted by created_at descending
3. THE System SHALL restrict credit grant/consume operations to system-internal callers and administrators—no user-initiated redemption endpoint in v1

### Requirement 53: Credit Ledger Integrity (Property-Based Testing)

**User Story:** As a system operator, I want to ensure credit ledger integrity, so that credit balances are accurately tracked and never lost or double-spent.

#### Acceptance Criteria

1. FOR ANY UserCredit record, credit_amount MUST EQUAL remaining_balance PLUS the sum of all CreditConsumption.consumed_amount records referencing that credit
2. FOR ANY credit consumption event, the CreditConsumption row and corresponding invoice/payment rows MUST be created in the same database transaction
3. IF a Razorpay charge for the remaining cash portion FAILS, the ENTIRE transaction—including the credit deduction—MUST roll back
4. FOR ANY UserCredit, credit_status MUST be CONSUMED ONLY when remaining_balance equals zero
5. FOR ANY UserCredit, credit_status MUST be EXPIRED ONLY when current timestamp exceeds valid_until and status was previously ACTIVE

### Requirement 54: Credit Integration with Proration (Downgrade Credit Notes)

**User Story:** As a user downgrading my plan, I want to receive credit for unused time that persists as a UserCredit record, so that I can apply it to future renewals.

#### Acceptance Criteria

1. WHEN downgrading to a lower-priced plan THEN THE Proration_Service SHALL calculate the prorated unused value as before
2. WHEN downgrading to a lower-priced plan THEN THE Proration_Service SHALL create a credit note for the unused amount (for GST compliance)
3. WHEN downgrading to a lower-priced plan THEN THE Proration_Service SHALL call Credit_Service.grant_credit() with credit_type PLAN_CREDIT in the same transaction
4. WHEN a UserCredit is created from a proration downgrade THEN valid_from SHALL be the downgrade timestamp and valid_until SHALL default to the end of the subscription's then-current billing cycle plus 12 months
5. WHEN displaying credit note amount on the billing dashboard THEN THE System SHALL source the figure from UserCredit.remaining_balance, not a recalculation from the credit note document

### Requirement 55: Credit Integration with Invoice Generation

**User Story:** As the billing system, I want to apply credit to invoices during generation, so that users' credit balances are properly offset against their charges.

#### Acceptance Criteria

1. WHEN generating an invoice for a subscription renewal THEN THE Invoice_Service SHALL call Credit_Service.apply_credit_to_invoice() with the invoice gross total
2. WHEN applying credit to an invoice THEN THE Credit_Service SHALL return the net cash amount due and the credit_applied amount (in rupees)
3. WHEN applying credit to an invoice THEN THE System SHALL update Invoice.credit_applied with the credit portion (rupees)
4. WHEN applying credit to an invoice THEN the CREDIT application MUST NOT affect subtotal, tax_amount, or total calculations (GST calculated on full price)
5. WHEN an invoice is marked PAID via credit alone THEN no Razorpay charge shall be initiated
6. WHEN an invoice is partially paid via credit AND cash THEN the Razorpay charge SHALL be for the remaining cash amount only

## Notes

- Credit amounts are stored in **paisa** (Decimal) across the system—this matches `Payment.amount` convention
- Invoice totals are stored in **rupees** (Decimal)—this is the existing billing system convention
- The single paisa-to-rupee conversion happens ONLY in `CreditService.apply_credit_to_invoice()`—no other component should perform raw paisa↔rupee conversions
- Credit is a **payment method choice**, not a discount: GST tax is calculated on the full invoice total before credit application
- No user-facing credit redemption endpoint in v1—credit is applied automatically by the system
- Consumption order is **soonest-expiring-first**, then oldest-created (credits with no expiry are consumed last)
