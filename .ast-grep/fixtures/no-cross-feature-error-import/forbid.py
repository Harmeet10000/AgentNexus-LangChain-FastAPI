# forbid: features/invoices/service.py must not import another feature's errors (cross-feature)
from app.features.subscriptions.errors import SubscriptionNotFoundError
from app.features.subscriptions.errors import SubscriptionCode

# also forbid: audit importing payments errors
from app.features.payments.errors import PaymentNotFoundError
