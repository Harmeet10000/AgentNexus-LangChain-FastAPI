# permit: src/app/features/subscriptions/errors.py — allowed (ignored via ignores)
from app.shared.result import FeatureError

class SubscriptionNotFoundError(FeatureError):
    pass

class SubscriptionDuplicateError(FeatureError):
    pass
