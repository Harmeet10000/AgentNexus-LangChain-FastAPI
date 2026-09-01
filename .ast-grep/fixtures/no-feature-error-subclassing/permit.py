# permit: src/app/features/subscriptions/errors.py — allowed (ignored via ignores)
from app.shared.result import FeatureError

class SubscriptionNotFoundError(FeatureError):
    pass

class SubscriptionDuplicateError(FeatureError):
    pass

# permit: src/app/shared/rag/errors.py — shared classifier owns its union (ADR-006)
class RagProviderError(FeatureError):
    pass
