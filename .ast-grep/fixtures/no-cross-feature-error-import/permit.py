# permit: same-feature relative import — allowed (subscriptions/service.py)
from .errors import SubscriptionNotFoundError
from .errors import SubscriptionCode

# permit: shared result import — allowed
from app.shared.result import FeatureError
