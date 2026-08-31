from app.features.subscriptions.errors import SubscriptionNotFoundError
error = None
match error:
    case SubscriptionNotFoundError():
        pass
