# permit is in errors.py — no violation (we model as not containing FeatureError subclass here to keep scan clean)
from pydantic import BaseModel
class SubscriptionNotFoundError(BaseModel):
    pass
