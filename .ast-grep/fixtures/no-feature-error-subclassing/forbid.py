# features/foo/service.py — must not subclass FeatureError outside errors.py
from app.shared.result import FeatureError

class BadFeatureError(FeatureError):
    pass
