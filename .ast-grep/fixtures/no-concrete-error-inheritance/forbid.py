from app.shared.result import FeatureError
class ConflictError(FeatureError):
    pass
class VersionConflictError(ConflictError):
    pass
