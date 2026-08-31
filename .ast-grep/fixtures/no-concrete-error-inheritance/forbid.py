from app.shared.result import FeatureError

class ConflictError(FeatureError):
    pass

# violation: concrete inherits concrete (generic — not just VersionConflictError)
class VersionConflictError(ConflictError):
    pass

class AnotherError(ConflictError):
    pass
