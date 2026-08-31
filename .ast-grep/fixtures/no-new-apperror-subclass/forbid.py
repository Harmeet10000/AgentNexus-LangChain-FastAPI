from app.shared.result import AppError

class NewAppError(AppError):
    pass

# also generic — any new AppError subclass is violation
class MyCustomAppError(AppError):
    pass
