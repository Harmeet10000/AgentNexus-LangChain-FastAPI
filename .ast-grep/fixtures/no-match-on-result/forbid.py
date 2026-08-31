from returns.result import Success, Failure
result = None
match result:
    case Success(value):
        pass
    case Failure(error):
        pass
