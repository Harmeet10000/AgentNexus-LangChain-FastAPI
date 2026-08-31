# permit: crawler policy guard — exempt (boolean check_rate_limit → raise TooManyRequestsException)
async def guard(rate_limiter, client_id):
    is_allowed, info = await rate_limiter.check_rate_limit(client_id, None)
    if not is_allowed:
        raise TooManyRequestsException(detail="rate limited")

# permit: ImportError capability flag — exempt
try:
    import foo
except ImportError:
    pass

# permit: correct render_result usage — not a raise
from app.shared.result import render_result

async def get_x(result, response):
    return render_result(result, response)

# permit: validation via Pydantic ValueError — exempt (raises ValueError, not expected-failure Exception)
def validate(v: str) -> None:
    if not v:
        raise ValueError("empty")
