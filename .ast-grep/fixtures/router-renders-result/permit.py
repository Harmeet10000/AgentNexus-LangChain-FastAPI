# permit: crawler policy guard
async def guard(rate_limiter, client_id):
    is_allowed, info = await rate_limiter.check_rate_limit(client_id, None)
    if not is_allowed:
        raise TooManyRequestsException(detail="rate limited")
