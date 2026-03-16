from fastapi import Request, Response
from fastapi_limiter.depends import RateLimiter


async def _ip_identifier(request: Request, response: Response) -> str:  # noqa: ARG001
    """Extract real client IP respecting reverse proxy forwarding headers.

    X-Forwarded-For can contain a comma-separated chain: client, proxy1, proxy2.
    The leftmost entry is the original client IP — take only that.
    Never trust the full header blindly; a client can spoof intermediate entries.
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
    elif request.client:
        ip = request.client.host
    else:
        ip = "unknown"
    return f"rl:{ip}"


# ── Preset named limiters — import and wrap with Depends() in router ──────────
#
# Intentionally conservative limits for auth endpoints:
# - Login: 5 attempts/min (brute force deterrent)
# - Register: 3/min (account farming deterrent)
# - Forgot password: 3 per 5 min (prevents email flooding)
# - Resend verification: 2 per 5 min (prevents email flooding)

LOGIN_RATE_LIMIT = RateLimiter(times=5, seconds=60)
REGISTER_RATE_LIMIT = RateLimiter(times=3, seconds=60)
FORGOT_PASSWORD_RATE_LIMIT = RateLimiter(times=3, seconds=300)
RESEND_VERIFICATION_RATE_LIMIT = RateLimiter(times=2, seconds=300)
