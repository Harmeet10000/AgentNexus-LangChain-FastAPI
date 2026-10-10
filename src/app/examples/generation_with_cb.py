"""FastAPI example showing an async circuit breaker around provider I/O."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Depends, Request

from app.resilience.circuit_breaker import (
    CircuitBreakerBackendError,
    CircuitBreakerOpenError,
    CircuitBreakerService,
    RedisCircuitBreakerStore,
)
from app.utils import ServiceUnavailableException

router = APIRouter()


def get_circuit_breaker(request: Request) -> CircuitBreakerService:
    return CircuitBreakerService(RedisCircuitBreakerStore(request.app.state.redis))


@router.post("/generate-text")
async def generate_text(
    cb_service: CircuitBreakerService = Depends(get_circuit_breaker),
) -> dict[str, object]:
    async def fetch_response() -> httpx.Response:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://api.example.com/generate")
            response.raise_for_status()
            return response

    try:
        response = await cb_service.call("text-generation-provider", fetch_response)
    except (CircuitBreakerBackendError, CircuitBreakerOpenError, httpx.HTTPError) as exc:
        msg = "External API call failed"
        raise ServiceUnavailableException(msg) from exc

    data = response.json()
    return {"generated_text": data["text"]}
