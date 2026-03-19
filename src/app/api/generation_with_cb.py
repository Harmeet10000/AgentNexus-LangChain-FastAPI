from fastapi import APIRouter, Depends, Request
from httpx import AsyncClient

from app.shared.circuit_breaker.service import CircuitBreakerService
from app.utils.exceptions import ServiceUnavailableException

router = APIRouter()


def get_circuit_breaker(request: Request) -> CircuitBreakerService:
    return CircuitBreakerService(request.app.state.redis)


@router.post("/generate-text")
async def generate_text(cb_service: CircuitBreakerService = Depends(get_circuit_breaker)):
    # If the breaker is OPEN, this context manager immediately raises ServiceUnavailableException
    async with cb_service.protect(
        service_name="openai_api", failure_threshold=3, recovery_timeout_seconds=30
    ):
        # Simulate an external API call (e.g., OpenAI)
        async with AsyncClient() as client:
            response = await client.get("https://api.example.com/generate")
            response.raise_for_status()
            data = response.json()

        return {"generated_text": data["text"]}
