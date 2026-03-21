# this is a sample endpoint that demonstrates how to use the CircuitBreakerService to protect an external API call. In this example, we simulate a call to an external text generation API. If the API fails repeatedly, the circuit breaker will open and prevent further calls until it recovers.


from fastapi import APIRouter, Depends, Request
from httpx import AsyncClient

from app.shared.circuit_breaker.service import CircuitBreakerService
from app.utils.exceptions import ServiceUnavailableException

router = APIRouter()


def get_circuit_breaker(request: Request) -> CircuitBreakerService:
    return CircuitBreakerService(request.app.state.redis)


@router.post("/generate-text")
async def generate_text(cb_service: CircuitBreakerService = Depends(get_circuit_breaker)) -> dict:
    # If the breaker is OPEN, this context manager immediately raises ServiceUnavailableException
    async with cb_service.protect(
        service_name="openai_api", failure_threshold=3, recovery_timeout_seconds=30
    ):
        try:
            # Simulate an external API call (e.g., OpenAI)
            async with AsyncClient() as client:
                response = await client.get("https://api.example.com/generate")
                response.raise_for_status()
                data = response.json()
        except Exception as e:
            # Any exception counts as a failure and will be recorded by the circuit breaker
            raise ServiceUnavailableException("External API call failed") from e

        return {"generated_text": data["text"]}
