"""Extract the OpenAPI snapshot for the Mintlify docs site (task 2.1).

Builds the real FastAPI application and fetches ``/swagger.json`` through an
in-process ASGI transport, validates the document shape, and writes it to
``docs-site/openapi.json``.

Two deliberate deviations from the original task sketch, both measured:

* **No subprocess server.** uvicorn-under-subprocess made readiness polling
  flaky (slow first-boot imports; a lifespan that retries a graph connection
  whose host does not resolve in CI-like environments). The ASGI transport
  exercises the same application object without opening a socket.
* **Lifespan never runs.** The docs build must not require live Postgres,
  Redis, Neo4j or RabbitMQ. Route tables are fully populated at app
  construction, which is what API documentation needs.

The app serves its schema at ``/swagger.json`` under production hardening —
see ``src/app/main.py``.

This is a standalone operator script: asserts and prints are its interface.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from httpx import ASGITransport, AsyncClient

OUT = Path(__file__).resolve().parents[1] / "openapi.json"
SPEC_PATH = "/swagger.json"


async def _fetch() -> dict[str, Any]:
    from app.main import create_app

    app = create_app()
    transport = ASGITransport(app=app)  # type: ignore[arg-type]
    async with AsyncClient(transport=transport, base_url="http://docs.local") as client:
        response = await client.get(SPEC_PATH)
        response.raise_for_status()
        return response.json()


def main() -> int:
    spec = asyncio.run(_fetch())

    # Shape validation: OpenAPI 3.x documents an info block and paths.
    assert spec.get("openapi", "").startswith("3."), "not an OpenAPI 3.x document"
    paths = spec.get("paths")
    assert isinstance(paths, dict) and paths, "no paths documented"

    OUT.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(spec['paths'])} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
