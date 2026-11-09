"""Create/refresh the LIVE-measurement testing user against the running API.

Prerequisite: the API app is running locally (uvicorn) — its lifespan wires
Beanie/Postgres/Redis against the configured cloud infra.

Idempotent flow, all through the production paths:
1. POST /auth/register (duplicate registration is tolerated).
2. Seed the verification-token hash directly in Mongo (dev has no Resend key,
   so the email loop cannot deliver) — then POST /auth/verify-email with the
   raw token, which flips ``is_verified`` through the production store.
3. POST /auth/login and capture the session cookie.
4. Store everything in ``.testing-credentials.json`` (gitignored) for reuse.

Run: ``uv run python scripts/live_measurement/testing_user.py``
Requires: the API app already listening on ``BASE_URL``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import secrets
from pathlib import Path

import httpx
from pymongo import AsyncMongoClient

BASE_URL = os.environ.get("LIVE_BASE_URL", "http://127.0.0.1:5000")
API = f"{BASE_URL}/api/v1"
TESTING_EMAIL = "testing@live-measurement.dev"
TESTING_PASSWORD = "LiveMeasurement!2026"
CREDENTIALS_PATH = Path(".testing-credentials.json")


def _sha(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


async def _mongo_flip_and_identity() -> tuple[str, str]:
    """Seed the verification hash; return (user_id, mongo_collection)."""
    from app.config import get_settings

    settings = get_settings()
    client = AsyncMongoClient(settings.MONGODB_URI)
    db = client.get_database(settings.MONGODB_DB_NAME)

    collection, doc = None, None
    for name in await db.list_collection_names():
        candidate = await db[name].find_one({"email": TESTING_EMAIL})
        if candidate is not None:
            collection, doc = name, candidate
            break
    if collection is None or doc is None:
        msg = f"testing user {TESTING_EMAIL} not found in any Mongo collection"
        raise RuntimeError(msg)

    raw_token = secrets.token_urlsafe(32)
    await db[collection].update_one(
        {"_id": doc["_id"]}, {"$set": {"verification_token_hash": _sha(raw_token)}}
    )
    user_id = str(doc.get("user_id") or doc.get("_id"))
    await client.close()
    return user_id, raw_token


async def main() -> None:
    transport = httpx.AsyncClient(base_url=API, timeout=60.0)

    register = await transport.post(
        "/auth/register",
        json={
            "email": TESTING_EMAIL,
            "password": TESTING_PASSWORD,
            "full_name": "Live Measurement Testing",
        },
    )
    if register.status_code not in (200, 201, 409, 429):
        raise RuntimeError(f"register failed [{register.status_code}]: {register.text[:300]}")

    user_id, raw_token = await _mongo_flip_and_identity()
    verify = await transport.post("/auth/verify-email", json={"token": raw_token})
    if verify.status_code != 200:
        raise RuntimeError(f"verify-email failed [{verify.status_code}]: {verify.text[:300]}")

    login = await transport.post(
        "/auth/login", json={"email": TESTING_EMAIL, "password": TESTING_PASSWORD}
    )
    if login.status_code != 200:
        raise RuntimeError(f"login failed [{login.status_code}]: {login.text[:300]}")

    cookie_header = dict(transport.cookies)
    creds = {
        "email": TESTING_EMAIL,
        "password": TESTING_PASSWORD,
        "user_id": user_id,
        "base_url": BASE_URL,
        "api_prefix": "/api/v1",
        "cookies": cookie_header,
        "note": "LIVE-measurement testing principal; reuse; never commit",
    }
    CREDENTIALS_PATH.write_text(json.dumps(creds, indent=2))
    CREDENTIALS_PATH.chmod(0o600)
    await transport.aclose()
    print(f"credentials stored: {CREDENTIALS_PATH} (user_id={user_id})")


if __name__ == "__main__":
    asyncio.run(main())
