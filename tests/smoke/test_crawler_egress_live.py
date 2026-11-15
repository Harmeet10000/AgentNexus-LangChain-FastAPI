"""Task 4.4 — opt-in live proof of the deployed crawler egress boundary.

This smoke test is disabled by default and, when explicitly enabled, proves
only the configured public probe and configured loopback denial through the
real deployment proxy without exposing internal destination details.
"""

from __future__ import annotations

import os

import pytest

from app.config import get_settings
from app.features.crawler.adapters.proxy.preflight import run_egress_preflight
from app.features.crawler.application.egress import build_egress_policy


@pytest.mark.skipif(
    os.environ.get("CRAWLER_EGRESS_LIVE_SMOKE") != "1",
    reason="set CRAWLER_EGRESS_LIVE_SMOKE=1 to run the live egress smoke",
)
@pytest.mark.asyncio
async def test_deployed_egress_allows_public_and_denies_loopback() -> None:
    policy = build_egress_policy(get_settings())
    result = await run_egress_preflight(policy)

    assert result.available is True
    assert result.public_probe_status is not None
    assert 200 <= result.public_probe_status < 400
    assert result.denial_status in {403, 407}
