"""Fail-closed capability preflight for the crawler egress boundary.

The crawler worker may only run when the deployment's proxy both permits an
operator-controlled public probe and explicitly denies a safe loopback probe.
A TCP-open proxy is not enough: a misconfigured policy can still expose
private infrastructure, and a reachable-but-broken proxy must not silently
degrade the worker to direct networking.

Smokescreen answers a policy denial with ``407`` and the ``X-Smokescreen-Error``
header, so an explicit denial is verifiable rather than inferred from a
generic failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
from pydantic import BaseModel, ConfigDict

from app.features.crawler.application.egress import (
    POLICY_DENIAL_HEADER,
    POLICY_DENIAL_STATUS,
)
from app.observability.logging import logger

if TYPE_CHECKING:
    from app.features.crawler.application.egress import (
        CrawlerEgressPolicy,
    )

#: Smokescreen's status for a request rejected by its address policy.
SMOKESCREEN_DENY_STATUS = POLICY_DENIAL_STATUS

#: Smokescreen's header carrying the denial reason.
SMOKESCREEN_ERROR_HEADER = POLICY_DENIAL_HEADER

#: Smokescreen's denial message template, used when the header is absent.
DENY_BODY_MARKER = "Egress proxying is denied"


class EgressPreflightResult(BaseModel):
    """Outcome of one boundary preflight."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    reason: str
    public_probe_status: int | None = None
    denial_status: int | None = None


def is_explicit_policy_denial(response: httpx.Response) -> bool:
    """Return True when the proxy answered with a verifiable policy denial."""
    if response.status_code not in {SMOKESCREEN_DENY_STATUS, 403}:
        return False
    if SMOKESCREEN_ERROR_HEADER in response.headers:
        return True
    try:
        body = response.text
    except (UnicodeDecodeError, httpx.ResponseNotRead):  # pragma: no cover - defensive
        return False
    return DENY_BODY_MARKER in body


async def _probe_policy(
    policy: CrawlerEgressPolicy,
    client: httpx.AsyncClient,
) -> EgressPreflightResult:
    """Prove the boundary permits the public probe and denies the loopback one."""
    try:
        public = await client.get(policy.public_probe_url)
    except httpx.HTTPError as exc:
        exc.add_note("operation=crawler_egress_preflight, probe=public")
        logger.bind(error_type=type(exc).__name__).exception("crawler_egress_public_probe_failed")
        return EgressPreflightResult(
            available=False,
            reason=f"the public probe could not reach the egress boundary: {type(exc).__name__}",
        )
    if public.status_code >= 400:
        return EgressPreflightResult(
            available=False,
            reason=f"the public probe returned {public.status_code} through the egress boundary",
            public_probe_status=public.status_code,
        )

    try:
        denied = await client.get(policy.loopback_probe_url)
    except httpx.HTTPError as exc:
        exc.add_note("operation=crawler_egress_preflight, probe=loopback")
        logger.bind(error_type=type(exc).__name__).exception("crawler_egress_loopback_probe_failed")
        return EgressPreflightResult(
            available=False,
            reason=(f"the loopback probe did not receive a policy denial: {type(exc).__name__}"),
            public_probe_status=public.status_code,
        )
    if not is_explicit_policy_denial(denied):
        return EgressPreflightResult(
            available=False,
            reason=(
                "the loopback probe was not explicitly denied by the egress policy "
                f"(status {denied.status_code})"
            ),
            public_probe_status=public.status_code,
            denial_status=denied.status_code,
        )

    return EgressPreflightResult(
        available=True,
        reason="the egress policy permits the public probe and denies the loopback probe",
        public_probe_status=public.status_code,
        denial_status=denied.status_code,
    )


async def run_egress_preflight(
    policy: CrawlerEgressPolicy,
    *,
    client: httpx.AsyncClient | None = None,
) -> EgressPreflightResult:
    """Prove the configured boundary behaves as the crawler depends on.

    The probe never falls back to a direct connection: an unreachable proxy is
    an unavailable crawler, not a licence to browse directly.
    """
    if client is not None:
        return await _probe_policy(policy, client)

    async with httpx.AsyncClient(
        proxy=policy.proxy_url,
        trust_env=False,
        timeout=policy.preflight_timeout_seconds,
        follow_redirects=False,
    ) as owned_client:
        result = await _probe_policy(policy, owned_client)

    logger.bind(
        available=result.available,
        public_probe_status=result.public_probe_status,
        denial_status=result.denial_status,
    ).info("crawler_egress_preflight_completed")
    return result
