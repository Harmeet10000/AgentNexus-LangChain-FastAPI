"""Task 3.6 — boundary failures reach the job contract as bounded, truthful outcomes.

A proxy denial, a proxy outage, and a TLS failure are three different facts.
The first is a decision, the second and third are transients. Each must reach
the existing worker/job error handling without changing task routing,
ownership, retry semantics, or the result schema — and without leaking the
proxy's reason or any internal network detail.
"""

from __future__ import annotations

import pytest

from app.features.crawler.application.egress import (
    BOUNDARY_DENIED_MESSAGE,
    POLICY_DENIAL_HEADER,
    POLICY_DENIAL_STATUS,
    boundary_denial_message,
    is_boundary_outage,
)
from app.utils import InfrastructureException

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "status_code",
    [
        "Page.goto: net::ERR_PROXY_CONNECTION_FAILED at https://example.com/",
        "Page.goto: net::ERR_NAME_NOT_RESOLVED at https://example.com/",
        "Page.goto: net::ERR_CONNECTION_RESET at https://example.com/",
        "Page.goto: net::ERR_CERT_AUTHORITY_INVALID at https://example.com/",
        "Page.goto: net::ERR_SSL_PROTOCOL_ERROR at https://example.com/",
    ],
)
def test_transient_boundary_failures_are_classified_as_outages(status_code: str) -> None:
    assert is_boundary_outage(status_code) is True


@pytest.mark.parametrize(
    "message",
    [
        "Page.goto: net::ERR_TUNNEL_CONNECTION_FAILED at https://example.com/",
        "Crawl timeout",
        "The crawler egress boundary refused the destination",
    ],
)
def test_a_refused_destination_is_not_classified_as_an_outage(message: str) -> None:
    assert is_boundary_outage(message) is False


def test_a_refusal_is_recognised_from_the_proxy_error_header() -> None:
    message = boundary_denial_message(200, {POLICY_DENIAL_HEADER: "denied by rule 'private'"})

    assert message == BOUNDARY_DENIED_MESSAGE
    assert "denied by rule" not in message


def test_a_refusal_is_recognised_from_the_policy_denial_status() -> None:
    assert boundary_denial_message(POLICY_DENIAL_STATUS, None) == BOUNDARY_DENIED_MESSAGE


def test_an_ordinary_origin_response_is_not_a_refusal() -> None:
    assert boundary_denial_message(404, {"content-type": "text/html"}) is None
    assert boundary_denial_message(200, None) is None


def test_the_refusal_message_carries_no_internal_detail() -> None:
    message = boundary_denial_message(
        POLICY_DENIAL_STATUS,
        {POLICY_DENIAL_HEADER: "no valid IP found among resolved addresses - 10.0.0.5"},
    )

    assert message is not None
    assert "10.0.0.5" not in message
    assert "resolved addresses" not in message


def test_a_boundary_outage_raises_a_retryable_infrastructure_failure() -> None:
    from app.features.crawler.adapters.inbound.worker import tasks as crawler_jobs

    with pytest.raises(InfrastructureException) as raised:
        crawler_jobs._boundary_outage_failure("crawl-1")

    assert raised.value.status_code == 503
