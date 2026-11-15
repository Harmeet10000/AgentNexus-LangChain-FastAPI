"""Typed declaration of the crawler worker's egress boundary.

The crawler worker reaches public destinations only through the deployment's
fail-closed proxy. This module owns the *declaration* of that boundary — the
proxy, the probes that prove the policy is live, the destinations allowed to
bypass it, and the browser destinations allowed at all — and rejects an
absent or unsafe combination instead of degrading to direct networking.

Nothing here performs I/O: the declaration is validated synchronously, and
the live proof of the policy lives in :mod:`app.features.crawler.adapters.proxy.preflight`.
"""

from __future__ import annotations

import ipaddress
from typing import TYPE_CHECKING
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from app.config import Settings

#: The only browser destinations the crawler may request. Anything else is
#: denied before it reaches the proxy. This is defense in depth: the proxy's
#: connection-time address policy remains the security boundary.
SUPPORTED_BROWSER_SCHEME_PORTS = frozenset({("http", 80), ("https", 443)})

#: Schemes that never leave the browser process and therefore never reach the
#: proxy. ``about:blank`` is the browser's own health-probe target.
NON_NETWORK_SCHEMES = frozenset({"about"})

#: Host names a public probe may never address.
BLOCKED_PROBE_HOSTS = frozenset(
    {
        "localhost",
        "localhost.localdomain",
        "metadata",
        "metadata.google",
        "metadata.google.internal",
        "metadata.aws.internal",
        "kubernetes.default.svc",
    }
)

#: Default value of ``CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS``.
DEFAULT_ALLOWED_SCHEME_PORTS = "http:80,https:443"

#: Smokescreen's status for a request rejected by its address policy.
POLICY_DENIAL_STATUS = 407

#: Smokescreen's header carrying the denial reason.
POLICY_DENIAL_HEADER = "X-Smokescreen-Error"

#: The bounded message a refused job reports. It never carries the proxy's
#: reason, the destination address, or any other internal network detail.
BOUNDARY_DENIED_MESSAGE = "The crawler egress boundary refused the destination"

#: Failure markers that describe a *transient* boundary outage. A refused
#: destination is not in this set: it is a terminal, truthful outcome.
BOUNDARY_OUTAGE_MARKERS = (
    "ERR_PROXY_CONNECTION_FAILED",
    "ERR_NAME_NOT_RESOLVED",
    "ERR_INTERNET_DISCONNECTED",
    "ERR_NETWORK_CHANGED",
    "ERR_CONNECTION_RESET",
    "ERR_CONNECTION_CLOSED",
    "ERR_CONNECTION_TIMED_OUT",
    "ERR_CONNECTION_REFUSED",
    "ERR_SSL_PROTOCOL_ERROR",
    "ERR_CERT_",
)


class CrawlerEgressConfigError(ValueError):
    """The crawler egress boundary is absent, incomplete, or unsafe."""


class CrawlerRedisRelaySettings(BaseModel):
    """Fixed-destination managed-Redis relay the crawler worker connects to."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    host: str = Field(min_length=1)
    port: int = Field(ge=1, le=65535)
    ca_file: str = Field(min_length=1)
    cert_file: str = Field(min_length=1)
    key_file: str = Field(min_length=1)

    @property
    def endpoint(self) -> str:
        """Human-readable relay endpoint, safe to log."""
        return f"{self.host}:{self.port}"


class CrawlerEgressPolicy(BaseModel):
    """The resolved, validated egress boundary the crawler worker must use."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    proxy_url: str = Field(min_length=1)
    public_probe_url: str = Field(min_length=1)
    loopback_probe_url: str = Field(min_length=1)
    preflight_timeout_seconds: float = Field(gt=0, le=60)
    allowed_scheme_ports: dict[str, tuple[int, ...]]
    bypass_destinations: tuple[str, ...] = ()
    redis_relay: CrawlerRedisRelaySettings | None = None
    external_processing_enabled: bool = False

    def is_browser_target_allowed(self, scheme: str, port: int) -> bool:
        """Return True when the browser may request this scheme/port pair."""
        allowed_ports = self.allowed_scheme_ports.get(scheme.lower())
        return allowed_ports is not None and port in allowed_ports

    def is_bypass_destination(self, host: str) -> bool:
        """Return True when this host is allowed to skip the HTTP proxy."""
        return host.lower() in self.bypass_destinations


def boundary_denial_message(
    status_code: int | None,
    headers: Mapping[str, str] | None,
) -> str | None:
    """Return a bounded message when the boundary refused the request.

    A refused plain-HTTP request still reaches the browser as a response, so
    the refusal is recognised from the response itself rather than assumed
    from an error. The proxy's own error header is the reliable signal: it
    distinguishes a boundary refusal from an origin that returned the same
    status itself.
    """
    if headers is not None:
        for name in headers:
            if name.lower() == POLICY_DENIAL_HEADER.lower():
                return BOUNDARY_DENIED_MESSAGE
    if status_code == POLICY_DENIAL_STATUS:
        return BOUNDARY_DENIED_MESSAGE
    return None


def is_boundary_outage(message: str) -> bool:
    """Return True when a failure message describes a transient outage."""
    return any(marker in message for marker in BOUNDARY_OUTAGE_MARKERS)


def _require_http_url(value: str | None, name: str) -> str:
    """Return an absolute http(s) URL or fail closed."""
    if value is None or not value.strip():
        message = f"{name} must be configured"
        raise CrawlerEgressConfigError(message)
    parsed = urlparse(value.strip())
    if parsed.scheme.lower() not in {"http", "https"}:
        message = f"{name} must use http or https"
        raise CrawlerEgressConfigError(message)
    if not parsed.hostname:
        message = f"{name} must contain a host"
        raise CrawlerEgressConfigError(message)
    return value.strip()


def _host_is_public_literal(host: str) -> bool:
    """Return True when the host is an IP literal that is publicly routable."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_global and not address.is_multicast


def _require_public_probe(url: str) -> None:
    """Reject a probe that could point back at internal infrastructure."""
    host = urlparse(url).hostname or ""
    if host.lower() in BLOCKED_PROBE_HOSTS:
        message = "CRAWLER_EGRESS_PUBLIC_PROBE_URL must address a public destination"
        raise CrawlerEgressConfigError(message)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return
    if not address.is_global or address.is_multicast:
        message = "CRAWLER_EGRESS_PUBLIC_PROBE_URL must address a public destination"
        raise CrawlerEgressConfigError(message)


def _require_loopback_probe(url: str) -> None:
    """Reject a negative probe that is not a safe loopback destination."""
    host = urlparse(url).hostname or ""
    if host.lower() == "localhost":
        return
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        message = "CRAWLER_EGRESS_LOOPBACK_PROBE_URL must address a loopback destination"
        raise CrawlerEgressConfigError(message) from None
    if not address.is_loopback:
        message = "CRAWLER_EGRESS_LOOPBACK_PROBE_URL must address a loopback destination"
        raise CrawlerEgressConfigError(message)


def parse_allowed_scheme_ports(raw: str) -> dict[str, tuple[int, ...]]:
    """Parse ``http:80,https:443`` into the browser's allowed destination set."""
    parsed: dict[str, tuple[int, ...]] = {}
    for entry in raw.split(","):
        item = entry.strip().lower()
        if not item:
            continue
        scheme, separator, port_text = item.partition(":")
        if not separator or not port_text.isdigit():
            message = f"CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS entry '{entry}' must be scheme:port"
            raise CrawlerEgressConfigError(message)
        port = int(port_text)
        if (scheme, port) not in SUPPORTED_BROWSER_SCHEME_PORTS:
            message = f"CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS entry '{entry}' is not a supported destination"
            raise CrawlerEgressConfigError(message)
        existing = parsed.get(scheme, ())
        parsed[scheme] = (*existing, port)
    if not parsed:
        message = "CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS must allow at least one destination"
        raise CrawlerEgressConfigError(message)
    return parsed


def parse_bypass_destinations(raw: str) -> tuple[str, ...]:
    """Parse the minimal internal bypass list, rejecting anything public.

    Only single-label service names and non-public IP literals may bypass the
    proxy: a dotted name could be a public domain, and a public IP literal
    would hand the worker an unmetered route out of the boundary.
    """
    destinations: list[str] = []
    for entry in raw.split(","):
        item = entry.strip().lower()
        if not item:
            continue
        if "*" in item:
            message = f"CRAWLER_EGRESS_NO_PROXY entry '{entry}' must not be a wildcard"
            raise CrawlerEgressConfigError(message)
        try:
            address = ipaddress.ip_address(item)
        except ValueError:
            if "://" in item or "/" in item or ":" in item:
                message = f"CRAWLER_EGRESS_NO_PROXY entry '{entry}' must be a bare host"
                raise CrawlerEgressConfigError(message) from None
            if "." in item:
                message = (
                    f"CRAWLER_EGRESS_NO_PROXY entry '{entry}' must be an internal service name"
                )
                raise CrawlerEgressConfigError(message) from None
        else:
            if address.is_global:
                message = f"CRAWLER_EGRESS_NO_PROXY entry '{entry}' must not be a public address"
                raise CrawlerEgressConfigError(message)
        destinations.append(item)
    return tuple(destinations)


def _build_redis_relay(settings: Settings) -> CrawlerRedisRelaySettings | None:
    """Resolve the fixed-destination relay, when the deployment declares one."""
    host = settings.CRAWLER_REDIS_RELAY_HOST
    if host is None or not host.strip():
        return None
    material = {
        "CRAWLER_REDIS_RELAY_CA_FILE": settings.CRAWLER_REDIS_RELAY_CA_FILE,
        "CRAWLER_REDIS_RELAY_CERT_FILE": settings.CRAWLER_REDIS_RELAY_CERT_FILE,
        "CRAWLER_REDIS_RELAY_KEY_FILE": settings.CRAWLER_REDIS_RELAY_KEY_FILE,
    }
    missing = [name for name, value in material.items() if value is None or not value.strip()]
    if missing:
        message = (
            "the crawler Redis relay requires deployment-issued mTLS material: "
            f"{', '.join(missing)}"
        )
        raise CrawlerEgressConfigError(message)
    return CrawlerRedisRelaySettings(
        host=host.strip(),
        port=settings.CRAWLER_REDIS_RELAY_PORT,
        ca_file=str(material["CRAWLER_REDIS_RELAY_CA_FILE"]),
        cert_file=str(material["CRAWLER_REDIS_RELAY_CERT_FILE"]),
        key_file=str(material["CRAWLER_REDIS_RELAY_KEY_FILE"]),
    )


def build_egress_policy(settings: Settings) -> CrawlerEgressPolicy:
    """Resolve the deployment's egress boundary, rejecting unsafe configuration.

    Raises:
        CrawlerEgressConfigError: The boundary is absent, incomplete, or
            would let the worker reach destinations the policy forbids.
    """
    if not settings.CRAWLER_EGRESS_ENABLED:
        message = (
            "the crawler egress boundary is disabled; the crawler worker cannot run without it"
        )
        raise CrawlerEgressConfigError(message)

    proxy_url = _require_http_url(settings.CRAWLER_EGRESS_PROXY_URL, "CRAWLER_EGRESS_PROXY_URL")
    public_probe_url = _require_http_url(
        settings.CRAWLER_EGRESS_PUBLIC_PROBE_URL, "CRAWLER_EGRESS_PUBLIC_PROBE_URL"
    )
    _require_public_probe(public_probe_url)

    loopback_probe_url = _require_http_url(
        settings.CRAWLER_EGRESS_LOOPBACK_PROBE_URL, "CRAWLER_EGRESS_LOOPBACK_PROBE_URL"
    )
    _require_loopback_probe(loopback_probe_url)

    allowed_scheme_ports = parse_allowed_scheme_ports(settings.CRAWLER_EGRESS_ALLOWED_SCHEME_PORTS)
    bypass_destinations = parse_bypass_destinations(settings.CRAWLER_EGRESS_NO_PROXY)

    return CrawlerEgressPolicy(
        proxy_url=proxy_url,
        public_probe_url=public_probe_url,
        loopback_probe_url=loopback_probe_url,
        preflight_timeout_seconds=settings.CRAWLER_EGRESS_PREFLIGHT_TIMEOUT_SECONDS,
        allowed_scheme_ports=allowed_scheme_ports,
        bypass_destinations=bypass_destinations,
        redis_relay=_build_redis_relay(settings),
        external_processing_enabled=settings.CRAWLER_EXTERNAL_PROCESSING_ENABLED,
    )


def relay_redis_url(redis_url: str, relay: CrawlerRedisRelaySettings) -> str:
    """Return ``redis_url`` re-pointed at the fixed-destination relay.

    Credentials, database selection, and every other part of the existing
    managed-Redis URL are preserved: the relay forwards the Redis protocol and
    must never become the place where Redis credentials live.
    """
    parsed = urlparse(redis_url)
    if not parsed.scheme.startswith("redis"):
        message = "the crawler Redis URL must be a redis:// or rediss:// URL"
        raise CrawlerEgressConfigError(message)
    host = relay.host
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    netloc = f"{host}:{relay.port}"
    if parsed.username is not None:
        credentials = parsed.username
        if parsed.password is not None:
            credentials = f"{credentials}:{parsed.password}"
        netloc = f"{credentials}@{netloc}"
    return parsed._replace(scheme="rediss", netloc=netloc).geturl()
