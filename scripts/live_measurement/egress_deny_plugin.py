"""Dev-egress deny plugin for proxy.py (crawler egress boundary contract).

Denies every destination with a body containing the exact marker the
crawler's egress preflight requires ("Egress proxying is denied"), and
allows everything else to pass through.  For the crawler boundary the
deployed policy is: public destinations allowed (public probe passes),
loopback/private destinations denied with the marker (loopback probe
receives the explicit policy denial).

This plugin denies a destination when its resolved host is loopback or
private (RFC1918 + link-local), which covers the default loopback probe.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Any

from proxy.http.parser import HttpParser
from proxy.http.proxy import HttpProxyBasePlugin


DENY_BODY = b"Egress proxying is denied"
DENY_PRIVATE_HOSTS = True


def _is_private_host(host: str) -> bool:
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        except OSError:
            return True  # unresolvable: deny (fail closed)
        for entry in resolved:
            candidate = entry[4][0]
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                return True
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved


class EgressBoundaryPlugin(HttpProxyBasePlugin):
    """Deny private/loopback destinations with the marker body; allow the rest."""

    def before_upstream_connection(self, request: HttpParser) -> HttpParser | None:
        host = (request.host or b"").decode("utf-8", "ignore")
        if DENY_PRIVATE_HOSTS and _is_private_host(host):
            self._deny()
            return None
        return request

    def handle_request(self, request: HttpParser) -> HttpParser | None:
        return request

    def handle_response(self, response: Any) -> Any:
        return response

    def _deny(self) -> None:
        self.client.flush()
        self.client.queue(
            b"HTTP/1.1 403 Forbidden\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: " + str(len(DENY_BODY)).encode() + b"\r\n"
            b"Connection: close\r\n"
            b"\r\n" + DENY_BODY
        )
