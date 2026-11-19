"""Launcher for the crawler's dev egress boundary proxy (proxy.py + deny plugin).

Programmatic launch instead of the console script so the plugin module
resolves from the repo root deterministically (the console script's
importer lost the PYTHONPATH context).  Contract: public destinations
forward; loopback/private destinations are denied with the marker body the
crawler's egress preflight requires.
"""

from __future__ import annotations

import sys
from typing import Any
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import proxy  # noqa: E402, PLC0415 — after sys.path is pinned
from proxy.http.proxy import HttpProxyBasePlugin  # noqa: E402, PLC0415

DENY_BODY = b"Egress proxying is denied"


def _is_private_host(host: str) -> bool:
    import ipaddress  # noqa: PLC0415

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = __import__("socket").getaddrinfo(host, None, proto=__import__("socket").IPPROTO_TCP)
        except OSError:
            return True
        for entry in resolved:
            try:
                address = ipaddress.ip_address(entry[4][0])
            except ValueError:
                continue
            if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                return True
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved


class EgressBoundaryPlugin(HttpProxyBasePlugin):
    """Deny private/loopback destinations with the marker body; allow the rest."""

    def before_upstream_connection(self, request: Any) -> Any:
        from proxy.http.parser import HttpParser  # noqa: PLC0415

        host = (request.host or b"").decode("utf-8", "ignore")
        if _is_private_host(host):
            self._deny()
            return None
        return request

    def handle_request(self, request: Any) -> Any:
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


def main() -> None:
    # proxy.py 2.4: main(**opts) — keyword form of the CLI flags.
    import ipaddress  # noqa: PLC0415

    proxy.main(
        hostname=ipaddress.ip_address("127.0.0.1"),
        port=8899,
        plugins=["scripts.live_measurement.egress_proxy.EgressBoundaryPlugin"],
    )


if __name__ == "__main__":
    main()
