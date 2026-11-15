"""Non-bypass evidence for the crawler worker's network namespace.

The worker must have no general route to public destinations: the proxy is the
only way out. This runner lives in the worker's namespace and proves that
removing or ignoring the proxy configuration does not restore a route, while
the internal control plane stays reachable.

Exit code 0 means the namespace held; 1 means a bypass was possible.
"""

from __future__ import annotations

import http.client
import os
import socket
import sys

PROXY_HOST = os.environ.get("EGRESS_PROXY_HOST", "crawler-egress-proxy-test")
PROXY_PORT = int(os.environ.get("EGRESS_PROXY_PORT", "4750"))
PUBLIC_FIXTURE_IP = os.environ.get("PUBLIC_FIXTURE_IP", "198.18.0.10")
PRIVATE_FIXTURE_IP = os.environ.get("PRIVATE_FIXTURE_IP", "10.10.0.10")
FIXTURE_PORT = int(os.environ.get("FIXTURE_HTTP_PORT", "8080"))
PUBLIC_HOST = os.environ.get("PUBLIC_FIXTURE_HOST", "public.fixture.internal")
BROKER_HOST = os.environ.get("BROKER_HOST", "rabbitmq")
BROKER_PORT = int(os.environ.get("BROKER_PORT", "5672"))
RELAY_HOST = os.environ.get("RELAY_HOST", "")
RELAY_PORT = int(os.environ.get("RELAY_PORT", "6379"))
INTERNET_PROBE = os.environ.get("INTERNET_PROBE", "1.1.1.1")


class EvidenceFailure(Exception):
    """The namespace did not hold."""


def _tcp_connect(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def case_no_direct_route_to_public_fixture() -> None:
    if _tcp_connect(PUBLIC_FIXTURE_IP, FIXTURE_PORT):
        raise EvidenceFailure("the worker reached the public fixture directly")


def case_no_direct_route_to_private_fixture() -> None:
    if _tcp_connect(PRIVATE_FIXTURE_IP, FIXTURE_PORT):
        raise EvidenceFailure("the worker reached the private fixture directly")


def case_no_direct_route_to_the_internet() -> None:
    if _tcp_connect(INTERNET_PROBE, 443):
        raise EvidenceFailure("the worker reached the public internet directly")


def case_proxy_is_the_only_route() -> None:
    connection = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=15)
    try:
        connection.request(
            "GET",
            f"http://{PUBLIC_HOST}:{FIXTURE_PORT}/",
            headers={"Host": f"{PUBLIC_HOST}:{FIXTURE_PORT}"},
        )
        response = connection.getresponse()
        response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise EvidenceFailure(f"the proxy did not serve the permitted fixture: {exc}") from exc
    finally:
        connection.close()
    if response.status != 200:
        raise EvidenceFailure(f"the proxy returned {response.status} for the permitted fixture")


def case_internal_control_plane_reachable() -> None:
    if not _tcp_connect(BROKER_HOST, BROKER_PORT):
        raise EvidenceFailure("the local broker is not reachable from the worker namespace")


def case_relay_exposes_no_target_selection() -> None:
    """A CONNECT request must never open a tunnel through the relay."""
    if not RELAY_HOST:
        print("skip case_relay_exposes_no_target_selection: no relay material in this profile")
        return
    try:
        with socket.create_connection((RELAY_HOST, RELAY_PORT), timeout=5) as sock:
            sock.sendall(
                f"CONNECT {INTERNET_PROBE}:443 HTTP/1.1\r\nHost: {INTERNET_PROBE}:443\r\n\r\n".encode()
            )
            response = sock.recv(256).decode("utf-8", errors="replace")
    except OSError as exc:
        raise EvidenceFailure(f"the relay listener did not accept a connection: {exc}") from exc
    if "200" in response.splitlines()[0] if response.splitlines() else False:
        raise EvidenceFailure("the relay answered a CONNECT request as a tunnel")


CASES = (
    case_no_direct_route_to_public_fixture,
    case_no_direct_route_to_private_fixture,
    case_no_direct_route_to_the_internet,
    case_proxy_is_the_only_route,
    case_internal_control_plane_reachable,
    case_relay_exposes_no_target_selection,
)


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        try:
            case()
        except (EvidenceFailure, http.client.HTTPException) as exc:
            failures.append(f"{case.__name__}: {exc}")
            print(f"FAIL {case.__name__}: {exc}", flush=True)
        else:
            print(f"ok   {case.__name__}", flush=True)

    if failures:
        print(f"\n{len(failures)} non-bypass case(s) failed", flush=True)
        return 1
    print(f"\nall {len(CASES)} non-bypass cases held", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
