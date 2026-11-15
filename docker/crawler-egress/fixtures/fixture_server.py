"""Deterministic HTTP and DNS fixtures for the crawler egress evidence.

The public fixture answers on ``198.18.0.0/24`` — the only range the test
Smokescreen policy allows. The private fixture answers on ``10.10.0.0/24`` and
records every request it receives, so "the denied fixture was never reached"
is a fact about the fixture rather than an expectation of the test.

The DNS fixture answers for a fixed internal zone, including one name whose
answer changes between queries and one name whose answer mixes a permitted
and a denied address. Those two names are what make the connection-time
policy observable: the proxy resolves again for every connection and selects
an address it classified as permitted.

Usage: ``fixture_server.py {public|private|dns}``
"""

from __future__ import annotations

import argparse
import http.server
import os
import socket
import socketserver
import struct
import sys
import threading

PUBLIC_IP = "198.18.0.10"
PRIVATE_IP = "10.10.0.10"
HTTP_PORT = int(os.environ.get("FIXTURE_HTTP_PORT", "8080"))
DNS_PORT = int(os.environ.get("FIXTURE_DNS_PORT", "53"))
HITS_LOG = os.environ.get("FIXTURE_HITS_LOG", "/tmp/fixture-hits.log")

PRIVATE_HOST = "private.fixture.internal"
PUBLIC_HOST = "public.fixture.internal"

STATIC_ZONE: dict[str, list[str]] = {
    PUBLIC_HOST: [PUBLIC_IP],
    "redirect.fixture.internal": [PUBLIC_IP],
    "subresource.fixture.internal": [PUBLIC_IP],
    PRIVATE_HOST: [PRIVATE_IP],
    "mixed.fixture.internal": [PUBLIC_IP, PRIVATE_IP],
}
CHANGING_HOST = "changing.fixture.internal"

_query_lock = threading.Lock()
_changing_queries = 0


def _record_hit(path: str) -> None:
    """Append one line per received request so the test can count arrivals."""
    with _lock_hits:
        with open(HITS_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{path}\n")


_lock_hits = threading.Lock()


def _parse_question(payload: bytes) -> tuple[str, int, int]:
    """Return the queried name, the query type, and the offset past the question."""
    labels: list[str] = []
    offset = 12
    while True:
        length = payload[offset]
        offset += 1
        if length == 0:
            break
        labels.append(payload[offset : offset + length].decode("ascii"))
        offset += length
    qtype, _qclass = struct.unpack("!HH", payload[offset : offset + 4])
    return ".".join(labels).lower(), qtype, offset + 4


def _encode_name(name: str) -> bytes:
    encoded = bytearray()
    for label in name.split("."):
        encoded.append(len(label))
        encoded += label.encode("ascii")
    encoded.append(0)
    return bytes(encoded)


def _build_response(query: bytes, answers: list[str]) -> bytes:
    """Build a DNS response for one A-record question."""
    _name, _qtype, question_end = _parse_question(query)
    header = query[:2] + b"\x81\x80"
    header += struct.pack("!HHHH", 1, len(answers), 0, 0)
    body = query[12:question_end]
    for address in answers:
        body += b"\xc0\x0c"
        body += struct.pack("!HHIH", 1, 1, 5, 4)
        body += socket.inet_aton(address)
    return header + body


def _answers_for(name: str) -> list[str]:
    """Resolve one fixture name, honouring the changing-answer rule."""
    global _changing_queries  # noqa: PLW0603 — the fixture zone is process state
    if name in STATIC_ZONE:
        return STATIC_ZONE[name]
    if name == CHANGING_HOST:
        with _query_lock:
            _changing_queries += 1
            first = _changing_queries == 1
        return [PUBLIC_IP] if first else [PRIVATE_IP]
    return []


class _DnsHandler(socketserver.BaseRequestHandler):
    """Answer one DNS question."""

    def handle(self) -> None:
        data, sock = self.request
        try:
            name, _qtype, _end = _parse_question(data)
        except (IndexError, struct.error):
            return
        try:
            sock.sendto(_build_response(data, _answers_for(name)), self.client_address)
        except OSError:
            return


class _DnsServer(socketserver.ThreadingUDPServer):
    allow_reuse_address = True
    daemon_threads = True


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    """Serve the fixture's HTTP surface."""

    protocol_version = "HTTP/1.1"
    role = "public"

    def do_GET(self) -> None:  # noqa: N802 — stdlib naming
        if self.role == "private":
            _record_hit(self.path)
            self._respond(200, b"PRIVATE FIXTURE REACHED\n")
            return
        if self.path == "/redirect-private":
            self.send_response(302)
            self.send_header("Location", f"http://{PRIVATE_HOST}:{HTTP_PORT}/private")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if self.path == "/subresource-page":
            body = (
                "<html><body>"
                f'<img src="http://{PRIVATE_HOST}:{HTTP_PORT}/private">'
                "</body></html>"
            ).encode()
            self._respond(200, body)
            return
        if self.path == "/subresource":
            self._respond(200, b"public subresource\n")
            return
        self._respond(200, b"public fixture\n")

    def _respond(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FixtureServer(http.server.ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True
    role = "public"


def _serve_http(role: str) -> None:
    server = _FixtureServer(("0.0.0.0", HTTP_PORT), _FixtureHandler)
    server.role = role
    server.serve_forever()


def _serve_dns() -> None:
    server = _DnsServer(("0.0.0.0", DNS_PORT), _DnsHandler)
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("role", choices=["public", "private", "dns"])
    arguments = parser.parse_args()

    if arguments.role == "dns":
        _serve_dns()
    else:
        _serve_http(arguments.role)
    return 0


if __name__ == "__main__":
    sys.exit(main())
