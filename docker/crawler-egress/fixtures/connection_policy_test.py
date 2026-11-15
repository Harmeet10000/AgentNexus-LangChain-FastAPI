"""Connection-time policy evidence for the crawler egress boundary.

Every case is driven through the deployment's proxy, because the proxy is the
component that resolves, classifies, and dials. The runner asserts what the
boundary must do — permit the public fixture and refuse every non-public
destination — and finally proves that the denied fixture never received a
request at all.

Exit code 0 means every case held; 1 means the boundary failed one.
"""

from __future__ import annotations

import http.client
import os
import sys
from urllib.parse import urlparse

from playwright.sync_api import Request, Response, sync_playwright

PROXY_HOST = os.environ.get("EGRESS_PROXY_HOST", "crawler-egress-proxy-test")
PROXY_PORT = int(os.environ.get("EGRESS_PROXY_PORT", "4750"))
HITS_LOG = os.environ.get("FIXTURE_HITS_LOG", "/tmp/fixture-hits.log")
PUBLIC_HOST = "public.fixture.internal"
PRIVATE_HOST = "private.fixture.internal"
MIXED_HOST = "mixed.fixture.internal"
CHANGING_HOST = "changing.fixture.internal"
FIXTURE_PORT = int(os.environ.get("FIXTURE_HTTP_PORT", "8080"))

DENIAL_STATUS = 407
DENIAL_HEADER = "x-smokescreen-error"


class EvidenceFailure(Exception):
    """One boundary case did not hold."""


def _proxied_get(url: str, timeout: float = 15.0) -> tuple[int, dict[str, str], str]:
    """Perform one GET through the proxy and return status, headers, and body."""
    parsed = urlparse(url)
    connection = http.client.HTTPConnection(PROXY_HOST, PROXY_PORT, timeout=timeout)
    try:
        connection.request("GET", url, headers={"Host": parsed.netloc, "Accept": "*/*"})
        response = connection.getresponse()
        body = response.read().decode("utf-8", errors="replace")
        headers = {name.lower(): value for name, value in response.getheaders()}
        return response.status, headers, body
    finally:
        connection.close()


def _follow(url: str, max_hops: int = 3) -> tuple[int, dict[str, str], str, list[str]]:
    """Follow redirects through the proxy and report the final exchange."""
    chain: list[str] = []
    current = url
    for _hop in range(max_hops):
        status, headers, body = _proxied_get(current)
        chain.append(f"{current} -> {status}")
        location = headers.get("location")
        if status in {301, 302, 303, 307, 308} and location:
            current = location
            continue
        return status, headers, body, chain
    raise EvidenceFailure(f"redirect chain did not settle: {chain}")


def _require_denial(url: str, chain: list[str] | None = None) -> None:
    """Assert the boundary refused this destination with a policy denial."""
    status, headers, body = _proxied_get(url)
    if status != DENIAL_STATUS:
        raise EvidenceFailure(f"{url} was not denied (status {status})")
    if DENIAL_HEADER not in headers:
        raise EvidenceFailure(f"{url} was denied without the proxy error header")


def _require_permitted(url: str) -> None:
    """Assert the boundary permitted this destination."""
    status, headers, body = _proxied_get(url)
    if status != 200:
        raise EvidenceFailure(f"{url} was not permitted (status {status})")


def _private_hits() -> int:
    """Count the requests the denied fixture received."""
    if not os.path.exists(HITS_LOG):
        return 0
    with open(HITS_LOG, encoding="utf-8") as handle:
        return len([line for line in handle if line.strip()])


def case_public_seed() -> None:
    _require_permitted(f"http://{PUBLIC_HOST}:{FIXTURE_PORT}/")


def case_private_seed() -> None:
    _require_denial(f"http://{PRIVATE_HOST}:{FIXTURE_PORT}/private")


def case_private_redirect() -> None:
    """A permitted page that redirects to a denied destination."""
    status, headers, body, chain = _follow(f"http://{PUBLIC_HOST}:{FIXTURE_PORT}/redirect-private")
    if status != DENIAL_STATUS:
        raise EvidenceFailure(f"the private redirect was not denied: {chain}")
    if DENIAL_HEADER not in headers:
        raise EvidenceFailure("the private redirect was denied without the proxy error header")


def _exercise_browser_subresource(page_url: str, private_url: str) -> None:
    """Load a public page in Chromium and observe its denied private asset."""
    attempted = False
    denial_statuses: list[int] = []
    request_failures: list[str] = []

    def on_request(request: Request) -> None:
        nonlocal attempted
        if request.url == private_url:
            attempted = True

    def on_response(response: Response) -> None:
        if response.url == private_url:
            denial_statuses.append(response.status)

    def on_request_failed(request: Request) -> None:
        if request.url == private_url:
            request_failures.append(request.failure or "browser request failed")

    proxy_url = f"http://{PROXY_HOST}:{PROXY_PORT}"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True,
                proxy={"server": proxy_url},
                args=["--no-sandbox"],
            )
            try:
                page = browser.new_page()
                page.on("request", on_request)
                page.on("response", on_response)
                page.on("requestfailed", on_request_failed)
                navigation = page.goto(page_url, wait_until="networkidle", timeout=15_000)
                if navigation is None or navigation.status != 200:
                    status = navigation.status if navigation is not None else "no response"
                    raise EvidenceFailure(
                        f"the public subresource page was not permitted (status {status})"
                    )
            finally:
                browser.close()
    except EvidenceFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - standalone evidence must fail with context
        raise EvidenceFailure(f"browser subresource exercise failed: {exc}") from exc

    if not attempted:
        raise EvidenceFailure("the browser never attempted the private subresource")
    if DENIAL_STATUS not in denial_statuses and not request_failures:
        raise EvidenceFailure(
            "the browser private subresource had no proxy-denial or request-failure evidence"
        )


def case_private_subresource() -> None:
    """A subresource of a permitted page that points at a denied destination."""
    page_url = f"http://{PUBLIC_HOST}:{FIXTURE_PORT}/subresource-page"
    private_url = f"http://{PRIVATE_HOST}:{FIXTURE_PORT}/private"
    _exercise_browser_subresource(page_url, private_url)


def case_changing_dns() -> None:
    """The answer for one name changes between two connections."""
    first_status, _headers, _body = _proxied_get(f"http://{CHANGING_HOST}:{FIXTURE_PORT}/")
    if first_status != 200:
        raise EvidenceFailure(f"the first answer for {CHANGING_HOST} was not permitted")
    second_status, _headers, _body = _proxied_get(f"http://{CHANGING_HOST}:{FIXTURE_PORT}/")
    if second_status != DENIAL_STATUS:
        raise EvidenceFailure(
            f"the changed answer for {CHANGING_HOST} was not denied (status {second_status})"
        )


def case_mixed_answers() -> None:
    """One answer carries a permitted and a denied address."""
    status, _headers, _body = _proxied_get(f"http://{MIXED_HOST}:{FIXTURE_PORT}/")
    if status != 200:
        raise EvidenceFailure(f"the permitted address of {MIXED_HOST} was not selected")


def case_denied_fixture_untouched() -> None:
    hits = _private_hits()
    if hits:
        raise EvidenceFailure(f"the denied fixture received {hits} request(s)")


CASES = (
    case_public_seed,
    case_private_seed,
    case_private_redirect,
    case_private_subresource,
    case_changing_dns,
    case_mixed_answers,
    case_denied_fixture_untouched,
)


def main() -> int:
    failures: list[str] = []
    for case in CASES:
        try:
            case()
        except (EvidenceFailure, OSError, http.client.HTTPException) as exc:
            failures.append(f"{case.__name__}: {exc}")
            print(f"FAIL {case.__name__}: {exc}", flush=True)
        else:
            print(f"ok   {case.__name__}", flush=True)

    if failures:
        print(f"\n{len(failures)} crawler egress policy case(s) failed", flush=True)
        return 1
    print(f"\nall {len(CASES)} crawler egress policy cases held", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
