"""Rendered-shape assertions for the crawler's deterministic egress evidence.

The runtime evidence for the connection-time policy runs the Compose profile
``crawler-egress-test``. These assertions pin the shape that makes that
evidence meaningful: the test containers have no route of their own, the
proxy under test is the only path out, and the permitted range exists only
in the test policy.
"""

import runpy
import threading
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
PROFILE = "crawler-egress-test"
FIXTURES = {
    "crawler-egress-fixture-public": "198.18.0.10",
    "crawler-egress-fixture-private": "10.10.0.10",
    "crawler-egress-fixture-dns": "198.18.0.53",
}


def _compose(profile: str | None = None) -> dict[str, object]:
    loaded = yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    if profile is None:
        return loaded
    services = loaded["services"]
    assert isinstance(services, dict)
    return {
        "services": {
            name: service
            for name, service in services.items()
            if profile in (service.get("profiles") or [])  # type: ignore[union-attr]
        },
        "networks": loaded["networks"],
    }


def _network_names(service: dict[str, object]) -> set[str]:
    networks = service.get("networks", [])
    if isinstance(networks, dict):
        return set(networks)
    assert isinstance(networks, list)
    return {str(item) for item in networks}


def test_every_evidence_service_is_opt_in() -> None:
    services = _compose()["services"]
    assert isinstance(services, dict)

    evidence = {
        name
        for name in services
        if name.startswith("crawler-egress-")
        and name not in {"crawler-egress-proxy", "crawler-egress-proxy-test"}
    }
    assert evidence

    for name in evidence:
        service = services[name]
        assert isinstance(service, dict)
        assert PROFILE in (service.get("profiles") or [])


def test_evidence_containers_have_no_route_of_their_own() -> None:
    services = _compose(PROFILE)["services"]
    assert isinstance(services, dict)

    for name in ("crawler-egress-tests", "crawler-egress-bypass-tests"):
        service = services[name]
        assert isinstance(service, dict)
        assert _network_names(service) == {"crawler-control"}


def test_the_proxy_under_test_is_the_only_path_out() -> None:
    services = _compose(PROFILE)["services"]
    assert isinstance(services, dict)

    proxy = services["crawler-egress-proxy-test"]
    assert isinstance(proxy, dict)
    assert _network_names(proxy) == {"crawler-control", "crawler-egress-test"}
    assert proxy["dns"] == ["198.18.0.53"]

    on_fixture_network = {
        name
        for name, service in services.items()
        if "crawler-egress-test" in _network_names(service)
    }
    assert on_fixture_network == {
        "crawler-egress-proxy-test",
        "crawler-egress-fixture-public",
        "crawler-egress-fixture-private",
        "crawler-egress-fixture-dns",
    }


def test_permitted_and_denied_fixtures_differ_by_address_policy() -> None:
    services = _compose(PROFILE)["services"]
    networks = _compose(PROFILE)["networks"]
    assert isinstance(services, dict)
    assert isinstance(networks, dict)

    egress_network = networks["crawler-egress-test"]
    assert isinstance(egress_network, dict)
    subnets = {
        entry["subnet"]
        for entry in egress_network["ipam"]["config"]  # type: ignore[index]
    }
    assert subnets == {"198.18.0.0/24", "10.10.0.0/24"}

    for name, address in FIXTURES.items():
        service = services[name]
        assert isinstance(service, dict)
        assert service["networks"]["crawler-egress-test"]["ipv4_address"] == address  # type: ignore[index]


def test_the_permitted_range_exists_only_in_the_test_policy() -> None:
    production = (ROOT / "docker" / "crawler-egress" / "smokescreen.production.yaml").read_text(
        encoding="utf-8"
    )
    test_policy = (ROOT / "docker" / "crawler-egress" / "smokescreen.test.yaml").read_text(
        encoding="utf-8"
    )

    assert "198.18.0.0/24" not in production
    assert "unsafe_allow_private_ranges: false" in production
    assert "198.18.0.0/24" in test_policy
    assert "unsafe_allow_private_ranges: false" in test_policy


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixtures_are_pinned_and_carry_no_credentials(name: str) -> None:
    services = _compose(PROFILE)["services"]
    assert isinstance(services, dict)

    service = services[name]
    assert isinstance(service, dict)
    image = service["image"]
    assert isinstance(image, str)
    assert "@sha256:" in image
    assert "environment" not in service


def test_private_fixture_records_every_received_request(tmp_path: Path) -> None:
    """Prove the zero-hit oracle changes when the private server is reached."""
    namespace = runpy.run_path(str(ROOT / "docker/crawler-egress/fixtures/fixture_server.py"))
    handler = namespace["_FixtureHandler"]
    handler.do_GET.__globals__["HITS_LOG"] = str(tmp_path / "private-hits.log")
    server = namespace["_FixtureServer"](("127.0.0.1", 0), handler)
    server.role = "private"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/private",
            timeout=2,
        ) as response:
            assert response.status == 200
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert (tmp_path / "private-hits.log").read_text(encoding="utf-8").strip() == "/private"


def test_connection_policy_runner_installs_a_real_browser() -> None:
    services = _compose(PROFILE)["services"]
    assert isinstance(services, dict)
    runner = services["crawler-egress-tests"]
    assert isinstance(runner, dict)
    build = runner["build"]
    assert isinstance(build, dict)
    assert build["dockerfile"] == "docker/crawler-egress/evidence.Dockerfile"

    dockerfile = (ROOT / str(build["dockerfile"])).read_text(encoding="utf-8")
    runner = (
        ROOT / "docker" / "crawler-egress" / "fixtures" / "connection_policy_test.py"
    ).read_text(encoding="utf-8")
    assert "FROM python:3.12-slim@sha256:" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "playwright install --with-deps chromium" in dockerfile
    assert "with sync_playwright()" in runner
    assert 'page.goto(page_url, wait_until="networkidle"' in runner


def test_subresource_case_uses_browser_navigation() -> None:
    """The subresource proof must use Chromium, not replay the asset with HTTP."""
    namespace = runpy.run_path(
        str(ROOT / "docker/crawler-egress/fixtures/connection_policy_test.py")
    )
    page_url = "http://public.fixture.internal:8080/subresource-page"
    private_url = "http://private.fixture.internal:8080/private"
    navigations: list[tuple[str, str]] = []

    def exercise_browser(observed_page_url: str, observed_private_url: str) -> None:
        navigations.append((observed_page_url, observed_private_url))

    case = namespace["case_private_subresource"]
    case.__globals__["_exercise_browser_subresource"] = exercise_browser
    case.__globals__["_proxied_get"] = lambda *_args, **_kwargs: pytest.fail(
        "the subresource case must not replay the asset with raw HTTP"
    )
    case()

    assert navigations == [(page_url, private_url)]
