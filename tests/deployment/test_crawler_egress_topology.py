"""Rendered-shape assertions for the crawler's non-bypass topology."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
GATEWAYS = {"crawler-egress-proxy", "crawler-redis-relay"}


def _compose(name: str) -> dict[str, object]:
    loaded = yaml.safe_load((ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _network_names(service: dict[str, object]) -> set[str]:
    networks = service.get("networks", [])
    if isinstance(networks, dict):
        return set(networks)
    assert isinstance(networks, list)
    return {str(item) for item in networks}


@pytest.mark.parametrize(
    ("compose_file", "environment"),
    [("docker-compose.yml", "development"), ("docker-compose.prod.yml", "production")],
)
def test_crawler_worker_has_only_internal_control_network(
    compose_file: str,
    environment: str,
) -> None:
    compose = _compose(compose_file)
    services = compose["services"]
    networks = compose["networks"]
    assert isinstance(services, dict)
    assert isinstance(networks, dict)

    crawler = services["celery-worker-crawler"]
    assert isinstance(crawler, dict)
    assert _network_names(crawler) == {"crawler-control"}
    assert crawler["environment"]["ENVIRONMENT"] == environment
    assert networks["crawler-control"]["internal"] is True


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.prod.yml"])
def test_exactly_two_constrained_gateways_are_dual_homed(compose_file: str) -> None:
    compose = _compose(compose_file)
    services = compose["services"]
    assert isinstance(services, dict)

    dual_homed = {
        name
        for name, raw_service in services.items()
        if isinstance(raw_service, dict)
        and _network_names(raw_service) == {"crawler-control", "langchain-network"}
    }
    assert dual_homed == GATEWAYS | {"rabbitmq"}

    for gateway in GATEWAYS:
        raw_gateway = services[gateway]
        assert isinstance(raw_gateway, dict)
        assert raw_gateway.get("restart") == "unless-stopped"
        assert "healthcheck" in raw_gateway


@pytest.mark.parametrize("compose_file", ["docker-compose.yml", "docker-compose.prod.yml"])
def test_local_rabbitmq_and_redis_relay_are_explicit_internal_destinations(
    compose_file: str,
) -> None:
    services = _compose(compose_file)["services"]
    assert isinstance(services, dict)
    rabbitmq = services["rabbitmq"]
    relay = services["crawler-redis-relay"]
    worker = services["celery-worker-crawler"]
    assert isinstance(rabbitmq, dict)
    assert isinstance(relay, dict)
    assert isinstance(worker, dict)

    assert "crawler-control" in _network_names(rabbitmq)
    assert _network_names(relay) == {"crawler-control", "langchain-network"}
    assert "crawler-redis-relay" in worker["environment"]["NO_PROXY"]
    assert "rabbitmq" in worker["environment"]["NO_PROXY"]
    assert "*" not in worker["environment"]["NO_PROXY"]


def test_non_compose_acceptance_contract_requires_route_evidence() -> None:
    runbook = (ROOT / "docs" / "runbooks" / "crawler-egress-boundary.md").read_text(
        encoding="utf-8"
    )

    assert "no default public route" in runbook
    assert "deny-all egress policy" in runbook
    assert "HTTP 407" in runbook
    assert "X-Smokescreen-Error" in runbook
    assert "exactly one configured backend" in runbook
