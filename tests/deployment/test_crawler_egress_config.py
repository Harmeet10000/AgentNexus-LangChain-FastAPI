"""Static assertions for production and deterministic Smokescreen policy."""

from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]
EGRESS_DIR = ROOT / "docker" / "crawler-egress"


def _yaml(name: str) -> dict[str, object]:
    loaded = yaml.safe_load((EGRESS_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def test_production_policy_pins_release_checksums_and_fails_closed() -> None:
    dockerfile = (EGRESS_DIR / "smokescreen.Dockerfile").read_text(encoding="utf-8")
    config = _yaml("smokescreen.production.yaml")

    assert "SMOKESCREEN_VERSION=v0.1.0" in dockerfile
    assert "4e1e447449e3cc653dae956208588722b5c8fe8ce894f633adc756a851d1ec46" in dockerfile
    assert "f15ddd0dc2b2cbc6d7318f441928939eb9bedfab2a1881508af3096a1e24b7c9" in dockerfile
    assert "sha256sum -c -" in dockerfile
    assert config["unsafe_allow_private_ranges"] is False
    assert config["connect_timeout"] == "10s"
    assert config["dns_timeout"] == "5s"
    assert int(config["max_concurrent_requests"]) > 0
    assert int(config["max_concurrent_connect_tunnels"]) > 0
    assert config["time_connect"] is True
    assert "allow_ranges" not in config


def test_test_policy_isolation_keeps_fixture_allowance_out_of_production() -> None:
    production = _yaml("smokescreen.production.yaml")
    test = _yaml("smokescreen.test.yaml")

    assert test["allow_ranges"] == ["198.18.0.0/24"]
    assert "allow_ranges" not in production
    assert production["unsafe_allow_private_ranges"] is False
    assert test["unsafe_allow_private_ranges"] is False
    assert production["acl_file"] == test["acl_file"]
    assert (EGRESS_DIR / "smokescreen.production.acl.yaml").read_bytes() != (
        EGRESS_DIR / "smokescreen.test.acl.yaml"
    ).read_bytes()

    production_compose = (ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
    assert "smokescreen.test.yaml" not in production_compose
    assert "198.18.0.0/24" not in production_compose
