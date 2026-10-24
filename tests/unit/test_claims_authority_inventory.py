"""Static guard for the claims-authority inventory in issue #78."""

from pathlib import Path

from app.security.claims_authority import find_claim_privilege_violations


def test_application_never_authorizes_from_jwt_role_or_permissions() -> None:
    root = Path(__file__).resolve().parents[2]
    assert find_claim_privilege_violations(root) == []


def test_static_guard_catches_getters_and_assigned_claim_aliases(tmp_path: Path) -> None:
    source = tmp_path / "src" / "app" / "future_endpoint.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
def handler(claims):
    token = claims
    if token.get(\"role\") == \"admin\":
        return token.permissions
    return token[\"permissions\"]
""",
        encoding="utf-8",
    )

    violations = find_claim_privilege_violations(tmp_path)

    assert len(violations) == 3
