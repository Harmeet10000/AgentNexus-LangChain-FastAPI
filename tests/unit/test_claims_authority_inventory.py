"""Static guard for the claims-authority inventory in issue #78."""

from pathlib import Path

from app.security.claims_authority import find_claim_privilege_violations


def test_application_never_authorizes_from_jwt_role_or_permissions() -> None:
    root = Path(__file__).resolve().parents[2]
    assert find_claim_privilege_violations(root) == []
