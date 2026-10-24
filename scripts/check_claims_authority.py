"""Reject direct JWT role/permission authorization in application code.

JWT role and permission fields are retained for wire compatibility and
diagnostics, but current privilege authority belongs to the live account
returned by ``require_permission``/``require_role``.  This small AST check is
deliberately conservative: any claim-like variable reading ``role`` or
``permissions`` is a review failure, so a new alias cannot silently bypass the
policy.
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.security.claims_authority import find_claim_privilege_violations


def main() -> int:
    violations = find_claim_privilege_violations(Path(__file__).resolve().parents[1])
    if violations:
        sys.stdout.write("JWT privilege claim usage found:\n")
        sys.stdout.write("\n".join(violations) + "\n")
        return 1
    sys.stdout.write("claims authority check passed\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
