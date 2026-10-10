"""Reusable security primitives: token claims, JWT encode/decode, password hashing.

No domain imports — account policy (OAuth provider config, revocation,
user lifecycle) stays in `features.auth`. Claim fields, token validation,
and key behavior are unchanged by the move.
"""
