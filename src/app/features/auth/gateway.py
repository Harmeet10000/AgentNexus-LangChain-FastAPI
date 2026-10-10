"""Outgoing gateway for the auth slice.

Reserved: external service calls initiated by this slice belong here behind
explicit functions (no direct httpx/boto3 use in services). No callers need
it today.
"""
