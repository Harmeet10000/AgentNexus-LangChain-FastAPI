"""Lifespan-owned billing provider dependency."""

from fastapi import Request

from .protocol import BillingProvider


def get_lifespan_billing_provider(request: Request) -> BillingProvider:
    provider = getattr(request.app.state, "billing_provider", None)
    if provider is None:
        message = "Billing provider is not initialized"
        raise RuntimeError(message)
    return provider
