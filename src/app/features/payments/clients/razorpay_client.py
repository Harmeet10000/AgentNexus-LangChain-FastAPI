"""Async Razorpay API client with tenacity retries.

Retry policy (Requirement 34):
- transient errors (503, 504, 429, timeouts) are retried with exponential
  backoff (multiplier=1, min=1s, max=10s, stop after 3 attempts);
- permanent errors (401, 403, 4xx) raise immediately with no retry.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.config import get_settings
from app.features.payments.exceptions import RazorpayRetryableError
from app.utils import ExternalServiceException, logger

type _JsonObject = dict[str, Any]

_RAZORPAY_TIMEOUT_CODES = (503, 504, 429)


class RazorpayPermanentError(ExternalServiceException):
    """Non-retryable Razorpay failure (401/403/bad request)."""


class CircuitOpenError(ExternalServiceException):
    """Razorpay circuit is open; request refused without a call (Requirement 25.6)."""


class _CircuitBreaker:
    """In-process per-operation circuit breaker.

    ponytail: single-process breaker; scale to Redis if multiple workers call
    Razorpay from the same key. ``failure_threshold`` consecutive failures
    open the circuit for ``recovery_timeout_seconds``.
    """

    def __init__(self, *, failure_threshold: int = 3, recovery_timeout_seconds: int = 60) -> None:
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout_seconds
        self._failures: dict[str, int] = defaultdict(int)
        self._opened_at: dict[str, float] = {}

    def allow(self, key: str) -> bool:
        opened_at = self._opened_at.get(key)
        if opened_at is None:
            return True
        if time.monotonic() - opened_at >= self._recovery_timeout:
            self._opened_at.pop(key, None)
            self._failures[key] = 0
            return True
        return False

    def record_success(self, key: str) -> None:
        self._failures.pop(key, None)
        self._opened_at.pop(key, None)

    def record_failure(self, key: str) -> None:
        self._failures[key] += 1
        if self._failures[key] >= self._failure_threshold:
            self._opened_at[key] = time.monotonic()


_circuit = _CircuitBreaker()


def _raise_for_status(response: httpx.Response, *, operation: str) -> None:
    if response.status_code < 400:
        return
    detail = f"{operation} returned HTTP {response.status_code}: {response.text[:500]}"
    logger.bind(operation=operation, status_code=response.status_code).warning(detail)
    if response.status_code in _RAZORPAY_TIMEOUT_CODES:
        raise RazorpayRetryableError(service="razorpay", detail=detail)
    raise RazorpayPermanentError(service="razorpay", detail=detail)


def _retry_policy() -> Any:
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(RazorpayRetryableError),
        reraise=True,
    )


class RazorpayClient:
    """Thin async wrapper around the Razorpay REST API."""

    def __init__(
        self,
        *,
        key_id: str | None = None,
        key_secret: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        settings = get_settings()
        self._key_id = key_id or settings.RAZORPAY_KEY_ID
        self._key_secret = key_secret or settings.RAZORPAY_KEY_SECRET.get_secret_value()
        self._base_url = (base_url or settings.RAZORPAY_API_BASE_URL).rstrip("/")
        self._timeout = timeout or settings.RAZORPAY_REQUEST_TIMEOUT_SECONDS

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            auth=(self._key_id, self._key_secret),
            timeout=self._timeout,
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> _JsonObject:
        if not _circuit.allow(operation):
            logger.bind(operation=operation).warning("Circuit open; Razorpay request refused")
            raise CircuitOpenError(service="razorpay", detail=f"{operation} refused by circuit")
        try:
            async with self._client() as client:
                response = await client.request(method, path, params=params, json=json)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            _circuit.record_failure(operation)
            raise RazorpayRetryableError(
                service="razorpay", detail=f"{operation} failed with transport error: {exc}"
            ) from exc
        if response.status_code >= 400:
            _circuit.record_failure(operation)
            _raise_for_status(response, operation=operation)
        _circuit.record_success(operation)
        return response.json()

    @_retry_policy()
    async def create_customer(
        self,
        *,
        email: str,
        contact: str | None,
        name: str | None,
        notes: dict[str, Any] | None = None,
    ) -> _JsonObject:
        return await self._request(
            "POST",
            "/v1/customers",
            operation="create_customer",
            json={
                "email": email,
                "contact": contact,
                "name": name,
                "notes": notes or {},
            },
        )

    @_retry_policy()
    async def find_customer_by_email(self, email: str) -> _JsonObject | None:
        data = await self._request(
            "GET",
            "/v1/customers",
            operation="find_customer_by_email",
            params={"count": 10},
        )
        for customer in data.get("items", []):
            if customer.get("email", "").lower() == email.lower():
                return customer
        return None

    @_retry_policy()
    async def create_plan(
        self, *, name: str, amount: int, interval: str, currency: str, period: int
    ) -> _JsonObject:
        return await self._request(
            "POST",
            "/v1/plans",
            operation="create_plan",
            json={
                "period": interval,
                "interval": period,
                "item": {"name": name, "amount": amount, "currency": currency},
            },
        )

    @_retry_policy()
    async def create_subscription(
        self,
        *,
        plan_id: str,
        customer_id: str,
        total_count: int,
        quantity: int,
        customer_notify: bool,
        notes: dict[str, Any] | None = None,
    ) -> _JsonObject:
        return await self._request(
            "POST",
            "/v1/subscriptions",
            operation="create_subscription",
            json={
                "plan_id": plan_id,
                "customer_id": customer_id,
                "total_count": total_count,
                "quantity": quantity,
                "customer_notify": 1 if customer_notify else 0,
                "notes": notes or {},
            },
        )

    @_retry_policy()
    async def cancel_subscription(
        self, subscription_id: str, *, cancel_at_cycle_end: bool = False
    ) -> _JsonObject:
        return await self._request(
            "POST",
            f"/v1/subscriptions/{subscription_id}/cancel",
            operation="cancel_subscription",
            json={"cancel_at_cycle_end": cancel_at_cycle_end},
        )

    @_retry_policy()
    async def update_subscription(
        self, subscription_id: str, *, values: dict[str, Any]
    ) -> _JsonObject:
        """PATCH a subscription (pause_at/resume_at/cancel_at_cycle_end)."""
        return await self._request(
            "PATCH",
            f"/v1/subscriptions/{subscription_id}",
            operation="update_subscription",
            json=values,
        )

    @_retry_policy()
    async def fetch_subscription(self, subscription_id: str) -> _JsonObject:
        return await self._request(
            "GET",
            f"/v1/subscriptions/{subscription_id}",
            operation="fetch_subscription",
        )

    @_retry_policy()
    async def list_subscriptions(self, *, params: dict[str, Any]) -> list[_JsonObject]:
        data = await self._request(
            "GET", "/v1/subscriptions", operation="list_subscriptions", params=params
        )
        return list(data.get("items", []))

    @_retry_policy()
    async def create_payment_link(
        self,
        *,
        amount: int,
        currency: str,
        description: str,
        customer_id: str | None = None,
        notes: dict[str, Any] | None = None,
    ) -> _JsonObject:
        return await self._request(
            "POST",
            "/v1/payment_links",
            operation="create_payment_link",
            json={
                "amount": amount,
                "currency": currency,
                "description": description,
                "customer_id": customer_id,
                "notes": notes or {},
                "accept_partial": False,
            },
        )

    @_retry_policy()
    async def list_payments(self, *, params: dict[str, Any]) -> list[_JsonObject]:
        data = await self._request("GET", "/v1/payments", operation="list_payments", params=params)
        return list(data.get("items", []))

    @_retry_policy()
    async def fetch_payment(self, payment_id: str) -> _JsonObject:
        return await self._request("GET", f"/v1/payments/{payment_id}", operation="fetch_payment")

    @_retry_policy()
    async def create_refund(
        self, *, payment_id: str, amount: int, notes: dict[str, Any] | None = None
    ) -> _JsonObject:
        return await self._request(
            "POST",
            f"/v1/payments/{payment_id}/refund",
            operation="create_refund",
            json={"amount": amount, "notes": notes or {}},
        )

    @_retry_policy()
    async def submit_dispute_evidence(
        self, *, dispute_id: str, evidence: dict[str, Any]
    ) -> _JsonObject:
        return await self._request(
            "POST",
            f"/v1/disputes/{dispute_id}/accept",
            operation="submit_dispute_evidence",
            json=evidence,
        )
