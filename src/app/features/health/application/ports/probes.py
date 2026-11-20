"""Outbound probe ports for the health application service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from app.features.health.application.dto.result import ProbeResult


class Probe(Protocol):
    """A named, asynchronous capability check.

    Implementations belong to outbound adapters. The application service only
    knows this contract, which keeps driver lifecycles and exception taxonomies
    outside orchestration.
    """

    name: str

    async def check(self) -> ProbeResult:
        """Return an explicit result; adapter failures must not escape normally."""
