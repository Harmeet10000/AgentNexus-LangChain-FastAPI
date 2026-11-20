"""Canonical per-slice module descriptor.

A slice's ``module.py`` exposes a :class:`Module` whose fields are immutable
tuples of zero-argument factories. Importing a ``module.py`` must never compose
a router, a broker task, a scheduler entry, or a database adapter: composition
roots call only the factories for the surface they are building. The descriptor
therefore imports no web, broker, or persistence framework, which keeps it
importable from the API, worker, and scheduler processes alike.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

type RoutesFactory = Callable[[], Sequence[Any]]
type ConsumersFactory = Callable[[], Sequence[Any]]
type SchedulesFactory = Callable[[], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class Module:
    """One slice's composition surface: routes, consumers, schedules."""

    name: str
    routes: tuple[RoutesFactory, ...] = ()
    consumers: tuple[ConsumersFactory, ...] = ()
    schedules: tuple[SchedulesFactory, ...] = ()
