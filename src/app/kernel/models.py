"""Purpose-keyed Pydantic base models.

One base class per model purpose. Every base declares the full configuration
so new models inherit the project's conventions by construction instead of
re-declaring ``ConfigDict`` per class (it was hand-written verbatim 132 times).

Subclass ``model_config`` merges per key: a subclass may loosen one key
(e.g. ``frozen=False``) and keeps every other inherited setting.

Kernel rule respected: these classes are pure schemas — no FastAPI, no
SQLAlchemy, no network I/O, no app-state access (same precedent as
``kernel/errors.py``, which already subclasses ``BaseModel``).

Performance notes:

- ``defer_build=True`` trades eager schema construction for first-use latency;
  it is useful when many models are imported but only a subset is used. It is
  not automatically a startup or throughput win, so it is deliberately
  overridable by a concrete model.
- Pydantic models are not a faster replacement for dataclasses. Use them where
  validation, JSON/schema support, or controlled copying matters. Keep
  high-frequency internal carriers as dataclasses when measurements favor it.
- ``frozen=True`` prevents attribute reassignment, but nested lists/dicts stay
  mutable and a frozen model is hashable only when all of its fields are
  hashable.
- Defaults are not validated by default. Turn on ``validate_default`` where
  validators or untrusted defaults make that behavior necessary.
"""

from typing import Final

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

#: Settings shared by every base: closed models and no ``model_*`` field-name
#: namespace, so fields like ``model_name`` need no per-class override
#: (renaming remains the cleaner long-term choice).
_COMMON: Final[ConfigDict] = ConfigDict(
    extra="forbid",
    protected_namespaces=("model_validate", "model_dump"),
)


class StrictModel(BaseModel):
    """Immutable, closed value models.

    Use for: domain state, projections, snapshots, results/summary carriers,
    authorization capability bundles, config objects, structured evidence.

    Gives: immutable instances (hashable, thread-safe), unknown-field
    rejection, fast attribute access, deferred schema build.
    """

    model_config = ConfigDict(
        **_COMMON,
        frozen=True,
    )


class ConfigModel(StrictModel):
    """Operational policy/bounding parameters (retry budgets, batch sizes).

    Same guarantees as :class:`StrictModel`; a distinct name so call sites
    document intent — a ``ConfigModel`` must only bound behaviour, never
    carry protocol content.
    """


class DeferredModel(StrictModel):
    """Opt-in schema-build deferral for rarely used, import-heavy models.

    Deferral can improve cold startup when most schemas are never exercised,
    but moves schema-build latency to first use. Do not use on latency-critical
    request or graph paths without measuring the trade-off.
    """

    model_config = ConfigDict(defer_build=True)


class ReadModel(StrictModel):
    """Read views assembled from repositories / ORM / document rows.

    Adds ``from_attributes`` so ``Model.model_validate(db_row)`` works.
    Wire aliases are an API-contract decision and do not belong on generic
    repository read models.
    """

    model_config = ConfigDict(
        from_attributes=True,
    )


class MutableModel(BaseModel):
    """Operational accumulators and mutable reports.

    Use for: counters/durations reports assembled across a loop, usage
    collectors, heartbeat carriers. Mutations are plain assignments (no
    re-validation — deliberate, these paths are hot and producers are
    trusted). Unknown fields still rejected.
    """

    model_config = ConfigDict(**_COMMON)


class ContextModel(BaseModel):
    """Request-invocation collaborator carriers (LangGraph runtime contexts).

    Holds live callables/clients that must not be captured at compile time.
    Mutable, closed, and accepts non-pydantic collaborator types.
    """

    model_config = ConfigDict(
        **_COMMON,
        arbitrary_types_allowed=True,
    )


class RequestModel(BaseModel):
    """HTTP request bodies — the trust boundary.

    Unknown fields are a validation error (never silently ignored), and
    validation failures do not echo the caller's input back (credential-safe
    error envelopes).
    """

    model_config = ConfigDict(
        **_COMMON,
        frozen=True,
        populate_by_name=True,
        hide_input_in_errors=True,
    )


class ResponseModel(BaseModel):
    """HTTP response bodies (camelCase wire contract).

    Opt-in only for API families whose documented wire contract is camelCase.
    Models with mixed or established aliases should use explicit field aliases
    or a dedicated subclass override instead.
    """

    model_config = ConfigDict(
        **_COMMON,
        frozen=True,
        populate_by_name=True,
        serialize_by_alias=True,
        alias_generator=to_camel,
    )
