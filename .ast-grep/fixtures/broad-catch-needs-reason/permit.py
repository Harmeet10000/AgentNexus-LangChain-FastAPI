"""Fixture: constructs the rule permits (clean).

- Reason-carrying suppression (em-dash and double-dash forms).
- Blind except ending in a bare raise: a logging pass-through, nothing survived.
- The three framework-contract sites: the raise is read by the framework
  (Pydantic validator ValueError, PEP 562 module __getattr__ AttributeError),
  never by project code — exempt from the union rules.
"""

from pydantic import field_validator


def degrade_with_reason() -> None:
    try:
        do_work()
    except Exception as exc:  # noqa: BLE001 — optional dependency; app degrades without it
        record(exc)


def degrade_with_double_dash_reason() -> None:
    try:
        do_work()
    except Exception as exc:  # noqa: BLE001 -- one bad item must not kill the run
        record(exc)


def logging_pass_through() -> None:
    try:
        do_work()
    except Exception:
        raise


class FrameworkSettingsContract:
    @field_validator("environment")
    @classmethod
    def _check_environment(cls, value: str) -> str:
        # src/app/config/settings.py:473 — Pydantic reads ValueError as validation failure.
        if value not in {"development", "production"}:
            msg = f"unknown environment: {value!r}"
            raise ValueError(msg)
        return value


class StrictEnvelopeContract:
    def _assert_envelope(self, response_model: object) -> None:
        # src/app/api/strict_envelope.py:26 — envelope shape assertion in a validator.
        if response_model is None:
            msg = "route must declare response_model=APIResponse[T]"
            raise ValueError(msg)


def __getattr__(name: str) -> object:
    # src/database/__init__.py:37 — PEP 562 module __getattr__; hasattr depends on it.
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
