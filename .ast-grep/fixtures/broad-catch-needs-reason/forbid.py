"""Fixture: constructs the rule forbids (flagged)."""


def swallow_without_reason() -> None:
    try:
        do_work()
    except Exception as exc:  # noqa: BLE001
        record(exc)
