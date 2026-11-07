"""behave hooks for the acceptance suite."""

import os


def before_all(context):
    if not os.environ.get("OPENSPEC_ACCEPTANCE"):
        raise RuntimeError(
            "Run the suite via `python run_acceptance.py` (not plain `behave`).\n"
            "The wrapper extracts Gherkin from openspec/**/spec.md and composes the effective spec."
        )

    # TODO: boot the application under test and record its base URL.


def after_all(context):
    # TODO: shut the application down.
    pass


def before_scenario(context, scenario):
    # TODO: reset per-scenario state and attach page objects.
    pass
