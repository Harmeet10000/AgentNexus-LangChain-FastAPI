"""Background-worker package.

Worker composition lives in :mod:`app.worker.celery`; task implementations
are imported from :mod:`app.worker.jobs` only by a selected worker process.
This package intentionally has no runtime aliases or eager imports.
"""

__all__: tuple[str, ...] = ()
