"""Compatibility namespace for the canonical :mod:`app.worker.jobs` tasks.

The modules in this package are deliberately not Celery include targets. They
only re-export the canonical decorator-bearing adapters so old imports remain
valid without creating a second registration path.
"""
