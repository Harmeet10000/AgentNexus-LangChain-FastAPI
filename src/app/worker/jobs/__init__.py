"""Canonical decorator-bearing worker job namespace.

Every task carries an explicit stable wire name. The old
``app.worker.tasks`` modules are one-way re-export shims and are never included
by Celery, so importing both namespaces cannot double-register a task.
"""
