"""Celery worker entry point.

Run with: uv run celery -A celery_config worker --loglevel=info
"""

from src.app.connections import celery_app

__all__ = ["celery_app"]
