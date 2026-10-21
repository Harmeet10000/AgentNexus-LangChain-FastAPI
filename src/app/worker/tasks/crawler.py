"""Compatibility shim for the canonical crawler worker job."""

from app.worker.jobs.crawler import crawl_job

__all__ = ["crawl_job"]
