"""Compatibility shim for the canonical transactional email worker jobs."""

from app.worker.jobs.auth import send_password_reset_email, send_verification_email

__all__ = ["send_password_reset_email", "send_verification_email"]
