"""Thin Celery task entry points, one module per feature.

Every task carries an explicit wire `name=` so queued messages survive
module moves. Business orchestration lives in feature jobs/services;
these adapters only invoke them.
"""
