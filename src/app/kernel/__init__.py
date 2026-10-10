"""Kernel: cross-feature vocabulary and the Result implementation.

`errors.py`, `clock.py`, `ids.py`, and `enums.py` are pure (no I/O, no
FastAPI, no SQLAlchemy) and importable from any layer without creating a
cycle. `result.py` is the deliberate exception: it carries the full Result
machinery including HTTP transport rendering, consolidated here by explicit
decision so the implementation lives in exactly one module.
"""
