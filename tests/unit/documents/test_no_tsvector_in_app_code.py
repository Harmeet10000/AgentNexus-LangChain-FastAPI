"""Prohibit full-text vector machinery in application code (retrieval-sql task 2.2).

The keyword leg uses the ``pg_textsearch`` BM25 access method directly. A
``tsvector`` column would double-count the lexical signal in the three-branch
fusion, so no full-text vector, query, or construction call may appear under
``src/app/``.

Migrations are an immutable historical record and are explicitly excluded from
this prohibition: they live under ``src/alembic/versions/``, outside the
``src/app/`` tree walked here, so a migration that once mentioned these tokens
can never fail this test — only application code can.
"""

from __future__ import annotations

from pathlib import Path

APP_ROOT = Path("src/app")

# Full-text vector surface: the column type, the query constructors, and the
# ranking/headline helpers. Any of these in application code fails the test.
FORBIDDEN_TOKENS = (
    "tsvector",
    "to_tsvector",
    "to_tsquery",
    "plainto_tsquery",
    "phraseto_tsquery",
    "websearch_to_tsquery",
    "ts_rank",
    "ts_headline",
)


def test_no_full_text_vector_machinery_in_app_code() -> None:
    offenders: list[str] = []
    for path in sorted(APP_ROOT.rglob("*.py")):
        # Migrations are excluded by construction: they live under
        # src/alembic/, never under src/app/. The guard below states the
        # exclusion explicitly so a future move cannot silently re-scope it.
        if "alembic" in path.parts or "migrations" in path.parts:
            continue
        lowered = path.read_text(encoding="utf-8").lower()
        hits = sorted({token for token in FORBIDDEN_TOKENS if token in lowered})
        if hits:
            offenders.append(f"{path}: {', '.join(hits)}")
    assert not offenders, (
        "full-text vector machinery in application code:\n" + "\n".join(offenders)
    )
