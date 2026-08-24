"""join credit and unified documents heads

Revision ID: 0012
Revises: 0005, 0011
Create Date: 2026-08-22 22:50:33.860075

Empty body, deliberately. The two heads joined here descend from a common
parent (``0009``) and touch disjoint relations — ``0010`` adds the credit
relations, ``0011`` is itself an empty merge whose other leg
(``0008``) adds the unified ``documents`` / ``chunks`` relations. Both
create their extensions with ``IF NOT EXISTS``. There is nothing to reconcile.

What this repairs
-----------------
``alembic upgrade head`` — the singular form — could not resolve while two
heads existed. It is used in ``Makefile:39``, ``README.md:272`` and
``.github/workflows/test.yml:105``. Joining the heads repairs all three
**without editing them**.

Phantom relations
-----------------
A *phantom* relation is one this migration chain records as created while no
executed DDL ever created it. Every relation below is phantom on any
environment whose ``alembic_version`` was **stamped** rather than migrated
forward — which is the state of the deployed instance (stamped ``0009``):

- ``chat_messages``, ``chat_sessions``, ``document_vectors`` — ``0006``
- ``search_documents``, ``search_chunks``               — ``0009``
- ``parent_documents``                                  — ``0010``
- ``outbox_events``, ``dead_letter_events``              — ``0006``

``clauses`` is phantom in a stronger sense: **no revision in this chain
creates it at all**, yet ``0010`` alters it and four query sites read
it: three sites in ``features/search/repository.py`` via ``clauses_bm25_idx``,
and one in ``langgraph_layer/ingestion_kb/nodes.py`` via ``bm25_force_merge``.

``0010`` is unrunnable
------------------------------
``0010`` opens with ``op.batch_alter_table("clauses")`` and then runs
``UPDATE clauses ...`` plus ``create_foreign_key`` against it. Because
``clauses`` is created by no revision, that revision **cannot execute against
any database built from this chain**. It is nonetheless in the ancestry of the
billing line (``0006`` revises it), so every environment stamped at ``0006``
or later has recorded it as applied without it ever having run.

The repair for this is a **forward, idempotent** revision that creates the
target shape — never a rewind through ``0010``. See ADR-6.

Reversal below this point is unsupported
----------------------------------------
Do not downgrade past this revision. The reversal path drops relations that
were never created, using unguarded DDL that will abort the transaction:

- ``0005.downgrade()`` → ``batch_alter_table("clauses")`` on a
  nonexistent relation, then a bare ``drop_table("parent_documents")``
- ``0004.downgrade()`` → bare ``drop_table("search_chunks")`` and
  ``drop_table("search_documents")``
- ``0001.downgrade()``        → bare ``drop_table("dead_letter_events")`` and
  ``drop_table("outbox_events")``

None of those carry ``IF EXISTS``. A reversal attempt is therefore not merely
lossy — it fails partway, leaving ``alembic_version`` disagreeing with the
schema it claims to describe.
"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | Sequence[str] | None = ("0010", "0011")
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: the joined branches touch disjoint relations."""


def downgrade() -> None:
    """No-op. See the module docstring — reversal below this point is unsupported."""
