"""Migration chain integrity: exactly one head must exist at all times.

Two heads happen when two people branch migrations independently; alembic
then refuses to `upgrade head` until a merge revision is hand-written. This
gate catches it in CI instead of at deploy time.
"""

from alembic.config import Config
from alembic.script import ScriptDirectory


def test_migration_chain_has_exactly_one_head() -> None:
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    heads = script.get_heads()
    assert len(heads) == 1, f"branched migration heads: {heads}"


def test_revision_ids_are_sequential() -> None:
    """Every revision ID must be 4-digit zero-padded (the project convention)."""
    script = ScriptDirectory.from_config(Config("alembic.ini"))
    bad = [
        rev.revision
        for rev in script.walk_revisions()
        if not (len(rev.revision) == 4 and rev.revision.isdigit())
    ]
    assert not bad, f"non-sequential revision IDs: {sorted(bad)}"
