"""Check regeneration preserves manually maintained material."""

import tempfile
import unittest
from pathlib import Path

from reindex_okf import rebuild


class ReindexTests(unittest.TestCase):
    def test_dry_run_and_repeat_preserve_manual_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manual = b"# Start\n\n* [A](notes/a.md) - A concept.\n"
            (root / "index.md").write_bytes(manual)
            (root / "notes").mkdir()
            (root / "notes/a.md").write_text("---\ntype: Concept\ntitle: A\n---\n\n# A\n")
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            rebuild(root, dry_run=True)
            after = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            if before != after:
                self.fail("Dry-run changed the bundle")
            rebuild(root)
            generated = {
                p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()
            }
            rebuild(root)
            repeated = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
            if generated != repeated or (root / "index.md").read_bytes() != manual:
                self.fail("Reindex was not idempotent or rewrote a manual index")

    def test_invalid_document_fails_before_any_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bad.md").write_text("Missing metadata")
            try:
                rebuild(root)
            except ValueError:
                pass
            else:
                self.fail("Invalid metadata was accepted")
            if (root / "index.md").exists() or (root / "maintenance").exists():
                self.fail("Invalid input caused a partial write")


if __name__ == "__main__":
    unittest.main()
