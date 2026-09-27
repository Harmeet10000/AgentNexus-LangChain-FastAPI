"""Migration checks must detect lost PDF text and edited enhancement prefixes."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_migration import audit
from validate_okf import END, START


class MigrationTests(unittest.TestCase):
    def test_extract_and_enhancement_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = b"---\ntype: Concept\n---\nExisting explanation.\n"
            payload = b"page one\r\n\x0cpage two\n"
            raw = prefix + START.encode() + payload + END.encode()
            (root / "note.md").write_bytes(raw)
            (root / "before.txt").write_bytes(prefix)
            (root / "text.txt").write_bytes(payload)
            data = {
                "files": [],
                "segments": [],
                "enhancement_snapshots": [
                    {
                        "path": "note.md",
                        "asset": "before.txt",
                        "bytes": len(prefix),
                        "sha256": hashlib.sha256(prefix).hexdigest(),
                    }
                ],
                "text_sources": [
                    {
                        "asset": "text.txt",
                        "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "segments": [
                            {
                                "path": "note.md",
                                "start": 0,
                                "end": len(payload),
                                "payload_offset": len(prefix) + len(START.encode()),
                                "payload_bytes": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                ],
            }
            manifest = root / "migration.json"
            manifest.write_text(json.dumps(data))
            if not audit(root, manifest)["ok"]:
                self.fail("Unmodified migration should pass")
            (root / "note.md").write_bytes(raw.replace(b"page two", b"page six"))
            if audit(root, manifest)["ok"]:
                self.fail("Tampered migration should fail")
            (root / "note.md").write_bytes(raw.replace(b"Existing", b"Replaced"))
            if audit(root, manifest)["ok"]:
                self.fail("Tampered migration should fail")
