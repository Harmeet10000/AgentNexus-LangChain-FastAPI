"""Behavior tests for format permissiveness, producer checks, and preservation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_okf import END, START, authored_body, validate


class FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def put(self, text, name="note.md") -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def fixture_manifest(self, original=b"alpha\r\nUnicode: \xe2\x98\x83\n") -> tuple[Path, bytes]:
        prefix = b"---\ntype: Reference\n---\n\n" + START.encode()
        raw = prefix + original + END.encode() + b"\n"
        (self.root / "note.md").write_bytes(raw)
        asset = self.root / "assets/originals/original.md.txt"
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_bytes(original)
        sha = hashlib.sha256(original).hexdigest()
        data = {
            "files": [{"path": "original.md", "bytes": len(original), "sha256": sha}],
            "segments": [
                {
                    "source": "original.md",
                    "path": "note.md",
                    "start": 0,
                    "end": len(original),
                    "sha256": sha,
                    "payload_offset": len(prefix),
                    "payload_bytes": len(original),
                }
            ],
        }
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(data))
        return (manifest, original)


class FormatTests(FixtureCase):
    def test_minimal_and_custom_type_are_valid(self) -> None:
        self.put("---\ntype: Unregistered type\ncustom_key: [1, 2]\n---\nBody")
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_missing_type_fails(self) -> None:
        self.put("---\ntitle: Example\n---\nBody")
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_non_string_type_fails(self) -> None:
        self.put("---\ntype: [Concept]\n---\nBody")
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_duplicate_yaml_keys_fail(self) -> None:
        self.put("---\ntype: Concept\ntype: Reference\n---\nBody")
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_unsafe_yaml_is_not_constructed(self) -> None:
        self.put('---\ntype: !!python/object/apply:os.system ["exit 99"]\n---\nBody')
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_broken_links_are_not_format_errors(self) -> None:
        self.put("---\ntype: Concept\n---\n[Future note](missing.md)")
        if not validate(self.root)["ok"]:
            self.fail("Expected: validate(self.root)['ok']")
        if validate(self.root, strict=True)["ok"]:
            self.fail("Expected: not validate(self.root, strict=True)['ok']")

    def test_bare_and_list_verification_are_equivalent(self) -> None:
        for value in (
            '{by: human:reviewer, at: "2026-09-11T00:00:00Z"}',
            '[{by: human:reviewer, at: "2026-09-11T00:00:00Z"}]',
        ):
            self.put(f"---\ntype: Concept\nverified: {value}\n---\nBody")
            result = validate(self.root, strict=True)
            if not result["ok"]:
                self.fail("Expected: result['ok']")
            if result["trust_tiers"] != {"human-reviewed": 1}:
                self.fail("Expected: result['trust_tiers'] == {'human-reviewed': 1}")

    def test_naive_datetime_is_producer_issue(self) -> None:
        self.put(
            '---\ntype: Concept\ngenerated: {by: tool/1, at: "2026-09-11T00:00:00"}\n---\nBody'
        )
        if not validate(self.root)["ok"]:
            self.fail("Expected: validate(self.root)['ok']")
        if validate(self.root, strict=True)["ok"]:
            self.fail("Expected: not validate(self.root, strict=True)['ok']")

    def test_source_scope_descriptor_is_valid(self) -> None:
        self.put("---\ntype: Concept\nsources:\n  - resource: all queries in project X\n---\nBody")
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_source_footnotes_require_matching_ids(self) -> None:
        self.put(
            '---\ntype: Concept\nsources: [{id: a, resource: "https://example.org"}]\n---\nClaim.[^b]\n\n[^b]: Evidence'
        )
        if validate(self.root, strict=True)["ok"]:
            self.fail("Expected: not validate(self.root, strict=True)['ok']")

    def test_matching_footnotes_and_code_examples(self) -> None:
        self.put(
            '---\ntype: Concept\nsources: [{id: a, resource: "https://example.org"}]\n---\nClaim.[^a]\n\n[^a]: Evidence\n\n```markdown\nClaim.[^not-real]\n```\n'
        )
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_nested_index_frontmatter_fails(self) -> None:
        self.put('---\nokf_version: "0.2"\n---\n# Notes\n* [Note](../note.md)', "child/index.md")
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_root_index_version_does_not_require_type(self) -> None:
        self.put("---\ntype: Concept\n---\nBody")
        self.put('---\nokf_version: "0.2"\n---\n# Notes\n* [Note](note.md)', "index.md")
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_invalid_log_date_fails(self) -> None:
        self.put("# Updates\n\n## 2026-02-30\n* Created a note.", "log.md")
        if validate(self.root)["ok"]:
            self.fail("Expected: not validate(self.root)['ok']")

    def test_encoded_path_with_spaces_and_balanced_parentheses(self) -> None:
        self.put("---\ntype: Concept\n---\nBody", "other (one).md")
        self.put("---\ntype: Concept\n---\n[Other](other%20%28one%29.md)")
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_attested_computation_is_not_executed(self) -> None:
        self.put(
            '---\ntype: Attested Computation\nruntime: python\n---\n# Computation\n\n```python\nraise RuntimeError("must not execute")\n```\n'
        )
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")

    def test_inherited_link_preserved_without_strict_failure(self) -> None:
        self.put("---\ntype: Reference\n---\n" + START + "[Old link](missing.md)" + END + "\n")
        result = validate(self.root, strict=True)
        if not result["ok"]:
            self.fail("Expected: result['ok']")
        if len(result["inherited_link_issues"]) != 1:
            self.fail("Expected: len(result['inherited_link_issues']) == 1")

    def test_index_is_not_required(self) -> None:
        self.put("---\ntype: Concept\n---\nBody")
        if not validate(self.root, strict=True)["ok"]:
            self.fail("Expected: validate(self.root, strict=True)['ok']")


class PreservationTests(FixtureCase):
    def test_explicit_revision_asset_preserves_initial_copy(self) -> None:
        manifest, original = self.fixture_manifest()
        data = json.loads(manifest.read_text())
        revision = self.root / "assets/revisions/reviewed.md.txt"
        revision.parent.mkdir(parents=True)
        revision.write_bytes(original)
        data["files"][0]["asset"] = "assets/revisions/reviewed.md.txt"
        manifest.write_text(json.dumps(data))
        initial = self.root / "assets/originals/original.md.txt"
        initial.write_bytes(b"an earlier preserved revision")
        if not validate(self.root, manifest=manifest)["ok"]:
            self.fail("Explicit asset must select the reviewed revision")
        if initial.read_bytes() != b"an earlier preserved revision":
            self.fail("Validation must retain the initial copy")
        revision.write_bytes(original + b"unreviewed addition")
        if validate(self.root, manifest=manifest)["ok"]:
            self.fail("Revision asset changes must fail preservation")

    def test_revision_asset_cannot_escape_bundle(self) -> None:
        manifest, _ = self.fixture_manifest()
        data = json.loads(manifest.read_text())
        data["files"][0]["asset"] = "../outside.txt"
        manifest.write_text(json.dumps(data))
        if validate(self.root, manifest=manifest)["ok"]:
            self.fail("Manifest asset traversal must be rejected")

    def test_preservation_handles_unicode_and_crlf(self) -> None:
        manifest, _ = self.fixture_manifest()
        if not validate(self.root, strict=True, manifest=manifest)["ok"]:
            self.fail("Expected: validate(self.root, strict=True, manifest=manifest)['ok']")

    def test_preservation_handles_empty_original(self) -> None:
        manifest, _ = self.fixture_manifest(b"")
        if not validate(self.root, strict=True, manifest=manifest)["ok"]:
            self.fail("Expected: validate(self.root, strict=True, manifest=manifest)['ok']")

    def test_changed_passage_fails_preservation(self) -> None:
        manifest, _ = self.fixture_manifest()
        p = self.root / "note.md"
        p.write_bytes(p.read_bytes().replace(b"alpha", b"omega"))
        if validate(self.root, manifest=manifest)["ok"]:
            self.fail("Expected: not validate(self.root, manifest=manifest)['ok']")

    def test_gap_in_source_coverage_fails(self) -> None:
        manifest, _ = self.fixture_manifest()
        data = json.loads(manifest.read_text())
        data["segments"][0]["start"] = 1
        manifest.write_text(json.dumps(data))
        if validate(self.root, manifest=manifest)["ok"]:
            self.fail("Expected: not validate(self.root, manifest=manifest)['ok']")

    def test_changed_original_is_detected(self) -> None:
        manifest, original = self.fixture_manifest()
        folder = self.root / "originals"
        folder.mkdir()
        (folder / "original.md").write_bytes(original + b"extra")
        if validate(self.root, manifest=manifest, originals=folder)["ok"]:
            self.fail(
                "Expected: not validate(self.root, manifest=manifest, originals=folder)['ok']"
            )


if __name__ == "__main__":
    unittest.main()


class MultiPassageTests(unittest.TestCase):
    def test_all_passages_are_separated_from_authored_text(self) -> None:
        body = "before" + START + "first" + END + "between" + START + "second" + END + "after"
        if authored_body(body) != ("beforebetweenafter", "first\n\nsecond"):
            self.fail("Every complete passage must be excluded")

    def test_unclosed_marker_remains_authored(self) -> None:
        body = "before" + START + "unfinished"
        if authored_body(body) != (body, ""):
            self.fail("Incomplete passage must remain authored")
