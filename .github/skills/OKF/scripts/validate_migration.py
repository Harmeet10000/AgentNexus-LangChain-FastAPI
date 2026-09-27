# /// script
# requires-python = ">=3.10"
# dependencies = ["PyYAML>=6,<7", "markdown-it-py>=3,<5", "mdit-py-plugins>=0.4,<1"]
# ///
"""Read-only verification of verbatim migration, extraction coverage, and enhancements."""

import argparse
import hashlib
import json
import operator
import sys
from pathlib import Path
from typing import Any

from validate_okf import reconstruct, safe_relative, validate


def asset(root, name) -> bytes:
    path = root / safe_relative(name)
    if not path.resolve().is_relative_to(root.resolve()):
        message = "Path escapes bundle"
        raise ValueError(message)
    return path.read_bytes()


def audit(root, manifest, originals=None) -> dict[str, Any]:
    root = Path(root).resolve()
    data = json.loads(Path(manifest).read_text(encoding="utf-8"))
    report = validate(root, strict=True, manifest=manifest, originals=originals)
    errors = report["preservation_errors"]
    for entry in data.get("text_sources", []):
        copied = asset(root, entry["asset"])
        problems = []
        if len(copied) != entry["bytes"] or hashlib.sha256(copied).hexdigest() != entry["sha256"]:
            problems.append("Extracted text asset changed")
        selected = sorted(entry["segments"], key=operator.itemgetter("start"))
        reconstruct(root, selected, copied, report, problems)
        errors.extend({"path": entry["asset"], "message": p} for p in problems)
    for entry in data.get("enhancement_snapshots", []):
        original = asset(root, entry["asset"])
        current = asset(root, entry["path"])
        if (
            hashlib.sha256(original).hexdigest() != entry["sha256"]
            or len(original) != entry["bytes"]
            or not current.startswith(original)
        ):
            errors.append(
                {"path": entry["path"], "message": "Pre-migration enhancement changed or lost"}
            )
    report["enhancements_checked"] = len(data.get("enhancement_snapshots", []))
    report["text_sources_checked"] = len(data.get("text_sources", []))
    report["ok"] = report["ok"] and not errors
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument(
        "--manifest", type=Path, help="Defaults to BUNDLE/maintenance/verbatim-migration.json"
    )
    parser.add_argument("--originals", type=Path)
    args = parser.parse_args()
    try:
        report = audit(
            args.bundle,
            args.manifest or args.bundle / "maintenance/verbatim-migration.json",
            args.originals,
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Migration audit failed: {exc}\n")
    sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
