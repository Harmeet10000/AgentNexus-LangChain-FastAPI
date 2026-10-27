"""Fail CI when raw corpus binaries are stored despite redistribution limits.

Walks ``<corpus_root>/artifacts/<SOURCE_ID>/`` and checks each directory's
acquisition record at ``<corpus_root>/acquisition/<SOURCE_ID>.json``.  When a
record's ``redistribution_status`` disallows storage of the raw original, the
artifact directory must NOT contain an ``original.*`` file.  Artifact
directories without an acquisition record are skipped with a recorded warning
(the acquisition wave may still be running).

Usage: uv run python scripts/check_corpus_binaries.py [corpus_root]
Dependency-free: standard library only, so it can run in any CI step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REDISTRIBUTION_DENY_TOKENS = (
    "disallow",
    "forbid",
    "not_permitted",
    "no_redistribution",
    "prohibit",
    "denied",
)
_DEFAULT_CORPUS_ROOT = Path("evals/corpus")


def _redistribution_disallows_storage(status: object) -> bool:
    """Whether a redistribution status forbids storing the raw artifact.

    Mirrors ``app.ai.evaluation.schema._redistribution_disallows_storage``; the
    token list is duplicated here so this CI guard stays dependency-free.
    """
    if not isinstance(status, str) or not status.strip():
        return False
    normalized = status.strip().lower().replace("-", "_").replace(" ", "_")
    return any(token in normalized for token in _REDISTRIBUTION_DENY_TOKENS)


def _declared_redistribution_status(record: object) -> object:
    """Read the redistribution status from an acquisition record tolerantly."""
    if not isinstance(record, dict):
        return None
    if "redistribution_status" in record:
        return record["redistribution_status"]
    nested = record.get("redistribution")
    if isinstance(nested, dict):
        return nested.get("status")
    return nested


def _load_acquisition_record(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"warning: unreadable acquisition record {path}: {exc}\n")
        return None


def check_corpus(corpus_root: Path) -> list[str]:
    """Return one violation message per stored-but-disallowed original file."""
    violations: list[str] = []
    artifacts_root = corpus_root / "artifacts"
    acquisition_root = corpus_root / "acquisition"
    if not artifacts_root.is_dir():
        sys.stderr.write(f"note: no artifact directory at {artifacts_root}; nothing to scan\n")
        return violations
    for artifact_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        source_id = artifact_dir.name
        originals = sorted(artifact_dir.glob("original.*"))
        if not originals:
            continue
        record_path = acquisition_root / f"{source_id}.json"
        if not record_path.is_file():
            sys.stderr.write(
                f"warning: {source_id}: no acquisition record at {record_path}; "
                "redistribution policy unchecked\n"
            )
            continue
        record = _load_acquisition_record(record_path)
        status = _declared_redistribution_status(record)
        if _redistribution_disallows_storage(status):
            violations.extend(
                f"{source_id}: raw original {original} is committed although "
                f"redistribution_status {status!r} disallows storage"
                for original in originals
            )
    return violations


def main() -> None:
    """Run the scan and exit non-zero when disallowed originals exist."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_root",
        nargs="?",
        type=Path,
        default=_DEFAULT_CORPUS_ROOT,
        help=f"corpus root containing artifacts/ and acquisition/ (default: {_DEFAULT_CORPUS_ROOT})",
    )
    args = parser.parse_args()
    violations = check_corpus(args.corpus_root)
    for violation in violations:
        sys.stderr.write(f"{violation}\n")
    if violations:
        sys.stderr.write(
            f"corpus binary check failed with {len(violations)} violation(s); "
            "remove the raw originals or fix the redistribution_status records\n"
        )
        raise SystemExit(1)
    sys.stdout.write(f"corpus binary check passed for {args.corpus_root}\n")


if __name__ == "__main__":
    main()
