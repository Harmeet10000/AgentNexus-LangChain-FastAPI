"""Fail CI when raw corpus binaries are stored despite redistribution limits.

Walks ``<corpus_root>/artifacts/<SOURCE_ID>/`` and checks each directory's
acquisition record at ``<corpus_root>/acquisition/<SOURCE_ID>.json``.  When a
record's ``redistribution_status`` disallows storage of the raw original, the
artifact directory must NOT contain an ``original.*`` file.  Artifact
directories without an acquisition record are skipped with a recorded warning
(the acquisition wave may still be running).  The check fails closed: an
EXISTING acquisition record that cannot be read or parsed is a hard error
(exit non-zero), never a silent allow, and an ``artifacts/`` path that exists
but is not a directory (or cannot be inspected) is a hard error too.  Only a
genuinely absent ``artifacts/`` path is a clean no-op.

Usage: uv run python scripts/check_corpus_binaries.py [corpus_root]
Dependency-free: standard library only, so it can run in any CI step.
"""

from __future__ import annotations

import argparse
import json
import stat
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


class AcquisitionRecordError(Exception):
    """An existing acquisition record exists but cannot be read or parsed."""

    def __init__(self, path: Path, cause: OSError | UnicodeDecodeError | ValueError) -> None:
        self.path = path
        self.cause = cause
        detail = f"unreadable acquisition record {path}: {cause}"
        super().__init__(detail)


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
    """Parse an existing acquisition record; raise when it cannot be read.

    Failing closed here is deliberate: an unreadable record must surface as a
    check failure, never fall through as ``None`` (which the redistribution
    vocabulary would treat as storage-allowed).
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AcquisitionRecordError(path, exc) from exc


def _artifact_directories(artifacts_root: Path) -> list[Path] | None:
    """Return the artifact directories to scan, or None when nothing exists.

    ``None`` is reserved for a genuinely absent ``artifacts`` path (clean
    no-op).  A path that exists but is not a directory, or cannot be inspected
    (EACCES, ENOTDIR, ...), raises ``OSError`` so the caller can fail closed.
    """
    try:
        info = artifacts_root.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISDIR(info.st_mode):
        detail = f"{artifacts_root} exists but is not a directory"
        raise NotADirectoryError(detail)
    return sorted(p for p in artifacts_root.iterdir() if p.is_dir())


def check_corpus(corpus_root: Path) -> list[str]:
    """Return one failure message per violation or hard error (exit non-zero)."""
    failures: list[str] = []
    artifacts_root = corpus_root / "artifacts"
    acquisition_root = corpus_root / "acquisition"
    try:
        artifact_dirs = _artifact_directories(artifacts_root)
    except OSError as exc:
        # NotADirectoryError (exists but not a dir), EACCES, ENOTDIR, ...
        return [f"error: cannot inspect artifact directory {artifacts_root}: {exc}"]
    if artifact_dirs is None:
        sys.stderr.write(f"note: no artifact directory at {artifacts_root}; nothing to scan\n")
        return failures
    try:
        for artifact_dir in artifact_dirs:
            failures.extend(_check_artifact_dir(artifact_dir, acquisition_root))
    except OSError as exc:
        return [f"error: cannot scan artifact directory {artifacts_root}: {exc}"]
    return failures


def _check_artifact_dir(artifact_dir: Path, acquisition_root: Path) -> list[str]:
    """Check one artifact directory; raise OSError when it cannot be inspected."""
    source_id = artifact_dir.name
    originals = sorted(artifact_dir.glob("original.*"))
    if not originals:
        return []
    record_path = acquisition_root / f"{source_id}.json"
    if not record_path.is_file():
        sys.stderr.write(
            f"warning: {source_id}: no acquisition record at {record_path}; "
            "redistribution policy unchecked\n"
        )
        return []
    try:
        record = _load_acquisition_record(record_path)
    except AcquisitionRecordError as exc:
        return [
            f"error: {source_id}: {exc}; the redistribution policy cannot be verified "
            "for stored originals, failing closed"
        ]
    status = _declared_redistribution_status(record)
    if _redistribution_disallows_storage(status):
        return [
            f"{source_id}: raw original {original} is committed although "
            f"redistribution_status {status!r} disallows storage"
            for original in originals
        ]
    return []


def main() -> None:
    """Run the scan and exit non-zero on any violation or hard error."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "corpus_root",
        nargs="?",
        type=Path,
        default=_DEFAULT_CORPUS_ROOT,
        help=f"corpus root containing artifacts/ and acquisition/ (default: {_DEFAULT_CORPUS_ROOT})",
    )
    args = parser.parse_args()
    failures = check_corpus(args.corpus_root)
    for failure in failures:
        sys.stderr.write(f"{failure}\n")
    if failures:
        sys.stderr.write(
            f"corpus binary check failed with {len(failures)} violation(s)/error(s); "
            "remove the raw originals, fix the redistribution_status records, or repair "
            "the unreadable acquisition records / artifacts path\n"
        )
        raise SystemExit(1)
    sys.stdout.write(f"corpus binary check passed for {args.corpus_root}\n")


if __name__ == "__main__":
    main()
