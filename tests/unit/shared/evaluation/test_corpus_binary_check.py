"""Unit tests for the fail-closed corpus binary check (scripts/check_corpus_binaries.py).

Only a genuinely absent ``artifacts/`` root is a clean no-op: an unreadable
existing acquisition record or an uninspectable artifacts path must fail the
check (exit non-zero), never fall through as storage-allowed.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from scripts.check_corpus_binaries import check_corpus, main


def _corpus(tmp_path: Path) -> tuple[Path, Path]:
    """Create a corpus root and return (root, acquisition dir)."""
    acquisition = tmp_path / "corpus" / "acquisition"
    acquisition.mkdir(parents=True)
    return tmp_path / "corpus", acquisition


def _record(acquisition: Path, source_id: str, **overrides: object) -> None:
    record: dict[str, Any] = {
        "source_id": source_id,
        "redistribution_status": "permitted_with_attribution",
    }
    record.update(overrides)
    (acquisition / f"{source_id}.json").write_text(json.dumps(record), encoding="utf-8")


def _stored(corpus: Path, source_id: str) -> None:
    artifact_dir = corpus / "artifacts" / source_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "original.pdf").write_bytes(b"%PDF-1.4 fixture")


def test_absent_artifacts_root_is_a_clean_noop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert check_corpus(tmp_path / "corpus") == []
    assert "nothing to scan" in capsys.readouterr().err


def test_artifacts_path_that_is_a_file_is_a_hard_error(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "artifacts").write_text("a file, not a directory", encoding="utf-8")

    failures = check_corpus(corpus)

    assert len(failures) == 1
    assert "cannot inspect artifact directory" in failures[0]
    assert "not a directory" in failures[0]


def test_enotdir_artifacts_parent_is_a_hard_error(tmp_path: Path) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("a file where the corpus root should be", encoding="utf-8")

    failures = check_corpus(blocked / "corpus")

    assert len(failures) == 1
    assert "cannot inspect artifact directory" in failures[0]


def test_eacces_artifacts_stat_is_a_hard_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    original_stat = Path.stat

    def denying_stat(self: Path, **kwargs: object) -> Any:
        if self.name == "artifacts":
            raise PermissionError(13, "Permission denied")
        return original_stat(self, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "stat", denying_stat)

    failures = check_corpus(corpus)

    assert len(failures) == 1
    assert "cannot inspect artifact directory" in failures[0]
    assert "Permission denied" in failures[0]


def test_unreadable_acquisition_record_fails_closed(tmp_path: Path) -> None:
    corpus, _ = _corpus(tmp_path)
    _stored(corpus, "SRC-99")
    (corpus / "acquisition" / "SRC-99.json").write_text("{not json", encoding="utf-8")

    failures = check_corpus(corpus)

    assert len(failures) == 1
    assert "SRC-99" in failures[0]
    assert "unreadable acquisition record" in failures[0]


def test_undecodable_acquisition_record_fails_closed(tmp_path: Path) -> None:
    corpus, _ = _corpus(tmp_path)
    _stored(corpus, "SRC-98")
    (corpus / "acquisition" / "SRC-98.json").write_bytes(b"\xff\xfe not utf-8")

    failures = check_corpus(corpus)

    assert len(failures) == 1
    assert "unreadable acquisition record" in failures[0]


def test_main_exits_non_zero_on_unreadable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus, _ = _corpus(tmp_path)
    _stored(corpus, "SRC-99")
    (corpus / "acquisition" / "SRC-99.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["check_corpus_binaries.py", str(corpus)])

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    assert "unreadable acquisition record" in capsys.readouterr().err


def test_missing_acquisition_record_still_warns_and_skips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus, _ = _corpus(tmp_path)
    _stored(corpus, "SRC-97")

    assert check_corpus(corpus) == []
    assert "no acquisition record" in capsys.readouterr().err


def test_disallowed_redistribution_still_flagged(tmp_path: Path) -> None:
    corpus, acquisition = _corpus(tmp_path)
    _stored(corpus, "SRC-96")
    _record(acquisition, "SRC-96", redistribution_status="not_permitted")

    failures = check_corpus(corpus)

    assert len(failures) == 1
    assert "disallows storage" in failures[0]


def test_permitted_original_passes(tmp_path: Path) -> None:
    corpus, acquisition = _corpus(tmp_path)
    _stored(corpus, "SRC-95")
    _record(acquisition, "SRC-95")

    assert check_corpus(corpus) == []
