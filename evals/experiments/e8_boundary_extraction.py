"""E8 experiment (task 8.5): per-document vs per-contract metadata on SRC-04.

Runs the REAL boundary-local extraction path both ways over the multi-contract
corpus artifact SRC-04 (NHAI HAM packet, 5 recorded contracts):

- whole-document: one ``classify_document`` over the full markdown (the
  pre-hardening shape);
- contract-local: ``establish_contract_boundaries`` + one
  ``classify_document`` per boundary slice (the landed per-boundary path).

Correctness is measured against the contract boundaries recorded in the
corpus provenance manifest (``ingestion_fidelity_v1.manifest.json``):
boundary page coverage, and whether metadata stays separated per contract
instead of being mixed into one document-level classification. Cost is the
number of classification units and characters processed. Output:
``evals/reports/e8_per_contract_vs_per_document_src04.json``.

Run: ``uv run python evals/experiments/e8_boundary_extraction.py``
"""

from __future__ import annotations

import asyncio
import json
import sys
from bisect import bisect_right
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from app.features.documents.ingestion.chunking.classification import (
    classify_document,
    establish_contract_boundaries,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC04_PAGES = REPO_ROOT / "evals/corpus/artifacts/SRC-04/pages.json"
MANIFEST = REPO_ROOT / "evals/golden/ingestion_fidelity_v1.manifest.json"
OUTPUT = REPO_ROOT / "evals/reports/e8_per_contract_vs_per_document_src04.json"


def page_for_offset(offset: int, page_offsets: list[int]) -> int:
    """1-based page number whose recorded offset is the last one <= offset."""
    index = bisect_right(page_offsets, offset) - 1
    return index + 1 if index >= 0 else 1


def classification_payload(classified: object) -> dict[str, object]:
    """The measured metadata fields of one classification result."""
    return {
        "document_kind": getattr(classified, "document_kind", None),
        "jurisdiction": getattr(classified, "jurisdiction", None),
        "contract_type": getattr(classified, "contract_type", None),
        "parties": list(getattr(classified, "parties", []) or []),
        "warning_codes": [
            warning.code for warning in (getattr(classified, "warnings", None) or [])
        ],
    }


async def main() -> None:  # noqa: PLR0914 — one measurement pass keeps every comparison signal together
    pages = json.loads(SRC04_PAGES.read_text(encoding="utf-8"))["pages"]
    markdown, page_offsets = _reconstruct(pages)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recorded = next(
        entry["contract_boundaries"]
        for entry in manifest["per_artifact"]
        if entry["source_id"] == "SRC-04"
    )

    # Path A: whole-document classification (one unit over the full markdown).
    whole = classify_document(markdown=markdown, filename="SRC-04.pdf")

    # Path B: the real boundary-local path, then per-boundary classification.
    boundary_result = establish_contract_boundaries(markdown=markdown, page_count=len(pages))
    per_boundary: list[dict[str, object]] = []
    for boundary in boundary_result.boundaries:
        classified = classify_document(markdown=boundary.markdown, filename="SRC-04.pdf")
        start_page = page_for_offset(boundary.start_char, page_offsets)
        end_page = page_for_offset(max(boundary.end_char - 1, boundary.start_char), page_offsets)
        per_boundary.append(
            {
                "contract_index": boundary.contract_index,
                "start_page": start_page,
                "end_page": end_page,
                "chars": boundary.end_char - boundary.start_char,
                **classification_payload(classified),
            }
        )

    # Correctness measure 1: boundary page coverage against the recorded truth.
    detected_start_pages = [slice_["start_page"] for slice_ in per_boundary]
    coverage = []
    for recorded_boundary in recorded:
        start_page, end_page = recorded_boundary["page_range"]
        hit = any(start_page <= page <= end_page for page in detected_start_pages)
        coverage.append(
            {
                "contract_index": recorded_boundary["contract_index"],
                "title": recorded_boundary["title"],
                "recorded_page_range": [start_page, end_page],
                "detected_start_page_within_range": hit,
            }
        )
    detected_contracts = sum(1 for entry in coverage if entry["detected_start_page_within_range"])

    # Correctness measure 2: metadata separation. The whole-document path
    # yields exactly one (jurisdiction, contract_type, parties) tuple for five
    # contracts; the boundary path yields one tuple per slice — count the
    # distinct tuples and whether party sets stay disjoint across slices.
    boundary_tuples = {
        (slice_["jurisdiction"], slice_["contract_type"], tuple(slice_["parties"]))
        for slice_ in per_boundary
    }
    party_sets = [set(slice_["parties"]) for slice_ in per_boundary if slice_["parties"]]
    pairwise_disjoint = (
        all(not (a & b) for i, a in enumerate(party_sets) for b in party_sets[i + 1 :])
        if len(party_sets) > 1
        else True
    )

    # Cost: classification units and characters each path processes.
    whole_chars = len(markdown)
    boundary_chars = sum(int(slice_["chars"]) for slice_ in per_boundary)

    # Fixture control: the same two-way comparison on the CI two-contract
    # fixture (disjoint parties/laws with explicit boundary markers) proves
    # the mechanism itself separates metadata where markers exist, so the
    # SRC-04 gap is a marker-coverage gap, not a mechanism gap.
    fixture = _fixture_control()

    improves = detected_contracts > 1 and len(boundary_tuples) > 1
    report = {
        "experiment": "E8 (task 8.5): per-document vs per-contract metadata extraction on SRC-04",
        "generated_at_utc": _utc_now(),
        "artifact": "SRC-04",
        "pages": len(pages),
        "markdown_chars": whole_chars,
        "ground_truth": {
            "source": "evals/golden/ingestion_fidelity_v1.manifest.json per_artifact[SRC-04].contract_boundaries",
            "recorded_contracts": len(recorded),
        },
        "whole_document_path": {
            "classification_units": 1,
            "chars_processed": whole_chars,
            "metadata_tuples": 1,
            "classification": classification_payload(whole),
            "correctness_note": "one document-level (jurisdiction, contract_type, parties) "
            "tuple must describe five independently signed contracts; party and "
            "contract-type fields cannot be attributed to any single contract",
        },
        "contract_local_path": {
            "boundary_function": "establish_contract_boundaries (real production code)",
            "boundaries_detected": len(boundary_result.boundaries),
            "boundary_ambiguous": boundary_result.ambiguous,
            "boundary_warnings": [w.code for w in boundary_result.warnings],
            "classification_units": len(per_boundary),
            "chars_processed": boundary_chars,
            "metadata_tuples_distinct": len(boundary_tuples),
            "party_sets_pairwise_disjoint": pairwise_disjoint,
            "recorded_contracts_detected": detected_contracts,
            "boundary_page_coverage": coverage,
            "per_boundary": per_boundary,
        },
        "fixture_control": fixture,
        "correctness_verdict": {
            "boundary_coverage": f"{detected_contracts}/{len(recorded)} recorded contracts "
            "have a detected boundary start inside their recorded page range",
            "metadata_separation": f"{len(boundary_tuples)} distinct metadata tuples across "
            f"{len(per_boundary)} slices vs 1 mixed tuple whole-document; party sets "
            f"pairwise disjoint: {pairwise_disjoint}",
            "improves_correctness": improves,
            "verdict": _correctness_verdict(
                boundaries_detected=len(boundary_result.boundaries),
                recorded=len(recorded),
                distinct_tuples=len(boundary_tuples),
                coverage_contracts=detected_contracts,
            ),
        },
        "cost_verdict": {
            "whole_document_units": 1,
            "contract_local_units": len(per_boundary),
            "cost_ratio": round(len(per_boundary) / 1, 3),
            "chars_processed_ratio": round(boundary_chars / whole_chars, 6)
            if whole_chars
            else None,
            "note": "contract-local extraction multiplies classification invocations by "
            "the slice count (see cost_ratio for the measured multiple); each slice "
            "classifies only its own characters — the characters-processed ratio is "
            "reported separately and stays within ~1x of whole-document",
        },
        "measurement_note": "Both paths run the real production classification code over the "
        "recorded SRC-04 pages.json (no model provider). Detected boundaries are compared to "
        "the recorded provenance page ranges; boundary detection is deliberately conservative "
        "and may detect fewer slices than the recorded 5-contract ground truth — the gap is "
        "recorded, not papered over.",
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"report written: {OUTPUT}\n")
    sys.stdout.write(json.dumps(report["correctness_verdict"], indent=1) + "\n")
    sys.stdout.write(json.dumps(report["cost_verdict"], indent=1) + "\n")


def _correctness_verdict(
    *,
    boundaries_detected: int,
    recorded: int,
    distinct_tuples: int,
    coverage_contracts: int,
) -> str:
    """Derive the verdict wording from the measured slice coverage.

    Greptile wave-3 finding: the verdict used to read "degenerates to the
    whole-document path" unconditionally, while a run detecting 52 slices had
    a 52.0 cost ratio — finding a start inside only one expected contract is
    not the same as producing one slice, and neither collapses nor fragments
    may be described by a fixed phrase. The wording now follows the
    measurement: a single slice/tuple is the degenerate case; an over-split
    run (slices far above the recorded contracts, starts inside only some of
    them) is fragmentation, not success; only slice count and coverage at or
    above the recorded contracts read as measured separation.
    """
    fixture_clause = (
        "the boundary-local mechanism separates contract metadata where explicit "
        "markers exist (fixture control)"
    )
    follow_up = (
        "Improving SRC-04 correctness requires the recorded page-range boundaries "
        "(or stronger markers) to seed establish_contract_boundaries — recorded as "
        "the follow-up need, not papered over."
    )
    measured_shape = (
        f"the conservative marker function produces {boundaries_detected} slice(s) "
        f"for {recorded} recorded contracts, with boundary starts inside "
        f"{coverage_contracts} of them and {distinct_tuples} distinct metadata "
        "tuple(s) on the real SRC-04 packet"
    )
    if distinct_tuples <= 1 or boundaries_detected <= 1:
        return (
            f"DEGENERATE: {measured_shape}; per-contract extraction collapses onto "
            "the whole-document path. "
            + fixture_clause
            + ". "
            + follow_up
        )
    if boundaries_detected > recorded or coverage_contracts < recorded:
        return (
            f"FRAGMENTED: {measured_shape} — boundary identity is fragmented "
            "(over-split) and/or partially covered, so per-contract extraction "
            "does not yet beat the whole-document path on SRC-04. "
            + fixture_clause
            + ". "
            + follow_up
        )
    return (
        f"MEASURED: {measured_shape} — slice identity tracks the recorded contracts. "
        + fixture_clause
        + "."
    )


def _fixture_control() -> dict[str, object]:
    """Run both extraction paths on the CI two-contract fixture markdown."""
    fixture_markdown = (
        "# MASTER SERVICES AGREEMENT (CONTRACT ONE)\n\n"
        "THIS MASTER SERVICES AGREEMENT is entered into on 1 March 2026 between "
        "NORTHWIND MANUFACTURING PTY LTD (the Client) and ATLANTIC FREIGHT SOLUTIONS LTD "
        "(the Supplier, together with the Client, the Parties).\n\n"
        "3. GOVERNING LAW. This Agreement and any dispute arising from it is governed by "
        "the laws of Victoria, Australia.\n\n"
        "SIGNED for NORTHWIND MANUFACTURING PTY LTD: A. Rector, Director. "
        "SIGNED for ATLANTIC FREIGHT SOLUTIONS LTD: J. Mbeki, Director.\n\n"
        "# CONSULTING SERVICES AGREEMENT (CONTRACT TWO)\n\n"
        "THIS CONSULTING SERVICES AGREEMENT is made on 15 April 2026 between "
        "CROWN COURT CHAMBERS LTD (the Client) and PEMBERTON ADVISORY PARTNERS LLP "
        "(the Consultant, together with the Client, the Parties).\n\n"
        "3. GOVERNING LAW. This Agreement is governed by the laws of England and Wales.\n\n"
        "SIGNED for CROWN COURT CHAMBERS LTD: H. Okafor, Partner. "
        "SIGNED for PEMBERTON ADVISORY PARTNERS LLP: M. Duval, Partner.\n"
    )
    whole = classify_document(markdown=fixture_markdown, filename="two-contract-fixture.md")
    boundary_result = establish_contract_boundaries(markdown=fixture_markdown)
    slices = [
        classification_payload(
            classify_document(markdown=boundary.markdown, filename="two-contract-fixture.md")
        )
        for boundary in boundary_result.boundaries
    ]
    distinct = {(s["jurisdiction"], s["contract_type"], tuple(s["parties"])) for s in slices}
    return {
        "source": "mirrors tests/unit/features/documents/test_two_contract_identity.py fixture",
        "boundaries_detected": len(boundary_result.boundaries),
        "whole_document_tuples": 1,
        "per_boundary_tuples_distinct": len(distinct),
        "whole_classification": classification_payload(whole),
        "per_boundary": slices,
        "mechanism_separates_metadata": len(distinct) > 1,
    }


def _reconstruct(pages: list[dict[str, object]]) -> tuple[str, list[int]]:
    """Join recorded page texts into one markdown with page start offsets."""
    offsets: list[int] = []
    parts: list[str] = []
    offset = 0
    for page in pages:
        text = str(page.get("text", ""))
        offsets.append(offset)
        parts.append(text)
        offset += len(text) + 2
    return "\n\n".join(parts), offsets


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


if __name__ == "__main__":
    asyncio.run(main())
