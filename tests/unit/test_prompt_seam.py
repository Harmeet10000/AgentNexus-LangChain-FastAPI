"""Band: agent-tools-unification group 7 — kinded prompt assembly + cache seam."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from app.shared.langchain_layer.prompts import (
    PromptSection,
    SectionKind,
    SystemPromptParts,
    assemble_kinded_sections,
    build_assembled_prompt,
)


def _section(kind: Any, body: str, label: str = "") -> PromptSection:
    return PromptSection(kind=kind, body=body, label=label)


# --- 7.1 kind ordering ---


def test_scrambled_sections_emit_in_kind_order() -> None:
    sections = [
        _section(SectionKind.TASK, "analyse the clause"),
        _section(SectionKind.EVIDENCE, "[1] precedent A"),
        _section(SectionKind.OUTPUT_CONTRACT, "return JSON"),
        _section(SectionKind.INSTRUCTION, "be precise"),
    ]
    out = assemble_kinded_sections(sections)
    positions = [
        out.index("be precise"),
        out.index("return JSON"),
        out.index("[1] precedent A"),
        out.index("analyse the clause"),
    ]
    assert positions == sorted(positions), "instruction -> contract -> evidence -> task"


def test_labels_never_influence_ordering() -> None:
    a = assemble_kinded_sections([_section(SectionKind.INSTRUCTION, "b", label="HEADER ONE")])
    b = assemble_kinded_sections([_section(SectionKind.INSTRUCTION, "b", label="totally different")])
    assert a.splitlines()[0] != b.splitlines()[0], "labels differ"
    # Same kind → same position in the emitted order.
    two = assemble_kinded_sections(
        [
            _section(SectionKind.EVIDENCE, "e", label="L-A"),
            _section(SectionKind.INSTRUCTION, "i", label="L-B"),
            _section(SectionKind.EVIDENCE, "e2", label="L-C"),
        ]
    )
    assert two.index("L-B") < two.index("L-A") and two.index("L-A") < two.index("L-C")


def test_an_empty_section_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _section(SectionKind.TASK, "   ")


# --- 7.2 cacheable preamble / ranked evidence ---


def test_preamble_is_byte_identical_across_different_evidence() -> None:
    parts = SystemPromptParts()
    p1 = build_assembled_prompt(parts, task="t1", evidence=["only evidence"])
    p2 = build_assembled_prompt(parts, task="t2", evidence=["a", "b", "c"])
    assert p1.preamble == p2.preamble
    assert p1.preamble in p1.render() and p1.preamble in p2.render()


def test_evidence_order_is_preserved_and_ranked() -> None:
    assembled = build_assembled_prompt(SystemPromptParts(), task="t", evidence=["first", "second", "third"])
    block = assembled.evidence_block
    assert block.index("first") < block.index("second") < block.index("third")
    assert "[1] first" in block and "[3] third" in block


def test_task_comes_after_evidence_in_render() -> None:
    assembled = build_assembled_prompt(SystemPromptParts(), task="the task", evidence=["ev"])
    rendered = assembled.render()
    assert rendered.index("ev") < rendered.index("the task")


# --- 7.4 tabular payloads through the toon seam ---


def test_tabular_evidence_is_serialised_at_the_seam() -> None:
    assembled = build_assembled_prompt(
        SystemPromptParts(),
        task="t",
        evidence=[{"statute": "s 73", "rank": 1}],
    )
    assert isinstance(assembled.evidence[0], str)
    assert "statute" in assembled.evidence[0]
