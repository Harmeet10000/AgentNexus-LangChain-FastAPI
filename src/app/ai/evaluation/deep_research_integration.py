"""Adapters joining bounded deep research to evaluation instrumentation."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from app.ai.evaluation.process_metrics import NodeTransition, ProcessMetricsTracker
from app.ai.evaluation.spot_check import Citation, ReportSection, run_spot_checks

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from app.ai.evaluation.process_metrics import ProcessMetricSnapshot
    from app.ai.evaluation.spot_check import AsyncIndexClient, RetrievedChunk, SpotCheckResult


class DeepResearchInstrumentation:
    """Collect deterministic process metrics for the bounded research graph."""

    def __init__(
        self,
        tracker: ProcessMetricsTracker | None = None,
        *,
        clock: Callable[[], datetime] = lambda: datetime.now(tz=UTC),
    ) -> None:
        self._tracker = tracker or ProcessMetricsTracker()
        self._clock = clock
        self._operation_id = ""
        self._start_time: datetime | None = None

    def start(self, objective: str, node_limit: int) -> None:
        """Begin a fresh instrumentation operation."""
        self._start_time = self._clock()
        self._operation_id = str(uuid4())
        self._tracker.start_tracking(objective, node_limit)

    def record_stage(
        self,
        stage: str,
        input_state: dict[str, object],
        output_state: dict[str, object],
        tool_calls: list[str] | None = None,
    ) -> None:
        """Record one graph-stage transition using the injected clock."""
        output = dict(output_state)
        evidence = output.get("evidence")
        if isinstance(evidence, list):
            output["retrieved_document_ids"] = [
                document_id for item in evidence if (document_id := _document_id(item)) is not None
            ]
        queries = output.get("queries")
        if stage == "plan" and isinstance(queries, list):
            output["primary_queries"] = queries[:1]
            output["primary_query_count"] = 1
            output["follow_up_queries"] = queries[1:]
        summary = output.get("summary")
        if stage == "synthesize" and isinstance(summary, str):
            output["report_word_count"] = len(summary.split())
        self._tracker.record_node_transition(
            NodeTransition(
                node_name=stage,
                timestamp=self._clock(),
                input_state=dict(input_state),
                output_state=output,
                tool_calls=tool_calls or [],
            )
        )

    def finish(self, report_word_count: int) -> ProcessMetricSnapshot:
        """Build a snapshot using the supplied final report word count."""
        if self._start_time is None:
            message = "finish called before start"
            raise RuntimeError(message)
        snapshot = self._tracker.build_snapshot(operation_id=self._operation_id)
        ratio = (
            snapshot.total_retrieved_documents / report_word_count if report_word_count > 0 else 0.0
        )
        return snapshot.model_copy(
            update={
                "start_time": self._start_time,
                "end_time": self._tracker.transitions[-1].timestamp
                if self._tracker.transitions
                else self._start_time,
                "final_report_word_count": report_word_count,
                "brief_to_report_coverage_ratio": ratio,
            }
        )


_CITATION_PATTERN = re.compile(r"\[([^\]]+)\]")


def _as_string_mapping(value: object) -> Mapping[str, object] | None:
    if not isinstance(value, dict):
        return None
    return cast("dict[str, object]", value)


def _document_id(value: object) -> str | None:
    item = _as_string_mapping(value)
    document_id = item.get("document_id") if item is not None else None
    return document_id if isinstance(document_id, str) else None


def _mapping_list(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for value_item in value if (item := _as_string_mapping(value_item)) is not None]


async def spot_check_deep_research_result(
    *,
    final_state: dict[str, object],
    retrieved_chunks: list[RetrievedChunk],
    index_client: AsyncIndexClient | None = None,
) -> list[SpotCheckResult]:
    """Adapt the bounded subgraph's metadata and notes for spot checking."""
    evidence = _mapping_list(final_state.get("evidence"))
    handles: dict[str, str] = {}
    for item in evidence:
        evidence_id = item.get("evidence_id")
        chunk_id = item.get("chunk_id")
        if isinstance(evidence_id, str) and isinstance(chunk_id, str):
            handles[evidence_id] = chunk_id

    citations = [Citation(chunk_id=chunk_id) for chunk_id in handles.values()]
    notes_value = final_state.get("read_notes")
    notes = (
        [note for note in notes_value if isinstance(note, str)]
        if isinstance(notes_value, list)
        else []
    )
    summary_value = final_state.get("summary")
    summary = summary_value if isinstance(summary_value, str) else ""

    sections: list[ReportSection] = []
    for index, note in enumerate(notes, start=1):
        parsed_handles = _CITATION_PATTERN.findall(note)
        note_citations = [
            Citation(chunk_id=handles[handle]) for handle in parsed_handles if handle in handles
        ]
        if not note_citations and index <= len(citations):
            note_citations = citations[index - 1 : index]
        sections.append(
            ReportSection(
                section_id=f"research-{index}",
                title=f"Research note {index}",
                content=f"{summary}\n\n{note}" if index == 1 and summary else note,
                citations=note_citations,
            )
        )
    if not sections and summary:
        sections.append(
            ReportSection(
                section_id="research-summary",
                title="Research summary",
                content=summary,
                citations=citations,
            )
        )

    return await run_spot_checks(
        report_sections=sections,
        retrieved_chunks=retrieved_chunks,
        index_client=index_client,
    )
