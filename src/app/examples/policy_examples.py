"""Combined policy-registration examples.

Run with ``uv run python -m app.examples.policy_examples``. The module
checks three registration patterns without mutating shipped registries:

- adding an extraction stage;
- adding a retrieval branch and DTO leaf call;
- adding a startup policy with degraded and fatal outcomes.

Expected checks return ``Result`` values. Third-party-style setup failures in
the startup example still raise so the policy runner can classify them.
"""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING

from fastapi import FastAPI
from returns.result import Failure, Success

from app.features.documents.service import (
    RETRIEVAL_BRANCHES,
    RetrievalBranchPolicy,
    RetrievalQuery,
    _BranchInput,
)
from app.features.health.health_check import ALL_PROBES
from app.lifecycle.lifespan import (
    STARTUP_POLICIES,
    StartupPolicy,
    _run_startup_policy,
)
from app.shared.rag.docling.docling_enhanced import (
    EXTRACTION_STAGES,
    ExtractionStage,
)
from app.shared.rag.docling.models import DoclingEnhancementConfig
from app.utils import logger

if TYPE_CHECKING:
    from typing import Any

    from returns.result import Result

    from app.shared.result import FeatureError

    type DocumentResult[T] = Result[T, FeatureError]


def _check(condition: bool, message: str) -> Result[None, str]:
    """Represent a failed example assertion as data."""
    return Success(None) if condition else Failure(message)


# ---------------------------------------------------------------------------
# Extraction-stage registration
# ---------------------------------------------------------------------------


async def _run_toc_stage(doc: Any, _source: str, _config: DoclingEnhancementConfig) -> list[str]:
    return [
        line[3:].strip() for line in doc.export_to_markdown().splitlines() if line.startswith("## ")
    ]


TOC_STAGE = ExtractionStage(
    name="toc",
    enabled=lambda config: config.generate_doctags,
    run=_run_toc_stage,
)


def _stub_doc() -> Any:
    doc = SimpleNamespace()
    doc.export_to_markdown = lambda: "# Title\n\n## Risk\n\n## Term\n"
    doc.export_to_doc_tags = lambda: "<doctags/>"
    return doc


async def _demo_extraction_stage() -> Result[None, str]:
    result = _check(
        [stage.name for stage in EXTRACTION_STAGES] == ["doctags", "tables", "code", "images"],
        "shipped stages keep their documented order",
    )
    if isinstance(result, Failure):
        return result

    extended = (*EXTRACTION_STAGES, TOC_STAGE)
    doc = _stub_doc()
    config = DoclingEnhancementConfig()
    outputs: dict[str, Any] = {}
    for stage in extended:
        if stage.enabled(config):
            outputs[stage.name] = await stage.run(doc, "example.md", config)

    result = _check(outputs["toc"] == ["Risk", "Term"], "the registered stage must run")
    if isinstance(result, Failure):
        return result
    result = _check(outputs["doctags"] == "<doctags/>", "shipped stages still run alongside")
    if isinstance(result, Failure):
        return result

    dark_config = DoclingEnhancementConfig(generate_doctags=False)
    result = _check(TOC_STAGE.enabled(dark_config) is False, "flag off must skip the stage")
    if isinstance(result, Failure):
        return result
    return _check(
        [stage.name for stage in extended if stage.enabled(dark_config)]
        == ["tables", "code", "images"],
        "only flag-gated stages run",
    )


# ---------------------------------------------------------------------------
# Retrieval-branch registration
# ---------------------------------------------------------------------------


class _StubRepo:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def bm25_search(self, **_kwargs: Any) -> DocumentResult[list[dict[str, Any]]]:
        self.calls.append("bm25_search")
        return Success([{"chunk_id": "bm25-1", "score": 0.9}])

    async def vector_search(self, **_kwargs: Any) -> DocumentResult[list[dict[str, Any]]]:
        self.calls.append("vector_search")
        return Success([{"chunk_id": "vec-1", "score": 0.8}])

    async def trigram_search(self, **_kwargs: Any) -> DocumentResult[list[dict[str, Any]]]:
        self.calls.append("trigram_search")
        return Success([{"chunk_id": "tri-1", "score": 0.7}])

    async def exact_phrase_search(self, **_kwargs: Any) -> DocumentResult[list[dict[str, Any]]]:
        self.calls.append("exact_phrase_search")
        return Success([{"chunk_id": "exact-1", "score": 1.0}])

    async def legal_rrf_search(self, **kwargs: Any) -> DocumentResult[list[dict[str, Any]]]:
        self.calls.append("legal_rrf_search")
        self.last_kwargs = kwargs
        return Success([])


async def _run_exact_phrase_branch(
    repo: Any, args: _BranchInput
) -> DocumentResult[list[dict[str, Any]]]:
    return await repo.exact_phrase_search(
        user_id=args.user_id,
        phrase=args.query_text,
        candidate_limit=args.candidate_limit,
        filter_params=args.filter_params,
    )


async def _demo_retrieval_policy() -> Result[None, str]:
    repo: Any = _StubRepo()
    query = RetrievalQuery(
        user_id="user-1",
        query_text="indemnity cap",
        query_embedding=[0.0, 0.1, 0.2],
        limit=20,
        vector_weight=0.4,
        keyword_weight=0.6,
        jurisdiction="US-CA",
        document_ids=["doc-1"],
        chunk_ids=None,
    )
    await repo.legal_rrf_search(**query.model_dump())
    result = _check(repo.calls == ["legal_rrf_search"], "DTO must unpack to one leaf call")
    if isinstance(result, Failure):
        return result
    result = _check(
        repo.last_kwargs["query_text"] == "indemnity cap"
        and repo.last_kwargs["jurisdiction"] == "US-CA",
        "DTO fields must survive the round-trip",
    )
    if isinstance(result, Failure):
        return result

    extended = (
        *RETRIEVAL_BRANCHES,
        RetrievalBranchPolicy(name="exact_phrase", run=_run_exact_phrase_branch),
    )
    branch_input = _BranchInput(
        user_id="user-1",
        query_text="indemnity cap",
        query_embedding=[0.0, 0.1, 0.2],
        candidate_limit=50,
        filter_params={},
    )
    results = await asyncio.gather(*(branch.run(repo, branch_input) for branch in extended))
    result = _check(
        [branch.name for branch in extended] == ["bm25", "vector", "trigram", "exact_phrase"],
        "the fourth branch registers alongside the shipped branches",
    )
    if isinstance(result, Failure):
        return result
    result = _check(len(results) == 4, "every registered branch must run")
    if isinstance(result, Failure):
        return result
    return _check(
        repo.calls[-4:]
        == ["bm25_search", "vector_search", "trigram_search", "exact_phrase_search"],
        "branches must execute in registry order",
    )


# ---------------------------------------------------------------------------
# Startup-policy registration
# ---------------------------------------------------------------------------


def _stub_app() -> Any:
    return SimpleNamespace(state=SimpleNamespace())


async def _setup_feature_flags(app: FastAPI, _settings: Any) -> None:
    app.state.feature_flags = {"new_ui": True}


def _report_feature_flags_degraded(exc: BaseException) -> None:
    exc.add_note("operation=setup_feature_flags")


FEATURE_FLAGS_POLICY = StartupPolicy(
    name="feature_flags",
    setup=_setup_feature_flags,
    state_attr="feature_flags",
    fatal_on=(),
    degrade_on=(ConnectionError, OSError),
    report=_report_feature_flags_degraded,
)


async def _failing_setup(_app: FastAPI, _settings: Any) -> None:
    message = "flags endpoint unreachable"
    raise ConnectionError(message)


async def _fatal_setup(_app: FastAPI, _settings: Any) -> None:
    message = "flags schema is from the future; refusing to guess"
    raise ValueError(message)


async def _demo_startup_policy() -> Result[None, str]:
    result = _check(
        [policy.name for policy in STARTUP_POLICIES]
        == ["cognee", "graphiti", "crawl4ai", "object_storage", "celery", "outbox_relay"],
        "policies must boot in dependency order",
    )
    if isinstance(result, Failure):
        return result
    linked = [policy.probe for policy in STARTUP_POLICIES if policy.probe is not None]
    result = _check(bool(linked), "startup policies must link dependency probes")
    if isinstance(result, Failure):
        return result
    result = _check(all(probe in ALL_PROBES for probe in linked), "linked probes must be canonical")
    if isinstance(result, Failure):
        return result

    app = _stub_app()
    await _run_startup_policy(app, object(), FEATURE_FLAGS_POLICY)
    result = _check(app.state.feature_flags == {"new_ui": True}, "setup must land state")
    if isinstance(result, Failure):
        return result

    app = _stub_app()
    degraded = FEATURE_FLAGS_POLICY._replace(setup=_failing_setup)
    await _run_startup_policy(app, object(), degraded)
    result = _check(app.state.feature_flags is None, "degradable failure must land None")
    if isinstance(result, Failure):
        return result

    fatal = FEATURE_FLAGS_POLICY._replace(setup=_fatal_setup, fatal_on=(ValueError,))
    propagated = False
    try:
        await _run_startup_policy(_stub_app(), object(), fatal)
    except ValueError:
        propagated = True
    return _check(propagated, "fatal setup error must propagate out of boot")


async def _demo() -> Result[None, str]:
    for demo in (_demo_extraction_stage, _demo_retrieval_policy, _demo_startup_policy):
        result = await demo()
        if isinstance(result, Failure):
            return result
    return Success(None)


if __name__ == "__main__":
    outcome = asyncio.run(_demo())
    if isinstance(outcome, Failure):
        logger.bind(check=outcome.failure()).error("Policy example check failed")
        sys.exit(1)
