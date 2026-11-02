"""D7 task 4.1 / ADR-3: the tool-calling seams keep their middleware boundary.

The validation layer owns semantic gates over values that already parsed; the
tool-calling seams' transport retry stays exactly where it was before the
adoption. This test pins the pre-adoption middleware configuration of Saul's
two ``create_agent`` seams (risk, compliance) so the semantic layer cannot
silently absorb transport retry (reference 14: output-validation-and-retry).
"""

from typing import Any

import pytest
from tests.unit.features.agent_saul.graph.test_factory_bindings import (
    _build_with_captured_create_agent,
    _bundle,
    _StubLLM,
)

from app.features.agent_saul.graph import factory as saul_factory

pytestmark = pytest.mark.unit

# The pre-adoption configuration, captured verbatim: both tool-calling seams
# are built with an EMPTY inline middleware list. ModelRetryMiddleware /
# ToolRetryMiddleware are deliberately absent here — the retry allowlist is
# supplied at the application middleware configuration seam (factory.py), and
# transport retry for these seams must not move into the validation layer.
_PRE_ADOPTION_MIDDLEWARE: list[Any] = []


def test_tool_calling_seams_keep_their_preadoption_middleware_stack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.features.agent_saul.graph.prompts import (
        _COMPLIANCE_SYSTEM_PROMPT,
        _RISK_ANALYSIS_SYSTEM_PROMPT,
    )

    captured = _build_with_captured_create_agent(monkeypatch)
    registry = saul_factory.build_agent_registry(_StubLLM(), _StubLLM(), tools=_bundle())  # type: ignore[arg-type]

    assert registry.risk_agent is not None
    for prompt in (_RISK_ANALYSIS_SYSTEM_PROMPT, _COMPLIANCE_SYSTEM_PROMPT):
        seam = captured[prompt[:24]]
        assert seam["middleware"] == _PRE_ADOPTION_MIDDLEWARE
        # Equality with the pre-adoption configuration means no transport
        # retry middleware was folded into the semantic layer.
        assert not any(
            type(m).__name__ in {"ModelRetryMiddleware", "ToolRetryMiddleware"}
            for m in seam["middleware"]
        )
