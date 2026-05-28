"""LangGraph implementation for Tavily-backed deep research."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, cast

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    filter_messages,
    get_buffer_string,
)
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from app.shared.langchain_layer.models import _build_chat_model
from app.utils import logger

from .config import Configuration
from .prompts import (
    _CLARIFY_WITH_USER_PROMPT,
    _COMPRESS_RESEARCH_SYSTEM_PROMPT,
    _COMPRESS_RESEARCH_USER_PROMPT,
    _FINAL_REPORT_GENERATION_PROMPT,
    _LEAD_RESEARCHER_SYSTEM_PROMPT,
    _RESEARCH_SYSTEM_PROMPT,
    _TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT,
)
from .state import (
    AgentInputState,
    AgentState,
    ClarifyWithUser,
    ConductResearch,
    ResearchComplete,
    ResearcherOutputState,
    ResearcherState,
    ResearchQuestion,
    SupervisorState,
)
from .utils import (
    get_all_tools,
    get_model_token_limit,
    get_notes_from_tool_calls,
    get_today_str,
    is_token_limit_exceeded,
    remove_up_to_last_ai_message,
    think_tool,
)

if TYPE_CHECKING:
    from typing import Any, Literal

    from langchain_core.runnables import RunnableConfig


def _build_model(model_name: str, max_tokens: int) -> Any:
    """Build a shared Gemini model for deep research nodes."""
    return _build_chat_model(
        model_name=model_name,
        max_tokens=max_tokens,
    ).with_config({"tags": ["langsmith:nostream"]})


async def clarify_with_user(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["write_research_brief", "__end__"]]:
    """Ask a clarifying question when the requested research scope is unclear."""
    configurable = Configuration.from_runnable_config(config)
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")

    clarification_model = (
        _build_model(configurable.research_model, configurable.research_model_max_tokens)
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    prompt_content = _CLARIFY_WITH_USER_PROMPT.format(
        messages=get_buffer_string(state["messages"]),
        date=get_today_str(),
    )
    response = cast(
        "ClarifyWithUser", await clarification_model.ainvoke([HumanMessage(content=prompt_content)])
    )

    if response.need_clarification:
        return Command(goto="__end__", update={"messages": [AIMessage(content=response.question)]})
    return Command(
        goto="write_research_brief",
        update={"messages": [AIMessage(content=response.verification)]},
    )


async def write_research_brief(
    state: AgentState,
    config: RunnableConfig,
) -> Command[Literal["research_supervisor"]]:
    """Transform user messages into a structured research brief."""
    configurable = Configuration.from_runnable_config(config)
    research_model = (
        _build_model(configurable.research_model, configurable.research_model_max_tokens)
        .with_structured_output(ResearchQuestion)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    prompt_content = _TRANSFORM_MESSAGES_INTO_RESEARCH_TOPIC_PROMPT.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    response = cast(
        "ResearchQuestion", await research_model.ainvoke([HumanMessage(content=prompt_content)])
    )
    supervisor_system_prompt = _LEAD_RESEARCHER_SYSTEM_PROMPT.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )
    return Command(
        goto="research_supervisor",
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief),
                ],
            },
        },
    )


async def supervisor(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:
    """Plan research and delegate focused topics to researcher subgraphs."""
    configurable = Configuration.from_runnable_config(config)
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]
    research_model = (
        _build_model(configurable.research_model, configurable.research_model_max_tokens)
        .bind_tools(lead_researcher_tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    response = await research_model.ainvoke(state.get("supervisor_messages", []))
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1,
        },
    )


async def supervisor_tools(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "__end__"]]:
    """Execute supervisor tool calls for reflection and research delegation."""
    configurable = Configuration.from_runnable_config(config)
    supervisor_messages = state.get("supervisor_messages", [])
    research_iterations = state.get("research_iterations", 0)
    most_recent_message = cast("Any", supervisor_messages[-1])

    exceeded_iterations = research_iterations > configurable.max_researcher_iterations
    no_tool_calls = not most_recent_message.tool_calls
    research_complete = any(
        tool_call["name"] == "ResearchComplete" for tool_call in most_recent_message.tool_calls
    )
    if exceeded_iterations or no_tool_calls or research_complete:
        return Command(
            goto="__end__",
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", ""),
            },
        )

    all_tool_messages: list[ToolMessage] = [
        ToolMessage(
            content=f"Reflection recorded: {tool_call['args']['reflection']}",
            name="think_tool",
            tool_call_id=tool_call["id"],
        )
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "think_tool"
    ]
    update_payload: dict[str, object] = {"supervisor_messages": []}

    conduct_research_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] == "ConductResearch"
    ]
    if conduct_research_calls:
        allowed_calls = conduct_research_calls[: configurable.max_concurrent_research_units]
        overflow_calls = conduct_research_calls[configurable.max_concurrent_research_units :]
        try:
            tool_results = await asyncio.gather(
                *(
                    researcher_subgraph.ainvoke(
                        cast(
                            "Any",
                            {
                                "researcher_messages": [
                                    HumanMessage(content=tool_call["args"]["research_topic"])
                                ],
                                "research_topic": tool_call["args"]["research_topic"],
                            },
                        ),
                        config,
                    )
                    for tool_call in allowed_calls
                )
            )
        except (RuntimeError, ValueError, AttributeError) as exc:
            logger.bind(error=str(exc)).warning("deep_research_supervisor_tool_failed")
            return Command(
                goto="__end__",
                update={
                    "notes": get_notes_from_tool_calls(supervisor_messages),
                    "research_brief": state.get("research_brief", ""),
                },
            )

        for observation, tool_call in zip(tool_results, allowed_calls, strict=True):
            all_tool_messages.append(
                ToolMessage(
                    content=observation.get(
                        "compressed_research",
                        "Error synthesizing research report: Maximum retries exceeded",
                    ),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        all_tool_messages.extend(
            [
                ToolMessage(
                    content=(
                        "Error: maximum concurrent research units exceeded. "
                        f"Retry with {configurable.max_concurrent_research_units} or fewer units."
                    ),
                    name="ConductResearch",
                    tool_call_id=overflow_call["id"],
                )
                for overflow_call in overflow_calls
            ]
        )

        raw_notes_concat = "\n".join(
            "\n".join(observation.get("raw_notes", [])) for observation in tool_results
        )
        if raw_notes_concat:
            update_payload["raw_notes"] = [raw_notes_concat]

    update_payload["supervisor_messages"] = all_tool_messages
    return Command(goto="supervisor", update=update_payload)


state_graph_factory = cast("Any", StateGraph)
supervisor_builder = state_graph_factory(SupervisorState, config_schema=Configuration)
supervisor_builder.add_node("supervisor", supervisor)
supervisor_builder.add_node("supervisor_tools", supervisor_tools)
supervisor_builder.add_edge(START, "supervisor")
supervisor_subgraph = supervisor_builder.compile()


async def researcher(
    state: ResearcherState,
    config: RunnableConfig,
) -> Command[Literal["researcher_tools"]]:
    """Conduct focused research on one supervisor-assigned topic."""
    configurable = Configuration.from_runnable_config(config)
    tools = await get_all_tools(config)
    if not tools:
        msg = "No research tools are available. Tavily search must be configured."
        raise ValueError(msg)

    researcher_prompt = _RESEARCH_SYSTEM_PROMPT.format(mcp_prompt="", date=get_today_str())
    research_model = (
        _build_model(configurable.research_model, configurable.research_model_max_tokens)
        .bind_tools(tools)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )
    messages = [SystemMessage(content=researcher_prompt), *state.get("researcher_messages", [])]
    response = await research_model.ainvoke(messages)
    return Command(
        goto="researcher_tools",
        update={
            "researcher_messages": [response],
            "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
        },
    )


async def execute_tool_safely(tool_to_call, args: dict[str, object], config: RunnableConfig) -> str:
    """Execute a research tool and convert failures into model-visible observations."""
    try:
        return str(await tool_to_call.ainvoke(args, config))
    except Exception as exc:  # noqa: BLE001 - LangChain tools may wrap arbitrary provider errors.
        return f"Error executing tool: {exc!s}"


async def researcher_tools(
    state: ResearcherState,
    config: RunnableConfig,
) -> Command[Literal["researcher", "compress_research"]]:
    """Execute researcher tool calls."""
    configurable = Configuration.from_runnable_config(config)
    researcher_messages = state.get("researcher_messages", [])
    most_recent_message = cast("Any", researcher_messages[-1])
    if not most_recent_message.tool_calls:
        return Command(goto="compress_research")

    tools = await get_all_tools(config)
    tools_by_name = {tool_to_call.name: tool_to_call for tool_to_call in tools}
    tool_calls = [
        tool_call
        for tool_call in most_recent_message.tool_calls
        if tool_call["name"] in tools_by_name
    ]
    observations = await asyncio.gather(
        *(
            execute_tool_safely(tools_by_name[tool_call["name"]], tool_call["args"], config)
            for tool_call in tool_calls
        )
    )
    tool_outputs = [
        ToolMessage(
            content=observation,
            name=tool_call["name"],
            tool_call_id=tool_call["id"],
        )
        for observation, tool_call in zip(observations, tool_calls, strict=True)
    ]

    exceeded_iterations = state.get("tool_call_iterations", 0) >= configurable.max_react_tool_calls
    research_complete = any(tool_call["name"] == "ResearchComplete" for tool_call in tool_calls)
    if exceeded_iterations or research_complete:
        return Command(goto="compress_research", update={"researcher_messages": tool_outputs})
    return Command(goto="researcher", update={"researcher_messages": tool_outputs})


async def compress_research(
    state: ResearcherState, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Compress raw researcher messages into a concise summary."""
    configurable = Configuration.from_runnable_config(config)
    synthesizer_model = _build_model(
        configurable.compression_model,
        configurable.compression_model_max_tokens,
    )
    researcher_messages = [
        *state.get("researcher_messages", []),
        HumanMessage(content=_COMPRESS_RESEARCH_USER_PROMPT),
    ]

    for _ in range(3):
        try:
            compression_prompt = _COMPRESS_RESEARCH_SYSTEM_PROMPT.format(date=get_today_str())
            response = await synthesizer_model.ainvoke(
                [SystemMessage(content=compression_prompt), *researcher_messages]
            )
            raw_notes_content = "\n".join(
                str(message.content)
                for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
            )
            return {"compressed_research": str(response.content), "raw_notes": [raw_notes_content]}
        except Exception as exc:  # noqa: BLE001 - model providers expose varied exception classes.
            if is_token_limit_exceeded(exc, configurable.compression_model):
                researcher_messages = remove_up_to_last_ai_message(researcher_messages)
                continue

    raw_notes_content = "\n".join(
        str(message.content)
        for message in filter_messages(researcher_messages, include_types=["tool", "ai"])
    )
    return {
        "compressed_research": "Error synthesizing research report: Maximum retries exceeded",
        "raw_notes": [raw_notes_content],
    }


researcher_builder = state_graph_factory(
    ResearcherState,
    output=ResearcherOutputState,
    config_schema=Configuration,
)
researcher_builder.add_node("researcher", researcher)
researcher_builder.add_node("researcher_tools", researcher_tools)
researcher_builder.add_node("compress_research", compress_research)
researcher_builder.add_edge(START, "researcher")
researcher_builder.add_edge("compress_research", END)
researcher_subgraph = researcher_builder.compile()


async def final_report_generation(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, object]:
    """Generate the final research report from compressed findings."""
    configurable = Configuration.from_runnable_config(config)
    writer_model = _build_model(
        configurable.final_report_model,
        configurable.final_report_model_max_tokens,
    )
    notes = state.get("notes", [])
    cleared_state = {"notes": {"type": "override", "value": []}}
    findings = "\n".join(notes)
    findings_token_limit: int | None = None

    for current_retry in range(4):
        try:
            final_report_prompt = _FINAL_REPORT_GENERATION_PROMPT.format(
                research_brief=state.get("research_brief", ""),
                messages=get_buffer_string(state.get("messages", [])),
                findings=findings,
                date=get_today_str(),
            )
            final_report = await writer_model.ainvoke([HumanMessage(content=final_report_prompt)])
        except Exception as exc:  # noqa: BLE001 - model providers expose varied exception classes.
            if not is_token_limit_exceeded(exc, configurable.final_report_model):
                return {
                    "final_report": f"Error generating final report: {exc}",
                    "messages": [AIMessage(content="Report generation failed due to an error")],
                    **cleared_state,
                }
            model_token_limit = get_model_token_limit(configurable.final_report_model)
            if not model_token_limit:
                return {
                    "final_report": (
                        "Error generating final report: token limit exceeded and the configured "
                        "model context length is unknown."
                    ),
                    "messages": [AIMessage(content="Report generation failed due to token limits")],
                    **cleared_state,
                }
            findings_token_limit = (
                model_token_limit * 4
                if current_retry == 0 or findings_token_limit is None
                else int(findings_token_limit * 0.9)
            )
            findings = findings[:findings_token_limit]
        else:
            return {
                "final_report": final_report.content,
                "messages": [final_report],
                **cleared_state,
            }

    return {
        "final_report": "Error generating final report: Maximum retries exceeded",
        "messages": [AIMessage(content="Report generation failed after maximum retries")],
        **cleared_state,
    }


deep_researcher_builder = state_graph_factory(
    AgentState,
    input=AgentInputState,
    config_schema=Configuration,
)
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)
deep_researcher_builder.add_node("write_research_brief", write_research_brief)
deep_researcher_builder.add_node("research_supervisor", supervisor_subgraph)
deep_researcher_builder.add_node("final_report_generation", final_report_generation)
deep_researcher_builder.add_edge(START, "clarify_with_user")
deep_researcher_builder.add_edge("research_supervisor", "final_report_generation")
deep_researcher_builder.add_edge("final_report_generation", END)
deep_researcher = deep_researcher_builder.compile()
