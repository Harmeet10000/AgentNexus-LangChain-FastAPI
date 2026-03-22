from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI

class AgentState(TypedDict):
    messages: Annotated[list, "The conversation history"]
    search_context: Annotated[list, "Aggregated code snippets"]

def build_search_graph() -> StateGraph:
    # 1. Initialize modern 2026 LLM with parallel tool calling
    llm = ChatOpenAI(model="gpt-5-turbo-preview").bind_tools([code_search_ripgrep, structural_search])

    workflow = StateGraph(AgentState)

    async def call_model(state: AgentState):
        response = await llm.ainvoke(state["messages"])
        return {"messages": [response]}

    # 2. Define nodes
    workflow.add_node("agent", call_model)
    workflow.add_node(
        "tools",
        ToolNode(
            [code_search_ripgrep, structural_search],
            handle_tool_errors=True,
        ),
    )

    # 3. Define edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", lambda x: "tools" if x["messages"][-1].tool_calls else END)
    workflow.add_edge("tools", "agent")

    return workflow.compile()
