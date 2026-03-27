1. add Field description for tool instead of simple docstrings
2. use structured ouput everywhere for llm output, tool output, MCP output
3. use toons for serialisation before sending to LLM
4. use toons for deserialisation after receiving from LLM
5. use toons for serialisation before sending to tools
6. use toons for deserialisation after receiving from tools
7. use async functions, methods and packages in langchain and langGraph
8. trim/remove tool output in a multi step agent conversation 
9. # CORRECT — use dedicated package imports
from langchain_tavily import TavilySearch 
# WRONG — deprecated community import path
from langchain_community.tools.tavily_search import TavilySearchResults
10. all methods, functions, model and agent invocation should have langsmith decorator for proper obervability
11. always normalise agent state after fetching from checkpointer so that there is no version mismatch
12. have proper retry mechanism for tools with idenpotent execution as mention in langchain docs
13.   use create_agent inside langraph graph node 
from langgraph.graph import StateGraph
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

class LegalState(TypedDict):
    messages: list

# Node 1: Researcher agent
def research_node(state: LegalState) -> LegalState:
    """Full agent as node with middleware."""
    researcher = create_agent(
        model="gpt-4o",
        tools=[search_caselaw, validate_cite],
        middleware=[
            SummarizationMiddleware(),  # Context compaction
        ],
        checkpointer=MemorySaver(),  # Per-node persistence
        system_prompt="Legal researcher. Cite sources.",
    )
    
    result = researcher.invoke(state)
    return {"messages": result["messages"]}

# Node 2: Reviewer agent
def review_node(state: LegalState) -> LegalState:
    reviewer = create_agent(
        model="claude-3-sonnet",
        tools=[analyze_risk, flag_clause],
        system_prompt="Risk reviewer. Conservative analysis.",
    )
    result = reviewer.invoke(state)
    return {"messages": result["messages"]}

# Graph orchestration
graph = StateGraph(LegalState)
graph.add_node("research", research_node)
graph.add_node("review", review_node)
graph.add_edge("research", "review")

app = graph.compile()
14.  use this 
from langgraph.graph.message import add_messages

# Messages are merged by ID (deduplication)
state["messages"] = add_messages(state["messages"], [new_msg])
15. 
