# LangChain/LangGraph v1.0+ Implementation Guide with Code Snippets

## Overview
Production-grade conversational chatbot implementation using LangChain, LangGraph v1.0+, and FastAPI with complete code examples.

---

## 📁 Project Structure

```
src/
├── main.py                          # FastAPI app entry
├── api/                             # FastAPI Layer
│   ├── endpoints/
│   │   ├── chat.py                 # Chat endpoints
│   │   ├── agents.py               # Agent endpoints (NEW)
│   │   └── workflows.py            # Workflow endpoints
│   ├── router.py
│   └── middleware/
├── agents/                          # LangChain Agents (NEW)
│   ├── react/
│   ├── reflection/
│   └── reflexion/
├── chains/                          # LangChain Chains
│   ├── conversation/
│   ├── rag/
│   └── structured/
├── graphs/                          # LangGraph Workflows
│   ├── workflows/
│   ├── states/
│   └── nodes/
├── tools/                           # LangChain Tools (NEW)
│   ├── search/
│   ├── computation/
│   └── information/
├── prompts/                         # Prompt Templates (NEW)
│   ├── system/
│   ├── reasoning/
│   └── fewshot/
├── memory/                          # Memory Systems (NEW)
│   ├── buffer/
│   ├── vector/
│   └── entity/
└── services/
    ├── langchain/
    ├── langgraph/
    └── langsmith/
```

---

## 1. 🔧 Tools Implementation

### 1.1 Base Tool Structure

```python
# src/tools/base_tool.py
from typing import Optional, Type, Any
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun

class BaseToolInput(BaseModel):
    """Base input schema for tools."""
    query: str = Field(description="The input query for the tool")

class CustomBaseTool(BaseTool):
    """Base class for custom tools with enhanced features."""

    name: str
    description: str
    args_schema: Type[BaseModel] = BaseToolInput
    return_direct: bool = False

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous execution."""
        raise NotImplementedError("Must implement _run method")

    async def _arun(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Async execution - preferred for production."""
        raise NotImplementedError("Must implement _arun method")
```

### 1.2 Search Tool Example

```python
# src/tools/search/web_search.py
from typing import Optional
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun
import httpx

class WebSearchInput(BaseModel):
    query: str = Field(description="Search query")
    num_results: int = Field(default=5, description="Number of results")

class WebSearchTool(BaseTool):
    """Web search tool using DuckDuckGo or Tavily."""

    name: str = "web_search"
    description: str = """
    Useful for searching the web for current information.
    Use this when you need recent data, news, or general knowledge.
    Input should be a search query string.
    """
    args_schema: type[BaseModel] = WebSearchInput

    async def _arun(
        self,
        query: str,
        num_results: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute web search asynchronously."""
        try:
            # Example using DuckDuckGo
            from duckduckgo_search import AsyncDDGS

            async with AsyncDDGS() as ddgs:
                results = []
                async for result in ddgs.text(query, max_results=num_results):
                    results.append({
                        "title": result.get("title", ""),
                        "snippet": result.get("body", ""),
                        "url": result.get("href", "")
                    })

                return self._format_results(results)

        except Exception as e:
            return f"Search error: {str(e)}"

    def _run(self, query: str, num_results: int = 5) -> str:
        """Sync wrapper - not recommended for production."""
        import asyncio
        return asyncio.run(self._arun(query, num_results))

    def _format_results(self, results: list) -> str:
        """Format search results."""
        if not results:
            return "No results found."

        formatted = "Search Results:\n\n"
        for i, result in enumerate(results, 1):
            formatted += f"{i}. {result['title']}\n"
            formatted += f"   {result['snippet']}\n"
            formatted += f"   URL: {result['url']}\n\n"

        return formatted
```

### 1.3 Calculator Tool

```python
# src/tools/computation/calculator.py
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Optional
import numexpr

class CalculatorInput(BaseModel):
    expression: str = Field(description="Mathematical expression to evaluate")

class CalculatorTool(BaseTool):
    """Calculator for mathematical operations."""

    name: str = "calculator"
    description: str = """
    Useful for mathematical calculations and numeric operations.
    Input should be a valid mathematical expression.
    Examples: "2 + 2", "sqrt(16)", "log(100)"
    """
    args_schema: type[BaseModel] = CalculatorInput

    def _run(self, expression: str) -> str:
        """Calculate mathematical expression."""
        try:
            # Use numexpr for safe evaluation
            result = numexpr.evaluate(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {str(e)}"

    async def _arun(self, expression: str) -> str:
        """Async version."""
        return self._run(expression)
```

### 1.4 Tool Registry

```python
# src/tools/registry.py
from typing import Dict, List
from langchain.tools import BaseTool
from src.tools.search.web_search import WebSearchTool
from src.tools.computation.calculator import CalculatorTool

class ToolRegistry:
    """Central registry for all tools."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        """Register default tools."""
        self.register(WebSearchTool())
        self.register(CalculatorTool())

    def register(self, tool: BaseTool):
        """Register a new tool."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> BaseTool:
        """Get tool by name."""
        return self._tools.get(name)

    def get_all(self) -> List[BaseTool]:
        """Get all registered tools."""
        return list(self._tools.values())

    def get_by_category(self, category: str) -> List[BaseTool]:
        """Get tools by category."""
        return [t for t in self._tools.values() if category in t.name]

# Global tool registry instance
tool_registry = ToolRegistry()
```

---

## 2. 🤖 ReAct Agent Implementation

### 2.1 ReAct Agent with LangGraph (Official Pattern)

```python
# src/agents/react/react_agent.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import operator

# Define agent state (used internally by create_react_agent)
class AgentState(TypedDict):
    """State for ReAct agent."""
    messages: Annotated[Sequence[BaseMessage], operator.add]

# ReAct prompt template
REACT_SYSTEM_PROMPT = """You are a helpful AI assistant with access to tools.

When solving problems:
1. Think step by step about what you need to do
2. Use available tools when needed
3. Reason about tool outputs
4. Provide a final answer

Available tools:
{tools}

Format your responses as:
Thought: [Your reasoning]
Action: [Tool to use]
Action Input: [Input for the tool]
Observation: [Tool result]
... (repeat Thought/Action/Observation as needed)
Final Answer: [Your final response]
"""

class ReActAgent:
    """ReAct (Reasoning + Acting) agent implementation."""

    def __init__(self, model_name: str = "gemini-pro", temperature: float = 0.7):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature
        )
        self.tools = tool_registry.get_all()
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create ReAct agent graph."""
        # Create prompt with tools
        prompt = ChatPromptTemplate.from_messages([
            ("system", REACT_SYSTEM_PROMPT.format(
                tools="\n".join([f"- {t.name}: {t.description}" for t in self.tools])
            )),
            MessagesPlaceholder(variable_name="messages"),
        ])

        # Bind tools to LLM
        llm_with_tools = self.llm.bind_tools(self.tools)

        # Define agent node
        def agent_node(state: AgentState):
            """Agent reasoning node."""
            messages = state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        # Create tool node
        tool_node = ToolNode(self.tools)

        # Define routing logic
        def should_continue(state: AgentState):
            """Determine if we should continue or end."""
            messages = state["messages"]
            last_message = messages[-1]

            # If no tool calls, we're done
            if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                return "end"
            return "continue"

        # Build graph
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)

        # Set entry point
        workflow.set_entry_point("agent")

        # Add conditional edges
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "continue": "tools",
                "end": END,
            }
        )

        # Tool node always goes back to agent
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def run(self, query: str) -> str:
        """Execute agent with query."""
        initial_state = {
            "messages": [HumanMessage(content=query)]
        }

        result = await self.graph.ainvoke(initial_state)
        return result["messages"][-1].content
```

### 2.2 ReAct Agent API Endpoint

```python
# src/api/endpoints/agents.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.agents.react.react_agent import ReActAgent

router = APIRouter(prefix="/agents", tags=["Agents"])

class AgentRequest(BaseModel):
    query: str
    agent_type: str = "react"
    temperature: float = 0.7

class AgentResponse(BaseModel):
    response: str
    agent_type: str
    steps: list

# Initialize agents
react_agent = ReActAgent()

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute an agent with a query."""
    try:
        if request.agent_type == "react":
            response = await react_agent.run(request.query)

            return AgentResponse(
                response=response,
                agent_type="react",
                steps=[]  # Extract from agent state if needed
            )
        else:
            raise HTTPException(400, f"Unknown agent type: {request.agent_type}")

    except Exception as e:
        raise HTTPException(500, str(e))
```

---

## 3. 🪞 Reflection Agent Implementation

### 3.1 Reflection Pattern with LangGraph

```python
# src/agents/reflection/reflection_agent.py
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END

class ReflectionState(TypedDict):
    """State for reflection agent."""
    query: str
    draft: str
    critique: str
    final_response: str
    iteration: int
    max_iterations: int

REFLECTION_SYSTEM_PROMPT = """You are an AI assistant that generates high-quality responses.
First, create a draft response. Then, critique your own work and improve it.
"""

CRITIQUE_PROMPT = """Review the following response and identify:
1. Factual errors or inaccuracies
2. Missing important information
3. Unclear explanations
4. Ways to improve clarity and completeness

Draft Response:
{draft}

Provide a detailed critique:"""

IMPROVE_PROMPT = """Based on this critique, generate an improved response.

Original Query: {query}
Previous Draft: {draft}
Critique: {critique}

Generate an improved response:"""

class ReflectionAgent:
    """Reflection agent that improves responses through self-critique."""

    def __init__(self, model_name: str = "gemini-pro", max_iterations: int = 2):
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
        self.max_iterations = max_iterations
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create reflection workflow graph."""

        def generate_draft(state: ReflectionState):
            """Generate initial draft response."""
            response = self.llm.invoke([
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=state["query"])
            ])
            return {"draft": response.content, "iteration": 1}

        def critique_draft(state: ReflectionState):
            """Critique the draft response."""
            critique_response = self.llm.invoke([
                SystemMessage(content="You are a critical reviewer."),
                HumanMessage(content=CRITIQUE_PROMPT.format(draft=state["draft"]))
            ])
            return {"critique": critique_response.content}

        def improve_response(state: ReflectionState):
            """Improve based on critique."""
            improved = self.llm.invoke([
                SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                HumanMessage(content=IMPROVE_PROMPT.format(
                    query=state["query"],
                    draft=state["draft"],
                    critique=state["critique"]
                ))
            ])
            return {
                "draft": improved.content,
                "iteration": state["iteration"] + 1
            }

        def should_continue(state: ReflectionState):
            """Check if should continue refining."""
            if state["iteration"] >= state["max_iterations"]:
                return "finalize"
            return "critique"

        def finalize(state: ReflectionState):
            """Finalize the response."""
            return {"final_response": state["draft"]}

        # Build graph
        workflow = StateGraph(ReflectionState)

        workflow.add_node("generate", generate_draft)
        workflow.add_node("critique", critique_draft)
        workflow.add_node("improve", improve_response)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("generate")

        workflow.add_conditional_edges(
            "generate",
            should_continue,
            {"critique": "critique", "finalize": "finalize"}
        )

        workflow.add_edge("critique", "improve")

        workflow.add_conditional_edges(
            "improve",
            should_continue,
            {"critique": "critique", "finalize": "finalize"}
        )

        workflow.add_edge("finalize", END)

        return workflow.compile()

    async def run(self, query: str) -> dict:
        """Execute reflection agent."""
        initial_state = {
            "query": query,
            "draft": "",
            "critique": "",
            "final_response": "",
            "iteration": 0,
            "max_iterations": self.max_iterations
        }

        result = await self.graph.ainvoke(initial_state)
        return {
            "response": result["final_response"],
            "iterations": result["iteration"],
            "final_critique": result["critique"]
        }
```

---

## 4. 📋 Structured Output Implementation

### 4.1 Structured Output with Pydantic

```python
# src/chains/structured/structured_output.py
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser

# Define output schemas
class Person(BaseModel):
    """Information about a person."""
    name: str = Field(description="Person's full name")
    age: Optional[int] = Field(description="Person's age", default=None)
    occupation: Optional[str] = Field(description="Person's occupation", default=None)

class Article(BaseModel):
    """Structured article information."""
    title: str = Field(description="Article title")
    summary: str = Field(description="Brief summary")
    key_points: List[str] = Field(description="Main points")
    sentiment: str = Field(description="Overall sentiment: positive, negative, neutral")

class StructuredOutputChain:
    """Chain for generating structured outputs."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            temperature=0
        )

    async def extract_person_info(self, text: str) -> Person:
        """Extract person information from text."""
        parser = PydanticOutputParser(pydantic_object=Person)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Extract person information from the text."),
            ("human", "{text}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser

        result = await chain.ainvoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })

        return result

    async def analyze_article(self, text: str) -> Article:
        """Analyze article and return structured output."""
        parser = PydanticOutputParser(pydantic_object=Article)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze the article and extract key information."),
            ("human", "{text}\n\n{format_instructions}")
        ])

        chain = prompt | self.llm | parser

        result = await chain.ainvoke({
            "text": text,
            "format_instructions": parser.get_format_instructions()
        })

        return result
```

### 4.2 Function Calling with Structured Output

```python
# src/chains/structured/function_calling.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from typing import List

@tool
def search_database(
    query: str,
    limit: int = 10,
    filters: dict = None
) -> str:
    """
    Search the database with query and filters.

    Args:
        query: Search query string
        limit: Maximum number of results
        filters: Optional filters as dict

    Returns:
        Search results as formatted string
    """
    # Implementation here
    return f"Searched for '{query}' with limit {limit}"

@tool
def send_email(
    to: str,
    subject: str,
    body: str
) -> str:
    """
    Send an email.

    Args:
        to: Recipient email address
        subject: Email subject
        body: Email body content

    Returns:
        Confirmation message
    """
    return f"Email sent to {to}"

class FunctionCallingAgent:
    """Agent with function calling capabilities."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.tools = [search_database, send_email]
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    async def execute(self, query: str) -> str:
        """Execute query with function calling."""
        response = await self.llm_with_tools.ainvoke(query)

        # Check if function was called
        if hasattr(response, 'tool_calls') and response.tool_calls:
            # Execute the function calls
            results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]

                # Find and execute tool
                tool = next(t for t in self.tools if t.name == tool_name)
                result = await tool.ainvoke(tool_args)
                results.append(result)

            return "\n".join(results)

        return response.content
```

---

## 5. 🔄 StateGraph Advanced Patterns

### 5.1 Multi-Agent Collaboration Graph

```python
# src/graphs/workflows/multi_agent_workflow.py
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
import operator

class MultiAgentState(TypedDict):
    """State for multi-agent collaboration."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    research_complete: bool
    analysis_complete: bool
    final_report: str

class MultiAgentWorkflow:
    """Workflow with multiple specialized agents."""

    def __init__(self):
        self.researcher_llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.analyst_llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.writer_llm = ChatGoogleGenerativeAI(model="gemini-pro")
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create multi-agent collaboration graph."""

        def researcher_node(state: MultiAgentState):
            """Research agent - gathers information."""
            query = state["messages"][-1].content

            prompt = f"Research the following topic and gather key facts:\n{query}"
            research = self.researcher_llm.invoke(prompt)

            return {
                "messages": [AIMessage(content=research.content)],
                "current_agent": "analyst",
                "research_complete": True
            }

        def analyst_node(state: MultiAgentState):
            """Analysis agent - analyzes research."""
            research = state["messages"][-1].content

            prompt = f"Analyze this research and identify key insights:\n{research}"
            analysis = self.analyst_llm.invoke(prompt)

            return {
                "messages": [AIMessage(content=analysis.content)],
                "current_agent": "writer",
                "analysis_complete": True
            }

        def writer_node(state: MultiAgentState):
            """Writer agent - creates final report."""
            analysis = state["messages"][-1].content

            prompt = f"Write a comprehensive report based on this analysis:\n{analysis}"
            report = self.writer_llm.invoke(prompt)

            return {
                "messages": [AIMessage(content=report.content)],
                "final_report": report.content
            }

        # Build graph
        workflow = StateGraph(MultiAgentState)

        workflow.add_node("researcher", researcher_node)
        workflow.add_node("analyst", analyst_node)
        workflow.add_node("writer", writer_node)

        workflow.set_entry_point("researcher")
        workflow.add_edge("researcher", "analyst")
        workflow.add_edge("analyst", "writer")
        workflow.add_edge("writer", END)

        return workflow.compile()

    async def run(self, query: str) -> str:
        """Execute multi-agent workflow."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "current_agent": "researcher",
            "research_complete": False,
            "analysis_complete": False,
            "final_report": ""
        }

        result = await self.graph.ainvoke(initial_state)
        return result["final_report"]
```

### 5.2 Human-in-the-Loop Workflow

```python
# src/graphs/workflows/human_in_loop.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

class HumanInLoopWorkflow:
    """Workflow with human intervention points."""

    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro")
        # Add checkpointer for persistence
        self.checkpointer = SqliteSaver.from_conn_string(":memory:")
        self.graph = self._create_graph()

    def _create_graph(self):
        """Create workflow with human checkpoints."""

        def draft_node(state):
            """Generate draft."""
            draft = self.llm.invoke(state["query"])
            return {"draft": draft.content, "status": "needs_review"}

        def human_review_node(state):
            """Human reviews draft - interruption point."""
            # This node will pause execution
            return {"status": "under_review"}

        def finalize_node(state):
            """Finalize after human approval."""
            return {
                "final_output": state["draft"],
                "status": "complete"
            }

        workflow = StateGraph(dict)

        workflow.add_node("draft", draft_node)
        workflow.add_node("human_review", human_review_node)
        workflow.add_node("finalize", finalize_node)

        workflow.set_entry_point("draft")
        workflow.add_edge("draft", "human_review")

        # Add conditional edge after human review
        workflow.add_conditional_edges(
            "human_review",
            lambda s: s.get("approved", False),
            {
                True: "finalize",
                False: "draft"  # Regenerate if not approved
            }
        )

        workflow.add_edge("finalize", END)

        return workflow.compile(
            checkpointer=self.checkpointer,
            interrupt_before=["human_review"]  # Pause here
        )

    async def run(self, query: str, thread_id: str):
        """Execute with thread for resumability."""
        config = {"configurable": {"thread_id": thread_id}}

        result = await self.graph.ainvoke(
            {"query": query, "draft": "", "status": "pending"},
            config=config
        )

        return result

    async def resume(self, thread_id: str, approved: bool):
        """Resume after human review."""
        config = {"configurable": {"thread_id": thread_id}}

        # Update state and continue
        current_state = self.graph.get_state(config)
        current_state.values["approved"] = approved

        result = await self.graph.ainvoke(None, config=config)
        return result
```

---

## 6. 💾 Memory Systems

### 6.1 Conversation Buffer Memory

```python
# src/memory/buffer/conversation_buffer.py
from langchain.memory import ConversationBufferMemory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from typing import List

class RedisBackedMemory(BaseChatMessageHistory):
    """Redis-backed chat history."""

    def __init__(self, session_id: str, redis_client):
        self.session_id = session_id
        self.redis = redis_client
        self.key = f"chat_history:{session_id}"

    def add_message(self, message: BaseMessage) -> None:
        """Add message to history."""
        import json
        msg_dict = {
            "type": message.type,
            "content": message.content
        }
        self.redis.lpush(self.key, json.dumps(msg_dict))
        self.redis.expire(self.key, 86400)  # 24 hours

    def messages(self) -> List[BaseMessage]:
        """Get all messages."""
        import json
        messages = []

        raw_messages = self.redis.lrange(self.key, 0, -1)
        for raw in reversed(raw_messages):
            msg_dict = json.loads(raw)
            if msg_dict["type"] == "human":
                messages.append(HumanMessage(content=msg_dict["content"]))
            elif msg_dict["type"] == "ai":
                messages.append(AIMessage(content=msg_dict["content"]))

        return messages

    def clear(self) -> None:
        """Clear history."""
        self.redis.delete(self.key)

class ConversationMemoryManager:
    """Manage conversation memory."""

    def __init__(self, redis_client):
        self.redis = redis_client

    def get_memory(self, session_id: str) -> ConversationBufferMemory:
        """Get memory for session."""
        chat_history = RedisBackedMemory(session_id, self.redis)

        return ConversationBufferMemory(
            chat_memory=chat_history,
            return_messages=True,
            memory_key="chat_history"
        )
```

### 6.2 Vector-based Memory

```python
# src/memory/vector/vector_memory.py
from langchain.memory import VectorStoreRetrieverMemory
from langchain.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorMemoryManager:
    """Semantic memory using vector store."""

    def __init__(self, pinecone_index):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.vectorstore = Pinecone(
            index=pinecone_index,
            embedding=self.embeddings,
            text_key="text"
        )

    def create_memory(self, namespace: str, k: int = 5):
        """Create vector-backed memory."""
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k, "namespace": namespace}
        )

        return VectorStoreRetrieverMemory(
            retriever=retriever,
            memory_key="relevant_history"
        )
```

---

## 7. 📝 Prompt Templates

### 7.1 Advanced Prompt Templates

```python
# src/prompts/system/agent_system.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate

# System prompts
REACT_AGENT_SYSTEM = """You are a helpful AI assistant with access to various tools.

Your capabilities:
- Web search for current information
- Calculator for math operations
- Database queries for data retrieval

Instructions:
1. Break down complex questions into steps
2. Use tools when needed
3. Reason through each step
4. Provide clear, accurate answers

Format your responses:
Thought: [Your reasoning]
Action: [Tool name]
Action Input: [Tool input]
Observation: [Tool result]
... (repeat as needed)
Final Answer: [Final response]
"""

# Few-shot examples
FEWSHOT_EXAMPLES = [
    {
        "input": "What is 25 * 4 + 10?",
        "output": """Thought: This is a mathematical calculation.
Action: calculator
Action Input: 25 * 4 + 10
Observation: Result: 110
Thought: I have the answer.
Final Answer: The result is 110."""
    },
    {
        "input": "What is the capital of France and its population?",
        "output": """Thought: I need to search for information about Paris.
Action: web_search
Action Input: Paris France capital population
Observation: Paris is the capital of France with a population of approximately 2.2 million.
Thought: I have the information needed.
Final Answer: Paris is the capital of France with a population of about 2.2 million people."""
    }
]

def create_fewshot_prompt():
    """Create few-shot prompt template."""
    example_prompt = ChatPromptTemplate.from_messages([
        ("human", "{input}"),
        ("ai", "{output}")
    ])

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=FEWSHOT_EXAMPLES,
    )

    final_prompt = ChatPromptTemplate.from_messages([
        ("system", REACT_AGENT_SYSTEM),
        few_shot_prompt,
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
    ])

    return final_prompt

# Chain-of-Thought prompt
COT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert problem solver. Always:
1. Break problems into steps
2. Show your reasoning
3. Verify your logic
4. Provide clear explanations"""),
    ("human", """Solve this step by step:

{question}

Let's approach this systematically:""")
])
```

### 7.2 Dynamic Prompt Builder

```python
# src/prompts/dynamic/template_builder.py
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, List

class DynamicPromptBuilder:
    """Build prompts dynamically based on context."""

    def __init__(self):
        self.base_templates = {}

    def build_prompt(
        self,
        task_type: str,
        context: Dict,
        examples: List = None
    ) -> ChatPromptTemplate:
        """Build prompt dynamically."""

        # Base system message
        system_message = self._get_system_message(task_type)

        # Add context
        if context:
            system_message += f"\n\nContext:\n{self._format_context(context)}"

        # Add examples
        messages = [("system", system_message)]

        if examples:
            for ex in examples:
                messages.append(("human", ex["input"]))
                messages.append(("ai", ex["output"]))

        messages.append(("human", "{query}"))

        return ChatPromptTemplate.from_messages(messages)

    def _get_system_message(self, task_type: str) -> str:
        """Get system message for task type."""
        templates = {
            "analysis": "You are an expert analyst. Analyze the data carefully and provide insights.",
            "generation": "You are a creative writer. Generate engaging and original content.",
            "extraction": "You are a data extraction specialist. Extract structured information accurately.",
            "summarization": "You are a summarization expert. Create concise, informative summaries."
        }
        return templates.get(task_type, "You are a helpful AI assistant.")

    def _format_context(self, context: Dict) -> str:
        """Format context dictionary."""
        return "\n".join([f"- {k}: {v}" for k, v in context.items()])
```

---

## 8. 🌐 FastAPI Integration

### 8.1 Complete Agent Endpoints

```python
# src/api/endpoints/agents.py (Complete version)
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from src.agents.react.react_agent import ReActAgent
from src.agents.reflection.reflection_agent import ReflectionAgent

router = APIRouter(prefix="/agents", tags=["Agents"])

# Request/Response models
class AgentRequest(BaseModel):
    query: str = Field(..., description="User query")
    agent_type: str = Field(default="react", description="Agent type: react, reflection, reflexion")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Agent configuration")

class AgentResponse(BaseModel):
    response: str
    agent_type: str
    metadata: Dict[str, Any] = {}

# Initialize agents
react_agent = ReActAgent()
reflection_agent = ReflectionAgent()

@router.post("/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute agent with specified type."""
    try:
        if request.agent_type == "react":
            result = await react_agent.run(request.query)
            return AgentResponse(
                response=result,
                agent_type="react",
                metadata={"tools_used": []}
            )

        elif request.agent_type == "reflection":
            result = await reflection_agent.run(request.query)
            return AgentResponse(
                response=result["response"],
                agent_type="reflection",
                metadata={
                    "iterations": result["iterations"],
                    "final_critique": result["final_critique"]
                }
            )

        else:
            raise HTTPException(400, f"Unknown agent type: {request.agent_type}")

    except Exception as e:
        raise HTTPException(500, f"Agent execution failed: {str(e)}")

@router.get("/types")
async def list_agent_types():
    """List available agent types."""
    return {
        "agent_types": [
            {
                "type": "react",
                "description": "Reasoning and Acting agent with tool use",
                "features": ["tool_calling", "multi_step_reasoning"]
            },
            {
                "type": "reflection",
                "description": "Self-improving agent through critique",
                "features": ["self_critique", "iterative_improvement"]
            },
            {
                "type": "reflexion",
                "description": "Strategy-learning agent with memory",
                "features": ["memory", "strategy_optimization"]
            }
        ]
    }
```

---

## 9. 📊 Complete Usage Examples

### 9.1 Complete Chat Application

```python
# examples/complete_chatbot.py
import asyncio
from src.agents.react.react_agent import ReActAgent
from src.memory.buffer.conversation_buffer import ConversationMemoryManager
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

async def main():
    """Complete chatbot example."""

    # Initialize agent
    agent = ReActAgent(temperature=0.7)

    # Initialize memory
    memory_manager = ConversationMemoryManager(redis_client)
    memory = memory_manager.get_memory("session_123")

    # Create chat loop
    print("Chatbot ready! (Type 'exit' to quit)")

    while True:
        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            break

        # Get response with memory
        response = await agent.run(user_input)

        # Save to memory
        memory.save_context(
            {"input": user_input},
            {"output": response}
        )

        print(f"\nAssistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 10. 🧪 Testing

### 10.1 Agent Testing

```python
# tests/test_agents.py
import pytest
from src.agents.react.react_agent import ReActAgent

@pytest.mark.asyncio
async def test_react_agent_basic():
    """Test basic ReAct agent functionality."""
    agent = ReActAgent()

    response = await agent.run("What is 2 + 2?")

    assert response is not None
    assert "4" in response.lower()

@pytest.mark.asyncio
async def test_react_agent_with_tools():
    """Test agent with tool usage."""
    agent = ReActAgent()

    response = await agent.run("Search for Python tutorials")

    assert response is not None
    # Verify tool was called
```

---

## 📚 Key Takeaways

1. **Tools**: Inherit from `BaseTool`, implement `_arun` for async
2. **Agents**: Use StateGraph for complex reasoning patterns
3. **Memory**: Choose between buffer, vector, or entity memory based on needs
4. **Structured Output**: Use Pydantic models for type safety
5. **Prompts**: Use templates with few-shot examples for better performance
6. **FastAPI**: Create clean endpoints that wrap agent logic

---

**Version**: 1.0
**Last Updated**: 2025-10-04
**Framework Versions**: LangChain 0.2+, LangGraph 0.1+
