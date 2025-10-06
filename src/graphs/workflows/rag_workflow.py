"""RAG workflow using LangGraph."""

from typing import TypedDict, List, Dict, Any, Optional
from enum import Enum

from langgraph.graph import StateGraph, START, END

# ... no prebuilt ToolExecutor used here
from langchain.schema import BaseMessage, HumanMessage, AIMessage

from src.core.config.logging_config import LoggerAdapter
from src.services.pinecone.client import vector_store_service
from src.services.langchain.gemini_service import gemini_service

logger = LoggerAdapter(__name__)


class RAGState(TypedDict):
    """State for RAG workflow."""

    query: str
    context: List[str]
    messages: List[BaseMessage]
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    error: Optional[str]


class RAGWorkflowSteps(str, Enum):
    """RAG workflow steps."""

    RETRIEVE = "retrieve"
    RERANK = "rerank"
    GENERATE = "generate"
    VALIDATE = "validate"


class RAGWorkflow:
    """RAG workflow implementation using LangGraph."""

    def __init__(self):
        """Initialize RAG workflow."""
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> Any:
        """Build the RAG workflow graph."""
        workflow = StateGraph(RAGState)

        # Add nodes (use string node names)
        workflow.add_node(RAGWorkflowSteps.RETRIEVE.value, self._retrieve_context)
        workflow.add_node(RAGWorkflowSteps.RERANK.value, self._rerank_results)
        workflow.add_node(RAGWorkflowSteps.GENERATE.value, self._generate_answer)
        workflow.add_node(RAGWorkflowSteps.VALIDATE.value, self._validate_answer)

        # Define entry and edges.
        # LangGraph examples typically use START to mark the entry point.
        # For now we wire a simple deterministic flow: START -> retrieve -> rerank -> generate -> validate -> END
        # Note: conditional branching using add_conditional_edges is not part of all LangGraph versions.
        workflow.add_edge(START, RAGWorkflowSteps.RETRIEVE.value)
        workflow.add_edge(
            RAGWorkflowSteps.RETRIEVE.value, RAGWorkflowSteps.RERANK.value
        )
        workflow.add_edge(
            RAGWorkflowSteps.RERANK.value, RAGWorkflowSteps.GENERATE.value
        )
        workflow.add_edge(
            RAGWorkflowSteps.GENERATE.value, RAGWorkflowSteps.VALIDATE.value
        )
        workflow.add_edge(RAGWorkflowSteps.VALIDATE.value, END)

        return workflow.compile()

    async def _retrieve_context(self, state: RAGState) -> RAGState:
        """Retrieve relevant context from vector store."""
        try:
            logger.info("Retrieving context", query=state["query"])

            # Perform similarity search
            results = (
                await vector_store_service.similarity_search_with_relevance_scores(
                    query=state["query"], k=5, score_threshold=0.4
                )
            )

            if not results:
                state["error"] = "No relevant documents found"
                state["context"] = []
                state["sources"] = []
            else:
                state["context"] = [r["content"] for r in results]
                state["sources"] = results
                state["error"] = None

            logger.info(f"Retrieved {len(results)} documents")
            return state

        except Exception as e:
            logger.error("Failed to retrieve context", error=str(e))
            state["error"] = f"Retrieval failed: {str(e)}"
            return state

    async def _rerank_results(self, state: RAGState) -> RAGState:
        """Rerank search results for better relevance."""
        try:
            logger.info("Reranking results")

            # Simple reranking based on keyword matching
            query_keywords = set(state["query"].lower().split())

            ranked_sources = []
            for source in state["sources"]:
                content_keywords = set(source["content"].lower().split())
                overlap = len(query_keywords & content_keywords)
                source["rerank_score"] = overlap
                ranked_sources.append(source)

            # Sort by rerank score and original score
            ranked_sources.sort(
                key=lambda x: (x["rerank_score"], x["score"]), reverse=True
            )

            # Update context with reranked results
            state["context"] = [s["content"] for s in ranked_sources[:3]]
            state["sources"] = ranked_sources[:3]

            logger.info(f"Reranked to {len(state['sources'])} documents")
            return state

        except Exception as e:
            logger.error("Failed to rerank results", error=str(e))
            return state

    async def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using context."""
        try:
            logger.info("Generating answer")

            if not state["context"]:
                state["answer"] = (
                    "I couldn't find relevant information to answer your question."
                )
                return state

            # Create prompt with context
            context_str = "\n\n".join(state["context"])
            prompt = f"""Based on the following context, provide a comprehensive answer to the question.
            If the context doesn't contain enough information, say so.

            Context:
            {context_str}

            Question: {state["query"]}

            Answer:"""

            # Generate answer
            answer = await gemini_service.generate_response(
                messages=[{"role": "user", "content": prompt}], temperature=0.7
            )

            state["answer"] = answer

            # Add to messages
            state["messages"].append(HumanMessage(content=state["query"]))
            state["messages"].append(AIMessage(content=answer))

            logger.info("Generated answer", length=len(answer))
            return state

        except Exception as e:
            logger.error("Failed to generate answer", error=str(e))
            state["error"] = f"Generation failed: {str(e)}"
            state["answer"] = "I encountered an error while generating the answer."
            return state

    async def _validate_answer(self, state: RAGState) -> RAGState:
        """Validate and potentially improve the answer."""
        try:
            logger.info("Validating answer")

            # Check if answer is too short or generic
            if len(state["answer"]) < 50 or "I don't know" in state["answer"]:
                state["metadata"]["needs_improvement"] = True
            else:
                state["metadata"]["needs_improvement"] = False

            # Add source citations
            if state["sources"]:
                citations = []
                for i, source in enumerate(state["sources"], 1):
                    if "filename" in source.get("metadata", {}):
                        citations.append(f"[{i}] {source['metadata']['filename']}")

                if citations:
                    state["answer"] += "\n\nSources:\n" + "\n".join(citations)

            logger.info("Validation complete")
            return state

        except Exception as e:
            logger.error("Failed to validate answer", error=str(e))
            return state

    def _should_rerank(self, state: RAGState) -> str:
        """Decide whether to rerank results."""
        if state.get("error"):
            return "end"

        if len(state.get("sources", [])) > 3:
            return "rerank"

        if state.get("sources"):
            return "generate"

        return "end"

    def _should_retry(self, state: RAGState) -> str:
        """Decide whether to retry generation."""
        if (
            state.get("metadata", {}).get("needs_improvement")
            and state.get("metadata", {}).get("retry_count", 0) < 2
        ):
            state["metadata"]["retry_count"] = (
                state["metadata"].get("retry_count", 0) + 1
            )
            return "retry"

        return "end"

    async def run(
        self, query: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Run the RAG workflow."""
        try:
            # Initialize state
            initial_state: RAGState = {
                "query": query,
                "context": [],
                "messages": [],
                "answer": "",
                "sources": [],
                "metadata": metadata or {},
                "error": None,
            }

            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)

            # Return results
            return {
                "answer": final_state["answer"],
                "sources": final_state["sources"],
                "error": final_state.get("error"),
                "metadata": final_state["metadata"],
            }

        except Exception as e:
            logger.error("Failed to run RAG workflow", error=str(e))
            return {
                "answer": "An error occurred while processing your request.",
                "sources": [],
                "error": str(e),
                "metadata": {},
            }


# Create global instance
rag_workflow = RAGWorkflow()
