"""Workflow service implementation."""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import Depends, HTTPException, BackgroundTasks
from shared.utils.id_generator import generate_session_id, generate_short_id

from features.workflows.api.schemas import (
    WorkflowInput, WorkflowState, WorkflowResult, GraphVisualization,
    WorkflowListResponse, WorkflowHistoryResponse, CheckpointResponse,
    CheckpointRestoreResponse
)
from core.config.logging_config import LoggerAdapter
from core.cache.redis_client import redis_cache
from graphs.workflows.rag_workflow import rag_workflow
from services.langgraph import workflow_service as langgraph_service
from services.langsmith.client import langsmith_service

logger = LoggerAdapter(__name__)


class WorkflowService:
    """Service for workflow operations."""
    
    def __init__(self):
        """Initialize workflow service."""
        pass
    
    async def list_workflows(self) -> WorkflowListResponse:
        """List all available workflows."""
        try:
            workflows = langgraph_service.get_available_workflows()
            
            return WorkflowListResponse(
                workflows=workflows,
                count=len(workflows)
            )
            
        except Exception as e:
            logger.error("Failed to list workflows", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_workflow_schema(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow input/output schema."""
        try:
            schema = langgraph_service.get_workflow_schema(workflow_id)
            return schema
            
        except Exception as e:
            logger.error("Failed to get workflow schema", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def execute_workflow(
        self, 
        workflow_request: WorkflowInput,
        background_tasks: BackgroundTasks
    ) -> WorkflowResult:
        """Execute a workflow."""
        try:
            # Generate thread ID if not provided
            thread_id = workflow_request.thread_id or f"thread_{generate_session_id()}"
            
            # Get LangSmith callbacks
            callbacks = langsmith_service.get_callbacks()
            
            # Create workflow state
            state = {
                "state_id": f"state_{generate_short_id()}",
                "workflow_id": workflow_request.workflow_id,
                "thread_id": thread_id,
                "current_node": "start",
                "status": "running",
                "data": workflow_request.inputs,
                "metadata": workflow_request.config,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Store state in cache
            await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            if workflow_request.stream:
                # Execute with streaming
                result = await langgraph_service.execute_workflow_stream(
                    workflow_id=workflow_request.workflow_id,
                    inputs=workflow_request.inputs,
                    config=workflow_request.config,
                    thread_id=thread_id,
                    callbacks=callbacks
                )
            else:
                # Execute synchronously
                result = await langgraph_service.execute_workflow(
                    workflow_id=workflow_request.workflow_id,
                    inputs=workflow_request.inputs,
                    config=workflow_request.config,
                    thread_id=thread_id,
                    callbacks=callbacks
                )
            
            # Update state
            state["status"] = "completed"
            state["updated_at"] = datetime.now().isoformat()
            await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            # Log to LangSmith
            await langsmith_service.log_run(
                run_type="workflow",
                inputs={"workflow_id": workflow_request.workflow_id, "inputs": workflow_request.inputs},
                outputs=result
            )
            
            return WorkflowResult(
                workflow_id=workflow_request.workflow_id,
                thread_id=thread_id,
                status="completed",
                result=result
            )
            
        except Exception as e:
            logger.error("Failed to execute workflow", error=str(e))
            
            # Update state with error
            if 'thread_id' in locals():
                state = await redis_cache.get(f"workflow_state:{thread_id}")
                if state:
                    state["status"] = "failed"
                    state["error"] = str(e)
                    state["updated_at"] = datetime.now().isoformat()
                    await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            raise HTTPException(status_code=500, detail=str(e))
    
    async def execute_rag_workflow(
        self,
        query: str,
        namespace: str = "default",
        top_k: int = 5,
        rerank: bool = True,
        use_cache: bool = True
    ) -> WorkflowResult:
        """Execute the RAG workflow."""
        try:
            # Check cache if enabled
            cache_key = f"rag_result:{hash(query)}:{namespace}:{top_k}"
            if use_cache:
                cached_result = await redis_cache.get(cache_key)
                if cached_result:
                    logger.info("Returning cached RAG result")
                    return WorkflowResult(
                        workflow_id="rag_workflow",
                        thread_id="cached",
                        status="completed",
                        result=cached_result
                    )
            
            # Execute RAG workflow
            result = await rag_workflow.run(
                query=query,
                metadata={
                    "namespace": namespace,
                    "top_k": top_k,
                    "rerank": rerank
                }
            )
            
            # Cache result
            if use_cache:
                await redis_cache.set(cache_key, result, ttl=300)  # 5 minutes
            
            return WorkflowResult(
                workflow_id="rag_workflow",
                thread_id=f"rag_{generate_short_id()}",
                status="completed",
                result=result
            )
            
        except Exception as e:
            logger.error("Failed to execute RAG workflow", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_workflow_state(self, thread_id: str) -> Optional[WorkflowState]:
        """Get workflow execution state."""
        try:
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            
            if not state:
                return None
            
            return WorkflowState(**state)
            
        except Exception as e:
            logger.error("Failed to get workflow state", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def pause_workflow(self, thread_id: str) -> Dict[str, str]:
        """Pause a running workflow."""
        try:
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            
            if not state:
                raise HTTPException(status_code=404, detail="Workflow state not found")
            
            if state["status"] != "running":
                raise HTTPException(status_code=400, detail=f"Workflow is not running (status: {state['status']})")
            
            # Update state
            state["status"] = "paused"
            state["updated_at"] = datetime.now().isoformat()
            await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            # Actual pause logic would be implemented in the workflow service
            await langgraph_service.pause_workflow(thread_id)
            
            return {"message": f"Workflow {thread_id} paused successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to pause workflow", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def resume_workflow(self, thread_id: str) -> Dict[str, str]:
        """Resume a paused workflow."""
        try:
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            
            if not state:
                raise HTTPException(status_code=404, detail="Workflow state not found")
            
            if state["status"] != "paused":
                raise HTTPException(status_code=400, detail=f"Workflow is not paused (status: {state['status']})")
            
            # Update state
            state["status"] = "running"
            state["updated_at"] = datetime.now().isoformat()
            await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            # Actual resume logic would be implemented in the workflow service
            await langgraph_service.resume_workflow(thread_id)
            
            return {"message": f"Workflow {thread_id} resumed successfully"}
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to resume workflow", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def visualize_workflow(self, workflow_id: str) -> Optional[GraphVisualization]:
        """Get workflow graph visualization data."""
        try:
            graph_data = langgraph_service.get_workflow_graph(workflow_id)
            
            if not graph_data:
                return None
            
            return GraphVisualization(
                nodes=graph_data["nodes"],
                edges=graph_data["edges"],
                metadata=graph_data.get("metadata")
            )
            
        except Exception as e:
            logger.error("Failed to visualize workflow", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_workflow_history(self, thread_id: str) -> WorkflowHistoryResponse:
        """Get workflow execution history."""
        try:
            # Get execution history from cache or database
            history_key = f"workflow_history:{thread_id}"
            history = await redis_cache.get(history_key)
            
            if not history:
                # Check if workflow exists
                state = await redis_cache.get(f"workflow_state:{thread_id}")
                if not state:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                
                # Return empty history if no history found
                history = []
            
            return WorkflowHistoryResponse(
                thread_id=thread_id,
                history=history,
                count=len(history)
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to get workflow history", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def create_checkpoint(
        self, 
        thread_id: str, 
        checkpoint_id: Optional[str] = None
    ) -> CheckpointResponse:
        """Create a checkpoint for workflow state."""
        try:
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            
            if not state:
                raise HTTPException(status_code=404, detail="Workflow state not found")
            
            # Generate checkpoint ID if not provided
            if not checkpoint_id:
                checkpoint_id = f"checkpoint_{generate_short_id()}"
            
            # Create checkpoint
            checkpoint = {
                "checkpoint_id": checkpoint_id,
                "thread_id": thread_id,
                "state": state,
                "created_at": datetime.now().isoformat()
            }
            
            # Store checkpoint
            await redis_cache.set(f"workflow_checkpoint:{thread_id}:{checkpoint_id}", checkpoint, ttl=86400)  # 24 hours
            
            return CheckpointResponse(
                message="Checkpoint created successfully",
                checkpoint_id=checkpoint_id
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to create checkpoint", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))
    
    async def restore_checkpoint(
        self, 
        thread_id: str, 
        checkpoint_id: str
    ) -> CheckpointRestoreResponse:
        """Restore workflow state from a checkpoint."""
        try:
            checkpoint = await redis_cache.get(f"workflow_checkpoint:{thread_id}:{checkpoint_id}")
            
            if not checkpoint:
                raise HTTPException(status_code=404, detail="Checkpoint not found")
            
            # Restore state
            state = checkpoint["state"]
            state["updated_at"] = datetime.now().isoformat()
            await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
            
            return CheckpointRestoreResponse(
                message="Checkpoint restored successfully",
                checkpoint_id=checkpoint_id,
                restored_at=datetime.now().isoformat()
            )
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error("Failed to restore checkpoint", error=str(e))
            raise HTTPException(status_code=500, detail=str(e))


def get_workflow_service() -> WorkflowService:
    """Dependency to get workflow service."""
    return WorkflowService()