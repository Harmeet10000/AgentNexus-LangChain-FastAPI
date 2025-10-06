"""LangGraph workflow API routes."""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Request

from features.workflows.api.schemas import (
    WorkflowInput,
    WorkflowState,
    WorkflowResult,
    GraphVisualization,
    WorkflowListResponse,
    WorkflowHistoryResponse,
    CheckpointResponse,
    CheckpointRestoreResponse,
)
from features.workflows.services.workflow_service import (
    WorkflowService,
    get_workflow_service,
)
from core.config.logging_config import LoggerAdapter
from shared.schemas.response import http_success, http_error

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


@router.get("/list", response_model=WorkflowListResponse)
async def list_workflows(
    request: Request, service: WorkflowService = Depends(get_workflow_service)
):
    """List all available workflows."""
    try:
        result = await service.list_workflows()

        return http_success(
            request, message="Workflows listed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        return http_error(request, e, 500)


@router.get("/{workflow_id}/schema")
async def get_workflow_schema(
    request: Request,
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Get workflow input/output schema."""
    try:
        result = await service.get_workflow_schema(workflow_id)

        if not result:
            return http_error(request, Exception("Workflow not found"), status_code=404)

        return http_success(
            request, message="Workflow schema retrieved successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to get workflow schema", error=str(e))
        return http_error(request, e, 500)


@router.post("/execute", response_model=WorkflowResult)
async def execute_workflow(
    request: Request,
    background_tasks: BackgroundTasks,
    workflow_request: WorkflowInput,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Execute a workflow."""
    try:
        result = await service.execute_workflow(workflow_request, background_tasks)

        return http_success(
            request, message="Workflow executed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to execute workflow", error=str(e))
        return http_error(request, e, 500)


@router.post("/rag", response_model=WorkflowResult)
async def execute_rag_workflow(
    request: Request,
    query: str,
    namespace: str = "default",
    top_k: int = 5,
    rerank: bool = True,
    use_cache: bool = True,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Execute the RAG workflow."""
    try:
        result = await service.execute_rag_workflow(
            query=query,
            namespace=namespace,
            top_k=top_k,
            rerank=rerank,
            use_cache=use_cache,
        )

        return http_success(
            request, message="RAG workflow executed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to execute RAG workflow", error=str(e))
        return http_error(request, e, 500)


@router.get("/state/{thread_id}", response_model=WorkflowState)
async def get_workflow_state(
    request: Request,
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Get workflow execution state."""
    try:
        result = await service.get_workflow_state(thread_id)

        if not result:
            return http_error(
                request, Exception("Workflow state not found"), status_code=404
            )

        return http_success(
            request, message="Workflow state retrieved successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to get workflow state", error=str(e))
        return http_error(request, e, 500)


@router.post("/pause/{thread_id}")
async def pause_workflow(
    request: Request,
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Pause a running workflow."""
    try:
        result = await service.pause_workflow(thread_id)

        return http_success(
            request, message=f"Workflow {thread_id} paused successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to pause workflow", error=str(e))
        return http_error(request, e, 500)


@router.post("/resume/{thread_id}")
async def resume_workflow(
    request: Request,
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Resume a paused workflow."""
    try:
        result = await service.resume_workflow(thread_id)

        return http_success(
            request, message=f"Workflow {thread_id} resumed successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to resume workflow", error=str(e))
        return http_error(request, e, 500)


@router.get("/{workflow_id}/visualize", response_model=GraphVisualization)
async def visualize_workflow(
    request: Request,
    workflow_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Get workflow graph visualization data."""
    try:
        result = await service.visualize_workflow(workflow_id)

        if not result:
            return http_error(request, Exception("Workflow not found"), status_code=404)

        return http_success(
            request,
            message="Workflow visualization retrieved successfully",
            data=result,
        )

    except Exception as e:
        logger.error("Failed to visualize workflow", error=str(e))
        return http_error(request, e, 500)


@router.get("/history/{thread_id}", response_model=WorkflowHistoryResponse)
async def get_workflow_history(
    request: Request,
    thread_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Get workflow execution history."""
    try:
        result = await service.get_workflow_history(thread_id)

        return http_success(
            request, message="Workflow history retrieved successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to get workflow history", error=str(e))
        return http_error(request, e, 500)


@router.post("/checkpoint/{thread_id}", response_model=CheckpointResponse)
async def create_checkpoint(
    request: Request,
    thread_id: str,
    checkpoint_id: Optional[str] = None,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Create a checkpoint for workflow state."""
    try:
        result = await service.create_checkpoint(thread_id, checkpoint_id)

        return http_success(
            request, message="Checkpoint created successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to create checkpoint", error=str(e))
        return http_error(request, e, 500)


@router.post(
    "/restore/{thread_id}/{checkpoint_id}", response_model=CheckpointRestoreResponse
)
async def restore_checkpoint(
    request: Request,
    thread_id: str,
    checkpoint_id: str,
    service: WorkflowService = Depends(get_workflow_service),
):
    """Restore workflow state from a checkpoint."""
    try:
        result = await service.restore_checkpoint(thread_id, checkpoint_id)

        return http_success(
            request, message="Checkpoint restored successfully", data=result
        )

    except Exception as e:
        logger.error("Failed to restore checkpoint", error=str(e))
        return http_error(request, e, 500)
