"""LangGraph workflow API schemas."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


class WorkflowInput(BaseModel):
    """Workflow input model."""
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    inputs: Dict[str, Any] = Field(..., description="Input data for the workflow")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Workflow configuration")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for stateful workflows")
    stream: bool = Field(default=False, description="Stream workflow events")


class WorkflowState(BaseModel):
    """Workflow state model."""
    state_id: str
    workflow_id: str
    thread_id: str
    current_node: str
    status: Literal["pending", "running", "completed", "failed", "paused"]
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class WorkflowResult(BaseModel):
    """Workflow execution result."""
    workflow_id: str
    thread_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    steps: Optional[List[Dict[str, Any]]] = None


class GraphVisualization(BaseModel):
    """Graph visualization data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


class WorkflowListResponse(BaseModel):
    """Workflow list response model."""
    workflows: List[Dict[str, Any]]
    count: int


class WorkflowHistoryResponse(BaseModel):
    """Workflow history response model."""
    thread_id: str
    history: List[Dict[str, Any]]
    count: int


class CheckpointResponse(BaseModel):
    """Checkpoint response model."""
    message: str
    checkpoint_id: str


class CheckpointRestoreResponse(BaseModel):
    """Checkpoint restore response model."""
    message: str
    checkpoint_id: str
    restored_at: str