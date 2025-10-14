"""LangSmith monitoring and tracing service."""

import os
from typing import Optional, Dict, Any, List
from datetime import datetime
import uuid

from langsmith import Client
from langsmith.evaluation import evaluate
from langchain.callbacks.tracers import LangChainTracer
from langchain.callbacks.manager import CallbackManager

from src.app.core.settings import Settings, get_settings
from app.utils.logger import logger


# Global LangSmith client
langsmith_client: Optional[Client] = None
langchain_tracer: Optional[LangChainTracer] = None


def initialize_langsmith():
    """Initialize LangSmith client for monitoring."""
    global langsmith_client, langchain_tracer

    try:
        if not get_settings().LANGSMITH_API_KEY:
            logger.warning("LangSmith API key not provided, monitoring disabled")
            return

        logger.info("Initializing LangSmith")

        # Set environment variables for LangChain tracing
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = get_settings().LANGSMITH_PROJECT
        os.environ["LANGCHAIN_ENDPOINT"] = get_settings().LANGSMITH_ENDPOINT
        os.environ["LANGCHAIN_API_KEY"] = get_settings().LANGSMITH_API_KEY

        # Initialize LangSmith client
        langsmith_client = Client(
            api_url=get_settings().LANGSMITH_ENDPOINT, api_key=get_settings().LANGSMITH_API_KEY
        )

        # Create tracer for callbacks
        langchain_tracer = LangChainTracer(
            project_name=get_settings().LANGSMITH_PROJECT, client=langsmith_client
        )

        logger.info("LangSmith initialized successfully")

    except Exception as e:
        logger.error("Failed to initialize LangSmith", error=str(e))
        # Don't raise - monitoring is optional


class LangSmithService:
    """Service for LangSmith monitoring and evaluation."""

    def __init__(self):
        """Initialize LangSmith service."""
        self.client = langsmith_client
        self.tracer = langchain_tracer
        self.project_name = get_settings().LANGSMITH_PROJECT

    def get_callback_manager(self) -> Optional[CallbackManager]:
        """Get callback manager with LangSmith tracer."""
        if self.tracer:
            return CallbackManager([self.tracer])
        return None

    def get_callbacks(self) -> List:
        """Get list of callbacks for LangChain."""
        if self.tracer:
            return [self.tracer]
        return []

    async def log_run(
        self,
        run_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Optional[str]:
        """Log a run to LangSmith."""
        try:
            if not self.client:
                return None

            run_id = run_id or str(uuid.uuid4())

            run_data = {
                "id": run_id,
                "name": run_type,
                "run_type": run_type,
                "inputs": inputs,
                "outputs": outputs,
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "extra": metadata or {},
                "error": error,
                "project_name": self.project_name,
            }

            # Create run
            self.client.create_run(**run_data)

            logger.debug(f"Logged run to LangSmith: {run_id}")
            return run_id

        except Exception as e:
            logger.error("Failed to log run to LangSmith", {error:str(e)})
            return None

    async def log_feedback(
        self,
        run_id: str,
        score: float,
        value: Optional[str] = None,
        comment: Optional[str] = None,
        feedback_type: str = "user",
    ) -> bool:
        """Log feedback for a run."""
        try:
            if not self.client:
                return False

            self.client.create_feedback(
                run_id=run_id,
                key=feedback_type,
                score=score,
                value=value,
                comment=comment,
            )

            logger.info(f"Logged feedback for run {run_id}")
            return True

        except Exception as e:
            logger.error("Failed to log feedback", {error:str(e)})
            return False

    async def evaluate_dataset(
        self, dataset_name: str, llm_chain, evaluators: Optional[List] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a chain against a dataset."""
        try:
            if not self.client:
                return None

            # Run evaluation
            results = evaluate(
                llm_chain,
                dataset_name=dataset_name,
                client=self.client,
                project_name=f"{self.project_name}_eval",
                evaluators=evaluators,
            )

            logger.info(f"Completed evaluation on dataset {dataset_name}")
            return results

        except Exception as e:
            logger.error("Failed to evaluate dataset", {error:str(e)})
            return None

    async def create_dataset(
        self, name: str, description: str, examples: List[Dict[str, Any]]
    ) -> bool:
        """Create a new dataset for evaluation."""
        try:
            if not self.client:
                return False

            # Create dataset
            dataset = self.client.create_dataset(
                dataset_name=name, description=description
            )

            # Add examples
            for example in examples:
                self.client.create_example(
                    dataset_id=dataset.id,
                    inputs=example.get("inputs", {}),
                    outputs=example.get("outputs", {}),
                )

            logger.info(f"Created dataset {name} with {len(examples)} examples")
            return True

        except Exception as e:
            logger.error("Failed to create dataset",{ error:str(e)})
            return False

    def get_run_url(self, run_id: str) -> str:
        """Get URL for viewing a run in LangSmith UI."""
        return (
            f"{get_settings().LANGSMITH_ENDPOINT}/projects/{self.project_name}/runs/{run_id}"
        )

    async def get_run_metrics(
        self, run_id: Optional[str] = None, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get metrics for recent runs."""
        try:
            if not self.client:
                return []

            # Get runs
            runs = self.client.list_runs(
                project_name=self.project_name, execution_order=1, limit=limit
            )

            metrics = []
            for run in runs:
                metrics.append(
                    {
                        "run_id": str(run.id),
                        "name": run.name,
                        "status": run.status,
                        "start_time": (
                            run.start_time.isoformat() if run.start_time else None
                        ),
                        "end_time": run.end_time.isoformat() if run.end_time else None,
                        "latency": run.latency,
                        "tokens": run.token_usage,
                        "error": run.error,
                        "feedback": run.feedback_stats,
                    }
                )

            return metrics

        except Exception as e:
            logger.error("Failed to get run metrics", {error:str(e)})
            return []


# Create global instance
langsmith_service = LangSmithService()
