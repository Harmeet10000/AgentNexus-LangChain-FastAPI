"""Deterministic retrieval evaluation primitives."""

from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .report import EvaluationReport, write_report
from .runner import RetrievalEvaluation, run_retrieval_eval
from .schema import GoldenQuery, GoldenSet, load_golden_set

__all__ = [
    "EvaluationReport",
    "GoldenQuery",
    "GoldenSet",
    "RetrievalEvaluation",
    "load_golden_set",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
    "run_retrieval_eval",
    "write_report",
]
