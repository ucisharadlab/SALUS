"""Labeling functions for SALUS error detection."""

from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT, TaskRecord, load_task_records

__all__ = [
    "CORRECT",
    "INCORRECT",
    "ABSTAIN",
    "TaskRecord",
    "load_task_records",
]
