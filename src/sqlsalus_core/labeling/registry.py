"""Registry of the SALUS labeling functions."""

from __future__ import annotations

from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT, TaskRecord

# Execution-based LFs (LF1-LF14)
from sqlsalus_core.labeling.execution_lfs import (
    lf_01,
    lf_02,
    lf_03,
    lf_04,
    lf_05,
    lf_06,
    lf_07,
    lf_08,
    lf_09,
    lf_10,
    lf_11,
    lf_12,
    lf_13,
    lf_14,
)

# NL-SQL alignment LFs (LF15-LF25)
from sqlsalus_core.labeling.alignment_lfs import (
    lf_15,
    lf_16,
    lf_17,
    lf_18,
    lf_19,
    lf_20,
    lf_21,
    lf_22,
    lf_23,
    lf_24,
    lf_25,
)

# SQL structural LFs (LF26-LF34)
from sqlsalus_core.labeling.structural_lfs import (
    lf_26,
    lf_27,
    lf_28,
    lf_29,
    lf_30,
    lf_31,
    lf_32,
    lf_33,
    lf_34,
)

# Schema integrity LF
from sqlsalus_core.labeling.schema_lfs import (
    lf_35,
)

# Agent consensus LFs (LF36-LF37)
from sqlsalus_core.labeling.consensus_lfs import (
    lf_36,
    lf_37,
)

# Known error pattern LFs (LF38-LF43)
from sqlsalus_core.labeling.error_pattern_lfs import (
    lf_38,
    lf_39,
    lf_40,
    lf_41,
    lf_42,
    lf_43,
)


# ---------------------------------------------------------------------------
# All task-level LFs
# ---------------------------------------------------------------------------
TASK_LFS = [
    # Execution-based (LF1-LF14)
    lf_01, lf_02, lf_03, lf_04,
    lf_05, lf_06, lf_07, lf_08, lf_09, lf_10, lf_11, lf_12, lf_13, lf_14,
    # NL-SQL alignment (LF15-LF25)
    lf_15, lf_16, lf_17, lf_18, lf_19, lf_20, lf_21, lf_22, lf_23, lf_24, lf_25,
    # SQL structural (LF26-LF34)
    lf_26, lf_27, lf_28, lf_29, lf_30, lf_31, lf_32, lf_33, lf_34,
    # Schema integrity
    lf_35,
    # Agent consensus (LF36-LF37)
    lf_36, lf_37,
    # Known error patterns (LF38-LF43)
    lf_38, lf_39, lf_40, lf_41, lf_42, lf_43,
]

from sqlsalus_core.labeling.meta_lfs import lf_44

# Meta LFs (these run on the output of TASK_LFS) — LF44 only
META_LFS = [lf_44]

# High-precision LF indices kept for reference.
HIGH_PRECISION_LF_INDICES = {
    TASK_LFS.index(lf_02): "lf_02",
    TASK_LFS.index(lf_07): "lf_07",
}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def apply_task_lfs(record: TaskRecord) -> list[int]:
    """Run all task-level LFs on a record, returning the vote vector."""
    return [lf(record) for lf in TASK_LFS]


def apply_meta_lfs(votes: list[int]) -> list[int]:
    """Run all meta LFs on a vote vector."""
    return [meta_lf(votes) for meta_lf in META_LFS]


def apply_all_lfs(record: TaskRecord) -> dict[str, list[int]]:
    """Run all LFs (task + meta) and return both vote vectors.

    Returns:
        {
            "task_votes": [vote for each TASK_LF],
            "meta_votes": [vote for each META_LF],
        }
    """
    task_votes = apply_task_lfs(record)
    meta_votes = apply_meta_lfs(task_votes)
    return {
        "task_votes": task_votes,
        "meta_votes": meta_votes,
    }
