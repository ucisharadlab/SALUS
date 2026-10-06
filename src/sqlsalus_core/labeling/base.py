"""Base definitions for labeling functions: labels, TaskRecord, and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlsalus_core.execution.executor import ExecutionResult

# ---------------------------------------------------------------------------
# Snorkel label convention
# ---------------------------------------------------------------------------
CORRECT = 1
INCORRECT = 0
ABSTAIN = -1


# ---------------------------------------------------------------------------
# TaskRecord -- all data for one benchmark task, ready for labeling functions
# ---------------------------------------------------------------------------

@dataclass
class TaskRecord:
    """All data for a single benchmark task, ready for labeling functions."""

    task_id: int
    db_id: str
    question: str
    evidence: str
    difficulty: str
    gold_sql: str
    gold_result: dict[str, Any]          # ExecutionResult.to_dict()
    agents: dict[str, dict[str, Any]]    # agent_name -> {"sql": str, "result": dict}

    # Pre-computed fields (filled by load_task_records)
    gold_exec: Any = None                # ExecutionResult (populated at load time)
    agent_execs: dict[str, Any] = field(default_factory=dict)   # agent_name -> ExecutionResult
    agent_sqls: dict[str, str] = field(default_factory=dict)    # agent_name -> sql string

    # Optional database schema
    schema: dict[str, list[dict[str, str]]] | None = None


def _rebuild_exec(d: dict[str, Any]) -> ExecutionResult:
    """Reconstruct an ExecutionResult from its dict representation."""
    return ExecutionResult(
        rows=[tuple(r) for r in d.get("rows", [])],
        columns=d.get("columns", []),
        error=d.get("error"),
        row_count=d.get("row_count", 0),
    )


def load_task_records(jsonl_path: str | Path) -> list[TaskRecord]:
    """Read agent_outputs.jsonl and populate pre-computed fields.

    Each line is a JSON object with keys matching TaskRecord fields:
        task_id, db_id, question, evidence, difficulty, gold_sql,
        gold_result (dict), agents (dict of agent dicts).
    """
    jsonl_path = Path(jsonl_path)
    records: list[TaskRecord] = []

    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            rec = TaskRecord(
                task_id=entry["task_id"],
                db_id=entry["db_id"],
                question=entry["question"],
                evidence=entry.get("evidence", ""),
                difficulty=entry.get("difficulty", "unknown"),
                gold_sql=entry["gold_sql"],
                gold_result=entry["gold_result"],
                agents=entry["agents"],
            )

            # Populate pre-computed fields
            rec.gold_exec = _rebuild_exec(rec.gold_result)

            rec.agent_execs = {}
            rec.agent_sqls = {}
            for agent_name, agent_data in rec.agents.items():
                rec.agent_execs[agent_name] = _rebuild_exec(agent_data["result"])
                rec.agent_sqls[agent_name] = agent_data["sql"]

            records.append(rec)

    return records
