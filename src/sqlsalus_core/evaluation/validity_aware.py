from __future__ import annotations

from dataclasses import dataclass

from sqlsalus_core.execution.comparator import results_agree
from sqlsalus_core.execution.executor import ExecutionResult


@dataclass(frozen=True)
class ValidityAwarePrediction:
    pred: int | None
    n_valid_agents: int


def execution_is_valid(exec_result: ExecutionResult | None) -> bool:
    return exec_result is not None and exec_result.error is None


def agent_prediction(
    agent_exec: ExecutionResult | None,
    gold_exec: ExecutionResult | None,
) -> int | None:
    """Return 1/0 for a valid agent prediction, else None.

    Semantics:
    - `1`: agent executed successfully and matched the gold execution result
    - `0`: agent executed successfully and disagreed with gold
    - `None`: agent did not produce a valid executable answer
    """
    if not execution_is_valid(agent_exec) or not execution_is_valid(gold_exec):
        return None
    return 1 if results_agree(agent_exec, gold_exec) else 0


def majority_vote_prediction(
    agent_execs: dict[str, ExecutionResult | None],
    gold_exec: ExecutionResult | None,
    agents: list[str],
) -> tuple[ValidityAwarePrediction, dict[str, int | None]]:
    """Return a majority-vote prediction over valid agents only."""
    verdicts: dict[str, int | None] = {}
    valid_verdicts: list[int] = []

    for agent in agents:
        pred = agent_prediction(agent_execs.get(agent), gold_exec)
        verdicts[agent] = pred
        if pred is not None:
            valid_verdicts.append(pred)

    if not valid_verdicts:
        return ValidityAwarePrediction(pred=None, n_valid_agents=0), verdicts

    vote_sum = sum(valid_verdicts)
    pred = 1 if vote_sum > (len(valid_verdicts) / 2) else 0
    return ValidityAwarePrediction(pred=pred, n_valid_agents=len(valid_verdicts)), verdicts
