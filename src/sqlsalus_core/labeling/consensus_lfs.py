"""Agent consensus pattern labeling functions (LF36-LF37).

These LFs analyze agreement patterns among agents to make labeling
decisions about the BIRD ground truth.
"""

from __future__ import annotations

from sqlsalus_core.execution.comparator import results_agree
from sqlsalus_core.execution.executor import ExecutionResult
from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT, TaskRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cluster_agent_results(record: TaskRecord) -> list[list[str]]:
    """Group agents into clusters by result agreement.

    Returns a list of clusters, where each cluster is a list of agent names
    that agree with each other. Agents with errors are excluded.
    """
    valid_agents: list[tuple[str, ExecutionResult]] = []
    for name, ex in record.agent_execs.items():
        if ex.error is None:
            valid_agents.append((name, ex))

    if not valid_agents:
        return []

    clusters: list[list[str]] = []
    cluster_reps: list[ExecutionResult] = []

    for name, ex in valid_agents:
        placed = False
        for i, rep in enumerate(cluster_reps):
            if results_agree(ex, rep):
                clusters[i].append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])
            cluster_reps.append(ex)

    # Sort clusters by size (largest first)
    clusters.sort(key=len, reverse=True)
    return clusters


def _weighted_vote_agrees_with_bird(record: TaskRecord, weights: dict[str, float] | None = None) -> bool | None:
    """Compute weighted agent vote and check agreement with BIRD.

    Returns True if weighted majority agrees with BIRD, False if not,
    None if inconclusive.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return None

    if not record.agent_execs:
        return None

    # Default: equal weights
    if weights is None:
        weights = {name: 1.0 for name in record.agent_execs}

    agree_weight = 0.0
    disagree_weight = 0.0
    total_weight = 0.0

    for name, ex in record.agent_execs.items():
        if ex.error is not None:
            continue
        w = weights.get(name, 1.0)
        total_weight += w
        if results_agree(ex, gold):
            agree_weight += w
        else:
            disagree_weight += w

    if total_weight == 0:
        return None

    if agree_weight > disagree_weight:
        return True
    elif disagree_weight > agree_weight:
        return False
    else:
        return None  # Tie


def _bird_cluster_index(record: TaskRecord, clusters: list[list[str]]) -> int | None:
    """Find which cluster (if any) the BIRD result matches.

    Returns the cluster index or None.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return None

    for i, cluster in enumerate(clusters):
        if not cluster:
            continue
        rep_name = cluster[0]
        rep_ex = record.agent_execs.get(rep_name)
        if rep_ex is None:
            continue
        if results_agree(gold, rep_ex):
            return i

    return None


# ---------------------------------------------------------------------------
# LF36: Weighted agent vote disagrees with BIRD -> INCORRECT
# ---------------------------------------------------------------------------

def lf_36(record: TaskRecord) -> int:
    """Bipolar: weighted vote agrees -> CORRECT; disagrees -> INCORRECT."""
    result = _weighted_vote_agrees_with_bird(record)
    if result is True:
        return CORRECT
    if result is False:
        return INCORRECT
    return ABSTAIN


# ---------------------------------------------------------------------------
# LF37: Full agent consensus (all same result) disagrees with BIRD -> INCORRECT
# ---------------------------------------------------------------------------

def lf_37(record: TaskRecord) -> int:
    """Bipolar: full consensus agrees with BIRD -> CORRECT; disagrees -> INCORRECT."""
    clusters = cluster_agent_results(record)

    # Full consensus means exactly 1 cluster
    if len(clusters) != 1:
        return ABSTAIN

    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    rep_name = clusters[0][0]
    rep_ex = record.agent_execs.get(rep_name)
    if rep_ex is None:
        return ABSTAIN

    if results_agree(gold, rep_ex):
        return CORRECT
    return INCORRECT

