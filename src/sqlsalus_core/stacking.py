"""Agent Stacking: a global verdict-pattern rule selected with weak labels.

Four agents yield 16 verdict patterns and 65,536 deterministic mappings.
The public driver is run_pipeline.py.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def build_verdict_matrix(
    task_ids: np.ndarray,
    verdicts: dict[str, dict[int, int]],
    agents: list[str],
    default: int = 1,
) -> np.ndarray:
    """Build (n_tasks, n_agents) array of verdicts for the given task_ids."""
    n = len(task_ids)
    na = len(agents)
    mat = np.full((n, na), default, dtype=np.int32)
    for j, agent in enumerate(agents):
        av = verdicts[agent]
        for i, tid in enumerate(task_ids):
            if tid in av:
                mat[i, j] = av[tid]
    return mat


def verdict_to_input_idx(mat: np.ndarray) -> np.ndarray:
    """Convert (n_tasks, n_agents) binary verdict matrix to input pattern indices.

    Pattern index = agents[0]*2^(n-1) + agents[1]*2^(n-2) + ... + agents[n-1]*2^0.
    First agent is MSB, last agent is LSB (consistent with logic-table string ordering).
    """
    n_agents = mat.shape[1]
    powers = (2 ** np.arange(n_agents - 1, -1, -1)).astype(np.int32)
    return mat.dot(powers).astype(np.int32)


# ---------------------------------------------------------------------------
# Exhaustive strategy search (vectorized)
# ---------------------------------------------------------------------------

def exhaustive_search(
    y: np.ndarray,
    input_idx: np.ndarray,
    n_agents: int,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Enumerate all 2^(2^n) truth-table strategies.

    Returns (top_strategy_ids, f1_scores, precisions, recalls) sorted by F1 desc.
    """
    n_combos = 2 ** n_agents
    n_strategies = 2 ** n_combos

    # Bit matrix: bit_matrix[s, p] = output of strategy s for input pattern p
    # (s >> (n_combos-1-p)) & 1
    shifts = (n_combos - 1 - np.arange(n_combos, dtype=np.int64))
    all_s = np.arange(n_strategies, dtype=np.int64)
    bit_matrix = ((all_s[:, None] >> shifts[None, :]) & 1).astype(np.uint8)

    # Predictions: (n_strategies, n_tasks)
    preds = bit_matrix[:, input_idx]  # broadcast: bit_matrix has shape (n_strat, n_combos), index by input_idx

    y0 = (y == 0).astype(np.int32)   # 1 where error
    y1 = (y == 1).astype(np.int32)   # 1 where correct
    p0 = (preds == 0).astype(np.int32)  # 1 where predicted error
    p1 = (preds == 1).astype(np.int32)  # 1 where predicted correct

    tp = (y0[None, :] * p0).sum(axis=1).astype(np.float32)
    fp = (y1[None, :] * p0).sum(axis=1).astype(np.float32)
    fn = (y0[None, :] * p1).sum(axis=1).astype(np.float32)

    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(
        2 * prec * rec,
        prec + rec,
        out=np.zeros_like(prec),
        where=(prec + rec) > 0,
    )

    # Filter trivial all-same strategies (predict all 0 or all 1)
    n_tasks = len(y)
    trivial_all0 = (preds.sum(axis=1) == 0)
    trivial_all1 = (preds.sum(axis=1) == n_tasks)
    valid = ~(trivial_all0 | trivial_all1)
    f1[~valid] = -1.0

    order = np.argsort(-f1)[:top_k]
    return all_s[order], f1[order], prec[order], rec[order]


# ---------------------------------------------------------------------------
# Decision tree display
# ---------------------------------------------------------------------------

def format_decision_tree(strategy_id: int, agents: list[str], display: dict[str, str]) -> str:
    """Render the truth-table strategy as a hierarchical decision tree string."""
    n = len(agents)
    n_combos = 2 ** n
    lines: list[str] = []

    def output_for_path(path: list[int]) -> int:
        idx = sum(v << (n - 1 - j) for j, v in enumerate(path))
        return (strategy_id >> (n_combos - 1 - idx)) & 1

    def recurse(level: int, path: list[int], prefix: str) -> None:
        agent_disp = display[agents[level]]
        for vi, val in enumerate([0, 1]):
            is_last = vi == 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + ("    " if is_last else "│   ")
            new_path = path + [val]

            if level == n - 1:
                out = output_for_path(new_path)
                lines.append(f"{prefix}{connector}If {agent_disp} == {val} ──> Predict: {out}")
            else:
                lines.append(f"{prefix}{connector}If {agent_disp} == {val}")
                recurse(level + 1, new_path, child_prefix)

    recurse(0, [], "   ")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def metrics(y: np.ndarray, preds: np.ndarray) -> dict:
    p  = float(precision_score(y, preds, pos_label=0, zero_division=0))
    r  = float(recall_score(y, preds, pos_label=0, zero_division=0))
    f1 = float(f1_score(y, preds, pos_label=0, zero_division=0))
    tp = int(((y == 0) & (preds == 0)).sum())
    fp = int(((y == 1) & (preds == 0)).sum())
    fn = int(((y == 0) & (preds == 1)).sum())
    tn = int(((y == 1) & (preds == 1)).sum())
    return {"f1": f1, "p": p, "r": r, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def apply_strategy(strategy_id: int, input_idx: np.ndarray, n_agents: int) -> np.ndarray:
    """Apply a strategy (integer truth-table) to an array of input pattern indices."""
    n_combos = 2 ** n_agents
    preds = np.array(
        [(strategy_id >> (n_combos - 1 - p)) & 1 for p in input_idx],
        dtype=int,
    )
    return preds
