"""Execution-based labeling functions (LF1-LF14).

These LFs compare SQL execution results between the BIRD ground truth
and agent outputs to detect annotation errors.
"""

from __future__ import annotations

from typing import Any

from sqlsalus_core.execution.comparator import compare_result_sets, results_agree
from sqlsalus_core.execution.executor import ExecutionResult
from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT, TaskRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _agent_results(record: TaskRecord) -> list[tuple[str, ExecutionResult]]:
    """Return list of (agent_name, ExecutionResult) pairs, skipping errors."""
    pairs = []
    for name, ex in record.agent_execs.items():
        if ex.error is None:
            pairs.append((name, ex))
    return pairs


def get_majority_result(record: TaskRecord) -> ExecutionResult | None:
    """Return the most common result among agents (by agreement clusters).

    Returns the ExecutionResult from the largest cluster, or None if no
    agents produced valid results.
    """
    agents = _agent_results(record)
    if not agents:
        return None

    # Build agreement clusters greedily
    clusters: list[list[tuple[str, ExecutionResult]]] = []
    for name, ex in agents:
        placed = False
        for cluster in clusters:
            rep_name, rep_ex = cluster[0]
            if results_agree(ex, rep_ex):
                cluster.append((name, ex))
                placed = True
                break
        if not placed:
            clusters.append([(name, ex)])

    # Return representative from largest cluster
    largest = max(clusters, key=len)
    return largest[0][1]


def get_agent_consensus(record: TaskRecord) -> list[str]:
    """Return agent names in the largest agreement cluster."""
    agents = _agent_results(record)
    if not agents:
        return []

    clusters: list[list[str]] = []
    cluster_reps: list[ExecutionResult] = []
    for name, ex in agents:
        placed = False
        for i, rep in enumerate(cluster_reps):
            if results_agree(ex, rep):
                clusters[i].append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])
            cluster_reps.append(ex)

    return max(clusters, key=len)


def _cluster_agent_results(record: TaskRecord) -> list[list[str]]:
    """Group agents into clusters by result agreement."""
    agents = _agent_results(record)
    if not agents:
        return []

    clusters: list[list[str]] = []
    cluster_reps: list[ExecutionResult] = []
    for name, ex in agents:
        placed = False
        for i, rep in enumerate(cluster_reps):
            if results_agree(ex, rep):
                clusters[i].append(name)
                placed = True
                break
        if not placed:
            clusters.append([name])
            cluster_reps.append(ex)

    return clusters


def _consensus_result(record: TaskRecord) -> ExecutionResult | None:
    """Return the ExecutionResult from the largest agent agreement cluster."""
    return get_majority_result(record)


# ---------------------------------------------------------------------------
# LF1: Any agent agrees with BIRD ground truth -> CORRECT
# ---------------------------------------------------------------------------

def lf_01(record: TaskRecord) -> int:
    """Bipolar: any agent agrees with BIRD -> CORRECT; none agree -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    if not record.agent_execs:
        return ABSTAIN

    for name, ex in record.agent_execs.items():
        if ex.error is None and results_agree(ex, gold):
            return CORRECT

    return INCORRECT


# ---------------------------------------------------------------------------
# LF2: BIRD SQL has syntax/runtime error -> INCORRECT
# ---------------------------------------------------------------------------

def lf_02(record: TaskRecord) -> int:
    """Unipolar: BIRD SQL has execution error -> INCORRECT; otherwise ABSTAIN."""
    gold = record.gold_exec
    if gold is None:
        return ABSTAIN

    if gold.error is not None:
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF3: Agent consensus result is strict superset/subset of BIRD -> INCORRECT
# ---------------------------------------------------------------------------

def lf_03(record: TaskRecord) -> int:
    """Agent consensus is strict superset/subset of BIRD result -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    consensus = _consensus_result(record)
    if consensus is None or consensus.error is not None:
        return ABSTAIN

    cmp = compare_result_sets(gold, consensus)

    # Strict subset: gold is subset of consensus but not equal
    if cmp.r1_subset_r2 and not cmp.exact_match:
        return INCORRECT

    # Strict superset: consensus is subset of gold but not equal
    if cmp.r2_subset_r1 and not cmp.exact_match:
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF4: Selected single-agent execution agreement
# ---------------------------------------------------------------------------

def lf_04(record: TaskRecord) -> int:
    """Bipolar: selected agent agrees with reference -> CORRECT; otherwise INCORRECT.

    Selects the first agent alphabetically. Abstains if its execution is invalid.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    if not record.agent_execs:
        return ABSTAIN

    # Deterministic single-agent selection
    strongest_name = sorted(record.agent_execs.keys())[0]
    strongest_ex = record.agent_execs[strongest_name]

    if strongest_ex.error is not None:
        return ABSTAIN

    if results_agree(strongest_ex, gold):
        return CORRECT

    return INCORRECT


# ---------------------------------------------------------------------------
# LF5: BIRD cardinality is 1, agent consensus cardinality >> 1 (or vice versa)
# ---------------------------------------------------------------------------

def lf_05(record: TaskRecord) -> int:
    """BIRD card=1 but consensus card>>1, or vice versa -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    consensus = _consensus_result(record)
    if consensus is None or consensus.error is not None:
        return ABSTAIN

    gold_card = gold.row_count
    cons_card = consensus.row_count

    # BIRD returns 1 row, consensus returns many (>=5)
    if gold_card == 1 and cons_card >= 5:
        return INCORRECT

    # BIRD returns many (>=5), consensus returns 1
    if gold_card >= 5 and cons_card == 1:
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF6: BIRD result has NULLs in every row for selected columns, agents don't
# ---------------------------------------------------------------------------

def lf_06(record: TaskRecord) -> int:
    """BIRD result has NULLs in every row for some column, agents don't -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN
    if gold.row_count == 0:
        return ABSTAIN

    # Find columns that are all-NULL in gold
    num_cols = len(gold.columns) if gold.columns else (len(gold.rows[0]) if gold.rows else 0)
    if num_cols == 0:
        return ABSTAIN

    all_null_cols: set[int] = set()
    for col_idx in range(num_cols):
        if all(row[col_idx] is None for row in gold.rows):
            all_null_cols.add(col_idx)

    if not all_null_cols:
        return ABSTAIN

    # Check if agent consensus does NOT have all-NULL for those columns
    consensus = _consensus_result(record)
    if consensus is None or consensus.error is not None:
        return ABSTAIN
    if consensus.row_count == 0:
        return ABSTAIN

    cons_num_cols = len(consensus.columns) if consensus.columns else (
        len(consensus.rows[0]) if consensus.rows else 0
    )
    if cons_num_cols == 0:
        return ABSTAIN

    # Check corresponding columns by index (may not align perfectly)
    # Only flag if consensus has same number of columns
    if cons_num_cols != num_cols:
        return ABSTAIN

    for col_idx in all_null_cols:
        cons_all_null = all(row[col_idx] is None for row in consensus.rows)
        if not cons_all_null:
            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF7: BIRD result has more rows than reasonable (Cartesian product smell)
# ---------------------------------------------------------------------------

def lf_07(record: TaskRecord) -> int:
    """BIRD result has >1000 rows (heuristic for Cartesian product) -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    if gold.row_count > 1000:
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF8: BIRD result has duplicate rows where columns suggest uniqueness
# ---------------------------------------------------------------------------

def lf_08(record: TaskRecord) -> int:
    """BIRD result has duplicates in columns containing 'id' or 'pk' -> INCORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN
    if gold.row_count < 2:
        return ABSTAIN
    if not gold.columns:
        return ABSTAIN

    # Find 'id' or 'pk' columns
    id_col_indices: list[int] = []
    for i, col_name in enumerate(gold.columns):
        lower = col_name.lower()
        if lower == "id" or lower.endswith("_id") or lower == "pk" or lower.endswith("_pk"):
            id_col_indices.append(i)

    if not id_col_indices:
        return ABSTAIN

    # Check for duplicate values in each id column
    for col_idx in id_col_indices:
        values = [row[col_idx] for row in gold.rows]
        if len(values) != len(set(values)):
            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF9: BIRD result values outside column domain range
# ---------------------------------------------------------------------------

def lf_09(record: TaskRecord) -> int:
    """BIRD result contains values outside the range observed in agent results -> INCORRECT.

    If the gold result has numeric values that are far outside the range of
    values returned by agent consensus, the gold query likely selects wrong data.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN
    if gold.row_count == 0:
        return ABSTAIN

    consensus = _consensus_result(record)
    if consensus is None or consensus.error is not None:
        return ABSTAIN
    if consensus.row_count == 0:
        return ABSTAIN

    # Only compare if same number of columns
    gold_ncols = len(gold.rows[0]) if gold.rows else 0
    cons_ncols = len(consensus.rows[0]) if consensus.rows else 0
    if gold_ncols != cons_ncols or gold_ncols == 0:
        return ABSTAIN

    # Check each column: if gold has numeric values outside consensus range
    for col_idx in range(gold_ncols):
        try:
            cons_nums = [float(row[col_idx]) for row in consensus.rows
                         if row[col_idx] is not None and _is_numeric(row[col_idx])]
            gold_nums = [float(row[col_idx]) for row in gold.rows
                         if row[col_idx] is not None and _is_numeric(row[col_idx])]
        except (ValueError, TypeError):
            continue

        if not cons_nums or not gold_nums:
            continue

        cons_min, cons_max = min(cons_nums), max(cons_nums)
        cons_range = cons_max - cons_min
        if cons_range == 0:
            continue

        # Flag if any gold value is more than 2x the range outside consensus bounds
        margin = cons_range * 2.0
        for v in gold_nums:
            if v < cons_min - margin or v > cons_max + margin:
                return INCORRECT

    return ABSTAIN


def _is_numeric(val) -> bool:
    """Check if a value can be interpreted as a number."""
    if isinstance(val, (int, float)):
        return True
    if isinstance(val, str):
        try:
            float(val)
            return True
        except (ValueError, OverflowError):
            return False
    return False


# ---------------------------------------------------------------------------
# LF10: BIRD and agent consensus identical values, different column order
# ---------------------------------------------------------------------------

def lf_10(record: TaskRecord) -> int:
    """BIRD and consensus have same values but different column order -> CORRECT."""
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    consensus = _consensus_result(record)
    if consensus is None or consensus.error is not None:
        return ABSTAIN

    # If they already agree exactly, let other LFs handle it
    if results_agree(gold, consensus):
        return ABSTAIN

    # Same row count?
    if gold.row_count != consensus.row_count:
        return ABSTAIN
    if gold.row_count == 0:
        return ABSTAIN

    # Different number of columns means not a column-reordering issue
    gold_ncols = len(gold.rows[0]) if gold.rows else 0
    cons_ncols = len(consensus.rows[0]) if consensus.rows else 0
    if gold_ncols != cons_ncols or gold_ncols == 0:
        return ABSTAIN

    # Check if sorting each row's values produces the same multiset of rows
    def row_sorted(row: tuple) -> tuple:
        return tuple(sorted(str(v) for v in row))

    gold_sorted = sorted(row_sorted(r) for r in gold.rows)
    cons_sorted = sorted(row_sorted(r) for r in consensus.rows)

    if gold_sorted == cons_sorted:
        return CORRECT

    return INCORRECT


# ---------------------------------------------------------------------------
# LF11: Column-aligned agreement — gold and any agent agree after column alignment
# ---------------------------------------------------------------------------

def lf_11(record: TaskRecord) -> int:
    """Gold and any agent agree after column-name alignment -> CORRECT.

    Only fires for agents that DON'T agree via standard comparison,
    adding unique signal beyond lf_01.
    """
    from sqlsalus_core.execution.comparator import results_agree_column_aware

    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    for name, ex in record.agent_execs.items():
        if ex.error is not None:
            continue
        # Skip agents that already agree exactly (lf_01 handles those)
        if results_agree(ex, gold):
            continue
        if results_agree_column_aware(ex, gold):
            return CORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF12: Approximate float match — gold and any agent agree with float tolerance
# ---------------------------------------------------------------------------

def lf_12(record: TaskRecord) -> int:
    """Gold and any agent agree with approximate float comparison -> CORRECT.

    Only fires for agents that DON'T agree via standard comparison,
    adding unique signal for rounding differences (e.g., 17.2414 vs 17.24).
    """
    from sqlsalus_core.execution.comparator import results_agree_approx

    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN

    for name, ex in record.agent_execs.items():
        if ex.error is not None:
            continue
        # Skip agents that already agree exactly
        if results_agree(ex, gold):
            continue
        if results_agree_approx(ex, gold):
            return CORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF13: Percentage factor equivalence — gold and agent differ by factor of ~100
# ---------------------------------------------------------------------------

def lf_13(record: TaskRecord) -> int:
    """Gold and agent single-numeric results differ by factor ~100 -> CORRECT.

    Catches percentage vs fraction discrepancies (e.g., 22.73% vs 0.2273)
    where both compute the same ratio but one multiplies by 100.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN
    if gold.row_count != 1 or not gold.rows:
        return ABSTAIN

    gold_row = gold.rows[0]
    # Find numeric values in gold row
    gold_nums = [(i, float(v)) for i, v in enumerate(gold_row)
                 if v is not None and _is_numeric(v)]
    if not gold_nums:
        return ABSTAIN

    for name, ex in record.agent_execs.items():
        if ex.error is not None:
            continue
        if ex.row_count != 1 or not ex.rows:
            continue
        agent_row = ex.rows[0]

        # For each gold numeric value, check if any agent value is ~100x or ~1/100x
        for gi, gv in gold_nums:
            if abs(gv) < 1e-10:
                continue
            for ai, av in enumerate(agent_row):
                if av is None or not _is_numeric(av):
                    continue
                av_f = float(av)
                if abs(av_f) < 1e-10:
                    continue
                ratio = gv / av_f
                # Check if ratio is approximately 100 or 1/100 (within 5%)
                if 95.0 <= ratio <= 105.0 or 0.0095 <= ratio <= 0.0105:
                    return CORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF14: Row-value-sorted match — same values in rows, different column positions
# ---------------------------------------------------------------------------

def lf_14(record: TaskRecord) -> int:
    """Gold and any agent have same row values after sorting within each row -> CORRECT.

    Catches column reordering without needing column name information.
    More robust than column-name alignment for cases where column names differ.
    """
    gold = record.gold_exec
    if gold is None or gold.error is not None:
        return ABSTAIN
    if gold.row_count == 0:
        return ABSTAIN

    from sqlsalus_core.execution.comparator import normalize_rows

    gold_rows = normalize_rows(gold.rows)

    # Sort values within each row for column-order-independent comparison
    def rows_sorted_values(rows):
        return sorted(tuple(sorted(str(v) for v in row)) for row in rows)

    gold_vsorted = rows_sorted_values(gold_rows)

    for name, ex in record.agent_execs.items():
        if ex.error is not None:
            continue
        if ex.row_count != gold.row_count:
            continue
        # Skip if they already agree (saves computation)
        if results_agree(ex, gold):
            continue

        agent_rows = normalize_rows(ex.rows)
        agent_vsorted = rows_sorted_values(agent_rows)

        if gold_vsorted == agent_vsorted:
            return CORRECT

    return ABSTAIN
