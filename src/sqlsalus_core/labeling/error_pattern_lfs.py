"""Known error pattern labeling functions (LF38-LF43).

These LFs detect specific, well-documented SQL anti-patterns and common
annotation mistakes in NL2SQL benchmarks.
"""

from __future__ import annotations

import re
from typing import Any

from sqlsalus_core.execution.comparator import results_agree
from sqlsalus_core.labeling.base import ABSTAIN, CORRECT, INCORRECT, TaskRecord

try:
    import sqlglot
    from sqlglot import exp as sqlglot_exp
    _HAS_SQLGLOT = True
except ImportError:
    _HAS_SQLGLOT = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_parse(sql: str) -> list[Any] | None:
    if not _HAS_SQLGLOT:
        return None
    try:
        return sqlglot.parse(sql, dialect="sqlite")
    except Exception:
        return None


def _get_agent_sqls(record: TaskRecord) -> list[str]:
    return [sql for sql in record.agent_sqls.values() if sql]


def _consensus_result(record: TaskRecord):
    from sqlsalus_core.labeling.execution_lfs import get_majority_result
    return get_majority_result(record)


# ---------------------------------------------------------------------------
# LF38: Aggregation outside GROUP BY context
# ---------------------------------------------------------------------------

def lf_38(record: TaskRecord) -> int:
    """BIRD SQL uses aggregation without GROUP BY when agents use GROUP BY -> INCORRECT.

    Detects cases where gold SQL applies an aggregate (COUNT, SUM, AVG, etc.)
    without GROUP BY, but agents include GROUP BY — suggesting the gold query
    collapses groups that should be preserved.
    """
    trees = _safe_parse(record.gold_sql)
    if not trees:
        return ABSTAIN

    gold_tree = trees[0]
    if gold_tree is None:
        return ABSTAIN

    agg_types = (
        sqlglot_exp.Count, sqlglot_exp.Sum, sqlglot_exp.Avg,
        sqlglot_exp.Min, sqlglot_exp.Max,
    )
    has_agg = any(gold_tree.find_all(*agg_types))
    has_group = bool(gold_tree.find(sqlglot_exp.Group))

    if not has_agg or has_group:
        return ABSTAIN  # Gold either has no agg or already has GROUP BY

    # Check if agents use GROUP BY
    for agent_sql in _get_agent_sqls(record):
        agent_trees = _safe_parse(agent_sql)
        if not agent_trees or agent_trees[0] is None:
            continue
        if agent_trees[0].find(sqlglot_exp.Group):
            # Agent uses GROUP BY where gold doesn't, and gold has aggregation
            # Check results actually differ
            gold = record.gold_exec
            consensus = _consensus_result(record)
            if gold and consensus and not results_agree(gold, consensus):
                return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF39: LIKE '%val%' vs exact match mismatch
# ---------------------------------------------------------------------------

def lf_39(record: TaskRecord) -> int:
    """BIRD uses LIKE '%val%' where agents use exact match (or vice versa), and results differ -> INCORRECT."""
    trees = _safe_parse(record.gold_sql)
    if not trees or trees[0] is None:
        return ABSTAIN

    gold_has_like = bool(list(trees[0].find_all(sqlglot_exp.Like)))

    # Check agent consensus
    agent_sqls = _get_agent_sqls(record)
    if not agent_sqls:
        return ABSTAIN

    # Count how many agents use LIKE
    agents_with_like = 0
    agents_without_like = 0
    for asql in agent_sqls:
        atrees = _safe_parse(asql)
        if not atrees or atrees[0] is None:
            continue
        if list(atrees[0].find_all(sqlglot_exp.Like)):
            agents_with_like += 1
        else:
            agents_without_like += 1

    # Mismatch: gold uses LIKE but no agent does, or vice versa
    if gold_has_like and agents_with_like == 0 and agents_without_like > 0:
        gold = record.gold_exec
        consensus = _consensus_result(record)
        if gold and consensus and not results_agree(gold, consensus):
            return INCORRECT

    if not gold_has_like and agents_with_like > 0 and agents_without_like == 0:
        gold = record.gold_exec
        consensus = _consensus_result(record)
        if gold and consensus and not results_agree(gold, consensus):
            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF40: Date boundary off-by-one (< vs <=)
# ---------------------------------------------------------------------------

def lf_40(record: TaskRecord) -> int:
    """BIRD has date boundary off-by-one (< vs <=), agents differ, results differ -> INCORRECT.

    Detects when gold and agents use different comparison operators on
    date-like columns, which often indicates an off-by-one boundary error.
    """
    trees = _safe_parse(record.gold_sql)
    if not trees or trees[0] is None:
        return ABSTAIN

    # Find comparison operators in gold WHERE clause
    where = trees[0].find(sqlglot_exp.Where)
    if not where:
        return ABSTAIN

    date_keywords = {"date", "year", "month", "day", "time", "created", "updated",
                     "start", "end", "begin", "deadline", "dob", "birth"}

    # Check for comparisons on date-like columns
    gold_date_ops: dict[str, str] = {}
    op_types = {
        sqlglot_exp.GT: ">", sqlglot_exp.GTE: ">=",
        sqlglot_exp.LT: "<", sqlglot_exp.LTE: "<=",
    }
    for op_type, op_str in op_types.items():
        for node in where.find_all(op_type):
            col = node.this
            if isinstance(col, sqlglot_exp.Column):
                col_name = col.name.lower() if col.name else ""
                if any(kw in col_name for kw in date_keywords):
                    gold_date_ops[col_name] = op_str

    if not gold_date_ops:
        return ABSTAIN

    # Check agent consensus for different operators on the same date columns
    for agent_sql in _get_agent_sqls(record):
        atrees = _safe_parse(agent_sql)
        if not atrees or atrees[0] is None:
            continue
        awhere = atrees[0].find(sqlglot_exp.Where)
        if not awhere:
            continue

        for op_type, op_str in op_types.items():
            for node in awhere.find_all(op_type):
                col = node.this
                if isinstance(col, sqlglot_exp.Column):
                    col_name = col.name.lower() if col.name else ""
                    if col_name in gold_date_ops and gold_date_ops[col_name] != op_str:
                        # Different operator on same date column
                        gold = record.gold_exec
                        consensus = _consensus_result(record)
                        if gold and consensus and not results_agree(gold, consensus):
                            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF41: Division by column containing zeros, agents guard against it
# ---------------------------------------------------------------------------

def lf_41(record: TaskRecord) -> int:
    """BIRD SQL has unguarded division, agents use NULLIF or CASE -> INCORRECT.

    Detects division operations where gold SQL divides directly but agents
    guard against division by zero with NULLIF, CASE, or similar constructs.
    """
    gold_sql = record.gold_sql

    # Check if gold has division
    if '/' not in gold_sql:
        return ABSTAIN

    # Check if gold has any division guards
    gold_upper = gold_sql.upper()
    gold_has_guard = (
        'NULLIF' in gold_upper or
        re.search(r'CASE\s+WHEN.*=\s*0', gold_upper, re.IGNORECASE) or
        re.search(r'CASE\s+WHEN.*<>\s*0', gold_upper, re.IGNORECASE) or
        re.search(r'CASE\s+WHEN.*!=\s*0', gold_upper, re.IGNORECASE) or
        re.search(r'WHERE.*<>\s*0', gold_upper, re.IGNORECASE) or
        re.search(r'WHERE.*!=\s*0', gold_upper, re.IGNORECASE)
    )

    if gold_has_guard:
        return ABSTAIN  # Gold already guards

    # Check if agents guard against division by zero
    for agent_sql in _get_agent_sqls(record):
        if '/' not in agent_sql:
            continue
        agent_upper = agent_sql.upper()
        agent_has_guard = (
            'NULLIF' in agent_upper or
            re.search(r'CASE\s+WHEN.*=\s*0', agent_upper, re.IGNORECASE) or
            re.search(r'WHERE.*<>\s*0', agent_upper, re.IGNORECASE) or
            re.search(r'WHERE.*!=\s*0', agent_upper, re.IGNORECASE)
        )
        if agent_has_guard:
            # Agent guards but gold doesn't — check if results differ
            gold = record.gold_exec
            consensus = _consensus_result(record)
            if gold and consensus and not results_agree(gold, consensus):
                return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF42: Implicit type coercion changing semantics
# ---------------------------------------------------------------------------

def lf_42(record: TaskRecord) -> int:
    """BIRD SQL compares string and numeric columns without explicit CAST -> INCORRECT.

    Detects potential type coercion issues where gold SQL compares values
    without explicit CAST that agents include.
    """
    gold_upper = record.gold_sql.upper()

    # Check if agents use CAST where gold doesn't
    gold_has_cast = 'CAST' in gold_upper or 'TYPEOF' in gold_upper

    if gold_has_cast:
        return ABSTAIN  # Gold already handles types

    agent_sqls = _get_agent_sqls(record)
    if not agent_sqls:
        return ABSTAIN

    agents_with_cast = sum(1 for s in agent_sqls if 'CAST' in s.upper())

    # If majority of agents use CAST but gold doesn't
    if agents_with_cast > len(agent_sqls) / 2:
        gold = record.gold_exec
        consensus = _consensus_result(record)
        if gold and consensus and not results_agree(gold, consensus):
            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF43: ORDER BY on column not in SELECT with DISTINCT
# ---------------------------------------------------------------------------

def lf_43(record: TaskRecord) -> int:
    """BIRD SQL uses ORDER BY on a column not in SELECT when DISTINCT is used -> INCORRECT.

    In standard SQL, ORDER BY with DISTINCT requires the ORDER BY column
    to be in the SELECT list. SQLite is lenient about this, but it's
    usually a logical error.
    """
    trees = _safe_parse(record.gold_sql)
    if not trees or trees[0] is None:
        return ABSTAIN

    tree = trees[0]

    # Check for DISTINCT
    if not list(tree.find_all(sqlglot_exp.Distinct)):
        return ABSTAIN

    # Check for ORDER BY
    order = tree.find(sqlglot_exp.Order)
    if not order:
        return ABSTAIN

    # Get SELECT column names
    select = tree.find(sqlglot_exp.Select)
    if not select:
        return ABSTAIN

    select_cols: set[str] = set()
    for expr in select.expressions:
        if isinstance(expr, sqlglot_exp.Column):
            select_cols.add(expr.name.lower() if expr.name else "")
        elif hasattr(expr, 'alias') and expr.alias:
            select_cols.add(expr.alias.lower())
        # Also get the SQL text for more robust matching
        select_cols.add(expr.sql().lower().strip())

    # Get ORDER BY column names
    for ordered_expr in order.expressions:
        col_expr = ordered_expr.this if hasattr(ordered_expr, 'this') else ordered_expr
        if isinstance(col_expr, sqlglot_exp.Column):
            order_col = col_expr.name.lower() if col_expr.name else ""
            order_sql = col_expr.sql().lower().strip()
            if order_col and order_col not in select_cols and order_sql not in select_cols:
                return INCORRECT

    return ABSTAIN
