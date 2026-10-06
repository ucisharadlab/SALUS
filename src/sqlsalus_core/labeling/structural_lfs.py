"""SQL structural labeling functions (LF26-LF34).

These LFs compare the AST structure of BIRD ground truth SQL against
agent consensus SQL using sqlglot.
"""

from __future__ import annotations

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
# SQL AST extraction helpers
# ---------------------------------------------------------------------------

def _safe_parse(sql: str) -> list[Any] | None:
    """Parse SQL with sqlglot, returning AST list or None on failure."""
    if not _HAS_SQLGLOT:
        return None
    try:
        return sqlglot.parse(sql, dialect="sqlite")
    except Exception:
        return None


def extract_tables(sql: str) -> set[str]:
    """Extract referenced table names from SQL."""
    trees = _safe_parse(sql)
    if not trees:
        return set()
    tables: set[str] = set()
    for tree in trees:
        if tree is None:
            continue
        for tbl in tree.find_all(sqlglot_exp.Table):
            name = tbl.name
            if name:
                tables.add(name.lower())
    return tables


def extract_joins(sql: str) -> list[dict[str, str]]:
    """Extract join info: list of {table, join_type, on_condition_sql}.

    Returns empty list if parsing fails.
    """
    trees = _safe_parse(sql)
    if not trees:
        return []
    joins: list[dict[str, str]] = []
    for tree in trees:
        if tree is None:
            continue
        for join_node in tree.find_all(sqlglot_exp.Join):
            info: dict[str, str] = {}

            # Get joined table
            tbl = join_node.find(sqlglot_exp.Table)
            info["table"] = tbl.name.lower() if tbl and tbl.name else ""

            # Get join type
            join_kind = join_node.args.get("kind", "")
            side = join_node.args.get("side", "")
            if side:
                info["join_type"] = f"{side} {join_kind}".strip().lower()
            elif join_kind:
                info["join_type"] = str(join_kind).strip().lower()
            else:
                info["join_type"] = "inner"

            # Get ON condition
            on_cond = join_node.args.get("on")
            info["on_condition"] = on_cond.sql() if on_cond else ""

            joins.append(info)
    return joins


def extract_aggregations(sql: str) -> set[str]:
    """Extract aggregation function names (COUNT, SUM, AVG, MIN, MAX, etc.)."""
    trees = _safe_parse(sql)
    if not trees:
        return set()
    aggs: set[str] = set()
    agg_types = (
        sqlglot_exp.Count, sqlglot_exp.Sum, sqlglot_exp.Avg,
        sqlglot_exp.Min, sqlglot_exp.Max,
    )
    for tree in trees:
        if tree is None:
            continue
        for node in tree.find_all(*agg_types):
            aggs.add(type(node).__name__.upper())
    return aggs


def extract_group_by(sql: str) -> list[str]:
    """Extract GROUP BY column names."""
    trees = _safe_parse(sql)
    if not trees:
        return []
    cols: list[str] = []
    for tree in trees:
        if tree is None:
            continue
        group = tree.find(sqlglot_exp.Group)
        if group:
            for expr in group.expressions:
                if isinstance(expr, sqlglot_exp.Column):
                    cols.append(expr.name.lower() if expr.name else "")
                else:
                    cols.append(expr.sql().lower())
    return cols


def extract_where_conditions(sql: str) -> list[str]:
    """Extract WHERE condition SQL strings."""
    trees = _safe_parse(sql)
    if not trees:
        return []
    conditions: list[str] = []
    for tree in trees:
        if tree is None:
            continue
        where = tree.find(sqlglot_exp.Where)
        if where:
            # Get individual conditions (ANDs)
            for cond in where.find_all(sqlglot_exp.EQ, sqlglot_exp.GT, sqlglot_exp.GTE,
                                        sqlglot_exp.LT, sqlglot_exp.LTE, sqlglot_exp.NEQ,
                                        sqlglot_exp.Like, sqlglot_exp.In, sqlglot_exp.Between,
                                        sqlglot_exp.Is):
                conditions.append(cond.sql().lower())
    return conditions


def extract_literals(sql: str) -> set[str]:
    """Extract literal values (strings and numbers) from SQL."""
    trees = _safe_parse(sql)
    if not trees:
        return set()
    literals: set[str] = set()
    for tree in trees:
        if tree is None:
            continue
        for lit in tree.find_all(sqlglot_exp.Literal):
            val = lit.this
            if val:
                literals.add(str(val).lower().strip("'\""))
    return literals


def extract_comparison_ops(sql: str) -> list[tuple[str, str, str]]:
    """Extract comparison operations as (left_sql, operator, right_sql) triples."""
    trees = _safe_parse(sql)
    if not trees:
        return []
    ops: list[tuple[str, str, str]] = []
    op_map = {
        sqlglot_exp.EQ: "=",
        sqlglot_exp.GT: ">",
        sqlglot_exp.GTE: ">=",
        sqlglot_exp.LT: "<",
        sqlglot_exp.LTE: "<=",
        sqlglot_exp.NEQ: "!=",
        sqlglot_exp.Like: "LIKE",
        sqlglot_exp.Is: "IS",
    }
    for tree in trees:
        if tree is None:
            continue
        where = tree.find(sqlglot_exp.Where)
        if where:
            for op_type, op_str in op_map.items():
                for node in where.find_all(op_type):
                    left = node.this.sql().lower() if node.this else ""
                    right = node.expression.sql().lower() if node.expression else ""
                    ops.append((left, op_str, right))
    return ops


def _has_having(sql: str) -> bool:
    """Check if SQL has a HAVING clause."""
    trees = _safe_parse(sql)
    if not trees:
        return False
    for tree in trees:
        if tree is None:
            continue
        if tree.find(sqlglot_exp.Having):
            return True
    return False


def _has_subquery(sql: str) -> bool:
    """Check if SQL has subqueries."""
    trees = _safe_parse(sql)
    if not trees:
        return False
    for tree in trees:
        if tree is None:
            continue
        # Count Select nodes -- more than 1 means subquery
        selects = list(tree.find_all(sqlglot_exp.Select))
        if len(selects) > 1:
            return True
    return False


def get_consensus_sql(record: TaskRecord) -> str | None:
    """Return the SQL string from the largest agent agreement cluster.

    Returns None if no consensus can be determined.
    """
    from sqlsalus_core.labeling.execution_lfs import get_agent_consensus

    consensus_names = get_agent_consensus(record)
    if not consensus_names:
        return None
    return record.agent_sqls.get(consensus_names[0])


def _get_agent_sqls(record: TaskRecord) -> list[str]:
    """Return all valid agent SQL strings."""
    return [sql for sql in record.agent_sqls.values() if sql]


# ---------------------------------------------------------------------------
# LF26: BIRD SQL missing a JOIN all agents include
# ---------------------------------------------------------------------------

def lf_26(record: TaskRecord) -> int:
    """BIRD SQL missing a JOIN all agents include -> INCORRECT."""
    try:
        gold_tables = extract_tables(record.gold_sql)
        agent_sqls = _get_agent_sqls(record)
        if not agent_sqls or not gold_tables:
            return ABSTAIN

        # Tables present in ALL agents
        agent_table_sets = [extract_tables(s) for s in agent_sqls]
        if not agent_table_sets:
            return ABSTAIN

        common_agent_tables = agent_table_sets[0]
        for ts in agent_table_sets[1:]:
            common_agent_tables = common_agent_tables & ts

        # Tables in all agents but not in gold
        missing = common_agent_tables - gold_tables
        if missing:
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF27: BIRD SQL has extra JOIN no agent uses
# ---------------------------------------------------------------------------

def lf_27(record: TaskRecord) -> int:
    """BIRD SQL has extra JOIN no agent uses -> INCORRECT."""
    try:
        gold_tables = extract_tables(record.gold_sql)
        agent_sqls = _get_agent_sqls(record)
        if not agent_sqls or not gold_tables:
            return ABSTAIN

        # Union of all tables used by any agent
        all_agent_tables: set[str] = set()
        for s in agent_sqls:
            all_agent_tables |= extract_tables(s)

        # Tables in gold but not in any agent
        extra = gold_tables - all_agent_tables
        if extra:
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF28: BIRD uses different aggregation function than agent consensus
# ---------------------------------------------------------------------------

def lf_28(record: TaskRecord) -> int:
    """BIRD uses different aggregation than agent consensus -> INCORRECT (with result guard)."""
    try:
        gold_aggs = extract_aggregations(record.gold_sql)
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        cons_aggs = extract_aggregations(consensus_sql)

        if not gold_aggs and not cons_aggs:
            return ABSTAIN  # Neither uses aggregations

        if gold_aggs != cons_aggs:
            # Guard: only fire INCORRECT if execution results actually disagree
            gold = record.gold_exec
            from sqlsalus_core.labeling.execution_lfs import _consensus_result
            consensus = _consensus_result(record)
            if gold and consensus and results_agree(gold, consensus):
                return ABSTAIN  # Results agree despite structural difference
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF29: BIRD uses GROUP BY on different columns than agent consensus
# ---------------------------------------------------------------------------

def lf_29(record: TaskRecord) -> int:
    """BIRD uses GROUP BY on different columns than consensus -> INCORRECT (with result guard)."""
    try:
        gold_gb = set(extract_group_by(record.gold_sql))
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        cons_gb = set(extract_group_by(consensus_sql))

        if not gold_gb and not cons_gb:
            return ABSTAIN

        if gold_gb != cons_gb:
            # Guard: only fire INCORRECT if execution results actually disagree
            gold = record.gold_exec
            from sqlsalus_core.labeling.execution_lfs import _consensus_result
            consensus = _consensus_result(record)
            if gold and consensus and results_agree(gold, consensus):
                return ABSTAIN  # Results agree despite structural difference
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF30: BIRD uses INNER JOIN where agents use LEFT JOIN (or vice versa)
# ---------------------------------------------------------------------------

def lf_30(record: TaskRecord) -> int:
    """BIRD uses INNER JOIN where agents use LEFT JOIN or vice versa, results differ -> INCORRECT."""
    try:
        gold_joins = extract_joins(record.gold_sql)
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        cons_joins = extract_joins(consensus_sql)

        if not gold_joins and not cons_joins:
            return ABSTAIN

        # Build table -> join_type maps
        gold_join_types = {j["table"]: j["join_type"] for j in gold_joins if j["table"]}
        cons_join_types = {j["table"]: j["join_type"] for j in cons_joins if j["table"]}

        # Check for type mismatches on shared tables
        shared_tables = set(gold_join_types.keys()) & set(cons_join_types.keys())
        for tbl in shared_tables:
            g_type = gold_join_types[tbl]
            c_type = cons_join_types[tbl]
            # Check for inner vs left mismatch
            g_is_inner = "inner" in g_type or g_type == ""
            g_is_left = "left" in g_type
            c_is_inner = "inner" in c_type or c_type == ""
            c_is_left = "left" in c_type
            if (g_is_inner and c_is_left) or (g_is_left and c_is_inner):
                # Check if results actually differ
                gold = record.gold_exec
                from sqlsalus_core.labeling.execution_lfs import _consensus_result
                consensus = _consensus_result(record)
                if gold and consensus and not results_agree(gold, consensus):
                    return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF31: BIRD SQL has hardcoded literal not in NL or schema metadata
# ---------------------------------------------------------------------------

def lf_31(record: TaskRecord) -> int:
    """BIRD has hardcoded literal not in NL or evidence -> INCORRECT."""
    try:
        gold_literals = extract_literals(record.gold_sql)
        if not gold_literals:
            return ABSTAIN

        # Build set of acceptable literals from question + evidence
        text = (record.question + " " + record.evidence).lower()

        # Also accept common SQL literals
        common_literals = {"0", "1", "2", "null", "true", "false", "%", "_", "*"}

        for lit in gold_literals:
            lit_lower = lit.lower().strip()
            if not lit_lower:
                continue
            if lit_lower in common_literals:
                continue
            # Check if literal appears in the NL text
            if lit_lower in text:
                continue
            # Try numeric check: small integers are common
            try:
                val = float(lit_lower)
                if val == int(val) and 0 <= int(val) <= 100:
                    continue
            except (ValueError, OverflowError):
                pass
            # Literal not found in NL text -- suspicious
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF32: BIRD uses HAVING, agents don't (or vice versa), results differ
# ---------------------------------------------------------------------------

def lf_32(record: TaskRecord) -> int:
    """BIRD uses HAVING but agents don't (or vice versa), results differ -> INCORRECT."""
    try:
        gold_has_having = _has_having(record.gold_sql)
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        cons_has_having = _has_having(consensus_sql)

        if gold_has_having == cons_has_having:
            return ABSTAIN

        # Different HAVING usage -- check if results differ
        gold = record.gold_exec
        from sqlsalus_core.labeling.execution_lfs import _consensus_result
        consensus = _consensus_result(record)
        if gold and consensus and not results_agree(gold, consensus):
            return INCORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF33: Different syntax (subquery vs JOIN) but results agree -> CORRECT
# ---------------------------------------------------------------------------

def lf_33(record: TaskRecord) -> int:
    """BIRD and agents use different syntax but results agree -> CORRECT."""
    try:
        gold_has_sub = _has_subquery(record.gold_sql)
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        cons_has_sub = _has_subquery(consensus_sql)

        # Only fire if structural difference exists
        if gold_has_sub == cons_has_sub:
            return ABSTAIN

        # Check if results agree despite structural difference
        gold = record.gold_exec
        from sqlsalus_core.labeling.execution_lfs import _consensus_result
        consensus = _consensus_result(record)
        if gold and consensus and results_agree(gold, consensus):
            return CORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF34: BIRD and agents semantically equivalent (same AST normal form)
# ---------------------------------------------------------------------------

def lf_34(record: TaskRecord) -> int:
    """BIRD and consensus SQL normalize to same AST -> CORRECT."""
    if not _HAS_SQLGLOT:
        return ABSTAIN

    try:
        consensus_sql = get_consensus_sql(record)
        if consensus_sql is None:
            return ABSTAIN

        # Normalize both SQLs via sqlglot optimize
        gold_trees = sqlglot.parse(record.gold_sql, dialect="sqlite")
        cons_trees = sqlglot.parse(consensus_sql, dialect="sqlite")

        if not gold_trees or not cons_trees:
            return ABSTAIN

        gold_norm = gold_trees[0]
        cons_norm = cons_trees[0]

        if gold_norm is None or cons_norm is None:
            return ABSTAIN

        # Compare normalized SQL strings
        if gold_norm.sql(dialect="sqlite").lower().strip() == cons_norm.sql(dialect="sqlite").lower().strip():
            return CORRECT

    except Exception:
        return ABSTAIN

    return ABSTAIN
