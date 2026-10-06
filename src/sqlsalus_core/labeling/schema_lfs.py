"""Schema integrity labeling functions (LF35).

LF35 checks whether the reference SQL and agent consensus SQL use the same tables.
"""

from __future__ import annotations

from typing import Any

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

def _parse_sql(sql: str) -> list[Any] | None:
    """Parse SQL with sqlglot, returning AST list or None on failure."""
    if not _HAS_SQLGLOT:
        return None
    try:
        return sqlglot.parse(sql, dialect="sqlite")
    except Exception:
        return None


def _extract_table_names(sql: str) -> set[str]:
    """Extract referenced table names from SQL using sqlglot."""
    trees = _parse_sql(sql)
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


def _extract_column_refs(sql: str) -> list[tuple[str | None, str]]:
    """Extract (table_or_alias, column_name) pairs from SQL.

    table_or_alias may be None if the column is not table-qualified.
    """
    trees = _parse_sql(sql)
    if not trees:
        return []
    refs: list[tuple[str | None, str]] = []
    for tree in trees:
        if tree is None:
            continue
        for col in tree.find_all(sqlglot_exp.Column):
            table_part = col.table
            col_name = col.name
            if col_name:
                refs.append((table_part.lower() if table_part else None, col_name.lower()))
    return refs


def _extract_join_columns(sql: str) -> list[tuple[str, str, str, str]]:
    """Extract join column pairs: (left_table, left_col, right_table, right_col).

    Best-effort extraction from ON conditions in JOINs.
    """
    trees = _parse_sql(sql)
    if not trees:
        return []
    joins: list[tuple[str, str, str, str]] = []
    for tree in trees:
        if tree is None:
            continue
        for join in tree.find_all(sqlglot_exp.Join):
            on_cond = join.args.get("on")
            if on_cond is None:
                continue
            # Look for EQ nodes in the ON clause
            for eq in on_cond.find_all(sqlglot_exp.EQ):
                left = eq.left
                right = eq.right
                if isinstance(left, sqlglot_exp.Column) and isinstance(right, sqlglot_exp.Column):
                    lt = left.table.lower() if left.table else ""
                    lc = left.name.lower() if left.name else ""
                    rt = right.table.lower() if right.table else ""
                    rc = right.name.lower() if right.name else ""
                    if lc and rc:
                        joins.append((lt, lc, rt, rc))
    return joins


def _schema_columns(schema: dict[str, list[dict[str, str]]]) -> dict[str, set[str]]:
    """Build table_name_lower -> {col_name_lower, ...} mapping."""
    result: dict[str, set[str]] = {}
    for table, cols in schema.items():
        result[table.lower()] = {c["name"].lower() for c in cols}
    return result


def _schema_column_types(schema: dict[str, list[dict[str, str]]]) -> dict[str, dict[str, str]]:
    """Build table_name_lower -> {col_name_lower: type_string_lower}."""
    result: dict[str, dict[str, str]] = {}
    for table, cols in schema.items():
        result[table.lower()] = {c["name"].lower(): c["type"].lower() for c in cols}
    return result


def _is_numeric_type(type_str: str) -> bool:
    """Check if a SQLite column type looks numeric."""
    type_str = type_str.lower()
    return any(kw in type_str for kw in ("int", "real", "float", "double", "numeric", "decimal"))


def _is_string_type(type_str: str) -> bool:
    """Check if a SQLite column type looks like text."""
    type_str = type_str.lower()
    return any(kw in type_str for kw in ("text", "char", "varchar", "clob", "string"))


def _get_consensus_tables(record: TaskRecord) -> set[str]:
    """Get the set of tables used by the agent consensus SQL."""
    from sqlsalus_core.labeling.execution_lfs import get_agent_consensus

    consensus_names = get_agent_consensus(record)
    if not consensus_names:
        return set()
    # Use first consensus agent's SQL
    first_agent = consensus_names[0]
    sql = record.agent_sqls.get(first_agent, "")
    return _extract_table_names(sql)


def _build_alias_map(sql: str) -> dict[str, str]:
    """Build alias_lower -> table_name_lower mapping from the SQL."""
    trees = _parse_sql(sql)
    if not trees:
        return {}
    alias_map: dict[str, str] = {}
    for tree in trees:
        if tree is None:
            continue
        for tbl in tree.find_all(sqlglot_exp.Table):
            name = tbl.name
            alias = tbl.alias
            if name:
                if alias:
                    alias_map[alias.lower()] = name.lower()
                alias_map[name.lower()] = name.lower()
    return alias_map


# ---------------------------------------------------------------------------
# LF35: BIRD SQL uses same tables as agent consensus -> CORRECT
# ---------------------------------------------------------------------------

def lf_35(record: TaskRecord) -> int:
    """BIRD SQL uses same tables as agent consensus -> CORRECT."""
    try:
        gold_tables = _extract_table_names(record.gold_sql)
    except Exception:
        return ABSTAIN

    consensus_tables = _get_consensus_tables(record)

    if not gold_tables or not consensus_tables:
        return ABSTAIN

    if gold_tables == consensus_tables:
        return CORRECT

    return ABSTAIN
