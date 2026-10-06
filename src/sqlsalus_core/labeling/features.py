"""SQL structural features used by the SALUS decision plane.

The parser returns ten structural features. The decision plane adds execution
result cardinality and an execution-error indicator to form its 12 features.
"""

from __future__ import annotations


def _parse_sql_features(sql: str) -> dict[str, float]:
    """Extract structural features from a SQL string using sqlglot.

    Returns a dict of feature_name -> float, with zero values if parsing fails.
    """
    defaults = {
        "gold_join_count": 0.0,
        "gold_where_count": 0.0,
        "gold_has_group_by": 0.0,
        "gold_has_subquery": 0.0,
        "gold_has_having": 0.0,
        "gold_has_order_by": 0.0,
        "gold_has_distinct": 0.0,
        "gold_has_limit": 0.0,
        "gold_aggregate_count": 0.0,
        "gold_table_count": 0.0,
    }

    try:
        import sqlglot
        from sqlglot import exp

        parsed = sqlglot.parse(sql, read="sqlite")
        if not parsed:
            return defaults

        tree = parsed[0]
        if tree is None:
            return defaults

        features = dict(defaults)

        # JOIN count: count all Join nodes in the AST
        features["gold_join_count"] = float(len(list(tree.find_all(exp.Join))))

        # WHERE clause count: each Where node
        features["gold_where_count"] = float(len(list(tree.find_all(exp.Where))))

        # GROUP BY
        features["gold_has_group_by"] = 1.0 if list(tree.find_all(exp.Group)) else 0.0

        # Subquery: Select nodes nested inside another Select
        subqueries = list(tree.find_all(exp.Subquery))
        features["gold_has_subquery"] = 1.0 if subqueries else 0.0

        # HAVING
        features["gold_has_having"] = 1.0 if list(tree.find_all(exp.Having)) else 0.0

        # ORDER BY
        features["gold_has_order_by"] = 1.0 if list(tree.find_all(exp.Order)) else 0.0

        # DISTINCT
        has_distinct = bool(list(tree.find_all(exp.Distinct)))
        features["gold_has_distinct"] = 1.0 if has_distinct else 0.0

        # LIMIT
        features["gold_has_limit"] = 1.0 if list(tree.find_all(exp.Limit)) else 0.0

        # Aggregate functions (COUNT, SUM, AVG, MAX, MIN)
        agg_types = (exp.Count, exp.Sum, exp.Avg, exp.Max, exp.Min)
        agg_count = 0
        for agg_type in agg_types:
            agg_count += len(list(tree.find_all(agg_type)))
        features["gold_aggregate_count"] = float(agg_count)

        # Table count: count Table nodes (FROM + JOINs)
        features["gold_table_count"] = float(len(list(tree.find_all(exp.Table))))

        return features

    except Exception:
        return defaults
