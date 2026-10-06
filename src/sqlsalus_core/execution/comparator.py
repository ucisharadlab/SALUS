from __future__ import annotations

import math
from dataclasses import dataclass

from sqlsalus_core.execution.executor import ExecutionResult


@dataclass
class ComparisonResult:
    exact_match: bool
    r1_subset_r2: bool
    r2_subset_r1: bool
    overlap_ratio: float  # Jaccard: |intersection| / |union|
    card_diff: int  # r1.row_count - r2.row_count


def normalize_value(val) -> object:
    """Normalize a single value for comparison.

    - Strings: strip + lowercase
    - None / "NULL" / "null" / "None": unified to None
    - Numeric strings: convert to float (then int if lossless)
    - Booleans: convert to int (0/1) for cross-type consistency
    """
    if val is None:
        return None

    if isinstance(val, bool):
        return int(val)

    if isinstance(val, float):
        if math.isnan(val):
            return None
        if val == int(val):
            return int(val)
        return val

    if isinstance(val, int):
        return val

    if isinstance(val, str):
        stripped = val.strip()
        if stripped.upper() in ("NULL", "NONE", ""):
            return None
        # Try numeric conversion
        try:
            f = float(stripped)
            if math.isnan(f):
                return None
            if f == int(f) and "." not in stripped and "e" not in stripped.lower():
                return int(f)
            return f
        except (ValueError, OverflowError):
            pass
        return stripped.lower()

    if isinstance(val, bytes):
        return val

    return val


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    normalized = [tuple(normalize_value(v) for v in row) for row in rows]
    # Sort for order-independent comparison. Rows may contain None,
    # so we use a key that handles mixed types.
    return sorted(normalized, key=_sort_key)


def _sort_key(row: tuple):
    """Produce a sort key that handles None and mixed types without crashing."""
    parts = []
    for v in row:
        if v is None:
            parts.append((0, ""))
        elif isinstance(v, (int, float)):
            parts.append((1, v))
        elif isinstance(v, str):
            parts.append((2, v))
        elif isinstance(v, bytes):
            parts.append((3, v.hex()))
        else:
            parts.append((4, str(v)))
    return parts


# ---------------------------------------------------------------------------
# Column alignment helpers for labeling functions; core results_agree remains strict.
# ---------------------------------------------------------------------------

def _align_rows_by_column_names(
    r1: ExecutionResult, r2: ExecutionResult
) -> tuple[list[tuple], list[tuple]] | None:
    """If r1 and r2 have the same column names (case-insensitive) but in
    different order, reorder r2's columns to match r1's order.

    Returns (r1_rows, reordered_r2_rows) or None if alignment is not possible
    (different column names, missing column info, etc.).
    """
    if not r1.columns or not r2.columns:
        return None
    if len(r1.columns) != len(r2.columns):
        return None

    r1_lower = [c.lower().strip() for c in r1.columns]
    r2_lower = [c.lower().strip() for c in r2.columns]

    # Same order already?
    if r1_lower == r2_lower:
        return None  # No reordering needed

    # Same set of column names?
    if sorted(r1_lower) != sorted(r2_lower):
        return None  # Different column names entirely

    # Build permutation: for each position in r1, find matching position in r2
    r2_used = [False] * len(r2_lower)
    perm = []
    for name in r1_lower:
        found = False
        for j, r2_name in enumerate(r2_lower):
            if r2_name == name and not r2_used[j]:
                perm.append(j)
                r2_used[j] = True
                found = True
                break
        if not found:
            return None  # Shouldn't happen, but safety check

    # Reorder r2 rows
    reordered = [tuple(row[perm[i]] for i in range(len(perm))) for row in r2.rows]
    return r1.rows, reordered


def results_agree_column_aware(r1: ExecutionResult, r2: ExecutionResult) -> bool:
    """Like results_agree but also handles column reordering.

    Used by column-alignment LFs; agent verdicts use results_agree.
    """
    if r1.error is not None or r2.error is not None:
        return False

    # Try direct comparison first
    nr1 = normalize_rows(r1.rows)
    nr2 = normalize_rows(r2.rows)
    if nr1 == nr2:
        return True

    # Try column-aligned comparison
    aligned = _align_rows_by_column_names(r1, r2)
    if aligned is not None:
        rows1, rows2_reordered = aligned
        return normalize_rows(rows1) == normalize_rows(rows2_reordered)

    return False


def results_agree_approx(r1: ExecutionResult, r2: ExecutionResult,
                         rel_tol: float = 1e-4) -> bool:
    """Like results_agree but with approximate float comparison.

    Used by float-tolerance LFs; agent verdicts use results_agree.
    """
    if r1.error is not None or r2.error is not None:
        return False

    nr1 = normalize_rows(r1.rows)
    nr2 = normalize_rows(r2.rows)

    if len(nr1) != len(nr2):
        return False

    for row1, row2 in zip(nr1, nr2):
        if len(row1) != len(row2):
            return False
        for v1, v2 in zip(row1, row2):
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                if not math.isclose(v1, v2, rel_tol=rel_tol, abs_tol=1e-9):
                    return False
            elif v1 != v2:
                return False

    return True


# ---------------------------------------------------------------------------
# Execution-result agreement for agent verdicts
# ---------------------------------------------------------------------------

def results_agree(r1: ExecutionResult, r2: ExecutionResult) -> bool:
    """Return True iff R(q1, D) == R(q2, D) after normalization.

    If either result carries an error, agreement is False.
    """
    if r1.error is not None or r2.error is not None:
        return False

    return normalize_rows(r1.rows) == normalize_rows(r2.rows)


# ---------------------------------------------------------------------------
# Richer comparison for labeling functions
# ---------------------------------------------------------------------------

def compare_result_sets(r1: ExecutionResult, r2: ExecutionResult) -> ComparisonResult:
    if r1.error is not None or r2.error is not None:
        return ComparisonResult(
            exact_match=False,
            r1_subset_r2=False,
            r2_subset_r1=False,
            overlap_ratio=0.0,
            card_diff=r1.row_count - r2.row_count,
        )

    nr1 = normalize_rows(r1.rows)
    nr2 = normalize_rows(r2.rows)

    set1 = set(nr1)
    set2 = set(nr2)

    intersection = set1 & set2
    union = set1 | set2

    overlap_ratio = len(intersection) / len(union) if union else 1.0

    return ComparisonResult(
        exact_match=(nr1 == nr2),
        r1_subset_r2=set1 <= set2,
        r2_subset_r1=set2 <= set1,
        overlap_ratio=overlap_ratio,
        card_diff=r1.row_count - r2.row_count,
    )
