"""NL-SQL alignment labeling functions (LF15-LF25).

These LFs check whether the BIRD ground truth SQL correctly reflects
the intent expressed in the natural language question, using regex-based
keyword matching.
"""

from __future__ import annotations

import re

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

def _sql_upper(sql: str) -> str:
    """Return uppercased SQL for keyword matching."""
    return sql.upper()


def _sql_has_construct(sql: str, construct: str) -> bool:
    """Check if SQL contains a given construct keyword (case-insensitive)."""
    return bool(re.search(r'\b' + construct + r'\b', sql, re.IGNORECASE))


def _nl_matches(text: str, *patterns: str) -> bool:
    """Check if NL text matches any of the given regex patterns."""
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True
    return False


def _extract_nl_literals(question: str) -> set[str]:
    """Extract quoted strings and standalone numbers from the question."""
    literals: set[str] = set()

    # Quoted strings (single or double quotes)
    for match in re.finditer(r"""['"]([^'"]+)['"]""", question):
        literals.add(match.group(1).lower().strip())

    # Numbers (standalone, not part of words)
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', question):
        literals.add(match.group(1))

    return literals


def _extract_sql_where_literals(sql: str) -> set[str]:
    """Extract literal values from WHERE clause using sqlglot."""
    if not _HAS_SQLGLOT:
        # Fallback: simple regex
        literals: set[str] = set()
        for match in re.finditer(r"""['"]([^'"]+)['"]""", sql):
            literals.add(match.group(1).lower().strip())
        for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', sql):
            literals.add(match.group(1))
        return literals

    try:
        trees = sqlglot.parse(sql, dialect="sqlite")
        literals: set[str] = set()
        for tree in trees:
            if tree is None:
                continue
            where = tree.find(sqlglot_exp.Where)
            if where is None:
                continue
            for lit in where.find_all(sqlglot_exp.Literal):
                val = lit.this
                if val:
                    literals.add(str(val).lower().strip("'\""))
        return literals
    except Exception:
        return set()


def _sql_has_distinct(sql: str) -> bool:
    """Check if SQL has DISTINCT keyword."""
    return _sql_has_construct(sql, "DISTINCT")


def _any_agent_has_distinct(record: TaskRecord) -> bool:
    """Check if any agent SQL uses DISTINCT."""
    for sql in record.agent_sqls.values():
        if sql and _sql_has_distinct(sql):
            return True
    return False


# ---------------------------------------------------------------------------
# LF15: NL says "how many"/"count" but BIRD SQL has no COUNT
# ---------------------------------------------------------------------------

def lf_15(record: TaskRecord) -> int:
    """Bipolar: NL says 'how many'/'count' — has COUNT -> CORRECT; missing -> INCORRECT."""
    question = record.question
    sql = record.gold_sql

    if _nl_matches(question, r'\bhow\s+many\b', r'\bcount\s+(?:the|of|all)\b',
                   r'\bnumber\s+of\b', r'\btotal\s+number\b'):
        if _sql_has_construct(sql, "COUNT"):
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF16: NL says "average"/"mean" but BIRD SQL has no AVG
# ---------------------------------------------------------------------------

def lf_16(record: TaskRecord) -> int:
    """Bipolar: NL says 'average'/'mean' — has AVG -> CORRECT; missing -> INCORRECT."""
    question = record.question

    if _nl_matches(question, r'\baverage\b', r'\bmean\b', r'\bavg\b'):
        if _sql_has_construct(record.gold_sql, "AVG"):
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF17: NL says "maximum"/"highest"/"most" but BIRD SQL has no MAX
# ---------------------------------------------------------------------------

def lf_17(record: TaskRecord) -> int:
    """Bipolar: NL says 'maximum'/'highest' — has MAX or DESC LIMIT -> CORRECT; missing -> INCORRECT."""
    question = record.question

    if _nl_matches(question, r'\bmaximum\b', r'\bhighest\b', r'\blargest\b',
                   r'\bgreatest\b', r'\bmax\b'):
        if _sql_has_construct(record.gold_sql, "MAX"):
            return CORRECT
        # Also check ORDER BY ... DESC LIMIT 1 as an alternative
        has_desc_limit = (
            _sql_has_construct(record.gold_sql, "DESC") and
            _sql_has_construct(record.gold_sql, "LIMIT")
        )
        if has_desc_limit:
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF18: NL says "minimum"/"lowest"/"least" but BIRD SQL has no MIN
# ---------------------------------------------------------------------------

def lf_18(record: TaskRecord) -> int:
    """Bipolar: NL says 'minimum'/'lowest' — has MIN or ASC LIMIT -> CORRECT; missing -> INCORRECT."""
    question = record.question

    # Exclude "at least" — it doesn't imply MIN
    if re.search(r'\bat\s+least\b', question, re.IGNORECASE):
        # Remove "at least" occurrences before checking for "least"
        question_filtered = re.sub(r'\bat\s+least\b', '', question, flags=re.IGNORECASE)
    else:
        question_filtered = question

    if _nl_matches(question_filtered, r'\bminimum\b', r'\blowest\b', r'\bsmallest\b',
                   r'\bleast\b', r'\bmin\b'):
        if _sql_has_construct(record.gold_sql, "MIN"):
            return CORRECT
        # Also check ORDER BY ... ASC LIMIT 1 as alternative
        has_asc_limit = (
            _sql_has_construct(record.gold_sql, "ASC") and
            _sql_has_construct(record.gold_sql, "LIMIT")
        )
        if has_asc_limit:
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF19: NL says "total"/"sum" but BIRD SQL has no SUM
# ---------------------------------------------------------------------------

def lf_19(record: TaskRecord) -> int:
    """Bipolar: NL says 'total'/'sum' — has SUM -> CORRECT; missing -> INCORRECT."""
    question = record.question

    # Be careful: "total number" maps to COUNT, not SUM
    # Only match "total" when not followed by "number"
    if _nl_matches(question, r'\bsum\b', r'\bsummation\b'):
        if _sql_has_construct(record.gold_sql, "SUM"):
            return CORRECT
        return INCORRECT

    if _nl_matches(question, r'\btotal\b') and not _nl_matches(question, r'\btotal\s+number\b'):
        if _sql_has_construct(record.gold_sql, "SUM"):
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF20: NL mentions "top N"/"first N" but BIRD SQL has no LIMIT
# ---------------------------------------------------------------------------

def lf_20(record: TaskRecord) -> int:
    """Bipolar: NL mentions 'top N'/'first N' — has LIMIT -> CORRECT; missing -> INCORRECT."""
    question = record.question

    if _nl_matches(question, r'\btop\s+\d+\b', r'\bfirst\s+\d+\b',
                   r'\bbottom\s+\d+\b', r'\blast\s+\d+\b'):
        if _sql_has_construct(record.gold_sql, "LIMIT"):
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF21: NL mentions ordering but BIRD has no ORDER BY
# ---------------------------------------------------------------------------

def lf_21(record: TaskRecord) -> int:
    """Bipolar: NL mentions ordering — has ORDER BY -> CORRECT; missing -> INCORRECT."""
    question = record.question

    if _nl_matches(question, r'\border(?:ed)?\s+by\b', r'\bsort(?:ed)?\s+by\b',
                   r'\branke?d?\s+by\b', r'\bin\s+(?:ascending|descending)\s+order\b',
                   r'\bfrom\s+(?:highest|lowest)\b'):
        if _sql_has_construct(record.gold_sql, "ORDER BY"):
            return CORRECT
        return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF22: NL mentions specific filter value not in BIRD WHERE clause
# ---------------------------------------------------------------------------

def lf_22(record: TaskRecord) -> int:
    """NL mentions specific value not in BIRD WHERE clause -> INCORRECT.

    Best-effort: checks quoted strings and standalone numbers from the question
    against literals in the SQL WHERE clause.
    """
    question = record.question
    nl_literals = _extract_nl_literals(question)
    if not nl_literals:
        return ABSTAIN

    sql_literals = _extract_sql_where_literals(record.gold_sql)

    # Also include literals from the full SQL (not just WHERE)
    all_sql_literals: set[str] = set()
    for match in re.finditer(r"""['"]([^'"]+)['"]""", record.gold_sql):
        all_sql_literals.add(match.group(1).lower().strip())
    for match in re.finditer(r'\b(\d+(?:\.\d+)?)\b', record.gold_sql):
        all_sql_literals.add(match.group(1))
    all_sql_literals |= sql_literals

    # Check if NL-mentioned literals are in the SQL
    missing = nl_literals - all_sql_literals
    if missing:
        # Filter out very common/small numbers that may not need to be literal filters
        significant_missing = {m for m in missing if len(m) > 1 or not m.isdigit()}
        if significant_missing:
            return INCORRECT

    return CORRECT


# ---------------------------------------------------------------------------
# LF23: NL says "distinct"/"unique" but BIRD SQL has no DISTINCT
# ---------------------------------------------------------------------------

def lf_23(record: TaskRecord) -> int:
    """Bipolar: NL says 'distinct' — has DISTINCT -> CORRECT; missing (agents have it) -> INCORRECT."""
    question = record.question

    if _nl_matches(question, r'\bdistinct\b', r'\bunique\b', r'\bdifferent\b'):
        if _sql_has_distinct(record.gold_sql):
            return CORRECT
        if _any_agent_has_distinct(record):
            return INCORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF24: NL asks yes/no question but BIRD returns full table
# ---------------------------------------------------------------------------

def lf_24(record: TaskRecord) -> int:
    """Bipolar: yes/no Q + small result -> CORRECT; full table -> INCORRECT."""
    question = record.question
    gold = record.gold_exec

    if gold is None or gold.error is not None:
        return ABSTAIN

    if _nl_matches(question, r'^is\s+there\b', r'^does\b', r'^do\b', r'^are\s+there\b',
                   r'^has\b', r'^have\b', r'^is\s+it\b', r'^can\b', r'^was\b',
                   r'^were\b', r'^did\b', r'^will\b', r'^would\b',
                   r'\byes\s+or\s+no\b', r'\btrue\s+or\s+false\b'):
        # Stricter: only flag as INCORRECT if gold returns a large multi-row table
        # A proper yes/no answer should be 1 row x 1 column
        n_cols = len(gold.columns) if gold.columns else 0
        if gold.row_count > 10 and n_cols == 1:
            return INCORRECT
        if gold.row_count > 1 and n_cols > 1:
            return INCORRECT
        return CORRECT

    return ABSTAIN


# ---------------------------------------------------------------------------
# LF25: All checked keyword-to-SQL mappings are satisfied -> CORRECT
# ---------------------------------------------------------------------------

def lf_25(record: TaskRecord) -> int:
    """All NL keyword-to-SQL mappings check out -> CORRECT.

    Runs a battery of keyword checks; if all applicable ones pass, vote CORRECT.
    """
    question = record.question
    sql = record.gold_sql
    checks_applied = 0
    checks_passed = 0

    # COUNT check
    if _nl_matches(question, r'\bhow\s+many\b', r'\bnumber\s+of\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "COUNT"):
            checks_passed += 1

    # AVG check
    if _nl_matches(question, r'\baverage\b', r'\bmean\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "AVG"):
            checks_passed += 1

    # MAX check
    if _nl_matches(question, r'\bmaximum\b', r'\bhighest\b', r'\blargest\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "MAX") or (
            _sql_has_construct(sql, "DESC") and _sql_has_construct(sql, "LIMIT")
        ):
            checks_passed += 1

    # MIN check
    if _nl_matches(question, r'\bminimum\b', r'\blowest\b', r'\bsmallest\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "MIN") or (
            _sql_has_construct(sql, "ASC") and _sql_has_construct(sql, "LIMIT")
        ):
            checks_passed += 1

    # SUM check
    if _nl_matches(question, r'\bsum\b', r'\btotal\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "SUM"):
            checks_passed += 1

    # LIMIT check
    if _nl_matches(question, r'\btop\s+\d+\b', r'\bfirst\s+\d+\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "LIMIT"):
            checks_passed += 1

    # ORDER BY check
    if _nl_matches(question, r'\border(?:ed)?\s+by\b', r'\bsort(?:ed)?\s+by\b'):
        checks_applied += 1
        if _sql_has_construct(sql, "ORDER BY"):
            checks_passed += 1

    # Need at least 3 checks to have confidence
    if checks_applied < 3:
        return ABSTAIN

    if checks_applied == checks_passed:
        return CORRECT

    return INCORRECT
