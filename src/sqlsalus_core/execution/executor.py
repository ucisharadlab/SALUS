from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ExecutionResult:
    rows: list[tuple] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    error: str | None = None
    row_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": [list(r) for r in self.rows],
            "columns": self.columns,
            "error": self.error,
            "row_count": self.row_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ExecutionResult:
        return cls(
            rows=[tuple(r) for r in d["rows"]],
            columns=d["columns"],
            error=d.get("error"),
            row_count=d["row_count"],
        )


def execute_sql(db_path: Path, sql: str, timeout: int = 30) -> ExecutionResult:
    """Execute a SQL query on a SQLite database, returning structured results.

    Thread-safe: uses threading.Timer + connection.interrupt() instead of
    signal.SIGALRM, so it works correctly from any thread.
    """
    conn = None
    timer = None
    timed_out = [False]

    try:
        conn = sqlite3.connect(str(db_path))

        def _interrupt():
            timed_out[0] = True
            try:
                conn.interrupt()
            except Exception:
                pass

        timer = threading.Timer(timeout, _interrupt)
        timer.start()

        cursor = conn.execute(sql)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        rows = cursor.fetchall()

        if timed_out[0]:
            return ExecutionResult(error=f"Query timed out after {timeout}s")

        return ExecutionResult(
            rows=rows,
            columns=columns,
            error=None,
            row_count=len(rows),
        )

    except sqlite3.OperationalError as e:
        if timed_out[0]:
            return ExecutionResult(error=f"Query timed out after {timeout}s")
        return ExecutionResult(error=f"OperationalError: {e}")

    except sqlite3.Error as e:
        return ExecutionResult(error=f"{type(e).__name__}: {e}")

    except Exception as e:
        return ExecutionResult(error=f"{type(e).__name__}: {e}")

    finally:
        if timer is not None:
            timer.cancel()
        if conn is not None:
            conn.close()
