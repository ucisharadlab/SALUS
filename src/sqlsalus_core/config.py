"""Paths and agent identities for offline SALUS runs."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = Path(os.getenv("SQLSALUS_RESULTS_DIR", str(PROJECT_ROOT / "results")))
CURRENT_FRONTIER_DATA_PATH = DATA_DIR / "processed_query" / "bird_dev_frontier.json.gz"
CURRENT_FRONTIER_AGENTS = ("gpt5", "gpt4o", "claude", "gemini")
CURRENT_FRONTIER_DISPLAY = {
    "gpt5": "GPT-5", "gpt4o": "GPT-4o",
    "claude": "Claude Opus 4.6", "gemini": "Gemini 2.5 Pro",
}
PROCESSED_DATA_PATH = Path(os.getenv("SQLSALUS_DATA_PATH", str(CURRENT_FRONTIER_DATA_PATH)))
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "GT_RS_200.csv"
SQL_TIMEOUT_SECONDS = 30
