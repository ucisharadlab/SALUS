"""Generate labeling-matrix artifacts for the SALUS LF set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from sqlsalus_core.labeling.base import TaskRecord, _rebuild_exec
from sqlsalus_core.labeling.registry import META_LFS, TASK_LFS
from sqlsalus_core.data_io import load_json


SRC_DIR = Path(__file__).resolve().parent.parent

CONFIGS: dict[str, dict[str, Path]] = {
    "frontier": {
        "data_path": SRC_DIR / "data" / "processed_query" / "bird_dev_frontier.json.gz",
        "output_dir": SRC_DIR / "data" / "weak_supervision" / "frontier",
    },
    "frontier_november": {
        "data_path": SRC_DIR / "data" / "processed_query" / "bird_dev_nov_4agent.json.gz",
        "output_dir": SRC_DIR / "data" / "weak_supervision" / "frontier_november",
    },
    "frontier_spider": {
        "data_path": SRC_DIR / "data" / "processed_query" / "spider_dev_frontier.json.gz",
        "output_dir": SRC_DIR / "data" / "weak_supervision" / "frontier_spider",
    },
    "legacy": {
        "data_path": SRC_DIR / "data" / "processed_query" / "bird_dev_legacy.json.gz",
        "output_dir": SRC_DIR / "data" / "weak_supervision" / "legacy",
    },
}


def load_processed_rows(path: Path) -> list[dict[str, Any]]:
    return load_json(path)


def build_task_records(path: Path) -> list[TaskRecord]:
    records: list[TaskRecord] = []
    for entry in load_processed_rows(path):
        gold_result = entry.get("gold_result", {})
        agents_data = entry.get("agents", {})
        rec = TaskRecord(
            task_id=int(entry.get("task_id", 0)),
            db_id=str(entry.get("db_id", "")),
            question=str(entry.get("question", "")),
            evidence=str(entry.get("evidence", "")),
            difficulty=str(entry.get("difficulty", "")),
            gold_sql=str(entry.get("gold_sql", "")),
            gold_result=gold_result,
            agents=agents_data,
        )
        rec.gold_exec = _rebuild_exec(gold_result) if gold_result else None
        rec.agent_execs = {}
        rec.agent_sqls = {}
        for agent_name, agent_data in agents_data.items():
            if isinstance(agent_data, dict):
                rec.agent_execs[agent_name] = _rebuild_exec(agent_data.get("result", {}))
                rec.agent_sqls[agent_name] = agent_data.get("sql", "")
        records.append(rec)
    return records


def sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_csv(path: Path, task_ids: np.ndarray, columns: list[str], matrix: np.ndarray) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", *columns])
        for task_id, row in zip(task_ids.tolist(), matrix.tolist()):
            writer.writerow([task_id, *row])


def generate_for_config(config_name: str, data_path: Path, output_dir: Path) -> dict[str, Any]:
    records = build_task_records(data_path)
    task_ids = np.asarray([record.task_id for record in records], dtype=int)
    task_columns = [lf.__name__ for lf in TASK_LFS]
    meta_columns = [lf.__name__ for lf in META_LFS]

    task_matrix = np.asarray([[lf(record) for lf in TASK_LFS] for record in records], dtype=np.int8)
    meta_matrix = np.asarray([[meta(votes.tolist()) for meta in META_LFS] for votes in task_matrix], dtype=np.int8)
    full_matrix = np.concatenate([task_matrix, meta_matrix], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "task_ids.npy", task_ids)
    np.save(output_dir / "labeling_matrix_task.npy", task_matrix)
    np.save(output_dir / "labeling_matrix_meta.npy", meta_matrix)
    np.save(output_dir / "labeling_matrix_full.npy", full_matrix)
    write_csv(output_dir / "labeling_matrix.csv", task_ids, task_columns + meta_columns, full_matrix)

    manifest = {
        "config": config_name,
        "data_path": str(data_path.relative_to(SRC_DIR.parent)),
        "output_dir": str(output_dir.relative_to(SRC_DIR.parent)),
        "n_tasks": int(len(records)),
        "task_columns": task_columns,
        "meta_columns": meta_columns,
        "task_shape": list(task_matrix.shape),
        "meta_shape": list(meta_matrix.shape),
        "full_shape": list(full_matrix.shape),
        "task_sha256": sha256_bytes(task_matrix),
        "meta_sha256": sha256_bytes(meta_matrix),
        "full_sha256": sha256_bytes(full_matrix),
    }
    with (output_dir / "labeling_matrix_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        action="append",
        choices=sorted(CONFIGS),
        help="Config(s) to regenerate. Defaults to all configs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_names = args.config or list(CONFIGS)
    manifest: dict[str, Any] = {}
    for config_name in config_names:
        config = CONFIGS[config_name]
        manifest[config_name] = generate_for_config(
            config_name,
            config["data_path"],
            config["output_dir"],
        )
        print(
            f"{config_name}: {manifest[config_name]['n_tasks']} tasks, "
            f"{manifest[config_name]['full_shape'][1]} LF columns"
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
