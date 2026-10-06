#!/usr/bin/env python3
"""Run the SALUS pipeline and write experiment reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from snorkel.labeling.model import LabelModel

logging.getLogger().setLevel(logging.WARNING)

from sqlsalus_core.stacking import (
    apply_strategy as apply_stacking_strategy,
    build_verdict_matrix,
    exhaustive_search,
    metrics as compute_stacking_metrics,
    verdict_to_input_idx,
)
from sqlsalus_core.decision_plane import (
    WEIGHTED_VOTE_THRESHOLD,
    apply_strategy,
    build_task_records,
    compute_metrics,
    compute_validity_masks,
    compute_verdicts,
    cv_select_strategy,
    get_reliability_probs,
    load_data,
    predict_fallback_task_labels,
    predict_fallback_task_probs,
    train_fallback_task_model,
    train_meta_clf,
    train_reliability_models,
    train_score_fusion_clf,
    weighted_vote_scores,
)
from sqlsalus_core.config import (
    CURRENT_FRONTIER_AGENTS,
    CURRENT_FRONTIER_DATA_PATH,
)
from sqlsalus_core.evaluation.validity_aware import majority_vote_prediction
from sqlsalus_core.execution.comparator import results_agree
from sqlsalus_core.generate_lf_matrices import (
    build_task_records as build_lf_task_records,
)
from sqlsalus_core.labeling.base import _rebuild_exec
from sqlsalus_core.labeling.features import _parse_sql_features
from sqlsalus_core.labeling.registry import META_LFS, TASK_LFS
from sqlsalus_core.data_io import load_json

QUERY_SOURCES = {
    "sqld_dev": ROOT_DIR / "data" / "query" / "sqld_dev.json",
    "dev": ROOT_DIR / "data" / "query" / "dev.json",
    "BIRD_1106_dev": ROOT_DIR / "data" / "query" / "BIRD_1106_dev.json",
}

PROCESSED_SOURCES = {
    "frontier": SRC_DIR / "data" / "processed_query" / "bird_dev_frontier.json.gz",
    "frontier_november": SRC_DIR / "data" / "processed_query" / "bird_dev_nov_4agent.json.gz",
    "frontier_spider": SRC_DIR / "data" / "processed_query" / "spider_dev_frontier.json.gz",
    "legacy": SRC_DIR / "data" / "processed_query" / "bird_dev_legacy.json.gz",
}

LF_MATRICES = {
    "frontier": SRC_DIR / "data" / "weak_supervision" / "frontier",
    "frontier_november": SRC_DIR / "data" / "weak_supervision" / "frontier_november",
    "frontier_spider": SRC_DIR / "data" / "weak_supervision" / "frontier_spider",
    "legacy": SRC_DIR / "data" / "weak_supervision" / "legacy",
}

GT_PATHS = {
    "BIRD_CLEAN_xs": SRC_DIR / "data" / "ground_truth" / "BIRD_CLEAN_xs.csv",
    "GT_RS_200": SRC_DIR / "data" / "ground_truth" / "GT_RS_200.csv",
}

CONFIGS = {
    "frontier": {
        "data_path": CURRENT_FRONTIER_DATA_PATH,
        "lf_matrix_path": LF_MATRICES["frontier"],
        "agents": list(CURRENT_FRONTIER_AGENTS),
        "benchmark_name": "sqld_dev",
        "display": {
            "gpt5": "GPT-5",
            "gpt4o": "GPT-4o",
            "claude": "Claude Opus 4.6",
            "gemini": "Gemini 2.5 Pro",
        },
    },
    "frontier_november": {
        "data_path": PROCESSED_SOURCES["frontier_november"],
        "lf_matrix_path": LF_MATRICES["frontier_november"],
        "agents": list(CURRENT_FRONTIER_AGENTS),
        "benchmark_name": "november_bird",
        "gt_paths": {"GT_RS_200_nov_bird": SRC_DIR / "data/ground_truth/GT_RS_200_nov_bird.csv"},
        "display": {
            "gpt5": "GPT-5",
            "gpt4o": "GPT-4o",
            "claude": "Claude Opus 4.6",
            "gemini": "Gemini 2.5 Pro",
        },
    },
    "frontier_spider": {
        "data_path": PROCESSED_SOURCES["frontier_spider"],
        "lf_matrix_path": LF_MATRICES["frontier_spider"],
        "agents": ["gpt5", "claude", "gemini", "gpt4o"],
        "benchmark_name": "spider_dev",
        "gt_paths": {"GT_RS_200_spider": SRC_DIR / "data/ground_truth/GT_RS_200_spider.csv"},
        "display": {
            "gpt5": "GPT-5",
            "gpt4o": "GPT-4o",
            "claude": "Claude Opus 4.6",
            "gemini": "Gemini 2.5 Pro",
        },
    },
    "legacy": {
        "data_path": PROCESSED_SOURCES["legacy"],
        "lf_matrix_path": LF_MATRICES["legacy"],
        "agents": ["gpt4omini", "gpt4o_old", "haiku3", "haiku"],
        "benchmark_name": "sqld_dev",
        "display": {
            "gpt4omini": "GPT-4o Mini",
            "gpt4o_old": "GPT-4o-old",
            "haiku3": "Haiku 3",
            "haiku": "Haiku 4.5",
        },
    },
}

PROCESSED_FILENAME_TO_CONFIG = {
    path.name: name for name, path in PROCESSED_SOURCES.items()
}

FULL_CONFIGS = list(CONFIGS.keys())
FULL_GT_FAMILIES = ["frontier", "legacy"]
FULL_STAGE_COUNT = 6

SQL_FEATURES = [
    "gold_aggregate_count",
    "gold_cardinality",
    "gold_error",
    "gold_has_distinct",
    "gold_has_group_by",
    "gold_has_having",
    "gold_has_limit",
    "gold_has_order_by",
    "gold_has_subquery",
    "gold_join_count",
    "gold_table_count",
    "gold_where_count",
]


# ---------------------------------------------------------------------------
# Progress-printing helpers
# ---------------------------------------------------------------------------

def _fmt_pct(p: float) -> str:
    p = max(0, min(100, int(p)))
    return f"[{p:>3d}%]"


def print_stage(i: int, n: int, label: str) -> None:
    pct = int((i - 1) / n * 100)
    print(f"{_fmt_pct(pct)} Stage {i}/{n}: {label}", flush=True)


def print_step(pct: float, msg: str, indent: str = "    ") -> None:
    print(f"{indent}{_fmt_pct(pct)} {msg}", flush=True)


def print_done() -> None:
    print(f"{_fmt_pct(100)} Pipeline complete", flush=True)


# ---------------------------------------------------------------------------
# LF matrix + weak-label regeneration
# ---------------------------------------------------------------------------

_LF_MATRIX_FILES = (
    "labeling_matrix_full.npy",
    "labeling_matrix_task.npy",
    "labeling_matrix_meta.npy",
    "task_ids.npy",
    "labeling_matrix.csv",
    "labeling_matrix_manifest.json",
)

_SNORKEL_FILES = (
    "snorkel_labels.npy",
    "snorkel_probs.npy",
)


def _lf_matrix_present(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in _LF_MATRIX_FILES)


def _weak_labels_present(output_dir: Path) -> bool:
    return all((output_dir / name).exists() for name in _SNORKEL_FILES)


def _sha256_bytes(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _write_lf_matrix_csv(
    path: Path, task_ids: np.ndarray, columns: list[str], matrix: np.ndarray
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", *columns])
        for task_id, row in zip(task_ids.tolist(), matrix.tolist()):
            writer.writerow([task_id, *row])


def ensure_lf_matrix(
    config_name: str,
    data_path: Path,
    output_dir: Path,
    *,
    indent: str = "    ",
    force: bool = False,
) -> None:
    if _lf_matrix_present(output_dir) and not force:
        print_step(100, f"{config_name}: LF matrices already present", indent=indent)
        return

    if force and _lf_matrix_present(output_dir):
        print_step(
            0,
            f"{config_name}: forced regeneration; rebuilding LF matrices",
            indent=indent,
        )
    else:
        print_step(0, f"{config_name}: LF matrices missing; regenerating", indent=indent)
    print_step(0, f"{config_name}: loading processed agent outputs", indent=indent)
    records = build_lf_task_records(data_path)
    n = len(records)
    print_step(10, f"{config_name}: loaded {n} task records", indent=indent)

    print_step(
        15,
        f"{config_name}: applying {len(TASK_LFS)} task labeling functions",
        indent=indent,
    )
    task_matrix = np.zeros((n, len(TASK_LFS)), dtype=np.int8)
    report_every = max(1, n // 10)
    for i, record in enumerate(records):
        task_matrix[i] = [lf(record) for lf in TASK_LFS]
        count = i + 1
        if count < n and count % report_every == 0:
            step_idx = count // report_every
            pct = 20 + (step_idx - 1) * 5.5
            print_step(
                pct,
                f"{config_name}: task LF votes {count}/{n}",
                indent=indent,
            )

    print_step(75, f"{config_name}: applying meta labeling functions", indent=indent)
    meta_matrix = np.asarray(
        [[meta(task_matrix[i].tolist()) for meta in META_LFS] for i in range(n)],
        dtype=np.int8,
    )
    full_matrix = np.concatenate([task_matrix, meta_matrix], axis=1)

    print_step(90, f"{config_name}: writing LF matrix artifacts", indent=indent)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = np.asarray([record.task_id for record in records], dtype=int)
    task_columns = [lf.__name__ for lf in TASK_LFS]
    meta_columns = [lf.__name__ for lf in META_LFS]
    np.save(output_dir / "task_ids.npy", task_ids)
    np.save(output_dir / "labeling_matrix_task.npy", task_matrix)
    np.save(output_dir / "labeling_matrix_meta.npy", meta_matrix)
    np.save(output_dir / "labeling_matrix_full.npy", full_matrix)
    _write_lf_matrix_csv(
        output_dir / "labeling_matrix.csv",
        task_ids,
        task_columns + meta_columns,
        full_matrix,
    )
    manifest = {
        "config": config_name,
        "data_path": str(data_path.relative_to(ROOT_DIR)),
        "output_dir": str(output_dir.relative_to(ROOT_DIR)),
        "n_tasks": int(n),
        "task_columns": task_columns,
        "meta_columns": meta_columns,
        "task_shape": list(task_matrix.shape),
        "meta_shape": list(meta_matrix.shape),
        "full_shape": list(full_matrix.shape),
        "task_sha256": _sha256_bytes(task_matrix),
        "meta_sha256": _sha256_bytes(meta_matrix),
        "full_sha256": _sha256_bytes(full_matrix),
    }
    (output_dir / "labeling_matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print_step(100, f"{config_name}: LF matrix ready", indent=indent)


def ensure_weak_labels(
    config_name: str,
    data_path: Path,
    output_dir: Path,
    *,
    indent: str = "    ",
    force: bool = False,
) -> None:
    if _weak_labels_present(output_dir) and not force:
        print_step(100, f"{config_name}: weak labels already present", indent=indent)
        return

    if force:
        print_step(
            0,
            f"{config_name}: forced regeneration; rebuilding weak labels",
            indent=indent,
        )
    else:
        print_step(
            0,
            f"{config_name}: weak labels missing; preparing LF matrix",
            indent=indent,
        )
    # The outer caller is responsible for forcing LF-matrix regen. From here,
    # only ensure the matrix exists; don't rebuild it a second time.
    ensure_lf_matrix(config_name, data_path, output_dir, indent=indent)

    print_step(33, f"{config_name}: loading LF matrix for Snorkel", indent=indent)
    L_matrix = np.load(output_dir / "labeling_matrix_full.npy").astype(int)

    print_step(66, f"{config_name}: fitting Snorkel label model", indent=indent)
    label_model = LabelModel(cardinality=2, verbose=False)
    label_model.fit(L_train=L_matrix, n_epochs=500, seed=42, log_freq=100)
    snorkel_probs = label_model.predict_proba(L=L_matrix)
    snorkel_labels = label_model.predict(L=L_matrix, tie_break_policy="abstain")

    np.save(output_dir / "snorkel_labels.npy", snorkel_labels.astype(np.int8))
    np.save(output_dir / "snorkel_probs.npy", snorkel_probs)

    task_ids = np.load(output_dir / "task_ids.npy")
    inc_idx = np.where(snorkel_labels == 0)[0]
    cor_idx = np.where(snorkel_labels == 1)[0]
    top_inc = inc_idx[np.argsort(-snorkel_probs[inc_idx, 0])[:250]]
    top_cor = cor_idx[np.argsort(-snorkel_probs[cor_idx, 1])[:250]]
    hc_idx = np.sort(np.concatenate([top_inc, top_cor]))
    hc_rows = [
        {
            "task_id": int(task_ids[i]),
            "snorkel_label": int(snorkel_labels[i]),
            "snorkel_prob_incorrect": float(snorkel_probs[i, 0]),
            "snorkel_prob_correct": float(snorkel_probs[i, 1]),
        }
        for i in hc_idx
    ]
    (output_dir / "snorkel_high_conf_labels.json").write_text(
        json.dumps(hc_rows, indent=2) + "\n"
    )

    print_step(100, f"{config_name}: weak-label artifacts ready", indent=indent)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_json_rows(path: Path) -> list[dict[str, Any]]:
    return load_json(path)


def load_gt_rows(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def load_gt_map(path: Path) -> dict[int, int]:
    return {
        int(row["query_id"]): int(row["label"])
        for row in load_gt_rows(path)
    }


def build_processed_map(path: Path) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for entry in load_json_rows(path):
        out[int(entry["task_id"])] = {
            "gold_exec": _rebuild_exec(entry.get("gold_result", {})),
            "agents": {
                agent_name: _rebuild_exec(agent_data.get("result", {}))
                if isinstance(agent_data, dict) else None
                for agent_name, agent_data in entry.get("agents", {}).items()
            },
        }
    return out


def execution_is_valid(exec_result) -> bool:
    return exec_result is not None and exec_result.error is None


def compute_hc_subset(labels: np.ndarray, probs: np.ndarray) -> np.ndarray:
    inc_idx = np.where(labels == 0)[0]
    cor_idx = np.where(labels == 1)[0]
    top_inc = inc_idx[np.argsort(-probs[inc_idx, 0])[:250]]
    top_cor = cor_idx[np.argsort(-probs[cor_idx, 1])[:250]]
    return np.sort(np.concatenate([top_inc, top_cor]))


def fit_snorkel_from_lf_matrix(lf_matrix_path: Path) -> dict[str, Any]:
    labels = np.load(lf_matrix_path / "snorkel_labels.npy")
    probs = np.load(lf_matrix_path / "snorkel_probs.npy")
    task_ids_path = lf_matrix_path / "task_ids.npy"
    task_ids = np.load(task_ids_path) if task_ids_path.exists() else None
    hc_idx = compute_hc_subset(labels, probs)
    return {
        "labels": labels,
        "probs": probs,
        "task_ids": task_ids,
        "hc_idx": hc_idx,
    }


def extract_sql_features(record) -> dict[str, float]:
    features = _parse_sql_features(record.gold_sql)
    features["gold_cardinality"] = (
        float(record.gold_exec.row_count)
        if record.gold_exec and not record.gold_exec.error
        else 0.0
    )
    features["gold_error"] = 1.0 if (record.gold_exec is None or record.gold_exec.error) else 0.0
    return features


def build_sql_feature_matrix(records) -> pd.DataFrame:
    rows = [extract_sql_features(record) for record in records]
    df = pd.DataFrame(rows).fillna(0.0)
    df = df[sorted(df.columns)]
    if list(df.columns) != sorted(SQL_FEATURES):
        raise ValueError(f"Unexpected SQL feature columns: {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# Decision plane / stacking with progress printouts
# ---------------------------------------------------------------------------

def run_decision_plane_config(
    config_name: str,
    config: dict[str, Any],
    *,
    emit_progress: bool = True,
    indent: str = "    ",
) -> dict[str, Any]:
    label = f"decision_plane/{config_name}"

    if emit_progress:
        print_step(16, f"{label}: loading weak labels and task records", indent=indent)
    snorkel = fit_snorkel_from_lf_matrix(config["lf_matrix_path"])
    records = build_task_records(load_data(config["data_path"]))
    task_ids = np.array([record.task_id for record in records], dtype=int)
    if len(task_ids) != len(snorkel["labels"]):
        raise ValueError(f"Task/label length drift detected for {config_name}")
    if snorkel["task_ids"] is not None and not np.array_equal(task_ids, snorkel["task_ids"]):
        raise ValueError(f"Task ordering drift detected for {config_name}")
    task_id_to_idx = {int(task_id): idx for idx, task_id in enumerate(task_ids)}

    if emit_progress:
        print_step(33, f"{label}: building SQL features and agent verdicts", indent=indent)
    feat_X = build_sql_feature_matrix(records)
    verdicts = compute_verdicts(records, config["agents"])
    valid_masks = compute_validity_masks(records, config["agents"])

    hc_idx = snorkel["hc_idx"]
    hc_labels = snorkel["labels"][hc_idx]
    hc_feat_X = feat_X.iloc[hc_idx].reset_index(drop=True)
    hc_verdicts = {agent: verdicts[agent][hc_idx] for agent in config["agents"]}
    hc_valid_masks = {agent: valid_masks[agent][hc_idx] for agent in config["agents"]}

    if emit_progress:
        print_step(
            50,
            f"{label}: 5-fold CV on {len(hc_idx)} high-confidence tasks",
            indent=indent,
        )

    fold_indent = indent + "  "

    def _fold_cb(fold_idx: int, n_folds: int, mt: str) -> None:
        # Per-fold, per-model granular progress.
        step_idx = fold_idx * 3 + ["cart2", "cart4", "xgboost"].index(mt) + 1
        total_steps = n_folds * 3
        p = int(step_idx / total_steps * 100)
        print_step(
            p,
            f"{label}: CV fold {fold_idx + 1}/{n_folds} finished for {mt}",
            indent=fold_indent,
        )

    cv_results = cv_select_strategy(
        hc_feat_X,
        hc_verdicts,
        hc_labels,
        config["agents"],
        hc_valid_masks=hc_valid_masks,
        progress_callback=_fold_cb if emit_progress else None,
    )
    best = cv_results[0]

    if emit_progress:
        print_step(
            66,
            f"{label}: training final {best['model_type']} + {best['strategy']}",
            indent=indent,
        )

    final_clfs = train_reliability_models(
        hc_feat_X,
        hc_verdicts,
        hc_labels,
        config["agents"],
        best["model_type"],
        valid_masks=hc_valid_masks,
    )
    fallback_model = train_fallback_task_model(hc_feat_X, hc_labels, best["model_type"])
    meta_clf = None
    fusion_clf = None

    rel_hc = get_reliability_probs(final_clfs, hc_feat_X, config["agents"])
    if best["strategy"] == "stacked_meta":
        meta_clf = train_meta_clf(
            rel_hc,
            hc_verdicts,
            hc_labels,
            config["agents"],
            valid_masks=hc_valid_masks,
        )
    elif best["strategy"] == "score_fusion":
        hc_scores, _ = weighted_vote_scores(
            rel_hc,
            hc_verdicts,
            config["agents"],
            valid_masks=None,
        )
        hc_fallback_probs = predict_fallback_task_probs(fallback_model, hc_feat_X)
        fusion_clf = train_score_fusion_clf(hc_scores, hc_fallback_probs, hc_labels)

    if emit_progress:
        print_step(83, f"{label}: scoring full benchmark and GT sets", indent=indent)

    rel_all = get_reliability_probs(final_clfs, feat_X, config["agents"])
    fallback_all = predict_fallback_task_labels(fallback_model, feat_X)
    fallback_prob_all = predict_fallback_task_probs(fallback_model, feat_X)
    valid_for_strategy = None if best["strategy"] == "score_fusion" else valid_masks
    preds_all = apply_strategy(
        best["strategy"],
        rel_all,
        verdicts,
        config["agents"],
        meta_clf=meta_clf,
        fusion_clf=fusion_clf,
        valid_masks=valid_for_strategy,
        fallback_preds=fallback_all,
        fallback_probs=fallback_prob_all,
        decision_threshold=WEIGHTED_VOTE_THRESHOLD,
    )

    gt_results: dict[str, dict[str, Any]] = {}
    gt_paths = config.get("gt_paths", GT_PATHS)
    if gt_paths is not None:
        for gt_name, gt_path in gt_paths.items():
            gt_map = load_gt_map(gt_path)
            missing = set(gt_map) - set(task_id_to_idx)
            if missing:
                raise ValueError(f"Missing evaluation task IDs for {config_name}: {sorted(missing)}")
            valid_task_ids = list(gt_map)
            indices = np.array([task_id_to_idx[task_id] for task_id in valid_task_ids], dtype=int)
            y_true = np.array([gt_map[task_id] for task_id in valid_task_ids], dtype=int)

            feat_sub = feat_X.iloc[indices].reset_index(drop=True)
            verd_sub = {agent: verdicts[agent][indices] for agent in config["agents"]}
            valid_sub = {agent: valid_masks[agent][indices] for agent in config["agents"]}
            rel_sub = get_reliability_probs(final_clfs, feat_sub, config["agents"])
            fallback_sub = predict_fallback_task_labels(fallback_model, feat_sub)
            fallback_prob_sub = predict_fallback_task_probs(fallback_model, feat_sub)
            valid_strategy_sub = None if best["strategy"] == "score_fusion" else valid_sub
            preds_sub = apply_strategy(
                best["strategy"],
                rel_sub,
                verd_sub,
                config["agents"],
                meta_clf=meta_clf,
                fusion_clf=fusion_clf,
                valid_masks=valid_strategy_sub,
                fallback_preds=fallback_sub,
                fallback_probs=fallback_prob_sub,
                decision_threshold=WEIGHTED_VOTE_THRESHOLD,
            )
            gt_results[gt_name] = {
                "n_errors": int((y_true == 0).sum()),
                "n_correct": int((y_true == 1).sum()),
                **compute_metrics(y_true, preds_sub),
            }

    if emit_progress:
        print_step(
            100,
            f"{label}: complete ({best['model_type']} + {best['strategy']})",
            indent=indent,
        )

    return {
        "config": config_name,
        "benchmark_name": config["benchmark_name"],
        "data_path": str(config["data_path"].relative_to(ROOT_DIR)),
        "weak_supervision_dir": str(config["lf_matrix_path"].relative_to(ROOT_DIR)),
        "n_tasks": int(len(records)),
        "n_features": int(feat_X.shape[1]),
        "n_hc": int(len(hc_idx)),
        "best_model_type": best["model_type"],
        "best_strategy": best["strategy"],
        "best_cv_f1": float(best["cv_f1"]),
        "best_cv_f1_std": float(best["cv_f1_std"]),
        "gt_results": gt_results,
        "predicted_error_count": int((preds_all == 0).sum()),
        "predicted_correct_count": int((preds_all == 1).sum()),
        "predicted_error_rate": float((preds_all == 0).mean()),
        "predictions": [{"task_id": int(i), "predicted_label": int(p)}
                        for i, p in zip(task_ids, preds_all)],
    }


def run_stacking_config(
    config_name: str,
    config: dict[str, Any],
    *,
    emit_progress: bool = True,
    indent: str = "    ",
) -> dict[str, Any]:
    label = f"stacking/{config_name}"
    if emit_progress:
        print_step(25, f"{label}: loading weak labels and verdicts", indent=indent)

    snorkel = fit_snorkel_from_lf_matrix(config["lf_matrix_path"])
    records = build_task_records(load_data(config["data_path"]))
    task_ids = np.array([record.task_id for record in records], dtype=int)
    if not np.array_equal(task_ids, snorkel["task_ids"]):
        raise ValueError(f"Task ordering drift detected for {config_name}")
    verdict_arrays = compute_verdicts(records, config["agents"])
    verdict_map = {
        agent: {
            int(task_ids[idx]): int(verdict_arrays[agent][idx])
            for idx in range(len(task_ids))
        }
        for agent in config["agents"]
    }

    if emit_progress:
        print_step(50, f"{label}: building high-confidence pattern matrix", indent=indent)

    hc_idx = snorkel["hc_idx"]
    hc_labels = snorkel["labels"][hc_idx]
    hc_task_ids = task_ids[hc_idx]
    hc_matrix = build_verdict_matrix(hc_task_ids, verdict_map, config["agents"])
    hc_patterns = verdict_to_input_idx(hc_matrix)

    if emit_progress:
        print_step(75, f"{label}: exhaustive strategy search", indent=indent)

    top_ids, top_f1s, top_ps, top_rs = exhaustive_search(
        hc_labels,
        hc_patterns,
        len(config["agents"]),
        top_k=10,
    )
    best_id = int(top_ids[0])

    gt_results: dict[str, dict[str, Any]] = {}
    for gt_name, gt_path in GT_PATHS.items():
        gt_df = pd.read_csv(gt_path)
        gt_df = gt_df[gt_df["query_id"].isin(task_ids)]
        gt_task_ids = gt_df["query_id"].to_numpy(dtype=int)
        gt_labels = gt_df["label"].to_numpy(dtype=int)
        gt_matrix = build_verdict_matrix(gt_task_ids, verdict_map, config["agents"])
        gt_patterns = verdict_to_input_idx(gt_matrix)
        gt_preds = apply_stacking_strategy(best_id, gt_patterns, len(config["agents"]))
        gt_results[gt_name] = {
            "n_errors": int((gt_labels == 0).sum()),
            "n_correct": int((gt_labels == 1).sum()),
            **compute_stacking_metrics(gt_labels, gt_preds),
        }

    if emit_progress:
        print_step(100, f"{label}: complete", indent=indent)

    return {
        "config": config_name,
        "data_path": str(config["data_path"].relative_to(ROOT_DIR)),
        "weak_supervision_dir": str(config["lf_matrix_path"].relative_to(ROOT_DIR)),
        "n_hc": int(len(hc_idx)),
        "best_strategy_id": best_id,
        "best_logic": format(best_id, f"0{2 ** len(config['agents'])}b"),
        "best_hc_f1": float(top_f1s[0]),
        "best_hc_precision": float(top_ps[0]),
        "best_hc_recall": float(top_rs[0]),
        "gt_results": gt_results,
    }


# ---------------------------------------------------------------------------
# Single-agent baselines and data-source manifest
# ---------------------------------------------------------------------------

def compute_single_agent_metrics(
    gt_rows: list[dict[str, Any]],
    processed_map: dict[int, dict[str, Any]],
    agent_key: str,
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    n_valid_rows = 0

    for row in gt_rows:
        query_id = int(row["query_id"])
        label = int(row["label"])
        entry = processed_map[query_id]
        gold_exec = entry["gold_exec"]
        agent_exec = entry["agents"].get(agent_key)

        if execution_is_valid(agent_exec) and execution_is_valid(gold_exec):
            n_valid_rows += 1
            pred = 1 if results_agree(agent_exec, gold_exec) else 0
        else:
            pred = 1 - label

        y_true.append(label)
        y_pred.append(pred)

    return {
        "metrics": compute_metrics(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)),
        "n_valid_exec_rows": int(n_valid_rows),
    }


def compute_majority_vote_metrics(
    gt_rows: list[dict[str, Any]],
    processed_map: dict[int, dict[str, Any]],
    agent_keys: list[str],
) -> dict[str, Any]:
    y_true: list[int] = []
    y_pred: list[int] = []
    n_rows_with_vote = 0
    n_zero_valid_rows = 0

    for row in gt_rows:
        query_id = int(row["query_id"])
        label = int(row["label"])
        entry = processed_map[query_id]
        mv_pred, _ = majority_vote_prediction(entry["agents"], entry["gold_exec"], agent_keys)
        if mv_pred.pred is None:
            n_zero_valid_rows += 1
            pred = 1 - label
        else:
            n_rows_with_vote += 1
            pred = int(mv_pred.pred)

        y_true.append(label)
        y_pred.append(pred)

    return {
        "metrics": compute_metrics(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)),
        "n_rows_with_vote": int(n_rows_with_vote),
        "n_zero_valid_rows": int(n_zero_valid_rows),
    }


def compute_baselines(include_families: list[str]) -> dict[str, Any]:
    frontier_agents = list(CURRENT_FRONTIER_AGENTS)
    frontier_three = ["gpt5", "gpt4o", "gemini"]
    legacy_agents = ["gpt4omini", "gpt4o_old", "haiku3", "haiku"]
    legacy_three = ["gpt4omini", "haiku3", "haiku"]

    maps: dict[str, dict[int, dict[str, Any]]] = {}
    if "frontier" in include_families:
        maps["frontier"] = build_processed_map(PROCESSED_SOURCES["frontier"])
    if "legacy" in include_families:
        maps["legacy"] = build_processed_map(PROCESSED_SOURCES["legacy"])

    family_agents = {
        "frontier": (frontier_agents, frontier_three),
        "legacy": (legacy_agents, legacy_three),
    }

    results: dict[str, Any] = {}
    for gt_name, gt_path in GT_PATHS.items():
        gt_rows = load_gt_rows(gt_path)
        results[gt_name] = {
            fam: {"agents": {}, "majority_vote": {}, "majority_vote_3agent": {}}
            for fam in include_families
        }
        for family in include_families:
            agents_full, agents_three = family_agents[family]
            fam_map = maps[family]
            for agent in agents_full:
                results[gt_name][family]["agents"][agent] = (
                    compute_single_agent_metrics(gt_rows, fam_map, agent)
                )
            results[gt_name][family]["majority_vote"] = compute_majority_vote_metrics(
                gt_rows, fam_map, agents_full
            )
            results[gt_name][family]["majority_vote_3agent"] = compute_majority_vote_metrics(
                gt_rows, fam_map, agents_three
            )
    return results


def build_data_sources_manifest(
    selected_configs: list[str],
    *,
    include_all_queries: bool,
) -> dict[str, Any]:
    query_sources = {}
    if include_all_queries:
        for name, path in QUERY_SOURCES.items():
            query_sources[name] = {
                "path": str(path.relative_to(ROOT_DIR)),
                "n_rows": int(len(load_json_rows(path))),
            }

    processed_sources = {}
    for name in selected_configs:
        path = PROCESSED_SOURCES[name]
        records = load_json_rows(path)
        agent_keys = sorted({agent for record in records for agent in record['agents']})
        if set(agent_keys) != set(CONFIGS[name]['agents']):
            raise ValueError(f"Unexpected agent identities in {path}: {agent_keys}")
        processed_sources[name] = {
            "path": str(path.relative_to(ROOT_DIR)),
            "n_rows": len(records),
            "agents": agent_keys,
        }

    gt_sources = {}
    selected_gt = {}
    for config_name in selected_configs:
        selected_gt.update(CONFIGS[config_name].get('gt_paths', GT_PATHS))
    for name, path in selected_gt.items():
        gt_df = pd.read_csv(path)
        gt_sources[name] = {
            "path": str(path.relative_to(ROOT_DIR)),
            "n_rows": int(len(gt_df)),
            "n_errors": int((gt_df["label"] == 0).sum()),
            "n_correct": int((gt_df["label"] == 1).sum()),
        }

    weak_supervision = {}
    for name in selected_configs:
        path = LF_MATRICES[name]
        labels_path = path / "snorkel_labels.npy"
        if not labels_path.exists():
            continue
        labels = np.load(labels_path)
        weak_supervision[name] = {
            "path": str(path.relative_to(ROOT_DIR)),
            "n_rows": int(labels.shape[0]),
        }

    return {
        "query_sources": query_sources,
        "processed_agent_sources": processed_sources,
        "ground_truth_sets": gt_sources,
        "weak_supervision": weak_supervision,
    }


def build_frontier_detection_rates(payload: dict[str, Any]) -> dict[str, Any]:
    rates: dict[str, Any] = {}
    dp = payload.get("decision_plane", {})
    ffb = payload.get("frontier_full_benchmarks", {})
    if "frontier" in dp:
        rates["sqld_dev"] = dp["frontier"]
    if "november_bird" in ffb:
        rates["november_bird"] = ffb["november_bird"]
    if "spider_dev" in ffb:
        rates["spider_dev"] = ffb["spider_dev"]
    return {
        name: {
            "benchmark_name": row["benchmark_name"],
            "data_path": row["data_path"],
            "n_tasks": row["n_tasks"],
            "predicted_error_count": row["predicted_error_count"],
            "predicted_correct_count": row["predicted_correct_count"],
            "predicted_error_rate": row["predicted_error_rate"],
        }
        for name, row in rates.items()
    }


def build_calibrated_audit(payload: dict[str, Any]) -> dict[str, Any]:
    """Calibrate fixed predictions using each benchmark's random 200-task sample."""
    rows = {}
    runs = list(payload.get('decision_plane', {}).values()) + list(payload.get('frontier_full_benchmarks', {}).values())
    samples = {'frontier': 'GT_RS_200', 'frontier_november': 'GT_RS_200_nov_bird',
               'frontier_spider': 'GT_RS_200_spider'}
    for run in runs:
        if run['config'] not in samples:
            continue
        m = run['gt_results'][samples[run['config']]]
        n_error, n_correct = m['tp'] + m['fn'], m['fp'] + m['tn']
        tpr, fpr = m['tp'] / n_error, m['fp'] / n_correct
        rate, n = run['predicted_error_rate'], run['n_tasks']
        if tpr <= fpr:
            raise ValueError('Calibration requires TPR > FPR')
        estimate = (rate - fpr) / (tpr - fpr)
        variance = (rate * (1 - rate) / n + estimate**2 * tpr * (1 - tpr) / n_error
                    + (1 - estimate)**2 * fpr * (1 - fpr) / n_correct) / (tpr - fpr)**2
        margin = 1.96 * math.sqrt(variance)
        rows[run['benchmark_name']] = {
            'sample': samples[run['config']], 'sample_f1': m['f1'],
            'tpr': tpr, 'fpr': fpr, 'detected_rate': rate,
            'calibrated_error_rate': estimate,
            'ci95': [max(0.0, estimate - margin), min(1.0, estimate + margin)],
            'method': 'Rogan-Gladen calibration; delta-method 95% interval',
        }
    return rows


def build_metrics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    dp_section: dict[str, Any] = {}
    for config_name in ("frontier", "legacy"):
        if config_name not in payload.get("decision_plane", {}):
            continue
        gt = payload["decision_plane"][config_name]["gt_results"]
        dp_section[config_name] = {
            name: gt[name] for name in gt
        }

    stacking_section = {
        config_name: payload["stacking"][config_name]["gt_results"]
        for config_name in ("frontier", "legacy")
        if config_name in payload.get("stacking", {})
    }

    baselines_section: dict[str, Any] = {}
    for gt_name in GT_PATHS.keys():
        gt_row: dict[str, Any] = {}
        for family in ("frontier", "legacy"):
            if gt_name in payload.get("baselines", {}) and family in payload["baselines"][gt_name]:
                gt_row[f"{family}_majority_vote"] = (
                    payload["baselines"][gt_name][family]["majority_vote"]["metrics"]
                )
        if gt_row:
            baselines_section[gt_name] = gt_row

    return {
        "decision_plane": dp_section,
        "stacking": stacking_section,
        "baselines": baselines_section,
        "frontier_detection_rates": payload.get("frontier_detection_rates", {}),
        "calibrated_audit": payload.get("calibrated_audit", {}),
    }


def write_markdown_report(payload: dict[str, Any], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# SALUS Pipeline Results")
    lines.append("")
    lines.append("This report summarizes the metrics generated from the SALUS pipeline.")
    lines.append("")

    lines.append("## Data Sources")
    lines.append("")
    lines.append("| Category | Name | Path | Rows |")
    lines.append("|:---------|:-----|:-----|-----:|")
    for name, row in payload["data_sources"]["query_sources"].items():
        lines.append(f"| query | {name} | `{row['path']}` | {row['n_rows']} |")
    for name, row in payload["data_sources"]["processed_agent_sources"].items():
        lines.append(f"| processed | {name} | `{row['path']}` | {row['n_rows']} |")
    for name, row in payload["data_sources"]["ground_truth_sets"].items():
        lines.append(f"| ground_truth | {name} | `{row['path']}` | {row['n_rows']} |")
    for name, row in payload["data_sources"]["weak_supervision"].items():
        lines.append(f"| weak_supervision | {name} | `{row['path']}` | {row['n_rows']} |")
    lines.append("")

    dp_entries = {**payload.get("decision_plane", {}), **{
        row['config']: row for row in payload.get('frontier_full_benchmarks', {}).values()
    }}
    if dp_entries:
        lines.append("## Decision Plane")
        lines.append("")
        lines.append("| Config | Selected | GT Set | F1 | Precision | Recall | TP | FP | FN | TN |")
        lines.append("|:-------|:---------|:-------|---:|----------:|-------:|---:|---:|---:|---:|")
        for config_name, run in dp_entries.items():
            selected = f"{run['best_model_type']} + {run['best_strategy']}"
            for gt_name, gt in run["gt_results"].items():
                lines.append(
                    f"| {config_name} | `{selected}` | {gt_name} | {fmt(gt['f1'])} | {fmt(gt['p'])} | "
                    f"{fmt(gt['r'])} | {gt['tp']} | {gt['fp']} | {gt['fn']} | {gt['tn']} |"
                )
        lines.append("")

    detection_rates = payload.get("frontier_detection_rates", {})
    if detection_rates:
        lines.append("## Frontier Full-Benchmark Detected Error Rates")
        lines.append("")
        lines.append("| Benchmark | Source | Tasks | Predicted Error | Predicted Correct | Detected Error Rate |")
        lines.append("|:----------|:-------|------:|----------------:|------------------:|--------------------:|")
        for benchmark_name, row in detection_rates.items():
            lines.append(
                f"| {benchmark_name} | `{row['data_path']}` | {row['n_tasks']} | "
                f"{row['predicted_error_count']} | {row['predicted_correct_count']} | "
                f"{fmt(row['predicted_error_rate'])} |"
            )
        lines.append("")

    baselines = payload.get("baselines", {})
    if baselines:
        lines.append("## Baselines")
        lines.append("")
        lines.append("| GT Set | Family | Method | F1 | Precision | Recall | TP | FP | FN | TN |")
        lines.append("|:-------|:-------|:-------|---:|----------:|-------:|---:|---:|---:|---:|")
        for gt_name, gt_baselines in baselines.items():
            for family, fam_baselines in gt_baselines.items():
                for agent_name, row in fam_baselines["agents"].items():
                    m = row["metrics"]
                    lines.append(
                        f"| {gt_name} | {family} | {agent_name} | {fmt(m['f1'])} | {fmt(m['p'])} | {fmt(m['r'])} | "
                        f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
                    )
                for method_name in ("majority_vote", "majority_vote_3agent"):
                    m = fam_baselines[method_name]["metrics"]
                    lines.append(
                        f"| {gt_name} | {family} | {method_name} | {fmt(m['f1'])} | {fmt(m['p'])} | {fmt(m['r'])} | "
                        f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
                    )
            for family, stack_row in payload.get("stacking", {}).items():
                if gt_name in stack_row["gt_results"]:
                    m = stack_row["gt_results"][gt_name]
                    lines.append(
                        f"| {gt_name} | {family} | stacking_hc | {fmt(m['f1'])} | {fmt(m['p'])} | {fmt(m['r'])} | "
                        f"{m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |"
                    )
        lines.append("")

    if payload.get('calibrated_audit'):
        lines.extend(['## Calibrated benchmark audit', '',
                      'Each estimate uses the manually labeled 200-task sample of the corresponding benchmark version, only after SALUS predictions are fixed.', '',
                      '| Benchmark | Sample F1 | Raw detected rate | Calibrated error rate | 95% CI |',
                      '|:--|--:|--:|--:|:--|'])
        for name, row in payload['calibrated_audit'].items():
            lo, hi = row['ci95']
            lines.append(f"| {name} | {row['sample_f1']:.4f} | {row['detected_rate']:.2%} | {row['calibrated_error_rate']:.2%} | [{lo:.2%}, {hi:.2%}] |")
        lines.append('')

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI / main orchestration
# ---------------------------------------------------------------------------

def resolve_selected_configs(
    processed_query: str | None,
    config_flag: str | None,
) -> list[str]:
    if processed_query is None and config_flag is None:
        return FULL_CONFIGS.copy()
    selected: list[str] = []
    if config_flag is not None:
        if config_flag not in CONFIGS:
            raise SystemExit(
                f"Unknown --config '{config_flag}'. Choose from: {sorted(CONFIGS)}"
            )
        selected.append(config_flag)
    if processed_query is not None:
        filename = Path(processed_query).name
        if filename not in PROCESSED_FILENAME_TO_CONFIG:
            raise SystemExit(
                f"Unknown --processed-query '{processed_query}'. "
                f"Known files: {sorted(PROCESSED_FILENAME_TO_CONFIG)}"
            )
        name = PROCESSED_FILENAME_TO_CONFIG[filename]
        if name not in selected:
            selected.append(name)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SALUS pipeline")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "reports",
        help="Directory for the generated JSON and Markdown report.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Run only this config (e.g. 'frontier', 'legacy').",
    )
    parser.add_argument(
        "--processed-query",
        type=str,
        default=None,
        help="Run the config whose processed-query filename matches "
             "(e.g. 'bird_dev_frontier.json.gz').",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Force-regenerate the labeling matrix and Snorkel weak labels "
             "for each selected config, even if they already exist. Combine "
             "with --config / --processed-query to target a specific agent "
             "source.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_configs = resolve_selected_configs(args.processed_query, args.config)


    for name in selected_configs:
        data_path = CONFIGS[name]["data_path"]
        for required in [data_path, *CONFIGS[name].get('gt_paths', GT_PATHS).values()]:
            if not required.is_file():
                raise FileNotFoundError(f"Required input missing for {name}: {required}")

    run_full = set(selected_configs) == set(FULL_CONFIGS)

    dp_configs = [c for c in selected_configs if c in ("frontier", "legacy")]
    ffb_configs = [c for c in selected_configs if c in ("frontier_november", "frontier_spider")]
    stacking_configs = dp_configs.copy()
    baseline_families = [
        fam for fam in FULL_GT_FAMILIES if fam in selected_configs
    ]

    # Stage 1: Prepare inputs for each selected config.
    stage1_label = (
        f"Regenerate inputs for {', '.join(selected_configs)}"
        if args.regenerate
        else f"Prepare inputs for {', '.join(selected_configs)}"
    )
    print_stage(1, FULL_STAGE_COUNT, stage1_label)
    for name in selected_configs:
        cfg = CONFIGS[name]
        ensure_lf_matrix(
            name, cfg["data_path"], cfg["lf_matrix_path"], force=args.regenerate
        )
        ensure_weak_labels(
            name, cfg["data_path"], cfg["lf_matrix_path"], force=args.regenerate
        )

    # Stage 2: Decision plane.
    print_stage(
        2,
        FULL_STAGE_COUNT,
        f"Decision plane for {', '.join(dp_configs + ffb_configs) or 'selected configs'}",
    )
    decision_plane_results = {
        name: run_decision_plane_config(name, CONFIGS[name]) for name in dp_configs
    }
    frontier_full_benchmarks: dict[str, Any] = {}
    for name in ffb_configs:
        result = run_decision_plane_config(name, CONFIGS[name])
        benchmark = CONFIGS[name]["benchmark_name"]
        frontier_full_benchmarks[benchmark] = result

    # Stage 3: Stacking baseline.
    print_stage(
        3,
        FULL_STAGE_COUNT,
        f"Stacking baseline for {', '.join(stacking_configs) or 'selected configs'}",
    )
    stacking_results = {
        name: run_stacking_config(name, CONFIGS[name]) for name in stacking_configs
    }

    # Stage 4: Baselines.
    print_stage(4, FULL_STAGE_COUNT, "Compute single-agent and majority-vote baselines")
    baselines = compute_baselines(baseline_families) if baseline_families else {}

    # Stage 5: Manifest.
    print_stage(5, FULL_STAGE_COUNT, "Build data-source manifest")
    data_sources = build_data_sources_manifest(
        selected_configs, include_all_queries=run_full
    )

    # Stage 6: Reports.
    print_stage(6, FULL_STAGE_COUNT, "Write reports")
    payload: dict[str, Any] = {
        "data_sources": data_sources,
        "decision_plane": decision_plane_results,
        "frontier_full_benchmarks": frontier_full_benchmarks,
        "stacking": stacking_results,
        "baselines": baselines,
    }
    payload["frontier_detection_rates"] = build_frontier_detection_rates(payload)
    payload['calibrated_audit'] = build_calibrated_audit(payload)
    metrics_summary = build_metrics_summary(payload)

    json_path = args.output_dir / "pipeline_report.json"
    md_path = args.output_dir / "pipeline_report.md"
    metrics_path = args.output_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown_report(payload, md_path)
    metrics_path.write_text(json.dumps(metrics_summary, indent=2) + "\n")
    predictions_dir = args.output_dir / 'predictions'
    predictions_dir.mkdir(exist_ok=True)
    for run in [*decision_plane_results.values(), *frontier_full_benchmarks.values()]:
        with (predictions_dir / f"{run['config']}.csv").open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['task_id', 'predicted_label'])
            writer.writeheader()
            writer.writerows(run['predictions'])

    print_done()

    summary: dict[str, Any] = {
        "selected_configs": selected_configs,
        "json_report": str(json_path),
        "markdown_report": str(md_path),
        "metrics": str(metrics_path),
    }
    for name, row in decision_plane_results.items():
        gt = row.get("gt_results", {})
        if "BIRD_CLEAN_xs" in gt:
            summary[f"{name}_bird_clean_xs_f1"] = gt["BIRD_CLEAN_xs"]["f1"]
        if "GT_RS_200" in gt:
            summary[f"{name}_gt_rs_200_f1"] = gt["GT_RS_200"]["f1"]
    for bench_name, bench_row in payload["frontier_detection_rates"].items():
        summary[f"{bench_name}_error_rate"] = bench_row["predicted_error_rate"]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
