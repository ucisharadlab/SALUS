"""SALUS reliability models and aggregation helpers.

The public driver is run_pipeline.py. Model selection uses high-confidence
weak labels; manual correctness labels are used only for evaluation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sqlsalus_core.data_io import load_json
from sqlsalus_core.execution.comparator import results_agree
from sqlsalus_core.labeling.base import TaskRecord, _rebuild_exec
from sqlsalus_core.labeling.features import _parse_sql_features

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_TYPES = ["cart2", "cart4", "xgboost"]
WEIGHTED_VOTE_THRESHOLD = 0.5
STRATEGY_TIE_PRIORITY = {
    # In exact CV ties, prefer richer learned fusion over fixed-threshold voting.
    "score_fusion": 0,
    "stacked_meta": 1,
    "weighted_voting": 2,
    "best_agent": 3,
    "top2_voting": 4,
    "top3_voting": 5,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(path: Path) -> list[dict]:
    return load_json(path)


def build_task_records(data: list[dict]) -> list[TaskRecord]:
    records = []
    for entry in data:
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
        for aname, adata in agents_data.items():
            if isinstance(adata, dict):
                rec.agent_execs[aname] = _rebuild_exec(adata.get("result", {}))
                rec.agent_sqls[aname] = adata.get("sql", "")
        records.append(rec)
    return records


def load_gt(path: Path) -> dict[int, int]:
    """Load GT CSV → {query_id: label} where label 0=error, 1=correct."""
    df = pd.read_csv(path)
    return {int(r["query_id"]): int(r["label"]) for _, r in df.iterrows()}


# ---------------------------------------------------------------------------
# SQL feature extraction shared across agents
# ---------------------------------------------------------------------------

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


def extract_sql_features(rec: TaskRecord) -> dict[str, float]:
    """Extract shared SQL structural features from gold SQL."""
    feats = _parse_sql_features(rec.gold_sql)
    feats["gold_cardinality"] = (
        float(rec.gold_exec.row_count)
        if rec.gold_exec and not rec.gold_exec.error
        else 0.0
    )
    feats["gold_error"] = 1.0 if (rec.gold_exec is None or rec.gold_exec.error) else 0.0
    return feats


def build_feature_matrix(records: list[TaskRecord]) -> pd.DataFrame:
    rows = [extract_sql_features(r) for r in records]
    df = pd.DataFrame(rows).fillna(0.0)
    df = df[sorted(df.columns)]
    expected = sorted(SQL_FEATURES)
    actual = list(df.columns)
    if actual != expected:
        raise ValueError(f"Decision-plane feature drift detected. Expected {expected}, got {actual}")
    return df


# ---------------------------------------------------------------------------
# Agent verdicts and reliability labels
# ---------------------------------------------------------------------------

def compute_verdicts(records: list[TaskRecord], agents: list[str]) -> dict[str, np.ndarray]:
    """Binary verdict per agent per task: 1 = agrees with gold, 0 = disagrees."""
    n = len(records)
    verdicts = {a: np.zeros(n, dtype=np.int8) for a in agents}
    for i, rec in enumerate(records):
        for a in agents:
            aex = rec.agent_execs.get(a)
            if aex and rec.gold_exec and results_agree(aex, rec.gold_exec):
                verdicts[a][i] = 1
    return verdicts


def compute_validity_masks(records: list[TaskRecord], agents: list[str]) -> dict[str, np.ndarray]:
    """Per-agent validity mask: 1 when the agent execution result is usable."""
    n = len(records)
    masks = {a: np.zeros(n, dtype=np.int8) for a in agents}
    for i, rec in enumerate(records):
        for a in agents:
            aex = rec.agent_execs.get(a)
            if aex is not None and aex.error is None:
                masks[a][i] = 1
    return masks


def compute_reliability_labels(
    verdicts: dict[str, np.ndarray],
    snorkel_labels: np.ndarray,
    agent: str,
) -> np.ndarray:
    """y_rel = 1 if agent verdict aligns with Snorkel label (agent is reliable), else 0.

    Reliable means:
      - Agent agrees (verdict=1) AND Snorkel says CORRECT (label=1)
      - Agent disagrees (verdict=0) AND Snorkel says INCORRECT (label=0)
    """
    return (verdicts[agent] == snorkel_labels).astype(int)


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def make_model(model_type: str):
    if model_type == "cart2":
        return DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=42)
    if model_type == "cart4":
        return DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42)
    if model_type == "xgboost":
        return XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.5, min_child_weight=5,
            reg_alpha=1.0, reg_lambda=1.0, eval_metric="logloss",
            random_state=42, verbosity=0,
        )
    raise ValueError(f"Unknown model_type: {model_type}")


def train_reliability_models(
    feat_X: pd.DataFrame,
    verdicts: dict[str, np.ndarray],
    snorkel_labels: np.ndarray,
    agents: list[str],
    model_type: str,
    valid_masks: dict[str, np.ndarray] | None = None,
) -> dict[str, object]:
    """Train one reliability classifier per agent on the given feature set."""
    clfs = {}
    for a in agents:
        if valid_masks is None:
            train_mask = np.ones(len(feat_X), dtype=bool)
        else:
            train_mask = valid_masks[a].astype(bool)
        if not train_mask.any():
            clfs[a] = None
            continue
        y_rel = (verdicts[a][train_mask] == snorkel_labels[train_mask]).astype(int)
        if len(np.unique(y_rel)) < 2:
            clfs[a] = None
            continue
        clf = make_model(model_type)
        clf.fit(feat_X.iloc[train_mask].reset_index(drop=True), y_rel)
        clfs[a] = clf
    return clfs


def get_reliability_probs(
    clfs: dict[str, object],
    feat_X: pd.DataFrame,
    agents: list[str],
) -> dict[str, np.ndarray]:
    """Predict P(agent reliable) for each task using trained classifiers."""
    probs = {}
    for a in agents:
        clf = clfs.get(a)
        if clf is None:
            probs[a] = np.full(len(feat_X), 0.5)
        else:
            probs[a] = clf.predict_proba(feat_X)[:, 1]
    return probs


def train_fallback_task_model(
    feat_X: pd.DataFrame,
    labels: np.ndarray,
    model_type: str,
):
    """Train a task-level automatic fallback used when no agent executions are valid."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return int(unique[0])
    clf = make_model(model_type)
    clf.fit(feat_X, labels)
    return clf


def predict_fallback_task_labels(
    fallback_model,
    feat_X: pd.DataFrame,
) -> np.ndarray:
    """Predict fallback task labels from SQL features only."""
    if isinstance(fallback_model, (int, np.integer)):
        return np.full(len(feat_X), int(fallback_model), dtype=int)
    return fallback_model.predict(feat_X).astype(int)


def predict_fallback_task_probs(
    fallback_model,
    feat_X: pd.DataFrame,
) -> np.ndarray:
    """Predict P(task correct) from fallback task model."""
    if isinstance(fallback_model, (int, np.integer)):
        return np.full(len(feat_X), float(int(fallback_model)), dtype=float)
    if hasattr(fallback_model, "predict_proba"):
        probs = fallback_model.predict_proba(feat_X)
        if probs.shape[1] == 1:
            return np.full(len(feat_X), 0.5, dtype=float)
        return probs[:, 1].astype(float)
    # Conservative fallback if model lacks probabilities.
    return fallback_model.predict(feat_X).astype(float)


def train_score_fusion_clf(
    weighted_scores: np.ndarray,
    fallback_probs: np.ndarray,
    labels: np.ndarray,
):
    """Train a compact GT-agnostic fusion model on [weighted_score, fallback_prob]."""
    unique = np.unique(labels)
    if len(unique) < 2:
        return int(unique[0])
    X_fuse = np.column_stack([weighted_scores.astype(float), fallback_probs.astype(float)])
    clf = LogisticRegression(max_iter=200, class_weight="balanced", random_state=42)
    clf.fit(X_fuse, labels)
    return clf


# ---------------------------------------------------------------------------
# Ensemble strategies
# ---------------------------------------------------------------------------

def _build_meta_X(
    rel_probs: dict[str, np.ndarray],
    verdicts: dict[str, np.ndarray],
    agents: list[str],
    valid_masks: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Build meta-learner feature matrix: [rel_probs, verdicts, valids] per agent."""
    cols: dict = {}
    for a in agents:
        cols[f"rel_{a}"] = rel_probs[a]
        cols[f"verdict_{a}"] = verdicts[a].astype(float)
        if valid_masks is not None:
            cols[f"valid_{a}"] = valid_masks[a].astype(float)
    df = pd.DataFrame(cols)
    return df[sorted(df.columns)]


def train_meta_clf(
    rel_probs: dict[str, np.ndarray],
    verdicts: dict[str, np.ndarray],
    labels: np.ndarray,
    agents: list[str],
    valid_masks: dict[str, np.ndarray] | None = None,
):
    """Train XGBoost on reliability scores, verdicts, and optional validity masks."""
    meta_X = _build_meta_X(rel_probs, verdicts, agents, valid_masks)
    clf = XGBClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        random_state=42, verbosity=0, eval_metric="logloss",
    )
    clf.fit(meta_X, labels)
    return clf


def weighted_vote_scores(
    rel_probs: dict[str, np.ndarray],
    verdicts: dict[str, np.ndarray],
    agents: list[str],
    valid_masks: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted-vote score P(correct) and zero-valid mask for each task."""
    n = len(next(iter(verdicts.values())))
    scores = np.full(n, 0.5, dtype=float)
    zero_valid = np.zeros(n, dtype=bool)

    for i in range(n):
        valid_agents = [
            a for a in agents
            if valid_masks is None or bool(valid_masks[a][i])
        ]
        if not valid_agents:
            zero_valid[i] = True
            continue
        num = sum(rel_probs[a][i] * verdicts[a][i] for a in valid_agents)
        den = sum(rel_probs[a][i] for a in valid_agents)
        scores[i] = (num / den) if den > 0 else 0.5

    return scores, zero_valid


def apply_strategy(
    strategy: str,
    rel_probs: dict[str, np.ndarray],
    verdicts: dict[str, np.ndarray],
    agents: list[str],
    meta_clf=None,
    fusion_clf=None,
    valid_masks: dict[str, np.ndarray] | None = None,
    fallback_preds: np.ndarray | None = None,
    fallback_probs: np.ndarray | None = None,
    decision_threshold: float = 0.5,
) -> np.ndarray:
    """Apply one ensemble strategy and return binary predictions (0=INC, 1=COR)."""
    n = len(next(iter(verdicts.values())))

    if strategy == "weighted_voting":
        scores, zero_valid = weighted_vote_scores(
            rel_probs, verdicts, agents, valid_masks=valid_masks
        )
        preds = (scores >= float(decision_threshold)).astype(int)
        if fallback_preds is not None:
            preds[zero_valid] = fallback_preds[zero_valid]
        elif zero_valid.any():
            preds[zero_valid] = 1
        return preds

    if strategy == "best_agent":
        preds = np.zeros(n, dtype=int)
        for i in range(n):
            valid_agents = [
                a for a in agents
                if valid_masks is None or bool(valid_masks[a][i])
            ]
            if not valid_agents:
                preds[i] = int(fallback_preds[i]) if fallback_preds is not None else 1
                continue
            best = max(valid_agents, key=lambda a: rel_probs[a][i])
            preds[i] = verdicts[best][i]
        return preds

    if strategy in ("top2_voting", "top3_voting"):
        k = 2 if strategy == "top2_voting" else 3
        preds = np.zeros(n, dtype=int)
        for i in range(n):
            valid_agents = [
                a for a in agents
                if valid_masks is None or bool(valid_masks[a][i])
            ]
            if not valid_agents:
                preds[i] = int(fallback_preds[i]) if fallback_preds is not None else 1
                continue
            ranked = sorted(valid_agents, key=lambda a: rel_probs[a][i], reverse=True)
            top = ranked[:k]
            preds[i] = 1 if sum(verdicts[a][i] for a in top) > k / 2 else 0
        return preds

    if strategy == "stacked_meta":
        meta_X = _build_meta_X(rel_probs, verdicts, agents, valid_masks)
        preds = meta_clf.predict(meta_X).astype(int)
        if valid_masks is not None and fallback_preds is not None:
            valid_count = np.zeros(n, dtype=int)
            for a in agents:
                valid_count += valid_masks[a].astype(int)
            zero_valid = valid_count == 0
            preds[zero_valid] = fallback_preds[zero_valid]
        return preds

    if strategy == "score_fusion":
        if fallback_probs is None:
            raise ValueError("score_fusion requires fallback_probs")
        scores, _ = weighted_vote_scores(
            rel_probs, verdicts, agents, valid_masks=valid_masks
        )
        fuse_X = np.column_stack([scores.astype(float), fallback_probs.astype(float)])
        if isinstance(fusion_clf, (int, np.integer)):
            return np.full(n, int(fusion_clf), dtype=int)
        if fusion_clf is None:
            raise ValueError("score_fusion requires a trained fusion_clf")
        return fusion_clf.predict(fuse_X).astype(int)

    raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute error-detection metrics (pos_label=0 = detecting errors)."""
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, pos_label=0, average="binary", zero_division=0,
    )
    tp = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 1) & (y_pred == 0)).sum())
    fn = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 1) & (y_pred == 1)).sum())
    return {"f1": float(f1), "p": float(p), "r": float(r),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


# ---------------------------------------------------------------------------
# 5-Fold CV for strategy selection (GT never seen)
# ---------------------------------------------------------------------------

def cv_select_strategy(
    hc_feat_X: pd.DataFrame,
    hc_verdicts: dict[str, np.ndarray],
    hc_labels: np.ndarray,
    agents: list[str],
    hc_valid_masks: dict[str, np.ndarray] | None = None,
    n_folds: int = 5,
    progress_callback=None,
) -> list[dict]:
    """Evaluate all (model_type, strategy) combinations via 5-fold CV.

    Training labels are Snorkel high-conf labels — GT is never consulted.
    Returns list sorted by cv_f1 descending.
    """
    strategies = ["weighted_voting", "best_agent"]
    if len(agents) >= 3:
        strategies.append("top2_voting")
    if len(agents) >= 4:
        strategies.append("top3_voting")
    strategies.append("stacked_meta")
    strategies.append("score_fusion")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    splits = list(skf.split(hc_feat_X, hc_labels))

    all_fold_f1s: dict[tuple, list] = {}
    for mt in MODEL_TYPES:
        for s in strategies:
            all_fold_f1s[(mt, s)] = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        X_tr = hc_feat_X.iloc[train_idx].reset_index(drop=True)
        X_vl = hc_feat_X.iloc[val_idx].reset_index(drop=True)
        y_tr = hc_labels[train_idx]
        y_vl = hc_labels[val_idx]
        verd_tr = {a: hc_verdicts[a][train_idx] for a in agents}
        verd_vl = {a: hc_verdicts[a][val_idx]   for a in agents}
        valid_tr = None if hc_valid_masks is None else {
            a: hc_valid_masks[a][train_idx] for a in agents
        }
        valid_vl = None if hc_valid_masks is None else {
            a: hc_valid_masks[a][val_idx] for a in agents
        }

        for mt in MODEL_TYPES:
            # Train reliability models on training fold
            clfs = train_reliability_models(
                X_tr, verd_tr, y_tr, agents, mt, valid_masks=valid_tr
            )

            # Reliability probs for val (unbiased) and train (for meta-clf)
            rel_tr = get_reliability_probs(clfs, X_tr, agents)
            rel_vl = get_reliability_probs(clfs, X_vl, agents)
            fallback_model = train_fallback_task_model(X_tr, y_tr, mt)
            fallback_tr = predict_fallback_task_labels(fallback_model, X_tr)
            fallback_vl = predict_fallback_task_labels(fallback_model, X_vl)
            fallback_prob_tr = predict_fallback_task_probs(fallback_model, X_tr)
            fallback_prob_vl = predict_fallback_task_probs(fallback_model, X_vl)

            for s in strategies:
                if s == "stacked_meta":
                    # Train meta on training fold, evaluate on val fold
                    if len(np.unique(y_tr)) >= 2:
                        meta_clf = train_meta_clf(
                            rel_tr, verd_tr, y_tr, agents, valid_masks=valid_tr
                        )
                        preds = apply_strategy(
                            s,
                            rel_vl,
                            verd_vl,
                            agents,
                            meta_clf,
                            valid_masks=valid_vl,
                            fallback_preds=fallback_vl,
                        )
                    else:
                        preds = np.ones(len(y_vl), dtype=int)
                elif s == "score_fusion":
                    tr_scores, _ = weighted_vote_scores(
                        rel_tr, verd_tr, agents, valid_masks=None
                    )
                    fusion_clf = train_score_fusion_clf(
                        tr_scores, fallback_prob_tr, y_tr
                    )
                    preds = apply_strategy(
                        s,
                        rel_vl,
                        verd_vl,
                        agents,
                        fusion_clf=fusion_clf,
                        valid_masks=None,
                        fallback_probs=fallback_prob_vl,
                    )
                elif s == "weighted_voting":
                    preds = apply_strategy(
                        s,
                        rel_vl,
                        verd_vl,
                        agents,
                        valid_masks=valid_vl,
                        fallback_preds=fallback_vl,
                        decision_threshold=WEIGHTED_VOTE_THRESHOLD,
                    )
                else:
                    preds = apply_strategy(
                        s,
                        rel_vl,
                        verd_vl,
                        agents,
                        valid_masks=valid_vl,
                        fallback_preds=fallback_vl,
                    )

                all_fold_f1s[(mt, s)].append(compute_metrics(y_vl, preds)["f1"])

            if progress_callback is not None:
                progress_callback(fold_idx, n_folds, mt)

    results = []
    for (mt, s), f1s in all_fold_f1s.items():
        results.append({
            "model_type": mt,
            "strategy": s,
            "cv_f1": float(np.mean(f1s)),
            "cv_f1_std": float(np.std(f1s)),
            "fold_f1s": [round(x, 4) for x in f1s],
        })

    return sorted(
        results,
        key=lambda x: (
            -x["cv_f1"],
            STRATEGY_TIE_PRIORITY.get(x["strategy"], 999),
            x["model_type"],
        ),
    )
