#!/usr/bin/env python3
"""Evaluate recorded automatic SQLDriller and SAR-Agent outputs on the two BIRD sets.

This script does not execute either auditing system or call any model API.
Manual correctness labels are loaded only to score the fixed predictions.
"""

import argparse
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'src/data'


def sqldriller_predictions():
    path = DATA / 'baselines/sqldriller/issues/dev/modified_gold.tsv'
    predictions = {}
    for line in path.read_text().splitlines():
        task_id, label, original_sql, proposed_sql = line.split('\t')
        task_id, label = int(task_id), int(label)
        if task_id in predictions or label not in (0, 1):
            raise ValueError(f'Invalid SQLDriller row for {task_id}')
        predictions[task_id] = label
    if len(predictions) != 1534:
        raise ValueError('Expected 1,534 explicit automatic SQLDriller predictions')
    return predictions


def sar_predictions():
    predictions = {}
    for path in sorted((DATA / 'baselines/sar_agent/analyze_result').glob('*/final_analyze_result.txt')):
        match = re.search(r'^Correctness:\s*(Yes|No)\b', path.read_text(), re.MULTILINE | re.IGNORECASE)
        if not match:
            raise ValueError(f'Missing unambiguous automatic correctness verdict: {path}')
        task_id = int(path.parent.name)
        if task_id in predictions:
            raise ValueError(f'Duplicate SAR-Agent task {task_id}')
        predictions[task_id] = int(match.group(1).lower() == 'yes')
    if len(predictions) != 298:
        raise ValueError('Expected 298 automatic SAR-Agent reports')
    return predictions


def evaluate(predictions, rows):
    ids = [int(row['query_id']) for row in rows]
    if len(ids) != len(set(ids)) or set(ids) - predictions.keys():
        raise ValueError('Duplicate evaluation IDs or missing baseline outputs')
    tp = fp = fn = tn = 0
    for row in rows:
        actual, predicted = int(row['label']), predictions[int(row['query_id'])]
        if actual not in (0, 1):
            raise ValueError('Unknown evaluation label')
        tp += actual == 0 and predicted == 0
        fp += actual == 1 and predicted == 0
        fn += actual == 0 and predicted == 1
        tn += actual == 1 and predicted == 1
    return {'n': len(rows), 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': tp / (tp + fp) if tp + fp else 0.0,
            'recall': tp / (tp + fn) if tp + fn else 0.0,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'reports/baselines')
    args = parser.parse_args()
    results = {}
    predictions = {'SQLDriller': sqldriller_predictions(), 'SAR-Agent (o3)': sar_predictions()}
    for name, count in [('BIRD_CLEAN_xs', 298), ('GT_RS_200', 200)]:
        with (DATA / 'ground_truth' / f'{name}.csv').open(newline='') as f:
            rows = list(csv.DictReader(f))
        if len(rows) != count:
            raise ValueError(f'Unexpected sample size for {name}')
        results[name] = {method: evaluate(pred, rows) for method, pred in predictions.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'metrics.json').write_text(json.dumps(results, indent=2) + '\n')
    lines = ['# Automatic baseline evaluation', '',
             'Positive class: annotation error (label 0). Both baselines use automatic outputs only.', '',
             '| Dataset | Method | F1 | Precision | Recall | TP | FP | FN | TN |',
             '|:--|:--|--:|--:|--:|--:|--:|--:|--:|']
    for dataset, methods in results.items():
        for method, m in methods.items():
            lines.append(f"| {dataset} | {method} | {m['f1']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['tp']} | {m['fp']} | {m['fn']} | {m['tn']} |")
    (args.output_dir / 'results.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
