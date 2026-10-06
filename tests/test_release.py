"""Check published results, artifact integrity, and prediction coverage."""

import csv
import hashlib
import json
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class PublishedResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ROOT / 'reports/pipeline_report.json').read_text())

    def test_detection(self):
        expected = {
            'frontier': {'BIRD_CLEAN_xs': (154, 15, 12, 117), 'GT_RS_200': (62, 15, 6, 117)},
            'legacy': {'BIRD_CLEAN_xs': (143, 31, 23, 101), 'GT_RS_200': (58, 31, 10, 101)},
        }
        for config, datasets in expected.items():
            for dataset, counts in datasets.items():
                with self.subTest(config=config, dataset=dataset):
                    m = self.report['decision_plane'][config]['gt_results'][dataset]
                    self.assertEqual(tuple(m[k] for k in ['tp', 'fp', 'fn', 'tn']), counts)
                    tp, fp, fn, tn = counts
                    self.assertAlmostEqual(m['f1'], 2 * tp / (2 * tp + fp + fn))

    def test_frontier_stacking(self):
        for name, expected in [('BIRD_CLEAN_xs', .9021), ('GT_RS_200', .8243)]:
            actual = self.report['stacking']['frontier']['gt_results'][name]['f1']
            self.assertEqual(round(actual, 4), expected)

    def test_overall_audit(self):
        expected = {'sqld_dev': (1534, 656, 39, (33, 45)),
                    'november_bird': (1534, 571, 37, (30, 43)),
                    'spider_dev': (1034, 325, 27, (21, 33))}
        for name, (n, flagged, rate, ci) in expected.items():
            with self.subTest(benchmark=name):
                raw = self.report['frontier_detection_rates'][name]
                self.assertEqual((raw['n_tasks'], raw['predicted_error_count']), (n, flagged))
                calibrated = self.report['calibrated_audit'][name]
                self.assertEqual(round(100 * calibrated['calibrated_error_rate']), rate)
                self.assertEqual(tuple(round(100 * x) for x in calibrated['ci95']), ci)
        for name, dataset, counts in [
            ('november_bird', 'GT_RS_200_nov_bird', (59, 12, 10, 119)),
            ('spider_dev', 'GT_RS_200_spider', (56, 12, 4, 128)),
        ]:
            m = self.report['frontier_full_benchmarks'][name]['gt_results'][dataset]
            self.assertEqual(tuple(m[k] for k in ['tp', 'fp', 'fn', 'tn']), counts)

    def test_prediction_exports(self):
        for config, n in [('frontier', 1534), ('legacy', 1534),
                          ('frontier_november', 1534), ('frontier_spider', 1034)]:
            with (ROOT / 'reports/predictions' / f'{config}.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), n)
            self.assertEqual(len({r['task_id'] for r in rows}), n)
            self.assertTrue(all(r['predicted_label'] in ('0', '1') for r in rows))

    def test_bundled_weak_supervision(self):
        for config, n in [('frontier', 1534), ('legacy', 1534),
                          ('frontier_november', 1534), ('frontier_spider', 1034)]:
            with self.subTest(config=config):
                path = ROOT / 'src/data/weak_supervision' / config
                manifest = json.loads((path / 'labeling_matrix_manifest.json').read_text())
                self.assertEqual(manifest['config'], config)
                self.assertEqual(manifest['n_tasks'], n)
                matrices = {}
                for kind, columns in [('task', 43), ('meta', 1), ('full', 44)]:
                    matrix = np.load(path / f'labeling_matrix_{kind}.npy')
                    matrices[kind] = matrix
                    self.assertEqual(matrix.shape, (n, columns))
                    self.assertTrue(np.isin(matrix, [-1, 0, 1]).all())
                    digest = hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()
                    self.assertEqual(digest, manifest[f'{kind}_sha256'])
                np.testing.assert_array_equal(
                    matrices['full'], np.column_stack([matrices['task'], matrices['meta']]))
                task_ids = np.load(path / 'task_ids.npy')
                self.assertEqual(task_ids.shape, (n,))
                self.assertEqual(len(np.unique(task_ids)), n)
                labels = np.load(path / 'snorkel_labels.npy')
                probs = np.load(path / 'snorkel_probs.npy')
                self.assertEqual(labels.shape, (n,))
                self.assertEqual(probs.shape, (n, 2))
                self.assertTrue(np.isfinite(probs).all())
                self.assertTrue(((probs >= 0) & (probs <= 1)).all())
                np.testing.assert_allclose(probs.sum(axis=1), 1)
                np.testing.assert_array_equal(labels, probs.argmax(axis=1))
                hc = json.loads((path / 'snorkel_high_conf_labels.json').read_text())
                self.assertEqual(len(hc), 500)
                self.assertEqual(len({row['task_id'] for row in hc}), 500)
                self.assertEqual(sum(row['snorkel_label'] == 0 for row in hc), 250)
                self.assertEqual(sum(row['snorkel_label'] == 1 for row in hc), 250)
                index = {int(task_id): i for i, task_id in enumerate(task_ids)}
                for row in hc:
                    i = index[row['task_id']]
                    self.assertEqual(row['snorkel_label'], int(labels[i]))
                    self.assertAlmostEqual(row['snorkel_prob_incorrect'], probs[i, 0])
                    self.assertAlmostEqual(row['snorkel_prob_correct'], probs[i, 1])
                with (ROOT / 'reports/predictions' / f'{config}.csv').open() as f:
                    exported_ids = {int(row['task_id']) for row in csv.DictReader(f)}
                self.assertEqual(exported_ids, set(task_ids.tolist()))

    def test_packaged_data_size(self):
        for directory in [ROOT / 'src/data', ROOT / 'data']:
            for path in directory.rglob('*'):
                if path.is_file():
                    self.assertLess(path.stat().st_size, 50 * 1024**2,
                                    f'{path.relative_to(ROOT)} exceeds the regular-Git size budget')

    def test_external_outputs(self):
        import sys
        sys.path.insert(0, str(ROOT))
        from evaluate_baselines import evaluate, sar_predictions, sqldriller_predictions
        preds = {'SQLDriller': sqldriller_predictions(), 'SAR-Agent (o3)': sar_predictions()}
        expected = {'BIRD_CLEAN_xs': {'SQLDriller': (102, 48, 64, 84), 'SAR-Agent (o3)': (138, 25, 28, 107)},
                    'GT_RS_200': {'SQLDriller': (42, 48, 26, 84), 'SAR-Agent (o3)': (53, 25, 15, 107)}}
        saved = json.loads((ROOT / 'reports/baselines/metrics.json').read_text())
        for dataset, methods in expected.items():
            with (ROOT / 'src/data/ground_truth' / f'{dataset}.csv').open() as f:
                rows = list(csv.DictReader(f))
            for name, counts in methods.items():
                actual = evaluate(preds[name], rows)
                self.assertEqual(tuple(actual[k] for k in ['tp', 'fp', 'fn', 'tn']), counts)
                self.assertEqual(actual, saved[dataset][name])


if __name__ == '__main__':
    unittest.main()
