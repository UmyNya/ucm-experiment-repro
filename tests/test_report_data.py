import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from report_data import aggregate
from generate_report import generate


class ReportDataTest(unittest.TestCase):
    def record(self, *, valid=True, synthetic=False, identity='one', throughput=10, samples=(1, 2)):
        return {'valid': valid, 'synthetic': synthetic, 'comparability_group': 'matching-input', 'configuration_group': identity,
                'label': 'test', 'comparability': {}, 'path': identity+str(throughput),
                'performance': {'duration_s': 1, 'requests_per_s': 2, 'input_tokens_per_s': throughput,
                                'output_tokens_per_s': 2, 'requests': len(samples), 'successful_requests': len(samples)},
                'samples': [{'ttft_ms': v, 'e2e_ms': v+1, 'tpot_ms': None} for v in samples]}

    def test_exclude_failed_and_mock_by_default(self):
        rows = [self.record(), self.record(valid=False, throughput=100), self.record(synthetic=True, identity='mock', throughput=999)]
        group = aggregate(rows)
        self.assertEqual(len(group), 1)
        self.assertEqual(group[0]['runs'], 1)
        self.assertEqual(group[0]['metrics']['input_tokens_per_s']['mean'], 10)
        self.assertIsNone(group[0]['metrics']['input_tokens_per_s']['sample_sd'])
        self.assertEqual(len(aggregate(rows, include_validation=True)), 2)

    def test_pooled_percentile_and_sample_sd_are_not_averaged_quantiles(self):
        rows = [self.record(throughput=10, samples=(0, 100)), self.record(throughput=20, samples=(1,))]
        group = aggregate(rows)[0]
        self.assertEqual(group['runs'], 2)
        self.assertEqual(group['pooled']['ttft_ms']['p50'], 1)
        self.assertAlmostEqual(group['pooled']['ttft_ms']['p95'], 90.1)
        self.assertAlmostEqual(group['metrics']['input_tokens_per_s']['sample_sd'], 7.0710678118654755)

    def test_other_workload_and_pipeline_scope_never_merge(self):
        a = self.record()
        b = copy.deepcopy(a)
        b['comparability_group'] = 'pipeline'
        c = copy.deepcopy(a)
        c['comparability_group'] = 'other-input'
        self.assertEqual(len(aggregate([a,b,c])), 3)

    def test_duplicate_selection_is_rejected_before_reading(self):
        with self.assertRaisesRegex(ValueError, 'more than once'):
            generate([Path('/tmp/nonexistent-report-fixture'), Path('/tmp/nonexistent-report-fixture')])


if __name__ == '__main__':
    unittest.main()
