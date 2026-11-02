import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from calibrate_capacity import calibration_config, write_calibration
from capacity_validation import calibration_record, validate_workers
from common import save_json
from configuration import compile_plan, load, model_identity
from generate_report import generate
from report_data import aggregate, collect
import test_service_contracts


class HbmAutoTest(unittest.TestCase):
    def config(self):
        return load(ROOT / 'config/experiment.json', ['capacity.hbm.mode="auto"', 'platform.kind="cuda"'])

    def test_both_platforms_and_all_profiles_delegate_hbm_to_engine(self):
        for platform in ['cuda', 'ascend']:
            config = self.config()
            config['platform']['kind'] = platform
            config['server']['gpu_memory_utilization'] = 0.8
            config['capacity']['hbm'].update(tokens=None, bytes=None, calculation={'method': 'calibration'})
            config['capacity']['calibration_file'] = '/unused-calibration.json'
            for profile in config['profiles']:
                with self.subTest(platform=platform, profile=profile):
                    plan = compile_plan(config, profile, ROOT / 'runs/auto-plan')
                    native = plan['vllm_configuration']
                    self.assertNotIn('kv-cache-memory-bytes', native)
                    self.assertEqual(native['gpu-memory-utilization'], 0.8)
                    self.assertEqual(plan['capacity']['hbm_mode'], 'auto')
                    self.assertIsNone(plan['capacity']['kv_cache_memory_bytes'])
                    self.assertIsNone(plan['capacity']['hbm_target_tokens'])
                    self.assertIsNone(plan['capacity']['calibration_sha256'])

    def test_explicit_modes_still_pass_byte_budgets(self):
        for mode, value in [('tokens', 5372160000), ('physical', 8 * (1 << 30))]:
            config = load(ROOT / 'config/experiment.json', [f'capacity.hbm.mode="{mode}"'])
            config['capacity']['hbm']['bytes'] = 8 * (1 << 30)
            plan = compile_plan(config, 'swe_w24_n3379_performance', ROOT / 'runs/explicit-plan')
            self.assertEqual(plan['vllm_configuration']['kv-cache-memory-bytes'], value)

    def test_invalid_modes_and_utilization_are_rejected(self):
        for override in ['capacity.l1.mode="auto"', 'capacity.hbm.mode="unknown"',
                         'server.gpu_memory_utilization=0', 'server.gpu_memory_utilization=1.1']:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load(ROOT / 'config/experiment.json', [override])

    def test_auto_accepts_actual_size_but_preserves_layout_checks(self):
        config = self.config()
        plan = compile_plan(config, 'aisbench_n40_performance', ROOT / 'runs/auto-validation')
        worker = test_service_contracts.ServiceContractsTest().worker()
        worker.update(hbm_blocks=8000, hbm_tokens=8000 * 64, kv_tensor_bytes=8000 * 1074432)
        self.assertTrue(validate_workers(plan, [worker])['passed'])
        bad = copy.deepcopy(worker)
        bad['kv_tensor_bytes'] -= 1
        self.assertFalse(validate_workers(plan, [bad])['passed'])
        bad = copy.deepcopy(worker)
        bad['l1_nodes'] += 1
        self.assertFalse(validate_workers(plan, [bad])['passed'])
        bad = copy.deepcopy(worker)
        bad.update(hbm_blocks=0, hbm_tokens=0, kv_tensor_bytes=0)
        with self.assertRaisesRegex(ValueError, 'no allocatable blocks'):
            validate_workers(plan, [bad])
        config['capacity']['hbm']['mode'] = 'physical'
        explicit = compile_plan(config, 'aisbench_n40_performance', ROOT / 'runs/explicit-validation')
        self.assertIn('Actual HBM KV allocation exceeds requested physical budget',
                      validate_workers(explicit, [worker])['errors'])

    def test_l1_calibration_remains_identity_bound_with_auto_hbm(self):
        config = self.config()
        config['capacity']['l1'].update(mode='tokens', calculation={'method': 'calibration'})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'calibration.json'
            save_json(path, {'identity': model_identity(config),
                             'workers': [test_service_contracts.ServiceContractsTest().worker()['capacity_layout']]})
            config['capacity']['calibration_file'] = str(path)
            plan = compile_plan(config, 'aisbench_n40_performance', temporary)
            self.assertIsNotNone(plan['capacity']['calibration_sha256'])
            self.assertEqual(plan['capacity']['cache_buffer_capacity_gb'], 19)
            config['server']['tensor_parallel_size'] = 2
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                compile_plan(config, 'aisbench_n40_performance', temporary)

    def test_calibration_preserves_auto_and_exports_measured_layout(self):
        config = calibration_config(ROOT / 'config/experiment.json',
                                    ['capacity.hbm.mode="auto"', 'platform.kind="cuda"',
                                     'capacity.hbm.bytes=null', 'capacity.calibration_file="/unused.json"'])
        self.assertEqual(config['capacity']['hbm']['mode'], 'auto')
        self.assertEqual(config['capacity']['l1']['mode'], 'physical')
        self.assertIsNone(config['capacity']['calibration_file'])
        with tempfile.TemporaryDirectory() as temporary:
            plan = compile_plan(config, 'aisbench_n40_performance', temporary)
            value = calibration_record(plan, [test_service_contracts.ServiceContractsTest().worker()])
            write_calibration(temporary, value)
            aggregate = json.loads((Path(temporary) / 'capacity-calibration.json').read_text())
            layout = json.loads((Path(temporary) / 'layouts/worker-000.json').read_text())
            self.assertEqual(layout['hbm_bytes_per_block'], 1074432)
            self.assertTrue(aggregate['validation']['passed'])
        for mode in ['tokens', 'physical']:
            config = calibration_config(ROOT / 'config/experiment.json', [f'capacity.hbm.mode="{mode}"'])
            self.assertEqual(config['capacity']['hbm']['mode'], 'physical')

    def report_run(self, run, blocks):
        plan = compile_plan(self.config(), 'aisbench_n40_performance', run)
        plan['validation_only'] = 'Synthetic automatic HBM report fixture'
        plan['input_fingerprint'] = 'fixed-synthetic-input'
        plan['capacity']['actual_capacity_verified'] = blocks is not None
        if blocks is not None:
            plan['capacity']['observed_hbm'] = [
                {'rank': 0, 'kv_tensor_bytes': blocks * 1074432, 'hbm_physical_tokens': blocks * 64}]
        save_json(run / 'plan.json', plan)
        save_json(run / 'actual-service-plan.json', plan)
        attempt = run / 'attempts/001'
        save_json(attempt / 'status.json', {'status': 'completed'})
        save_json(attempt / 'performance.json', {'passed': True, 'requests': 1, 'successful_requests': 1,
                  'duration_s': 1, 'requests_per_s': 1, 'input_tokens_per_s': 64, 'output_tokens_per_s': 1})
        save_json(attempt / 'request-samples.json', [])
        return collect(run)[0]

    def test_reports_group_auto_runs_by_actual_capacity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [self.report_run(root / str(i), blocks) for i, blocks in enumerate([5000, 5000, 8000])]
            self.assertTrue(all(row['valid'] for row in rows))
            self.assertEqual(rows[0]['configuration_group'], rows[1]['configuration_group'])
            self.assertNotEqual(rows[0]['configuration_group'], rows[2]['configuration_group'])
            self.assertEqual(sorted(group['runs'] for group in aggregate(rows, include_validation=True)), [1, 2])

    def test_report_exports_and_displays_measured_hbm(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.report_run(root / 'input', 8000)
            with patch('generate_report.render', return_value=[]):
                report = generate([root / 'input'], include_validation=True, output_root=root / 'reports')
            text = (report / 'REPORT.md').read_text()
            self.assertIn('自动 HBM 实测容量', text)
            self.assertIn('8595456000', text)
            results = json.loads((report / 'RESULTS.json').read_text())
            self.assertEqual(results['records'][0]['observed_hbm'][0]['hbm_physical_tokens'], 512000)

    def test_auto_report_without_actual_capacity_is_excluded(self):
        with tempfile.TemporaryDirectory() as temporary:
            row = self.report_run(Path(temporary), None)
            self.assertFalse(row['valid'])
            self.assertIn('自动 HBM 缺少通过校验的实际容量记录', row['excluded_reasons'])


if __name__ == '__main__':
    unittest.main()
