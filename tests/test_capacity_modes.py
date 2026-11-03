"""Check mode-specific token targets while retaining physical capacity checks."""
from contextlib import redirect_stderr
import copy
import io
from itertools import product
import json
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from calibrate_capacity import calibration_config, write_calibration
from capacity_validation import calibration_record, validate_workers
from common import read_json, save_json
from configuration import compile_plan, load, validate
from service import Service
import test_service_contracts


class CapacityModesTest(unittest.TestCase):
    def config(self, hbm_mode='auto', l1_mode='physical'):
        return load(ROOT/'config/experiment.json', [
            f'capacity.hbm.mode="{hbm_mode}"', f'capacity.l1.mode="{l1_mode}"', 'platform.kind="cuda"'])

    def test_non_token_modes_ignore_unused_calibration_and_token_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root/'missing.json'
            malformed = root/'malformed.json'
            malformed.write_text('invalid JSON')
            stale = root/'stale.json'
            save_json(stale, {'identity': {}, 'workers': []})
            for mode, platform, path in product(['auto', 'physical'], ['cuda', 'ascend'],
                                                 [missing, malformed, stale]):
                config = self.config(mode)
                config['platform']['kind'] = platform
                config['capacity']['calibration_file'] = str(path)
                for tier in ['hbm', 'l1']:
                    config['capacity'][tier].update(tokens=-1, calculation={'method': 'calibration'})
                original = copy.deepcopy(config)
                for profile in config['profiles']:
                    with self.subTest(mode=mode, platform=platform, path=path.name, profile=profile):
                        plan = compile_plan(config, profile, root/'plan')
                        capacity = plan['capacity']
                        self.assertIsNone(capacity['calibration_sha256'])
                        self.assertIsNone(capacity['hbm_target_tokens'])
                        self.assertIsNone(capacity['l1_target_tokens'])
                        self.assertEqual(capacity['cache_buffer_capacity_gb'], config['capacity']['l1']['gib'])
                        expected_bytes = None if mode == 'auto' else config['capacity']['hbm']['bytes']
                        self.assertEqual(capacity['kv_cache_memory_bytes'], expected_bytes)
                        if 'vllm_configuration' in plan:
                            native = plan['vllm_configuration']
                            self.assertEqual('kv-cache-memory-bytes' in native, mode == 'physical')
                        else:
                            self.assertEqual('--kv-cache-memory-bytes' in plan['server_command'], mode == 'physical')
                        self.assertEqual(config, original)

    def test_mixed_modes_still_require_calibration_for_the_token_tier(self):
        for hbm, l1, tier in [('auto', 'tokens', 'l1'), ('physical', 'tokens', 'l1'),
                              ('tokens', 'physical', 'hbm')]:
            with self.subTest(hbm=hbm, l1=l1), tempfile.TemporaryDirectory() as temporary:
                config = self.config(hbm, l1)
                config['capacity'][tier]['calculation']['method'] = 'calibration'
                with self.assertRaisesRegex(ValueError, 'requires capacity.calibration_file'):
                    compile_plan(config, 'aisbench_n40_performance', temporary)
                path = Path(temporary)/'calibration.json'
                save_json(path, {'identity': {}, 'workers': []})
                config['capacity']['calibration_file'] = str(path)
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    compile_plan(config, 'aisbench_n40_performance', temporary)

    def test_validation_records_only_active_token_targets(self):
        for hbm, l1 in product(['auto', 'physical', 'tokens'], ['physical', 'tokens']):
            with self.subTest(hbm=hbm, l1=l1), tempfile.TemporaryDirectory() as temporary:
                config = self.config(hbm, l1)
                plan = compile_plan(config, 'aisbench_n40_performance', temporary)
                result = validate_workers(plan, [test_service_contracts.ServiceContractsTest().worker()])
                self.assertTrue(result['passed'], result['errors'])
                targets = {'hbm': hbm == 'tokens', 'l1': l1 == 'tokens'}
                self.assertEqual(result['enforced_token_targets_by_tier'], targets)
                self.assertEqual(result['enforced_token_targets'], any(targets.values()))

    def test_physical_checks_remain_active_without_token_targets(self):
        for mode, enforce in product(['auto', 'physical'], [True, False]):
            with self.subTest(mode=mode, enforce=enforce), tempfile.TemporaryDirectory() as temporary:
                plan = compile_plan(self.config(mode), 'aisbench_n40_performance', temporary)
                row = test_service_contracts.ServiceContractsTest().worker()
                row['kv_tensor_bytes'] -= 1
                result = validate_workers(plan, [row], enforce_targets=enforce)
                self.assertFalse(result['passed'])
                self.assertIn('Actual KV tensor bytes disagree with block allocation', result['errors'])
                self.assertFalse(result['enforced_token_targets'])
                row = test_service_contracts.ServiceContractsTest().worker()
                row['l1_nodes'] += 1
                result = validate_workers(plan, [row], enforce_targets=enforce)
                self.assertTrue(any('L1 nodes differ from physical budget' in error for error in result['errors']))
                if mode == 'physical':
                    row = test_service_contracts.ServiceContractsTest().worker()
                    row.update(hbm_blocks=8000, hbm_tokens=8000*64, kv_tensor_bytes=8000*1074432)
                    result = validate_workers(plan, [row], enforce_targets=enforce)
                    self.assertIn('Actual HBM KV allocation exceeds requested physical budget', result['errors'])

    def test_token_mismatch_and_explicit_target_bypass_keep_existing_behavior(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = compile_plan(self.config('tokens'), 'aisbench_n40_performance', temporary)
            row = test_service_contracts.ServiceContractsTest().worker()
            row.update(hbm_blocks=4000, hbm_tokens=4000*64, kv_tensor_bytes=4000*1074432)
            result = validate_workers(plan, [row])
            self.assertFalse(result['passed'])
            self.assertTrue(any('HBM tokens differ' in error for error in result['errors']))
            result = validate_workers(plan, [row], enforce_targets=False)
            self.assertTrue(result['passed'], result['errors'])
            self.assertFalse(result['enforced_token_targets'])
            self.assertEqual(result['enforced_token_targets_by_tier'], {'hbm': False, 'l1': False})

    def test_explicit_calibration_still_measures_non_token_modes(self):
        for mode in ['auto', 'physical']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                config = calibration_config(ROOT/'config/experiment.json', [
                    f'capacity.hbm.mode="{mode}"', 'platform.kind="cuda"',
                    'capacity.calibration_file="unused.json"'])
                self.assertEqual(config['capacity']['hbm']['mode'], mode)
                self.assertIsNone(config['capacity']['calibration_file'])
                plan = compile_plan(config, 'aisbench_n40_performance', temporary)
                record = calibration_record(plan, [test_service_contracts.ServiceContractsTest().worker()])
                write_calibration(temporary, record)
                saved = read_json(Path(temporary)/'capacity-calibration.json')
                self.assertTrue(saved['validation']['passed'])
                self.assertEqual(saved['workers'][0]['hbm_bytes_per_block'], 1074432)

    def test_missing_or_invalid_modes_are_not_silently_defaulted(self):
        for tier in ['hbm', 'l1']:
            config = self.config()
            del config['capacity'][tier]['mode']
            with self.subTest(tier=tier), self.assertRaises(KeyError):
                validate(config)
            for mode in ['AUTO', ' physical ', 'unknown']:
                with self.subTest(tier=tier, mode=mode), self.assertRaises(ValueError):
                    load(ROOT/'config/experiment.json', [f'capacity.{tier}.mode={json.dumps(mode)}'])

    def test_cpu_service_logs_skips_and_retains_actual_capacity(self):
        for mode in ['auto', 'physical']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                model = root/'model'
                model.mkdir()
                config = self.config(mode)
                config['paths'].update(model=str(model), tokenizer=str(model))
                config['capacity']['calibration_file'] = str(root/'missing.json')
                for tier in ['hbm', 'l1']:
                    config['capacity'][tier]['tokens'] = -1
                with socket.socket() as sock:
                    sock.bind(('127.0.0.1', 0))
                    config['server']['port'] = sock.getsockname()[1]
                config['execution'].update(startup_timeout_s=8, shutdown_timeout_s=2, health_poll_s=.05)
                service = Service(config, 'aisbench_n40_performance', root/'service')
                row = test_service_contracts.ServiceContractsTest().worker()
                row.update(hbm_blocks=4000, hbm_tokens=4000*64, kv_tensor_bytes=4000*1074432,
                           ucm_source=str(root/'mock-ucm/__init__.py'))
                save_json(service.run/'fixture-worker.json', row)
                service.plan['server_command'] = [sys.executable, '-B', str(ROOT/'tests/fake_model_service.py'),
                                                  str(config['server']['port'])]
                attestation = {'installed': {'ucm_root': str(root/'mock-ucm')}}
                log = io.StringIO()
                with patch('service.attest_environment', return_value=attestation), redirect_stderr(log):
                    with service:
                        result = read_json(service.run/'capacity-validation.json')
                        self.assertTrue(result['passed'], result['errors'])
                        self.assertFalse(result['enforced_token_targets'])
                        saved = read_json(service.run/'plan.json')
                        self.assertTrue(saved['capacity']['actual_capacity_verified'])
                        self.assertEqual(saved['capacity']['observed_hbm'][0]['hbm_physical_tokens'], 256000)
                self.assertIn(f'hbm (mode={mode})', log.getvalue())
                self.assertIn('l1 (mode=physical)', log.getvalue())
                self.assertTrue(read_json(service.run/'resource-cleanup.json')['released'])


if __name__ == '__main__':
    unittest.main()
