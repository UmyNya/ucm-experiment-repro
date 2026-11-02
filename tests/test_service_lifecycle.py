"""Exercise the real lifecycle controller against a CPU-only HTTP fixture."""
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from configuration import compile_plan, load
from capacity_validation import calibration_record
from common import save_json
from processes import same_process
from service import Service
import test_service_contracts


class ServiceLifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        c = load(ROOT/'config/experiment.json')
        c['platform']['kind'] = 'cuda'
        c['paths'].update(model=str(self.root/'model'), tokenizer=str(self.root/'model'))
        (self.root/'model').mkdir()
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            c['server']['port'] = sock.getsockname()[1]
        c['execution'].update(startup_timeout_s=8, shutdown_timeout_s=2, health_poll_s=.05)
        self.server = Service(c, 'aisbench_n40_performance', self.root/'service')
        row = test_service_contracts.ServiceContractsTest().worker()
        row['ucm_source'] = str(self.root/'mock-ucm/__init__.py')
        save_json(self.server.run/'fixture-worker.json', row)
        self.server.plan['server_command'] = [sys.executable, '-B', str(ROOT/'tests/fake_model_service.py'), str(c['server']['port'])]
        self.attestation = {'installed': {'ucm_root': str(self.root/'mock-ucm')}}

    def test_ready_snapshot_and_owned_process_stop(self):
        with patch('service.attest_environment', return_value=self.attestation):
            with self.server as server:
                leader = server.process.leader
                self.assertIsNotNone(same_process(leader))
                self.assertEqual(server.rows[0]['pid'], leader['pid'])
                (server.storage.path/'sample-cache').write_bytes(b'owned')
            self.assertIsNone(same_process(leader))
            self.assertFalse(server.storage.path.exists())
            result = json.loads((server.run/'resource-cleanup.json').read_text())
            self.assertTrue(result['released'])

    def test_occupied_port_leaves_existing_socket_untouched(self):
        with socket.socket() as occupied:
            occupied.bind(('127.0.0.1', self.server.plan['configuration']['server']['port']))
            occupied.listen(1)
            with patch('service.attest_environment', return_value=self.attestation):
                with self.assertRaises(OSError):
                    with self.server:
                        pass
            self.assertGreater(occupied.fileno(), 0)
            self.assertIsNone(self.server.process)
            self.assertFalse(self.server.storage.created)

    def test_auto_hbm_records_observed_capacity_in_service_plan(self):
        self.server.plan['configuration']['capacity']['hbm']['mode'] = 'auto'
        compiled = compile_plan(self.server.plan['configuration'], 'aisbench_n40_performance', self.server.run)
        self.server.plan['capacity'] = compiled['capacity']
        with patch('service.attest_environment', return_value=self.attestation):
            with self.server:
                saved = json.loads((self.server.run / 'plan.json').read_text())
                capacity = saved['capacity']
                self.assertIsNone(capacity['kv_cache_memory_bytes'])
                self.assertTrue(capacity['actual_capacity_verified'])
                self.assertEqual(capacity['observed_hbm'], [
                    {'rank': 0, 'kv_tensor_bytes': 5372160000, 'hbm_physical_tokens': 320000}])

    def test_failed_launcher_releases_its_session_and_storage(self):
        self.server.plan['server_command'] = [sys.executable, '-B', '-c', 'raise SystemExit(7)']
        with patch('service.attest_environment', return_value=self.attestation):
            with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
                with self.server:
                    pass
        self.assertTrue(self.server.process.cleanup['released'])
        self.assertFalse(self.server.storage.path.exists())

    def reconfigure(self, config):
        command = self.server.plan['server_command']
        row = json.loads((self.server.run/'fixture-worker.json').read_text())
        self.server = Service(config, 'aisbench_n40_performance', self.root/'reconfigured-service')
        save_json(self.server.run/'fixture-worker.json', row)
        self.server.plan['server_command'] = command

    def test_omitted_block_records_worker_value_without_changing_engine_config(self):
        c = self.server.plan['configuration']
        del c['server']['block_size']
        c['capacity']['hbm']['mode'] = 'auto'
        self.reconfigure(c)
        with patch('service.attest_environment', return_value=self.attestation):
            with self.server as server:
                saved = json.loads((server.run/'plan.json').read_text())
                self.assertIsNone(saved['configuration']['server']['block_size'])
                self.assertEqual(saved['observed_block_size'], 64)
                self.assertEqual(saved['model_identity']['block_size'], 64)
                self.assertNotIn('block-size', json.loads((server.run/'vllm-effective.yaml').read_text()))
                record = calibration_record(saved, server.rows)
                self.assertEqual(record['identity']['block_size'], 64)
        self.assertTrue(json.loads((self.server.run/'resource-cleanup.json').read_text())['released'])

    def test_calibrated_block_mismatch_stops_even_with_relaxed_capacity_validation(self):
        row = json.loads((self.server.run/'fixture-worker.json').read_text())
        calibration = calibration_record(self.server.plan, [row])
        path = self.root/'calibration.json'
        save_json(path, calibration)
        c = self.server.plan['configuration']
        c['server']['block_size'] = None
        c['capacity']['hbm']['mode'] = 'auto'
        c['capacity']['l1'].update(mode='tokens', calculation={'method': 'calibration'})
        c['capacity']['calibration_file'] = str(path)
        c['validation']['require_exact_capacity'] = False
        self.reconfigure(c)
        row['capacity_layout']['block_tokens'] = 128
        row['hbm_tokens'] *= 2
        row['l1_ordinary_tokens'] *= 2
        save_json(self.server.run/'fixture-worker.json', row)
        with patch('service.attest_environment', return_value=self.attestation):
            with self.assertRaisesRegex(ValueError, 'Actual block_size differs'):
                with self.server:
                    pass
        self.assertTrue(json.loads((self.server.run/'resource-cleanup.json').read_text())['released'])


if __name__ == '__main__':
    unittest.main()
