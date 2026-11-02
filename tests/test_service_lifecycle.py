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
from configuration import load
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

    def test_failed_launcher_releases_its_session_and_storage(self):
        self.server.plan['server_command'] = [sys.executable, '-B', '-c', 'raise SystemExit(7)']
        with patch('service.attest_environment', return_value=self.attestation):
            with self.assertRaisesRegex(RuntimeError, 'exited during startup'):
                with self.server:
                    pass
        self.assertTrue(self.server.process.cleanup['released'])
        self.assertFalse(self.server.storage.path.exists())


if __name__ == '__main__':
    unittest.main()
