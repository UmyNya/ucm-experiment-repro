import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch, Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from capacity_validation import validate_workers, calibration_record
from configuration import load, compile_plan
from service import Service
from storage import PrivateStorage


class ServiceContractsTest(unittest.TestCase):
    def setUp(self):
        self.c = load(ROOT/'config/experiment.json')
        self.c['platform']['kind'] = 'cuda'

    def worker(self):
        return {'pid': 123, 'worker_rank': 0, 'unique_id': 'fixture', 'platform': 'cuda',
                'capacity_layout': {'block_tokens': 64, 'hbm_bytes_per_block': 1074432, 'hbm_reserved_blocks': 1,
                                    'l1_node_bytes': 1093632, 'l1_nodes_per_token_block': 1, 'l1_reserved_nodes': 1024},
                'hbm_blocks': 5000, 'hbm_tokens': 320000, 'kv_tensor_bytes': 5372160000,
                'l1_nodes': 19*(1<<30)//1093632, 'l1_ordinary_tokens': 1128320, 'shared_memory': None}

    def test_measured_layout_calibrates_and_target_mismatch_fails(self):
        plan = compile_plan(self.c, 'aisbench_n40_performance', ROOT/'runs/contract')
        row = self.worker()
        self.assertTrue(validate_workers(plan, [row])['passed'])
        calibration = calibration_record(plan, [row])
        self.assertEqual(calibration['workers'][0]['hbm_bytes_per_block'], 1074432)
        row.update(hbm_blocks=4999, hbm_tokens=319936, kv_tensor_bytes=4999*1074432)
        self.assertFalse(validate_workers(plan, [row])['passed'])
        self.assertTrue(validate_workers(plan, [row], enforce_targets=False)['passed'])

    def test_missing_rank_wrong_backend_and_bad_node_count_fail(self):
        c = copy.deepcopy(self.c)
        c['platform']['visible_devices'] = ['0', '1']
        c['server']['tensor_parallel_size'] = 2
        plan = compile_plan(c, 'aisbench_n40_performance', ROOT/'runs/contract')
        self.assertFalse(validate_workers(plan, [self.worker()])['passed'])
        plan = compile_plan(self.c, 'aisbench_n40_performance', ROOT/'runs/contract')
        row = self.worker()
        row['platform'] = 'npu'
        self.assertFalse(validate_workers(plan, [row])['passed'])
        row = self.worker()
        row['l1_nodes'] += 1
        self.assertFalse(validate_workers(plan, [row])['passed'])

    def test_private_l2_cleanup_preserves_sibling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = compile_plan(self.c, 'aisbench_n40_performance', root)
            storage = PrivateStorage(plan)
            sibling = root/'scratch/unrelated'
            sibling.mkdir(parents=True)
            (sibling/'keep').write_text('keep')
            storage.create()
            (storage.path/'data').write_bytes(b'payload')
            result = storage.cleanup()
            self.assertTrue(result['released'])
            self.assertFalse(storage.path.exists())
            self.assertEqual((sibling/'keep').read_text(), 'keep')

    def test_large_shared_filesystem_is_not_reported_as_hard_quota(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['capacity']['l2'].update(gib=1, enforcement='filesystem', filesystem_limit_gib=1)
            storage = PrivateStorage(compile_plan(c, 'aisbench_n40_performance', tmp))
            with patch('storage.filesystem', return_value={'total_bytes': 2<<30}):
                with self.assertRaisesRegex(ValueError, 'not an enforced quota'):
                    storage.create()
            self.assertTrue(storage.cleanup()['released'])

    def test_loopback_wrong_backing_is_retained_without_unmount(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['capacity']['l2'].update(gib=1,enforcement='loopback',filesystem_limit_gib=1)
            storage = PrivateStorage(compile_plan(c,'aisbench_n40_performance',tmp))
            commands=[]
            with patch.object(storage,'command',side_effect=lambda name,argv:commands.append(name)),patch('storage.os.path.ismount',return_value=True):
                storage.create()
                with patch('storage.subprocess.check_output',side_effect=['/dev/loop-fixture','{"loopdevices":[{"back-file":"/different-owner.img"}]}']):
                    with self.assertRaisesRegex(RuntimeError,'ownership changed'):storage.cleanup()
            self.assertNotIn('l2-unmount',commands)
            self.assertTrue(storage.image.exists())
            self.assertTrue(storage.path.exists())

    def test_partially_successful_mount_is_cleaned_only_after_backing_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['capacity']['l2'].update(gib=1,enforcement='loopback',filesystem_limit_gib=1)
            storage = PrivateStorage(compile_plan(c,'aisbench_n40_performance',tmp))
            commands=[];mounted=[False]
            def command(name,argv):
                commands.append(name)
                if name=='l2-mount':
                    mounted[0]=True
                    raise RuntimeError('synthetic partial mount failure')
                if name=='l2-unmount':mounted[0]=False
            with patch.object(storage,'command',side_effect=command),patch('storage.os.path.ismount',side_effect=lambda p:mounted[0]):
                with self.assertRaisesRegex(RuntimeError,'partial mount'):storage.create()
                import json
                with patch('storage.subprocess.check_output',side_effect=['/dev/loop-fixture',json.dumps({'loopdevices':[{'back-file':str(storage.image)}]})]):
                    result=storage.cleanup()
            self.assertTrue(result['released'])
            self.assertIn('l2-unmount',commands)
            self.assertFalse(storage.path.exists())
            self.assertFalse(storage.image.exists())

    def test_startup_failure_still_cleans_created_private_l2(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = Service(self.c, 'aisbench_n40_performance', tmp)
            def fail():
                server.storage.create()
                (server.storage.path/'partial-data').write_bytes(b'x')
                raise RuntimeError('simulated startup failure')
            with patch.object(server, 'start', side_effect=fail):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    with server:
                        pass
            self.assertTrue(json.loads((Path(tmp)/'resource-cleanup.json').read_text())['released'])
            self.assertFalse(server.storage.path.exists())

    def test_resource_observer_failure_does_not_skip_process_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            server = Service(self.c, 'aisbench_n40_performance', tmp)
            server.process = Mock()
            with patch.object(server, 'capture_resources', side_effect=RuntimeError('capture failed')):
                with self.assertRaisesRegex(RuntimeError, 'capture failed'):
                    server.stop()
            server.process.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
