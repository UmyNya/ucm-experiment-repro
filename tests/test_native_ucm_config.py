import copy
import importlib.util
import logging
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import new_run, read_json, save_json, sha256
from configuration import load, compile_plan, model_identity
from service import Service
from ucm_configuration import store_config


class NativeUcmConfigTest(unittest.TestCase):
    def setUp(self):
        self.c = load(ROOT / 'config/experiment.json')

    def legacy(self):
        c = copy.deepcopy(self.c)
        native = c['ucm']
        c['ucm'] = {**c.pop('ucm_runtime'),
                    'launch': {k: v for k, v in native.items() if k != 'ucm_connectors'},
                    'store': native['ucm_connectors'][0]['ucm_connector_config']}
        return c

    def reload(self, c):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'config.json'
            save_json(path, c)
            return load(path)

    def test_native_file_contains_no_experiment_metadata(self):
        raw = read_json(ROOT / 'config/ucm.json')
        self.assertFalse({'launch', 'store', 'enabled', 'source_ref'} & raw.keys())
        self.assertTrue(raw['enable_early_prefetch'])
        self.assertEqual(raw['ucm_connectors'][0]['ucm_connector_name'], 'UcmPipelineStore')
        self.assertEqual(store_config(self.c)['cache_dump_capacity_policy'], 'wait')
        self.assertEqual(self.c['ucm_runtime'], {'enabled': True, 'source_ref': 'HEAD'})

    def test_legacy_and_native_configs_have_identical_plans_on_both_backends(self):
        legacy = self.reload(self.legacy())
        self.assertEqual(legacy['ucm'], self.c['ucm'])
        self.assertEqual(model_identity(legacy), model_identity(self.c))
        for platform in ['cuda', 'ascend']:
            legacy['platform']['kind'] = self.c['platform']['kind'] = platform
            for profile in self.c['profiles']:
                run = ROOT / 'runs/native-ucm-plan'
                a, b = [compile_plan(c, profile, run) for c in [legacy, self.c]]
                for key in ['ucm_configuration', 'server_command', 'server_environment', 'capacity', 'model_identity']:
                    self.assertEqual(a[key], b[key], (platform, profile, key))

    def test_native_paths_legacy_aliases_and_nested_extensions(self):
        native = load(ROOT / 'config/experiment.json', [
            'ucm.enable_early_prefetch=false',
            'ucm.ucm_connectors.0.ucm_connector_config.cache_dump_capacity_policy="spill"',
            'ucm.ucm_connectors.0.ucm_connector_config.store_health.enabled=false',
            'ucm.ucm_connectors.0.ucm_connector_config.posix_packed_layer_bytes=[4096,8192]',
            'ucm.metrics_config_path="/config/metrics.yaml"',
            'ucm_runtime.source_ref="test-ref"'])
        old = load(ROOT / 'config/experiment.json', [
            'ucm.launch.enable_early_prefetch=false',
            'ucm.store.cache_dump_capacity_policy="spill"',
            'ucm.store.store_health.enabled=false',
            'ucm.store.posix_packed_layer_bytes=[4096,8192]',
            'ucm.launch.metrics_config_path="/config/metrics.yaml"',
            'ucm.source_ref="test-ref"'])
        self.assertEqual(old['ucm'], native['ucm'])
        self.assertEqual(old['ucm_runtime'], native['ucm_runtime'])
        plan = compile_plan(native, 'aisbench_n40_performance', ROOT / 'runs/native-ucm-options')
        result = plan['ucm_configuration']
        store = result['ucm_connectors'][0]['ucm_connector_config']
        self.assertFalse(result['enable_early_prefetch'])
        self.assertEqual(result['metrics_config_path'], '/config/metrics.yaml')
        self.assertEqual(store['store_health'], {'enabled': False})
        self.assertEqual(store['posix_packed_layer_bytes'], [4096, 8192])

    def test_mixed_shapes_and_duplicate_metadata_are_rejected(self):
        c = self.legacy()
        c['ucm_runtime'] = {'source_ref': 'another-ref'}
        with self.assertRaisesRegex(ValueError, 'never both'):
            self.reload(c)
        c = self.legacy()
        c['ucm']['use_layerwise'] = True
        with self.assertRaisesRegex(ValueError, 'mix legacy'):
            self.reload(c)

    def test_capacity_path_overrides_and_unsupported_connector_shapes_fail(self):
        for name, value in [('cache_buffer_capacity_gb', '128'), ('posix_capacity_gb', '512'),
                            ('storage_backends', '"/shared-cache"')]:
            for prefix in ['ucm.store.', 'ucm.ucm_connectors.0.ucm_connector_config.']:
                with self.subTest(name=name, prefix=prefix), self.assertRaisesRegex(ValueError, 'through'):
                    load(ROOT / 'config/experiment.json', [prefix + name + '=' + value])
        for override in ['ucm.ucm_connectors=[]', 'ucm.ucm_connectors.1.ucm_connector_name="Other"',
                         'ucm.ucm_connectors.0.ucm_connector_name="Other"',
                         'ucm.ucm_connectors.0.ucm_connector_config.store_pipeline="Posix"',
                         'ucm_runtime.source_reff="HEAD"', 'ucm_runtime.enabled=1',
                         'ucm.ucm_connectors.0.ucm_connector_config=false']:
            with self.subTest(override=override), self.assertRaises(ValueError):
                load(ROOT / 'config/experiment.json', [override])

    def test_effective_file_and_snapshot_preserve_source_and_fill_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['paths']['output_root'] = tmp
            c['capacity']['l1']['gib'] = 128
            c['capacity']['l2']['gib'] = 512
            run = new_run(c, 'aisbench_n40_performance')
            service = Service(c, 'aisbench_n40_performance', run)
            native = read_json(run / 'ucm.json')
            store = native['ucm_connectors'][0]['ucm_connector_config']
            self.assertEqual(store['cache_buffer_capacity_gb'], 128)
            self.assertEqual(store['posix_capacity_gb'], 512)
            self.assertEqual(store['storage_backends'], service.plan['l2_path'])
            self.assertNotIn('posix_packed_layer_bytes', store)
            self.assertFalse({'launch', 'store', 'enabled', 'source_ref'} & native.keys())
            self.assertEqual(native, service.plan['ucm_configuration'])
            self.assertEqual(sha256(run / 'config-sources/ucm.json'), sha256(ROOT / 'config/ucm.json'))
            self.assertIsNone(service.process)

    def test_real_ucm_config_reader_accepts_generated_json_without_engine_import(self):
        root = os.environ.get('UCM_SOURCE_ROOT') or self.c['paths']['ucm_repository']
        source = Path(root) / 'ucm/utils.py'
        if not source.is_file():
            if os.environ.get('UCM_SOURCE_ROOT'):
                self.fail('UCM_SOURCE_ROOT does not contain ucm/utils.py')
            self.skipTest('Set UCM_SOURCE_ROOT to test the actual UCM Config reader')
        if importlib.util.find_spec('yaml') is None:
            self.skipTest('The actual UCM Config reader requires PyYAML in the test environment')
        before = sha256(source)
        logger = ModuleType('ucm.logger')
        logger.init_logger = logging.getLogger
        with patch.dict(sys.modules, {'ucm.logger': logger}):
            scope = {}
            exec(compile(source.read_text(encoding='utf-8-sig'), str(source), 'exec'), scope)
            with tempfile.TemporaryDirectory() as tmp:
                service = Service(self.c, 'aisbench_n40_performance', Path(tmp))
                config = SimpleNamespace(kv_connector_extra_config={'UCM_CONFIG_FILE': str(Path(tmp) / 'ucm.json')})
                self.assertEqual(scope['Config'](config).get_config(), service.plan['ucm_configuration'])
                self.assertIsNone(service.process)
        self.assertEqual(before, sha256(source))


if __name__ == '__main__':
    unittest.main()
