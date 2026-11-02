import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import new_run, read_json, save_json
from configuration import compile_plan, load, model_identity
from service import Service


class NativeVllmConfigTest(unittest.TestCase):
    def setUp(self):
        self.c = load(ROOT / 'config/experiment.json')

    def plan(self, config=None, run=None):
        return compile_plan(config or self.c, 'aisbench_n40_performance', run or ROOT / 'runs/native-plan')

    def load_inline(self, config):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'experiment.json'
            save_json(path, config)
            return load(path)

    def test_native_scalar_nested_list_and_boolean_options(self):
        c = load(ROOT / 'config/experiment.json', [
            'server.api_server_count=1', 'server.quantization="ascend"',
            'server.enable_auto_tool_choice=true', 'server.reasoning_parser="glm47"',
            'server.tool_call_parser="glm47"', 'server.attention_config.indexer_kv_dtype="fp8"',
            'server.additional_config={"enable_flashcomm1":true,"enable_cpu_binding":true}',
            'server.allowed_origins=["http://localhost:3000", "http://localhost:8080"]',
            'server.enable-prefix-caching=false', 'server.optional_example=null',
        ])
        plan = self.plan(c)
        native = plan['vllm_configuration']
        self.assertEqual(native['api-server-count'], 1)
        self.assertEqual(native['quantization'], 'ascend')
        self.assertTrue(native['enable-auto-tool-choice'])
        self.assertEqual(native['tool-call-parser'], 'glm47')
        self.assertEqual(native['reasoning-parser'], 'glm47')
        self.assertEqual(native['attention-config'], {'indexer_kv_dtype': 'fp8'})
        self.assertEqual(native['additional-config'], {'enable_flashcomm1': True, 'enable_cpu_binding': True})
        self.assertEqual(native['allowed-origins'], ['http://localhost:3000', 'http://localhost:8080'])
        self.assertNotIn('enable-prefix-caching', native)
        self.assertTrue(native['no-enable-prefix-caching'])
        self.assertNotIn('optional-example', native)
        self.assertFalse({'environment', 'platform-environment', 'extra-args'} & native.keys())
        self.assertEqual(plan['server_command'], [c['paths']['server_python'], '-B', '-m',
            'vllm.entrypoints.cli.main', 'serve', c['paths']['model'], '--config', plan['vllm_config_file']])

    def test_old_inline_metadata_and_new_files_generate_same_plan(self):
        old = self.load_inline(copy.deepcopy(self.c))
        for key in ['server_command', 'server_environment', 'vllm_configuration', 'model_identity', 'capacity']:
            self.assertEqual(self.plan()[key], self.plan(old)[key], key)
        old['server']['extra_args'] = ['--quantization', 'ascend']
        old = self.load_inline(old)
        self.assertEqual(self.plan(old)['server_command'][-2:], ['--quantization', 'ascend'])

    def test_duplicate_environment_source_and_option_aliases_fail(self):
        c = copy.deepcopy(self.c)
        c['config_files'] = {'server_env': str(ROOT / 'config/vllm-env.json')}
        with self.assertRaisesRegex(ValueError, 'never both'):
            self.load_inline(c)
        c = copy.deepcopy(self.c)
        c['server']['tensor-parallel-size'] = 2
        with self.assertRaisesRegex(ValueError, 'Duplicate vLLM option alias'):
            self.load_inline(c)
        del c['server']['tensor_parallel_size']
        c['server']['tensor-parallel-size'] = 1
        self.assertEqual(self.load_inline(c)['server']['tensor_parallel_size'], 1)

    def test_managed_values_cannot_be_overridden_by_native_or_legacy_aliases(self):
        for key in ['model', 'kv_cache_memory_bytes', 'kv-transfer-config', 'worker_extension_cls', 'config']:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'through'):
                load(ROOT / 'config/experiment.json', [f'server.{key}="override"'])
        for flag in ['--port=1', '--tensor_parallel_size=8', '--no-enable-prefix-caching',
                     '--kv-transfer-config.kv_connector=Other', '--config=other.yaml', '-tp',
                     '--kv-cache-memory-b=1', '--no-async-sched']:
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                load(ROOT / 'config/experiment.json', ['server.extra_args=' + json.dumps([flag])])
        with self.assertRaisesRegex(ValueError, 'instead of'):
            load(ROOT / 'config/experiment.json', ['server.no_enable_prefix_caching=true'])

    def test_effective_file_and_source_snapshots_are_written_without_starting_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = copy.deepcopy(self.c)
            c['paths']['output_root'] = tmp
            c['server']['platform_environment']['ascend']['HCCL_BUFFSIZE'] = '400'
            c['server']['platform_environment']['cuda']['HCCL_BUFFSIZE'] = 'wrong-platform'
            run = new_run(c, 'aisbench_n40_performance')
            service = Service(c, 'aisbench_n40_performance', run)
            native = read_json(run / 'vllm-effective.yaml')
            self.assertEqual(native, service.plan['vllm_configuration'])
            self.assertEqual(native['model'], c['paths']['model'])
            self.assertEqual(native['max-model-len'], c['profiles']['aisbench_n40_performance']['max_model_len'])
            self.assertEqual(native['kv-transfer-config']['kv_connector_extra_config']['UCM_CONFIG_FILE'], str(run / 'ucm.json'))
            self.assertEqual(native['worker-extension-cls'], 'runtime_observer.BundleObserver')
            self.assertTrue(native['no-async-scheduling'])
            self.assertEqual(service.plan['server_environment']['HCCL_BUFFSIZE'], '400')
            self.assertEqual(service.plan['server_environment']['ENABLE_UCM_PATCH'], '1')
            self.assertEqual(read_json(run / 'config-sources/server_env.json'), read_json(ROOT / 'config/vllm-env.json'))
            self.assertIsNone(service.process)

    def test_extension_options_invalidate_old_calibration(self):
        baseline = model_identity(self.c)
        self.assertNotIn('vllm_extensions', baseline)
        for option in [{'quantization': 'ascend'}, {'attention_config': {'indexer_kv_dtype': 'fp8'}},
                       {'extra_args': ['--dtype', 'float16']}]:
            c = copy.deepcopy(self.c)
            c['server'].update(option)
            self.assertNotEqual(baseline, model_identity(c))
            with tempfile.TemporaryDirectory() as tmp:
                calibration = Path(tmp) / 'calibration.json'
                save_json(calibration, {'identity': baseline, 'workers': []})
                c['capacity']['calibration_file'] = str(calibration)
                c['capacity']['hbm']['calculation']['method'] = 'calibration'
                with self.assertRaisesRegex(ValueError, 'identity differs'):
                    self.plan(c)


if __name__ == '__main__':
    unittest.main()
