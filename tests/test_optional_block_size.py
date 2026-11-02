"""No engine block-size defaults: use explicit configuration or worker evidence."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from block_size import input_block_size, observed_block_size
from capacity_validation import calibration_record, validate_workers
from common import read_json, save_json
from configuration import compile_plan, load, validate
from generate_report import generate
from report_data import collect
from run_experiment import execute
from workloads import ais_config
import run_client
import test_hbm_auto
import test_service_contracts


class OptionalBlockSizeTest(unittest.TestCase):
    def config(self):
        return load(ROOT/'config/experiment.json',
                    ['server.block_size=null', 'capacity.hbm.mode="auto"', 'platform.kind="cuda"'])

    def plan(self, config=None, run=None):
        return compile_plan(config or self.config(), 'aisbench_n40_performance', run or ROOT/'runs/block-fixture')

    def worker(self, block=64):
        row = test_service_contracts.ServiceContractsTest().worker()
        row['capacity_layout']['block_tokens'] = block
        row['hbm_tokens'] = row['hbm_blocks'] * block
        row['l1_ordinary_tokens'] = (row['l1_nodes'] - 1024) * block
        return row

    def test_missing_and_null_delegate_to_engine_on_both_platforms_all_profiles(self):
        for omitted in [True, False]:
            for backend in ['cuda', 'ascend']:
                c = self.config()
                c['platform']['kind'] = backend
                if omitted:
                    del c['server']['block_size']
                validate(c)
                for profile in c['profiles']:
                    with self.subTest(omitted=omitted, backend=backend, profile=profile):
                        plan = compile_plan(c, profile, ROOT/'runs/block-fixture')
                        self.assertNotIn('block-size', plan['vllm_configuration'])
                        self.assertNotIn('--block-size', plan['server_command'])
                        self.assertIsNone(plan['configuration']['server']['block_size'])
                        self.assertIsNone(plan['capacity']['calculation_block_size'])
                        self.assertIsNone(plan['model_identity']['block_size'])

    def test_missing_option_loads_from_native_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = self.config()
            server = c.pop('server')
            del server['block_size']
            c['config_files'] = {'server': 'vllm.json'}
            save_json(Path(tmp)/'vllm.json', server)
            save_json(Path(tmp)/'experiment.json', c)
            loaded = load(Path(tmp)/'experiment.json')
            self.assertIsNone(loaded['server']['block_size'])
            self.assertNotIn('block-size', self.plan(loaded)['vllm_configuration'])

    def test_invalid_explicit_values_rejected_and_positive_values_preserved(self):
        for value in ['0', '-1', 'true', '64.5', '"64"']:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'server.block_size'):
                load(ROOT/'config/experiment.json', ['server.block_size='+value])
        c = self.config()
        c['server']['block_size'] = 32
        self.assertEqual(self.plan(c)['vllm_configuration']['block-size'], 32)

    def test_physical_hbm_does_not_need_block_size(self):
        c = self.config()
        c['capacity']['hbm']['mode'] = 'physical'
        plan = self.plan(c)
        self.assertEqual(plan['capacity']['kv_cache_memory_bytes'], 5372160000)
        self.assertTrue(validate_workers(plan, [self.worker(128)])['passed'])

    def test_tokens_bytes_per_token_preserved_and_alignment_checked_after_start(self):
        c = self.config()
        c['capacity']['hbm']['mode'] = 'tokens'
        plan = self.plan(c)
        self.assertEqual(plan['capacity']['kv_cache_memory_bytes'], 5372160000)
        self.assertTrue(validate_workers(plan, [self.worker()])['passed'])
        c['capacity']['hbm']['tokens'] += 1
        check = validate_workers(self.plan(c), [self.worker()])
        self.assertFalse(check['passed'])
        self.assertIn('HBM token target is not aligned to the observed block size', check['errors'])

    def test_l1_layout_cannot_be_guessed_without_a_block_size(self):
        c = self.config()
        c['capacity']['l1']['mode'] = 'tokens'
        with self.assertRaisesRegex(ValueError, 'use physical L1 or calibrate'):
            self.plan(c)

    def test_calibration_records_actual_size_and_converts_without_setting_engine_option(self):
        c = self.config()
        plan = self.plan(c)
        record = calibration_record(plan, [self.worker()])
        self.assertIsNone(plan['model_identity']['block_size'])
        self.assertEqual(record['identity']['block_size'], 64)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calibration.json'
            save_json(path, record)
            c['capacity']['calibration_file'] = str(path)
            for tier in ['hbm', 'l1']:
                c['capacity'][tier].update(mode='tokens', calculation={'method': 'calibration'})
            calibrated = self.plan(c)
            self.assertNotIn('block-size', calibrated['vllm_configuration'])
            self.assertEqual(calibrated['capacity']['calculation_block_size'], 64)
            self.assertEqual(calibrated['capacity']['kv_cache_memory_bytes'], 5372160000)
            self.assertEqual(calibrated['capacity']['cache_buffer_capacity_gb'], 19)
            self.assertTrue(validate_workers(calibrated, [self.worker()])['passed'])
            check = validate_workers(calibrated, [self.worker(128)])
            self.assertIn('Observed token block size differs from the capacity calculation/calibration', check['errors'])
            c['server']['kv_cache_dtype'] = 'different'
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                self.plan(c)

    def test_calibration_rejects_inconsistent_or_missing_recorded_block(self):
        c = self.config()
        record = calibration_record(self.plan(c), [self.worker()])
        c['capacity']['l1'].update(mode='tokens', calculation={'method': 'calibration'})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'calibration.json'
            c['capacity']['calibration_file'] = str(path)
            for value in [None, 0, True, 128]:
                with self.subTest(value=value):
                    record['identity']['block_size'] = value
                    save_json(path, record)
                    with self.assertRaises(ValueError):
                        self.plan(c)

    def test_workers_must_report_one_positive_block_size(self):
        for rows in [[], [{}], [self.worker(0)], [self.worker(True)], [self.worker(64), self.worker(128)]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                observed_block_size(rows)
        c = self.config()
        c['server']['block_size'] = 64
        self.assertIn('Observed token block size differs from config',
                      validate_workers(self.plan(c), [self.worker(128)])['errors'])

    def test_ais_inputs_require_observation_and_prefer_actual_size(self):
        plan = self.plan()
        with self.assertRaisesRegex(ValueError, 'No default is assumed'):
            ais_config(plan)
        plan['configuration']['server']['block_size'] = 64
        self.assertEqual(input_block_size(plan), 64)
        plan['observed_block_size'] = 128
        self.assertEqual(input_block_size(plan), 128)

    def campaign(self, root, sizes, fail_preparation=False):
        c = self.config()
        c['execution'].update(repetitions=2, calibration_repetitions=1)
        events = []
        blocks = iter(sizes)
        class FixtureService:
            def __init__(self, config, profile, run, compile_cache):
                self.plan = compile_plan(config, profile, run)
                self.run = run
            def __enter__(self):
                events.append('start')
                self.plan['observed_block_size'] = next(blocks)
                self.plan['model_identity']['block_size'] = self.plan['observed_block_size']
                return self
            def __exit__(self, *_):
                events.append('stop')
                save_json(self.run/'resource-cleanup.json', {'released': True})
        def prepare(plan, run):
            events.append('inputs')
            self.assertEqual(input_block_size(plan), sizes[0])
            self.assertIsNone(plan['configuration']['server']['block_size'])
            if fail_preparation:
                raise RuntimeError('synthetic preparation failure')
            save_json(run/'inputs/PREPARATION.json', {'input_fingerprint': 'exact-fixture'})
        def client(plan, run, output, inputs, observe):
            events.append('client')
            self.assertEqual(input_block_size(plan), sizes[0])
            save_json(output/'performance.json', {'passed': True, 'requests': plan['request_count']})
            save_json(output/'compilation-check.json', {'passed': True})
        error = None
        try:
            execute(c, 'aisbench_n40_performance', root, service_factory=FixtureService,
                    prepare_inputs=prepare, run_client=client)
        except (ValueError, RuntimeError) as caught:
            error = caught
        return events, error

    def test_campaign_prepares_once_after_first_ready_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events, error = self.campaign(root, [128, 128, 128])
            self.assertIsNone(error)
            self.assertEqual(events, ['start', 'inputs', 'client', 'stop', 'start', 'client', 'stop', 'start', 'client', 'stop'])
            self.assertEqual(read_json(root/'plan.json')['observed_block_size'], 128)
            self.assertEqual(read_json(root/'status.json')['formal_repetitions_completed'], 2)

    def test_campaign_rejects_changed_alignment_and_cleans_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, error = self.campaign(Path(tmp), [128, 64, 64])
            self.assertRegex(str(error), 'block_size changed')
            self.assertEqual(events, ['start', 'inputs', 'client', 'stop', 'start', 'stop'])
            self.assertEqual(read_json(Path(tmp)/'status.json')['status'], 'failed')

    def test_campaign_preparation_failure_cleans_first_service(self):
        with tempfile.TemporaryDirectory() as tmp:
            events, error = self.campaign(Path(tmp), [128], fail_preparation=True)
            self.assertRegex(str(error), 'preparation failure')
            self.assertEqual(events, ['start', 'inputs', 'stop'])

    def test_standalone_client_observation_attachment_and_missing_evidence(self):
        for mode in ['attached', 'observer', 'unknown']:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                c = self.config()
                argv = ['run_aisbench.py']
                if mode == 'attached':
                    service = root/'service'
                    plan = self.plan(c, service)
                    plan['observed_block_size'] = 128
                    plan['model_identity']['block_size'] = 128
                    save_json(service/'plan.json', plan)
                    save_json(service/'service-status.json', {'status': 'ready', 'supervisor': {}, 'base_url': plan['base_url']})
                    argv += ['--service-run', str(service)]
                elif mode == 'observer':
                    argv += ['--base-url', 'http://127.0.0.1:8000', '--observe']
                def prepare(plan, run):
                    self.assertEqual(input_block_size(plan), 128)
                    self.assertIsNone(plan['configuration']['server']['block_size'])
                with patch.object(sys, 'argv', argv), patch.object(run_client, 'load', return_value=c), \
                     patch.object(run_client, 'new_run', return_value=root/'client'), \
                     patch.object(run_client, 'same_process', return_value=True), \
                     patch.object(run_client, 'rpc', return_value=[self.worker(128)]) as rpc, \
                     patch.object(run_client, 'prepare', side_effect=prepare), \
                     patch.object(run_client, 'attempt') as attempt:
                    if mode == 'unknown':
                        with self.assertRaisesRegex(ValueError, 'No default is assumed'):
                            run_client.main('aisbench')
                        attempt.assert_not_called()
                    else:
                        run_client.main('aisbench')
                        self.assertEqual(attempt.call_count, 1)
                        self.assertEqual(read_json(root/'client/plan.json')['observed_block_size'], 128)
                    self.assertEqual(rpc.call_count, int(mode == 'observer'))

    def test_reports_distinguish_observed_blocks_and_exclude_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for index, block in enumerate([64, 64, 128, None]):
                run = root/str(index)
                test_hbm_auto.HbmAutoTest().report_run(run, 5000)
                plan = read_json(run/'plan.json')
                plan['configuration']['server']['block_size'] = None
                if block is not None:
                    plan['observed_block_size'] = block
                save_json(run/'plan.json', plan)
                save_json(run/'actual-service-plan.json', plan)
                rows.append(collect(run)[0])
            self.assertEqual(rows[0]['configuration_group'], rows[1]['configuration_group'])
            self.assertNotEqual(rows[0]['configuration_group'], rows[2]['configuration_group'])
            self.assertTrue(all(row['valid'] for row in rows[:3]))
            self.assertFalse(rows[3]['valid'])
            self.assertTrue(any('block_size' in reason for reason in rows[3]['excluded_reasons']))
            with patch('generate_report.render', return_value=[]):
                report = generate([root/'2'], output_root=root/'reports', include_validation=True)
            self.assertIn('未指定，由引擎决定；worker 实测：128 tokens/block', (report/'REPORT.md').read_text())


if __name__ == '__main__':
    unittest.main()
