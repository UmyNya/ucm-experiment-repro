import copy
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from common import read_json, save_json
from configuration import load, compile_plan
from run_experiment import execute


class CampaignTest(unittest.TestCase):
    def test_each_profile_preserves_one_configuration_full_counts_and_fresh_services(self):
        for profile in load(ROOT/'config/experiment.json')['profiles']:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                c = load(ROOT/'config/experiment.json')
                c['execution'].update(repetitions=2, calibration_repetitions=1)
                events, services = [], []
                class FixtureService:
                    def __init__(self, config, selected, run, compile_cache):
                        self.plan = compile_plan(config, selected, run)
                        self.run = Path(run)
                        self.compile_cache = compile_cache
                        services.append(self)
                    def __enter__(self):
                        events.append(('start', self.run))
                        return self
                    def __exit__(self, *_):
                        events.append(('stop', self.run))
                        save_json(self.run/'resource-cleanup.json', {'released': True})
                def prepare(plan, run):
                    events.append(('inputs', plan['request_count']))
                    save_json(run/'inputs/PREPARATION.json', {'input_fingerprint': 'exact-fixture'})
                def client(plan, run, output, inputs, observe):
                    self.assertTrue(observe)
                    self.assertEqual(plan['configuration']['load'], c['load'])
                    save_json(output/'performance.json', {'passed': True, 'requests': plan['request_count']})
                    save_json(output/'compilation-check.json', {'passed': True})
                    events.append(('client', plan['measurement_phase'], plan['request_count'], plan['mode']))
                state = execute(c, profile, Path(tmp), service_factory=FixtureService, prepare_inputs=prepare, run_client=client,
                                analyze_pipeline=lambda plan,output: {'passed':True})
                self.assertEqual(state['formal_repetitions_completed'], 2)
                self.assertEqual([x[0] for x in events], ['inputs', 'start', 'client', 'stop', 'start', 'client', 'stop', 'start', 'client', 'stop'])
                self.assertEqual(len({s.plan['l2_path'] for s in services}), 3)
                self.assertEqual(len({s.compile_cache for s in services}), 1)
                calls = [x for x in events if x[0]=='client']
                self.assertEqual([x[1] for x in calls], ['calibration', 'formal', 'formal'])
                self.assertEqual(len({x[2] for x in calls}), 1)
                self.assertEqual(calls[0][3], 'performance')
                self.assertEqual(calls[1][3], c['profiles'][profile]['mode'])

    def test_client_failure_stops_service_and_prevents_next_repetition(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = load(ROOT/'config/experiment.json')
            c['execution']['repetitions'] = 3
            closed = []
            class Fixture:
                def __init__(self, config, profile, run, compile_cache):
                    self.plan = compile_plan(config, profile, run)
                def __enter__(self):
                    return self
                def __exit__(self, *_):
                    closed.append(True)
            def prepare(plan, run):
                save_json(run/'inputs/PREPARATION.json', {'input_fingerprint': 'fixture'})
            def fail(*args, **kwargs):
                raise RuntimeError('synthetic client failure')
            with self.assertRaisesRegex(RuntimeError, 'synthetic'):
                execute(c, 'swe_w24_n192_performance', Path(tmp), service_factory=Fixture, prepare_inputs=prepare, run_client=fail)
            self.assertEqual(closed, [True])
            state = read_json(Path(tmp)/'status.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(len(state['stages']), 1)


if __name__ == '__main__':
    unittest.main()
