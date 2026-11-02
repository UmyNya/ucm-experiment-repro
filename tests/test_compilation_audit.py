import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from compilation_audit import inventory, evaluate


class CompilationAuditTest(unittest.TestCase):
    def setUp(self):
        self.before = [{'pid': 1, 'events': [], 'triton_package_present': True,
                        'installed': {'dynamo': True, 'triton': True, 'device_graph_capture': True}}]
        self.cache = {'files': {}}

    def check(self, after, before_cache=None, after_cache=None, log=''):
        return evaluate(self.before, after, before_cache or self.cache, after_cache or self.cache,
                        log, graph_enabled=True, platform='cuda')

    def test_cache_hit_warning_is_not_codegen(self):
        after = copy.deepcopy(self.before)
        after[0]['events'] = [{'kind': 'triton_compile', 'cache_hit': True}]
        result = self.check(after, log='Triton kernel JIT compilation during inference: cached kernel')
        self.assertTrue(result['passed'])
        self.assertEqual(result['workers'][0]['triton_disk_cache_hits'], 1)

    def test_codegen_unknown_capture_and_missing_hook_reject(self):
        for event in [{'kind': 'triton_compile', 'cache_hit': False}, {'kind': 'triton_compile', 'cache_hit': None},
                      {'kind': 'device_graph_capture'}, {'kind': 'dynamo_compile_start'}]:
            after = copy.deepcopy(self.before)
            after[0]['events'] = [event]
            self.assertFalse(self.check(after)['passed'])
        after = copy.deepcopy(self.before)
        after[0]['installed']['triton'] = False
        self.assertFalse(self.check(after)['passed'])

    def test_changed_binary_and_stage_logs_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp)/'kernel.cubin'
            file.write_bytes(b'first')
            before = inventory([tmp])
            file.write_bytes(b'second')
            after = inventory([tmp])
            self.assertFalse(self.check(copy.deepcopy(self.before), before, after)['passed'])
        self.assertFalse(self.check(copy.deepcopy(self.before), log='torch.compile took 0.1s')['passed'])


if __name__ == '__main__':
    unittest.main()
