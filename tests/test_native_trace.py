"""Compile/run only the portable CPU trace recorder; no UCM build or device access."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from processes import OwnedProcess


@unittest.skipUnless(shutil.which('g++'), 'CPU C++ compiler is not installed')
class NativeTraceTest(unittest.TestCase):
    def test_disabled_ranges_async_end_once_json_escaping_and_capture_switch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            source = run/'test.cc'
            source.write_text(r'''
#include "profile_nvtx.h"
#include <thread>
int main(int argc, char** argv) {
    setenv("BUNDLE_NATIVE_TRACE_DIR", argv[1], 1);
    { UcmProfile::Range no_capture("absent"); }
    setenv("UCM_PIPELINE_CAPTURE", "1", 1);
    { UcmProfile::Range range("quote\"\nline"); }
    auto span = UcmProfile::Start("async");
    std::thread a([span]{ span->End(); });
    std::thread b([span]{ span->End(); });
    a.join(); b.join(); span.reset();
    UcmProfile::Mark("point");
    setenv("BUNDLE_NATIVE_TRACE_DIR", argv[2], 1);
    UcmProfile::Mark("second_capture");
}
''')
            for name in ['first','second']:
                (run/name).mkdir()
            command = ['g++', '-std=c++17', '-pthread', '-I'+str(ROOT/'native'), str(source), '-o', str(run/'test')]
            with OwnedProcess(command, cwd=run, environment=os.environ.copy(), log_path=run/'compile.log', name='compiler') as p:
                self.assertEqual(p.wait(60), 0, (run/'compile.log').read_text())
            with OwnedProcess([str(run/'test'), str(run/'first'), str(run/'second')], cwd=run,
                              environment=os.environ.copy(), log_path=run/'execute.log', name='fixture') as p:
                self.assertEqual(p.wait(10), 0)
            first = [json.loads(x) for f in (run/'first').glob('*.jsonl') for x in f.read_text().splitlines()]
            second = [json.loads(x) for f in (run/'second').glob('*.jsonl') for x in f.read_text().splitlines()]
            self.assertEqual([x['label'] for x in first], ['quote"\nline', 'async', 'point'])
            self.assertEqual([x['label'] for x in second], ['second_capture'])
            self.assertTrue(all(x['end_ns'] >= x['start_ns'] and x['pid'] > 0 and x['tid'] > 0 for x in first+second))
            self.assertTrue(first[-1]['point'])


if __name__ == '__main__':
    unittest.main()
