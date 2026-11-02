import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from processes import OwnedProcess, identity, same_process


class ProcessTest(unittest.TestCase):
    def test_timeout_cleans_child_but_preserves_unrelated_process(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
            command = [sys.executable, '-c',
                       'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"]); time.sleep(60)']
            try:
                owned = OwnedProcess(command, cwd=root, environment=os.environ.copy(), log_path=root/'process.log', shutdown_timeout=1)
                with self.assertRaises(subprocess.TimeoutExpired):
                    owned.wait(1)
                self.assertTrue(owned.cleanup['released'])
                self.assertGreaterEqual(len(owned.known), 2)
                self.assertIsNone(unrelated.poll())
                self.assertTrue(all(not (r := same_process(x)) or r['state'] == 'Z' for x in owned.known.values()))
            finally:
                unrelated.terminate()
                unrelated.wait(timeout=5)

    def test_normal_completion_has_cleanup_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with OwnedProcess([sys.executable, '-c', 'print("ok")'], cwd=root,
                              environment=os.environ.copy(), log_path=root/'log') as process:
                self.assertEqual(process.wait(5), 0)
            self.assertTrue(process.cleanup['released'])
            self.assertTrue((root/'process-owner.json').exists())
            self.assertTrue((root/'process-cleanup.json').exists())

    def test_identity_requires_matching_start_ticks(self):
        record = identity(os.getpid())
        record['start_ticks'] += 1
        self.assertIsNone(same_process(record))

    def test_launcher_exit_does_not_abandon_its_child(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            command = [sys.executable, '-c',
                       'import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time;time.sleep(60)"])']
            with OwnedProcess(command, cwd=root, environment=os.environ.copy(), log_path=root/'log', shutdown_timeout=1) as process:
                time.sleep(0.3)
                self.assertEqual(process.wait(5), 0)
                self.assertGreaterEqual(len(process.known), 2)
            self.assertTrue(process.cleanup['released'])
