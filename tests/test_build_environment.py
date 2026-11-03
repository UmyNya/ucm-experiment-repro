"""Resolve build tools through real subprocesses without installing or building UCM."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from configuration import load
from prepare_environment import build_environment


class BuildEnvironmentTest(unittest.TestCase):
    def tool(self, directory, name, result):
        directory.mkdir(parents=True, exist_ok=True)
        path = directory/name
        path.write_text(f'#!{sys.executable}\nprint({result!r})\n')
        path.chmod(0o700)

    def test_private_cmake_runs_when_not_on_parent_path_and_keeps_system_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run with spaces'
            inherited = Path(tmp)/'inherited-bin'
            self.tool(run/'venv/bin', 'cmake', 'private-cmake')
            self.tool(inherited, 'build-tool-fixture', 'inherited-tool')
            with patch.dict(os.environ, {'PATH': str(inherited)}):
                parent = dict(os.environ)
                with self.assertRaises(FileNotFoundError):
                    subprocess.check_output(['cmake', '--version'], env=parent)
                env = build_environment(load(ROOT/'config/experiment.json'), run)
                self.assertEqual(subprocess.check_output(['cmake', '--version'], env=env, text=True).strip(),
                                 'private-cmake')
                self.assertEqual(subprocess.check_output(['build-tool-fixture'], env=env, text=True).strip(),
                                 'inherited-tool')
                self.assertEqual(dict(os.environ), parent)

    def test_private_cmake_takes_precedence_over_an_inherited_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)/'run'
            inherited = Path(tmp)/'inherited-bin'
            self.tool(run/'venv/bin', 'cmake', 'private-cmake')
            self.tool(inherited, 'cmake', 'inherited-cmake')
            with patch.dict(os.environ, {'PATH': str(inherited)}):
                config = load(ROOT/'config/experiment.json')
                for platform in ['cuda', 'ascend']:
                    with self.subTest(platform=platform):
                        config['platform']['kind'] = platform
                        env = build_environment(config, run)
                        self.assertEqual(subprocess.check_output(['cmake', '--version'], env=env, text=True).strip(),
                                         'private-cmake')


if __name__ == '__main__':
    unittest.main()
