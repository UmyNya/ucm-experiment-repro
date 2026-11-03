"""Resolve build tools through real subprocesses without installing or building UCM."""
import contextlib
import errno
import io
import json
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
import prepare_environment
from prepare_environment import build_environment, validate_wheelhouse


class BuildEnvironmentTest(unittest.TestCase):
    def test_wheelhouse_preflight_rejects_missing_file_empty_and_unreadable_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file = root/'not-a-directory'
            file.write_text('fixture')
            empty = root/'empty'
            empty.mkdir()
            config = load(ROOT/'config/experiment.json')
            config['build']['install_dependencies'] = True
            for directory, message in [(root/'missing', 'Cannot read build.wheelhouse'),
                                       (file, 'Cannot read build.wheelhouse'),
                                       (empty, 'directory is empty')]:
                with self.subTest(directory=directory.name):
                    config['build']['wheelhouse'] = str(directory)
                    with self.assertRaisesRegex(ValueError, message):
                        validate_wheelhouse(config)
            config['build']['wheelhouse'] = str(root)
            with patch('prepare_environment.os.scandir', side_effect=PermissionError(errno.EACCES, 'Permission denied')):
                with self.assertRaisesRegex(ValueError, 'Cannot read build.wheelhouse'):
                    validate_wheelhouse(config)

    def test_wheelhouse_preflight_checks_local_source_without_resolving_packages(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'fixture.whl').write_bytes(b'fixture')
            config = load(ROOT/'config/experiment.json')
            config['build'].update(install_dependencies=True, wheelhouse=str(root))
            validate_wheelhouse(config)
            config['build']['wheelhouse'] = str(root/'not-downloaded-yet')
            validate_wheelhouse(config, check_directory=False)
            config['build']['install_dependencies'] = False
            validate_wheelhouse(config)
            config['build'].update(install_dependencies=True, wheelhouse=None)
            for check in [True, False]:
                with self.subTest(check_directory=check), self.assertRaisesRegex(ValueError, 'requires build.wheelhouse'):
                    validate_wheelhouse(config, check_directory=check)

    def test_missing_wheelhouse_records_failure_before_source_export_or_process_start(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root/'run'
            run.mkdir()
            config = load(ROOT/'config/experiment.json')
            config['build'].update(install_dependencies=True, wheelhouse=str(root/'missing'))
            mask = os.umask(0o077)
            try:
                with patch.object(sys, 'argv', ['prepare_environment.py', '--build']), \
                        patch('prepare_environment.load', return_value=config), \
                        patch('prepare_environment.new_run', return_value=run), \
                        patch('prepare_environment.export_source') as export, \
                        patch('prepare_environment.OwnedProcess') as process, \
                        contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(ValueError, 'Cannot read build.wheelhouse'):
                        prepare_environment.main()
                    export.assert_not_called()
                    process.assert_not_called()
            finally:
                os.umask(mask)
            status = json.loads((run/'status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertIn(str(root/'missing'), status['error'])
            self.assertFalse((run/'source').exists())
            self.assertFalse((run/'venv').exists())

    def test_build_cli_reports_resolved_missing_wheelhouse_without_running_pip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing = root/'missing wheels'
            output = root/'runs'
            result = subprocess.run([
                sys.executable, '-B', str(ROOT/'scripts/prepare_environment.py'), '--build',
                '--set', 'build.install_dependencies=true',
                '--set', 'build.wheelhouse='+json.dumps(str(missing)),
                '--set', 'paths.output_root='+json.dumps(str(output)),
            ], cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Cannot read build.wheelhouse directory', result.stdout)
            self.assertIn(str(missing), result.stdout)
            runs = list(output.iterdir())
            self.assertEqual(len(runs), 1)
            self.assertEqual(json.loads((runs[0]/'status.json').read_text())['status'], 'failed')
            self.assertFalse((runs[0]/'source').exists())
            self.assertFalse(list(runs[0].glob('build-*.log')))

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
