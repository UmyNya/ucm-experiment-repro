"""Exercise CMake configuration, local Git fetching, and compilation without installs."""
import importlib.util
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from configuration import load
from prepare_environment import build_environment


@unittest.skipUnless(all(shutil.which(tool) for tool in ['cmake', 'git', 'c++', 'make']),
                     'CMake, Git, a C++ compiler, and make are required')
class NativeDependencyBuildTest(unittest.TestCase):
    def environment(self, run, *, download):
        config = load(ROOT/'config/experiment.json')
        if download:
            config['build']['cmake_args'] = [
                '-DDOWNLOAD_DEPENDENCE=ON', '-DFETCHCONTENT_FULLY_DISCONNECTED=OFF']
        env = build_environment(config, run)
        # FetchContent must use only fixture repositories, even if a URL rewrite fails.
        env.update(GIT_CONFIG_COUNT='2', GIT_CONFIG_KEY_0='protocol.allow', GIT_CONFIG_VALUE_0='never',
                   GIT_CONFIG_KEY_1='protocol.file.allow', GIT_CONFIG_VALUE_1='always')
        return env

    def command(self, argv, cwd, env, *, check=True):
        result = subprocess.run(argv, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, timeout=60)
        if check:
            self.assertEqual(result.returncode, 0, result.stdout)
        return result

    def repository(self, path, cmake, *, tag='fixture-v1', files=None):
        path.mkdir(parents=True)
        (path/'CMakeLists.txt').write_text(cmake)
        for name, content in (files or {}).items():
            (path/name).write_text(content)
        env = self.environment(path, download=False)
        self.command(['git', 'init', '--quiet', str(path)], path.parent, env)
        self.command(['git', 'add', '.'], path, env)
        self.command(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                      '-c', 'commit.gpgsign=false', 'commit', '--quiet', '-m', 'Fixture'], path, env)
        self.command(['git', 'tag', tag], path, env)

    def project(self, root):
        dependency = root/'dependency'
        self.repository(dependency, 'cmake_minimum_required(VERSION 3.18)\n'
                        'project(fixture_dependency LANGUAGES CXX)\n'
                        'add_library(fixture_dep STATIC value.cc)\n',
                        files={'value.cc': 'int fixture_value() { return 42; }\n'})
        source = root/'source'
        source.mkdir()
        (source/'main.cc').write_text('int fixture_value();\nint main() { return fixture_value() == 42 ? 0 : 1; }\n')
        (source/'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.18)\nproject(probe LANGUAGES CXX)\n'
            'option(DOWNLOAD_DEPENDENCE "" ON)\n'
            'set(FETCHCONTENT_FULLY_DISCONNECTED OFF CACHE BOOL "")\n'
            'if(DOWNLOAD_DEPENDENCE)\ninclude(FetchContent)\n'
            f'FetchContent_Declare(fixture_dep GIT_REPOSITORY "{dependency.as_uri()}" GIT_TAG fixture-v1)\n'
            'FetchContent_MakeAvailable(fixture_dep)\nelse()\nadd_subdirectory(vendor/fixture_dep)\nendif()\n'
            'add_executable(probe main.cc)\ntarget_link_libraries(probe PRIVATE fixture_dep)\n')
        return source, dependency

    def configure(self, root, source, env, *, check=True):
        return self.command(['cmake', '-S', str(source), '-B', str(root/'build'), '-G', 'Unix Makefiles',
                             *shlex.split(env['BUNDLE_CMAKE_ARGS'])], root, env, check=check)

    def build_and_run(self, root, env):
        self.command(['cmake', '--build', str(root/'build'), '--parallel', '2'], root, env)
        self.command([str(root/'build/probe')], root, env)

    def test_offline_default_fails_without_vendored_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.project(root)
            env = self.environment(root, download=False)
            result = self.configure(root, source, env, check=False)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('vendor/fixture_dep', result.stdout)
            self.assertFalse((root/'build/_deps/fixture_dep-src').exists())

    def test_explicit_download_fetches_then_builds_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.project(root)
            env = self.environment(root, download=True)
            self.configure(root, source, env)
            self.assertTrue((root/'build/_deps/fixture_dep-src/value.cc').is_file())
            self.build_and_run(root, env)

    def test_offline_default_builds_with_vendored_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, dependency = self.project(root)
            shutil.copytree(dependency, source/'vendor/fixture_dep', ignore=shutil.ignore_patterns('.git'))
            env = self.environment(root, download=False)
            self.configure(root, source, env)
            self.assertFalse((root/'build/_deps').exists())
            self.build_and_run(root, env)

    @unittest.skipUnless(all(importlib.util.find_spec(name) for name in ['pip', 'setuptools', 'wheel']),
                         'Existing pip, setuptools, and wheel are required; tests never install them')
    def test_wheel_backend_runs_cmake_and_packages_without_installing_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = self.project(root)
            package = source/'build_orchestration_fixture'
            package.mkdir()
            (package/'__init__.py').write_text('')
            (source/'pyproject.toml').write_text(
                '[build-system]\nrequires = ["setuptools", "wheel"]\nbuild-backend = "setuptools.build_meta"\n')
            (source/'setup.py').write_text(
                'import os, shlex, subprocess\nfrom pathlib import Path\n'
                'from setuptools import setup\nfrom setuptools.command.build_py import build_py\n'
                'class BuildWithCMake(build_py):\n'
                '    def run(self):\n'
                '        native = Path("native-build").resolve()\n'
                '        subprocess.check_call(["cmake", "-S", ".", "-B", str(native),\n'
                '                               "-G", "Unix Makefiles", *shlex.split(os.environ["BUNDLE_CMAKE_ARGS"])])\n'
                '        subprocess.check_call(["cmake", "--build", str(native), "--parallel", "2"])\n'
                '        subprocess.check_call([str(native / "probe")])\n'
                '        super().run()\n'
                'setup(name="build-orchestration-fixture", version="0.0.0",\n'
                '      packages=["build_orchestration_fixture"], cmdclass={"build_py": BuildWithCMake})\n')
            env = self.environment(root, download=True)
            (root/'scratch/tmp').mkdir(parents=True)
            self.command([sys.executable, '-B', '-m', 'pip', 'wheel', '--no-index', '--no-deps',
                          '--no-build-isolation', '--wheel-dir', str(root/'wheels'), str(source)], root, env)
            wheels = list((root/'wheels').glob('*.whl'))
            self.assertEqual(len(wheels), 1)
            with zipfile.ZipFile(wheels[0]) as wheel:
                self.assertIn('build_orchestration_fixture/__init__.py', wheel.namelist())
            self.assertTrue((source/'native-build/_deps/fixture_dep-src/value.cc').is_file())
            self.command([str(source/'native-build/probe')], root, env)

    def test_actual_ucm_vendor_modules_fetch_all_four_dependencies(self):
        config = load(ROOT/'config/experiment.json')
        repository = Path(os.environ.get('UCM_SOURCE_ROOT') or config['paths']['ucm_repository'])
        vendor = repository/'ucm/shared/vendor'
        if not (vendor/'helper.cmake').is_file():
            if os.environ.get('UCM_SOURCE_ROOT'):
                self.fail('UCM_SOURCE_ROOT does not contain the vendor CMake modules')
            self.skipTest('Set UCM_SOURCE_ROOT to check the actual UCM vendor modules')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            copied_vendor = source/'vendor'
            copied_vendor.mkdir(parents=True)
            shutil.copyfile(vendor/'helper.cmake', copied_vendor/'helper.cmake')
            for name in ['fmt', 'spdlog', 'pybind11', 'zlib']:
                module = (vendor/f'dep-{name}.cmake').read_text()
                match = re.search(r'set\(DEP_[A-Z0-9]+_TAG\s+([^\s)]+)\)', module)
                self.assertIsNotNone(match, f'Unrecognized dependency tag in dep-{name}.cmake')
                tag = match.group(1)
                dependency = root/'dependencies'/name
                cmake = 'cmake_minimum_required(VERSION 3.18)\n'
                files = None
                if name == 'zlib':
                    cmake += 'add_library(zlibstatic STATIC value.cc)\n'
                    files = {'value.cc': 'int fixture_value() { return 42; }\n'}
                elif name == 'pybind11':
                    cmake += 'function(pybind11_add_module)\nendfunction()\n'
                else:
                    cmake += f'add_library({name} INTERFACE)\n'
                self.repository(dependency, cmake, tag=tag, files=files)
                rewritten, count = re.subn(r'https://[^\s)]+', lambda _: dependency.as_uri(), module)
                self.assertGreater(count, 0)
                (copied_vendor/f'dep-{name}.cmake').write_text(rewritten)
            (copied_vendor/'CMakeLists.txt').write_text('\n'.join(
                f'include(dep-{name}.cmake)' for name in ['fmt', 'spdlog', 'pybind11', 'zlib'])+'\n')
            (source/'main.cc').write_text('int fixture_value();\nint main() { return fixture_value() == 42 ? 0 : 1; }\n')
            (source/'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.18)\nproject(ucm_vendor_probe LANGUAGES CXX)\n'
                'list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/vendor")\n'
                'add_subdirectory(vendor)\n'
                'if(NOT COMMAND pybind11_add_module)\nmessage(FATAL_ERROR "pybind11 was not loaded")\nendif()\n'
                'add_executable(probe main.cc)\ntarget_link_libraries(probe PRIVATE zlibstatic)\n')
            env = self.environment(root, download=True)
            self.configure(root, source, env)
            for name in ['fmt', 'spdlog', 'pybind11', 'zlib']:
                self.assertTrue((root/f'build/_deps/{name}-src/CMakeLists.txt').is_file())
            self.build_and_run(root, env)


if __name__ == '__main__':
    unittest.main()
