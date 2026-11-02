#!/usr/bin/env python3
"""Export UCM to a private source copy and optionally build a private service venv.

Runs inside the user's existing container. Never starts Docker, checks out a
branch, changes a source repository, or installs into the base Python environment.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tarfile

from common import BUNDLE, new_run, read_json, save_json, sha256, within
from configuration import load
from processes import OwnedProcess


def git(repo, *args):
    return subprocess.check_output(['git', '--no-optional-locks', '-C', str(repo), *args], text=True,
                                   env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0', 'GIT_TERMINAL_PROMPT': '0'}).strip()


def export_source(config, run):
    repo = Path(config['paths']['ucm_repository'])
    status = git(repo, 'status', '--porcelain=v1', '--untracked-files=all')
    if status and config['validation']['strict_source_clean']:
        raise RuntimeError('UCM worktree is dirty; original files will not be changed or reset')
    commit = git(repo, 'rev-parse', '--verify', '--end-of-options', config['ucm_runtime']['source_ref'] + '^{commit}')
    if not re.fullmatch(r'[0-9a-f]{40,64}', commit):
        raise ValueError('Source ref did not resolve to a commit')
    archive = run / 'ucm-source.tar'
    with archive.open('xb') as stream:
        subprocess.run(['git', '--no-optional-locks', '-C', str(repo), 'archive', '--format=tar', commit],
                       stdout=stream, check=True)
    source = run / 'source'
    source.mkdir()
    with tarfile.open(archive) as data:
        data.extractall(source, filter='data')
    provenance = {'repository': str(repo), 'working_tree_head': git(repo, 'rev-parse', 'HEAD'),
                  'working_tree_status': status, 'exported_commit': commit,
                  'archive_sha256': sha256(archive), 'source_tree_modified_only_in_copy': True}
    save_json(run / 'source-provenance.json', provenance)
    return source, provenance


def adapt_metrics_dispatcher(text):
    init = '        self._buffers = {consumer: self._empty_stats() for consumer in CONSUMERS}'
    merge = '            for consumer in CONSUMERS:\n'
    if text.count(init) != 1 or text.count(merge) != 1:
        raise RuntimeError('Unknown UCM metrics dispatcher; cannot establish independent cumulative counters')
    text = text.replace(init, init+'\n        self._bundle_observer_buffer = self._empty_stats()')
    return text.replace(merge, '            self._merge_stats(self._bundle_observer_buffer, counter_stats,\n'
                        '                              gauge_stats, histogram_stats, MULTIPROC_CONSUMER)\n\n'+merge)


def adapt_copy(source, run, commit, jobs):
    """Record build-only controls and disable UCM's broad stale-SHM sweep in this copy."""
    changes = []

    def rewrite(relative, transform):
        path = source / relative
        before = path.read_text()
        after = transform(before)
        if after != before:
            path.write_text(after)
            changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                           fromfile='a/' + relative, tofile='b/' + relative))

    def cmake(text):
        old = 'execute_process(COMMAND git rev-parse HEAD OUTPUT_VARIABLE UCM_COMMIT_ID OUTPUT_STRIP_TRAILING_WHITESPACE)'
        if text.count(old) != 1:
            raise RuntimeError('Unknown UCM CMake commit-stamp logic; adapt the copied source explicitly')
        return text.replace(old, f'set(UCM_COMMIT_ID "{commit}")')

    def setup(text):
        marker = '        subprocess.check_call(\n            ["cmake", *cmake_args, ext.cmake_file_path], cwd=build_dir\n        )'
        if text.count(marker) != 1 or text.count('"-j8"') != 1:
            raise RuntimeError('Unknown setup.py build interface; do not guess CMake flags')
        injection = '        import shlex\n        cmake_args += shlex.split(os.environ.get("BUNDLE_CMAKE_ARGS", ""))\n\n'
        return text.replace(marker, injection + marker).replace('"-j8"', repr('-j' + str(jobs)))

    def python_cleanup(text):
        bom = '\ufeff' if text.startswith('\ufeff') else ''
        text = text.removeprefix(bom) if bom else text
        tree = ast.parse(text)
        targets = []
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                segment = ast.get_source_segment(text, node.iter) or ''
                if any(x in segment for x in ['/dev/shm/ucm_uniqueid_*', '/dev/shm/ucm_blocksize_*']):
                    targets.append(node)
        lines = text.splitlines(True)
        for node in sorted(targets, key=lambda n: n.lineno, reverse=True):
            lines[node.lineno-1:node.end_lineno] = [' ' * node.col_offset + 'pass  # Bundle: cleanup is limited to recorded owned objects.\n']
        adapted = ''.join(lines)
        if any(x in adapted for x in ['/dev/shm/ucm_uniqueid_*', '/dev/shm/ucm_blocksize_*']):
            raise RuntimeError('Unrecognized broad Python SHM cleanup remains')
        marker = '        local_uid = uuid.uuid4().hex'
        if adapted.count(marker) != 1:
            raise RuntimeError('Unknown UCM shared-memory UUID generator')
        adapted = adapted.replace(marker, '        local_uid = os.environ["BUNDLE_SHM_PREFIX"] + uuid.uuid4().hex')
        # Keep scheduler/worker PID lookup semantics, but move coordination files
        # into the owned run. All native SHM names carry the random run namespace.
        for kind in ['uniqueid', 'blocksize']:
            adapted = adapted.replace('f"/dev/shm/ucm_' + kind + '_',
                                      'f"{os.environ[\'BUNDLE_COORDINATION_DIR\']}/ucm_' + kind + '_')
        return bom + adapted

    def native_cleanup(text):
        signature = 'static void CleanUpShmFileExceptMe('
        if signature not in text:
            return text
        start = text.index('{', text.index(signature))
        nesting, end = 1, start + 1
        while nesting and end < len(text):
            nesting += (text[end] == '{') - (text[end] == '}')
            end += 1
        if nesting:
            raise RuntimeError('Cannot locate native stale-SHM cleanup body')
        return text[:start] + '{ (void)me; /* Bundle owns cleanup; never sweep another service. */ }' + text[end:]

    rewrite('CMakeLists.txt', cmake)
    rewrite('setup.py', setup)
    rewrite('ucm/integration/vllm/ucm_connector.py', python_cleanup)
    rewrite('ucm/store/cache/cc/trans_buffer.cc', native_cleanup)
    rewrite('ucm/metrics_dispatcher.py', adapt_metrics_dispatcher)
    patch = run / 'build-and-resource-isolation.patch'
    patch.write_text(''.join(changes))
    return {'patch': patch.name, 'sha256': sha256(patch),
            'scope': 'Offline build flags/parallelism, truthful source stamp, run-scoped coordination/SHM names, disable broad stale-SHM deletion, independent cumulative native metric snapshot; no cache scheduling changes'}


def apply_user_patches(config, source, run):
    records = []
    for index, path in enumerate(config['profiling']['patch_files']):
        original = Path(path)
        copy = run / f'profile-{index:03d}.patch'
        copy.write_bytes(original.read_bytes())
        # git apply rejects paths escaping cwd; no --unsafe-paths or shell expansion.
        for extra in [['--check'], []]:
            subprocess.run(['git', 'apply', *extra, str(copy)], cwd=source, check=True,
                           env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
        records.append({'source': str(original), 'copy': copy.name, 'sha256': sha256(copy)})
    return records


def native_instrumentation(config, source, run):
    if not config['profiling']['instrument_native']:
        return {'enabled': False}
    original = BUNDLE/'reference/patches/native-pipeline.patch'
    copied = run/'native-pipeline.patch'
    shutil.copyfile(original, copied)
    for extra in [['--check'], []]:
        subprocess.run(['git', 'apply', *extra, str(copied)], cwd=source, check=True,
                       env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    python_patch = run/'python-pipeline.patch'
    shutil.copyfile(BUNDLE/'reference/patches/python-pipeline.patch', python_patch)
    for extra in [['--check'], []]:
        subprocess.run(['git', 'apply', *extra, str(python_patch)], cwd=source, check=True,
                       env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    include = source/'bundle-profiling'
    include.mkdir()
    header = include/'profile_nvtx.h'
    shutil.copyfile(BUNDLE/'native/profile_nvtx.h', header)
    cmake = source/'CMakeLists.txt'
    text = cmake.read_text()
    marker = 'add_compile_definitions(UCM_COMMIT_ID="${UCM_COMMIT_ID}")'
    if text.count(marker) != 1:
        raise RuntimeError('Unknown UCM CMake insertion point for native trace header')
    text = text.replace(marker, marker+'\ninclude_directories("${CMAKE_CURRENT_SOURCE_DIR}/bundle-profiling")\n'
                        'add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-include>" '
                        '"$<$<COMPILE_LANGUAGE:CXX>:${CMAKE_CURRENT_SOURCE_DIR}/bundle-profiling/profile_nvtx.h>")')
    cmake.write_text(text)
    return {'enabled': True, 'patch_sha256': sha256(copied), 'python_patch_sha256': sha256(python_patch), 'header_sha256': sha256(header),
            'capture_default': 'disabled; UCM_PIPELINE_CAPTURE enables capture only around a profiling workload',
            'event_clock': 'CLOCK_MONOTONIC nanoseconds', 'implementation': 'portable JSONL native recorder, no CUDA/NVTX dependency'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BUNDLE/'config/experiment.json')
    parser.add_argument('--set', action='append', default=[])
    parser.add_argument('--build', action='store_true', help='Actually build/install into the newly created private venv')
    args = parser.parse_args()
    os.umask(0o077)
    config = load(args.config, args.set)
    run = new_run(config, 'service', 'environment')
    try:
        source, provenance = export_source(config, run)
        adjustment = adapt_copy(source, run, provenance['exported_commit'], config['build']['jobs'])
        native = native_instrumentation(config, source, run)
        patches = apply_user_patches(config, source, run)
        save_json(run/'applied-patches.json', {'build_and_resource_isolation': adjustment, 'native_pipeline': native, 'profiling': patches})
        save_json(run/'source-files.json', {str(p.relative_to(source)): sha256(p)
                  for p in sorted(source.rglob('*')) if p.is_file() and not p.is_symlink()})
        venv = run/'venv'
        python = str(venv/'bin/python')
        wheelhouse = config['build']['wheelhouse']
        env = {**os.environ, 'PLATFORM': config['platform']['kind'], 'PYTHONDONTWRITEBYTECODE': '1',
               'PIP_NO_INDEX': '1', 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
               'PIP_CACHE_DIR': str(run/'scratch/pip'), 'TMPDIR': str(run/'scratch/tmp'),
               'ENABLE_SPARSE': 'FALSE', 'BUILD_UCM_ASU': '0', 'UCM_ENABLE_MINDIE': '0',
               'BUILD_KV_CLIENT_PROVIDER_AIV': '0', 'BUILD_KV_CLIENT_PROVIDER_AICPU': '0',
               'ENABLE_GDR': '0', 'ENABLE_UCM_PATCH': '0',
               'CUDA_VISIBLE_DEVICES': '', 'ASCEND_RT_VISIBLE_DEVICES': '',
               'BUNDLE_CMAKE_ARGS': shlex.join(config['build']['cmake_args'] +
                  ['-DDOWNLOAD_DEPENDENCE=OFF', '-DFETCHCONTENT_FULLY_DISCONNECTED=ON'])}
        (run/'scratch/tmp').mkdir(parents=True)
        commands = [[config['paths']['server_python'], '-m', 'venv', '--system-site-packages', str(venv)]]
        if config['build']['install_dependencies']:
            if not wheelhouse:
                raise ValueError('Offline dependency installation requires build.wheelhouse')
            commands.append([python, '-m', 'pip', 'install', '--no-index', '--find-links', wheelhouse,
                             '-r', str(source/'requirements-build.txt'), '-r', str(source/'requirements.txt')])
        commands.append([python, '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
                         '--wheel-dir', str(run/'wheels'), str(source)])
        save_json(run/'build-plan.json', {'commands': commands, 'environment_changes': {k: env[k] for k in [
                  'PLATFORM', 'PIP_NO_INDEX', 'TMPDIR', 'ENABLE_SPARSE', 'BUILD_UCM_ASU', 'BUNDLE_CMAKE_ARGS',
                  'UCM_ENABLE_MINDIE', 'BUILD_KV_CLIENT_PROVIDER_AIV', 'BUILD_KV_CLIENT_PROVIDER_AICPU',
                  'ENABLE_GDR', 'ENABLE_UCM_PATCH', 'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES']},
                  'base_image_label_not_verified_digest': config['build']['image_label'],
                  'next_config_value': {'paths.server_python': python}, 'build_requested': args.build})
        if args.build:
            for index, command in enumerate(commands):
                with OwnedProcess(command, cwd=run, environment=env, log_path=run/f'build-{index}.log', name=f'build-{index}') as process:
                    if process.wait(7200):
                        raise RuntimeError(f'Environment build phase {index} failed; see its log')
            wheels = list((run/'wheels').glob('*.whl'))
            if len(wheels) != 1:
                raise RuntimeError('Expected exactly one newly built UCM wheel')
            install = [python, '-m', 'pip', 'install', '--no-index', '--no-deps', '--ignore-installed', str(wheels[0])]
            save_json(run/'install-command.json', install)
            with OwnedProcess(install, cwd=run, environment=env, log_path=run/'install.log', name='install') as process:
                if process.wait(600):
                    raise RuntimeError('Private UCM wheel installation failed')
            packages = subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], env=env, text=True)
            save_json(run/'packages.json', json.loads(packages))
            save_json(run/'wheel.json', {'file': str(wheels[0]), 'sha256': sha256(wheels[0])})
            probe = [python, '-B', str(BUNDLE/'scripts/runtime_inventory.py'), str(run/'installed-inventory.json')]
            with OwnedProcess(probe, cwd=run, environment=env, log_path=run/'inventory.log', name='inventory') as process:
                if process.wait(120):
                    raise RuntimeError('Installed environment inventory failed')
            actual = read_json(run/'installed-inventory.json')
            if not within(actual['ucm_root'], venv):
                raise RuntimeError('Service Python resolved UCM outside the new private venv')
            connector = Path(actual['ucm_root'])/'integration/vllm/ucm_connector.py'
            if 'BUNDLE_SHM_PREFIX' not in connector.read_text(encoding='utf-8-sig'):
                raise RuntimeError('Installed UCM lacks the owned-resource patch')
            if not any('.so' in name for name in actual['ucm_files']):
                raise RuntimeError('Built UCM package contains no native library')
            save_json(run/'service-environment.json', {'schema_version': 1, 'server_python': python,
                      'source': provenance, 'platform': config['platform']['kind'],
                      'resource_namespace': 'bundle-shm-v1', 'metrics_snapshot_contract': 'bundle-cumulative-v1', 'installed': actual,
                      'patches': read_json(run/'applied-patches.json'),
                      'wheel_sha256': sha256(wheels[0]), 'target_inference_tested': False})
        save_json(run/'status.json', {'status': 'built' if args.build else 'source_prepared',
                  'target_inference_tested': False, 'server_python': python if args.build else None})
    except BaseException as error:
        save_json(run/'status.json', {'status': 'failed', 'error': str(error)})
        print(run)
        raise
    print(run)


if __name__ == '__main__':
    main()
