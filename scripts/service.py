"""Owned vLLM service lifecycle, shared by standalone serving and campaigns."""
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import shutil
import socket
import time
import uuid

from boundaries import rpc
from capacity_validation import validate_workers
from common import BUNDLE, http, read_json, save_json, sha256, utc
from configuration import compile_plan
from prepare_environment import git
from processes import OwnedProcess, identity, same_process
from storage import PrivateStorage, file_identity


@contextmanager
def interrupt_cleanup():
    previous = {}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Received {signal.Signals(signum).name}')
    for sig in [signal.SIGTERM, signal.SIGINT]:
        previous[sig] = signal.signal(sig, interrupted)
    try:
        yield
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def attest_environment(plan, run):
    c = plan['configuration']
    python = Path(c['paths']['server_python']).absolute()
    manifest = python.parent.parent.parent/'service-environment.json'
    if not manifest.is_file():
        raise ValueError('Run prepare_environment.py --build inside the target container, then set paths.server_python to its private venv/bin/python')
    env = read_json(manifest)
    if env.get('resource_namespace') != 'bundle-shm-v1' or env['platform'] != c['platform']['kind']:
        raise ValueError('Service environment backend/resource-isolation contract differs from configuration')
    if env.get('metrics_snapshot_contract') != 'bundle-cumulative-v1':
        raise ValueError('Rebuild the private UCM environment with the independent cumulative metrics observer')
    commit = git(c['paths']['ucm_repository'], 'rev-parse', '--verify', '--end-of-options', c['ucm']['source_ref']+'^{commit}')
    if commit != env['source']['exported_commit']:
        raise ValueError('Selected UCM commit differs from prepared environment; rebuild a new private environment')
    if c['validation']['strict_source_clean'] and git(c['paths']['ucm_repository'], 'status', '--porcelain=v1', '--untracked-files=all'):
        raise ValueError('Original UCM repository is dirty; no files will be reset')
    expected_patches = [x['sha256'] for x in env['patches']['profiling']]
    if expected_patches != [sha256(p) for p in c['profiling']['patch_files']]:
        raise ValueError('Profiling patch list differs from prepared environment')
    native = env['patches'].get('native_pipeline', {})
    if bool(native.get('enabled')) != c['profiling']['instrument_native']:
        raise ValueError('Native pipeline instrumentation setting differs from prepared environment')
    if native.get('enabled') and native['header_sha256'] != sha256(BUNDLE/'native/profile_nvtx.h'):
        raise ValueError('Portable native trace header changed; rebuild the private environment')
    variables = {**os.environ, 'ENABLE_UCM_PATCH': '0', 'CUDA_VISIBLE_DEVICES': '', 'ASCEND_RT_VISIBLE_DEVICES': '',
                 'PYTHONDONTWRITEBYTECODE': '1'}
    command = [str(python), '-B', str(BUNDLE/'scripts/runtime_inventory.py'), str(run/'runtime-inventory.json')]
    with OwnedProcess(command, cwd=run, environment=variables, log_path=run/'inventory.log', name='inventory') as process:
        if process.wait(120):
            raise RuntimeError('Runtime environment inventory failed')
    actual = read_json(run/'runtime-inventory.json')
    if actual['ucm_root'] != env['installed']['ucm_root'] or actual['ucm_files'] != env['installed']['ucm_files']:
        raise ValueError('Installed UCM files changed after environment preparation')
    save_json(run/'service-environment.json', env)
    return env


class Service:
    def __init__(self, config, profile, run, *, compile_cache=None, enforce_capacity=True):
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=True)
        self.plan = compile_plan(config, profile, self.run)
        self.enforce_capacity = enforce_capacity
        self.process = None
        self.resources = {}
        self.prefix = 'bundle_' + uuid.uuid4().hex + '_'
        self.storage = PrivateStorage(self.plan)
        self.closed = False
        env = self.plan['server_environment']
        env.update(BUNDLE_SHM_PREFIX=self.prefix, BUNDLE_COORDINATION_DIR=str(self.run/'scratch/coordination'))
        env['PYTHONPATH'] = os.pathsep.join([str(BUNDLE/'scripts'), *config['profiling']['extra_python_paths']])
        if self.plan['mode'] == 'pipeline':
            if not config['profiling']['instrument_native']:
                raise ValueError('Pipeline requires native instrumentation in the prepared UCM build')
            match_dir = self.run/'match-events'
            match_dir.mkdir()
            env['BUNDLE_PIPELINE_MATCH_DIR'] = str(match_dir)
            if self.plan['profiling_backend'] == 'nsys':
                nsys = shutil.which(config['profiling']['nsys'])
                if not nsys:
                    raise FileNotFoundError('Set profiling.nsys to an installed Nsight Systems executable')
                # Repeated capture ranges flush at process shutdown; analysis
                # runs after the owned service stops. Never terminate an
                # independent service just to flush its report.
                self.plan['server_command'] = [nsys, 'profile', '--trace=cuda,nvtx', '--sample=none',
                    '--cpuctxsw=none', '--cuda-graph-trace=node', '--cuda-event-trace=false',
                    '--capture-range=cudaProfilerApi', '--capture-range-end=repeat', '--kill=none',
                    '--trace-fork-before-exec=true', '--output='+str(self.run/'device-profile'),
                    *self.plan['server_command']]
        if compile_cache:
            for key in ['XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'ASCEND_CACHE_PATH']:
                env[key] = str(Path(compile_cache)/Path(env[key]).name)
        cache_root = Path(compile_cache) if compile_cache else self.run/'compile-cache'
        self.plan['compile_cache_roots'] = [str(cache_root.resolve())]
        self.plan['server_log'] = str(self.run/'server.log')
        # TorchAir's default is relative to cwd. Only this campaign's cache is
        # shared across its fresh services; historical cache paths are not used.
        torchair_cache = cache_root/'torchair'
        torchair_cache.mkdir(parents=True, exist_ok=True)
        (self.run/'.torchair_cache').symlink_to(torchair_cache.resolve(), target_is_directory=True)
        save_json(self.run/'plan.json', self.plan)
        save_json(self.run/'ucm.json', self.plan['ucm_configuration'])
        save_json(self.run/'resource-namespace.json', {'prefix': self.prefix, 'supervisor': identity(os.getpid())})

    def capture_resources(self):
        for prefix in ['uc_shm_cache_', 'ucm_early_']:
            for path in Path('/dev/shm').glob(prefix+self.prefix+'*'):
                try:
                    if path.is_symlink():
                        continue
                    actual = file_identity(path)
                    if str(path) in self.resources and self.resources[str(path)] != actual:
                        raise RuntimeError('Owned shared-memory path changed identity')
                    self.resources[str(path)] = actual
                except FileNotFoundError:
                    pass
        save_json(self.run/'owned-shared-memory.json', self.resources)

    def start(self):
        c = self.plan['configuration']
        if not c['ucm']['enabled']:
            raise ValueError('This service entry requires UCM; OFF means synchronous UCM, not disabling the connector')
        if c['ucm']['store']['store_pipeline'] != 'Cache|Posix':
            raise ValueError('Owned-resource lifecycle currently requires the Cache|Posix store pipeline')
        self.environment = attest_environment(self.plan, self.run)
        # Never attach to or stop a service merely because it occupies this port.
        with socket.socket() as probe:
            # A fresh repetition may reuse the port while previous connections
            # remain in TIME_WAIT. A live listener still prevents this bind.
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', c['server']['port']))
            probe.listen(1)
        for key in ['model', 'tokenizer']:
            if not Path(c['paths'][key]).is_dir():
                raise FileNotFoundError(c['paths'][key])
        Path(self.plan['server_environment']['BUNDLE_COORDINATION_DIR']).mkdir(parents=True)
        for key in ['XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'VLLM_CACHE_ROOT',
                    'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'ASCEND_CACHE_PATH', 'ASCEND_WORK_PATH']:
            Path(self.plan['server_environment'][key]).mkdir(parents=True, exist_ok=True)
        self.storage.create()
        env = {**os.environ, **self.plan['server_environment']}
        self.process = OwnedProcess(self.plan['server_command'], cwd=self.run, environment=env,
                                    log_path=self.run/'server.log', name='server',
                                    shutdown_timeout=c['execution']['shutdown_timeout_s'])
        deadline = time.monotonic()+c['execution']['startup_timeout_s']
        last_error = None
        while time.monotonic() < deadline:
            self.capture_resources()
            if self.process.poll() is not None:
                raise RuntimeError('vLLM exited during startup; see server.log')
            try:
                http(self.plan['base_url'], '/health', timeout=5)
                models = http(self.plan['base_url'], '/v1/models', timeout=10)
                if c['server']['served_model_name'] not in {m['id'] for m in models['data']}:
                    raise RuntimeError('The endpoint serves an unexpected model name')
            except (OSError, ValueError, TimeoutError) as error:
                last_error = str(error)
                time.sleep(c['execution']['health_poll_s'])
                continue
            # A healthy endpoint with an incompatible observer is a hard
            # compatibility error, not a reason to wait for the startup timeout.
            rows = rpc(self.plan, 'bundle_snapshot', [True], timeout=60)
            members = {r['pid'] for r in self.process.capture()}
            if any(r['pid'] not in members or r['run_dir'] != str(self.run) or not r['unique_id'].startswith(self.prefix) for r in rows):
                raise RuntimeError('Ready endpoint workers do not belong to this service invocation')
            installed = self.environment['installed']['ucm_root']
            if any(str(Path(r['ucm_source']).parent) != installed for r in rows):
                raise RuntimeError('Worker imported an unexpected UCM package')
            save_json(self.run/'startup-workers.json', rows)
            check = validate_workers(self.plan, rows, enforce_targets=self.enforce_capacity)
            save_json(self.run/'capacity-validation.json', check)
            if not check['passed'] and c['validation']['require_exact_capacity']:
                raise ValueError('; '.join(check['errors']))
            save_json(self.run/'service-status.json', {'status': 'ready', 'base_url': self.plan['base_url'],
                      'supervisor': identity(os.getpid()), 'started_utc': utc(), 'capacity_passed': check['passed']})
            self.rows = rows
            return rows
        raise TimeoutError(f'Service startup timeout; last readiness error: {last_error}')

    def stop(self):
        if self.closed:
            return
        try:
            try:
                self.capture_resources()
            finally:
                if self.process:
                    self.process.stop()
            # No process is left to recreate these paths. Compare the inode
            # recorded while the owned session was alive before deleting.
            removed = []
            for name, expected in self.resources.items():
                path = Path(name)
                if not path.exists():
                    continue
                if path.is_symlink() or file_identity(path) != expected:
                    raise RuntimeError('Shared-memory ownership changed; refusing unlink')
                path.unlink()
                removed.append(name)
            storage = self.storage.cleanup()
            save_json(self.run/'resource-cleanup.json', {'released': True, 'shared_memory_removed': removed,
                      'l2': storage, 'checked_utc': utc(), 'logs_inputs_results_retained': True})
            self.closed = True
        except BaseException as error:
            save_json(self.run/'resource-cleanup.json', {'released': False, 'error': str(error), 'checked_utc': utc()})
            raise

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *_):
        self.stop()
