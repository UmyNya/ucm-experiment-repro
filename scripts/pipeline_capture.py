"""Worker-side portable pipeline collection; timing is correlated later using raw device traces."""
from contextlib import contextmanager
import array
import ctypes
import functools
import hashlib
import inspect
import os
from pathlib import Path
import re
import sys
import threading
import time
import types

from common import save_json


def token_hash(tokens):
    values = array.array('I', tokens)
    if sys.byteorder != 'little':
        values.byteswap()
    return hashlib.sha256(values.tobytes()).hexdigest()


def cuda_profiler_api():
    import torch
    torch.cuda.cudart()  # Ensure this worker's runtime is loaded before inspecting it.
    paths={line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
           if '/libcudart.so' in line and line.split()[-1].startswith('/')}
    if len(paths)!=1:
        raise RuntimeError('Cannot identify one loaded CUDA runtime for GIL-free profiler control; provide a version-specific worker adapter')
    runtime=ctypes.CDLL(next(iter(paths)))
    for name in ['cudaProfilerStart','cudaProfilerStop']:
        fn=getattr(runtime,name);fn.argtypes=[];fn.restype=ctypes.c_int
    # CDLL releases the GIL during profiler flush so Early heartbeat threads
    # continue running. Loading a guessed CUDA toolkit runtime is avoided.
    return runtime


class Capture:
    def __init__(self, worker, capture_id, metadata, backend):
        if not re.fullmatch(r'[A-Za-z0-9_]{1,100}', capture_id):
            raise ValueError('Capture ID must contain only letters, digits and underscores')
        self.worker, self.metadata, self.backend = worker, metadata, backend
        self.capture_id = capture_id
        self.directory = Path(os.environ['BUNDLE_RUN_DIR'])/'profiles'/capture_id/f'worker_{os.getpid()}'
        self.directory.mkdir(parents=True, exist_ok=False)
        self.events, self.requests, self.originals, self.anchors = [], {}, [], []
        self.lock = threading.Lock()
        self.batch = 0
        self.profiler = None
        self.cuda_api = None
        self.active = False
        self.hash_positions = {}
        for row in metadata['requests']:
            self.hash_positions.setdefault(row['sha256_u32le'], []).append(row['ordinal'])
        self.expected = {r['ordinal']: r for r in metadata['requests']}

    def append(self, event):
        with self.lock:
            self.events.append({'pid': os.getpid(), 'tid': threading.get_native_id(), **event})

    @contextmanager
    def scope(self, label):
        import torch
        begin = time.monotonic_ns()
        if self.backend == 'nsys':
            torch.cuda.nvtx.range_push(label)
            try:
                yield
            finally:
                end = time.monotonic_ns()
                torch.cuda.nvtx.range_pop()
                self.append({'event': 'scope', 'label': label, 'start_ns': begin, 'end_ns': end})
        else:
            with torch.autograd.profiler.record_function(label):
                try:
                    yield
                finally:
                    self.append({'event': 'scope', 'label': label, 'start_ns': begin, 'end_ns': time.monotonic_ns()})

    def anchor(self):
        for _ in range(5):
            start = time.monotonic_ns()
            with self.scope(f'BUNDLE_CLOCK|{start}'):
                pass
            self.anchors.append({'monotonic_ns': start, 'end_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()})

    def identify(self, request):
        ids, rid = request.prompt_token_ids, request.req_id
        if ids is None:
            raise ValueError('Pipeline capture requires the actual prompt token IDs')
        digest = token_hash(ids)
        positions = self.hash_positions.get(digest, [])
        if self.metadata['client'] == 'swe':
            match = re.search(re.escape(self.metadata['request_id_prefix'])+r'(\d+)(?:-|$)', rid)
            if not match:
                raise ValueError(f'Unexpected SWE request ID during pipeline capture: {rid}')
            ordinal = int(match[1])
            if ordinal not in positions:
                raise ValueError('SWE request ID and exact token hash disagree')
        else:
            if len(positions) != 1:
                raise ValueError('AIS prompt hash is missing or ambiguous; duplicate inputs require an explicit request-ID adapter')
            ordinal = positions[0]
        if len(ids) != self.expected[ordinal]['tokens']:
            raise ValueError('Observed prompt token count differs from frozen input')
        value = {'ordinal': ordinal, 'sha256_u32le': digest, 'tokens': len(ids),
                 'initial_computed_tokens': request.num_computed_tokens}
        if rid in self.requests and self.requests[rid] != value:
            raise ValueError('Pipeline request identity changed within one capture')
        self.requests[rid] = value

    def replace(self, owner, name, replacement):
        original = getattr(owner, name)
        self.originals.append((owner, name, original))
        setattr(owner, name, replacement)

    def instrument(self):
        from ucm.integration.vllm.rank_consistency import RankConsistencyManager
        from runtime_observer import _context
        _, connector = _context()
        runner = self.worker.model_runner
        original = runner.execute_model
        @functools.wraps(original)
        def execute(runner_self, scheduler_output, *args, **kwargs):
            for request in scheduler_output.scheduled_new_reqs:
                self.identify(request)
            self.batch += 1
            batch = self.batch
            event = {'event': 'execute', 'batch': batch, 'start_ns': time.monotonic_ns(),
                     'scheduled_tokens': dict(scheduler_output.num_scheduled_tokens)}
            try:
                with self.scope(f'BUNDLE_EXECUTE|{batch}'):
                    return original(scheduler_output, *args, **kwargs)
            finally:
                event['end_ns'] = time.monotonic_ns()
                self.append(event)
        self.replace(runner, 'execute_model', types.MethodType(execute, runner))
        for method, label in [('_model_forward', 'FORWARD'), ('sample_tokens', 'SAMPLE')]:
            if hasattr(runner, method):
                fn = getattr(runner, method)
                def wrap(fn=fn, label=label):
                    @functools.wraps(fn)
                    def call(runner_self, *args, **kwargs):
                        with self.scope(f'BUNDLE_{label}|{self.batch}'):
                            return fn(*args, **kwargs)
                    return call
                self.replace(runner, method, types.MethodType(wrap(), runner))
        for name, kind in [('submit_load', 'load_submit'), ('submit_dump', 'dump_submit')]:
            fn = getattr(RankConsistencyManager, name)
            signature = inspect.signature(fn)
            required = {'block_ids_by_request', 'block_ids', 'shard_indices'}
            if not required.issubset(signature.parameters):
                raise RuntimeError(f'Unsupported rank-consistency {name} interface; provide a compatible observer')
            def wrap(fn=fn, signature=signature, kind=kind):
                @functools.wraps(fn)
                def call(*args, **kwargs):
                    bound = signature.bind(*args, **kwargs).arguments
                    start = time.monotonic_ns()
                    task = fn(*args, **kwargs)
                    logical = {rid: sorted(bytes(b).hex() for b in blocks) for rid,blocks in bound['block_ids_by_request'].items()}
                    transform = connector.request_hasher if connector.tp_rank != 0 and not connector.is_mla else bytes
                    mapping = {rid: sorted(transform(bytes(b)).hex() for b in blocks)
                               for rid,blocks in bound['block_ids_by_request'].items()}
                    if set(bytes(b).hex() for b in bound['block_ids']) != {b for bs in mapping.values() for b in bs}:
                        raise ValueError('Observed store blocks differ from logical request-to-rank block mapping')
                    self.append({'event': kind, 'task': int(task.task_id), 'blocks_by_request': mapping,
                                 'logical_blocks_by_request': logical,
                                 'layers': sorted(set(map(int, bound['shard_indices']))),
                                 'block_count': len(bound['block_ids']), 'start_ns': start,
                                 'end_ns': time.monotonic_ns(), 'finalize': bound.get('finalize', False)})
                    return task
                return call
            self.replace(RankConsistencyManager, name, wrap())

    def synchronize(self):
        import torch
        (torch.cuda if self.backend == 'nsys' else torch.npu).synchronize()

    def start(self):
        import torch
        try:
            self.instrument()
            self.synchronize()
            if self.backend == 'nsys':
                self.cuda_api = cuda_profiler_api()
                code = self.cuda_api.cudaProfilerStart()
                if int(code) != 0:
                    raise RuntimeError(f'cudaProfilerStart returned {code}')
            elif self.backend == 'torch_npu':
                import torch_npu
                self.profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                                      torch_npu.profiler.ProfilerActivity.NPU],
                                                          record_shapes=False, profile_memory=False, with_stack=False)
                self.profiler.start()
            else:
                raise ValueError(f'Unsupported profiling backend: {self.backend}')
            self.active = True
            os.environ['BUNDLE_NATIVE_TRACE_DIR'] = str(self.directory)
            os.environ['UCM_PIPELINE_CAPTURE'] = '1'
            self.anchor()
            return {'pid': os.getpid(), 'directory': str(self.directory), 'backend': self.backend,
                    'start_monotonic_ns': time.monotonic_ns(), 'clock_anchors': self.anchors}
        except BaseException:
            self.stop()
            raise

    def restore(self):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
        self.originals = []

    def stop(self):
        import torch
        failure = None
        try:
            if self.active:
                try:
                    self.synchronize()
                    self.anchor()
                finally:
                    os.environ.pop('UCM_PIPELINE_CAPTURE', None)
                    if self.profiler is not None:
                        self.profiler.stop()
                        self.profiler.export_chrome_trace(str(self.directory/f'device_trace_{os.getpid()}.json'))
                    else:
                        code = self.cuda_api.cudaProfilerStop()
                        if int(code) != 0:
                            raise RuntimeError(f'cudaProfilerStop returned {code}')
        except BaseException as error:
            failure = str(error)
            raise
        finally:
            os.environ.pop('UCM_PIPELINE_CAPTURE', None)
            self.restore()
            self.active = False
            save_json(self.directory/'worker-events.json', {'pid': os.getpid(), 'events': self.events,
                      'requests': self.requests, 'clock_anchors': self.anchors, 'failure': failure,
                      'end_monotonic_ns': time.monotonic_ns(), 'backend': self.backend})
        return {'pid': os.getpid(), 'directory': str(self.directory), 'requests': len(self.requests),
                'events': len(self.events), 'stopped': True}
