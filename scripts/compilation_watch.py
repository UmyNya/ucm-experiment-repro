"""Worker-side observation of compiler callbacks and graph capture, without changing outputs."""
import functools
import importlib.util
import os
import sys
import threading
import time

_watch = None


class Watch:
    def __init__(self):
        self.events, self.hooks, self.unavailable = [], {}, {}
        self.lock = threading.Lock()
        self.dynamo_handler = self.triton_knobs = None
        self.install()

    def record(self, kind, **data):
        with self.lock:
            self.events.append({'kind': kind, 'monotonic_ns': time.monotonic_ns(), **data})

    def install(self):
        import torch
        try:
            from torch._dynamo.callback import callback_handler, on_compile_start
            def start(*args, **kwargs):
                self.record('dynamo_compile_start')
            on_compile_start(start)
            self.dynamo_handler, self.dynamo_callback = callback_handler, start
        except (ImportError, AttributeError) as error:
            self.unavailable['dynamo'] = str(error)
        try:
            from triton import knobs
            previous = knobs.compilation.listener
            def listener(**kwargs):
                self.record('triton_compile', cache_hit=kwargs.get('cache_hit'),
                            kernel=str(getattr(kwargs.get('src'), 'name', 'unknown')))
                if previous is not None:
                    previous(**kwargs)
            knobs.compilation.listener = listener
            self.triton_knobs, self.triton_listener = knobs, listener
        except (ImportError, AttributeError) as error:
            self.unavailable['triton'] = str(error)
        # Observe graph capture, never replay. Python wrappers only add an event
        # and call the original implementation with unchanged arguments/results.
        platform = os.environ.get('BUNDLE_PLATFORM')
        module = getattr(torch, 'npu' if platform == 'ascend' else 'cuda', None)
        cls = getattr(module, 'NPUGraph' if platform == 'ascend' else 'CUDAGraph', None)
        self.wrap('device_graph_capture', cls, 'capture_begin')

    def wrap(self, label, owner, name):
        if label in self.hooks:
            return
        if owner is None or not hasattr(owner, name):
            self.unavailable[label] = 'API not available in this runtime'
            return
        original = getattr(owner, name)
        @functools.wraps(original)
        def observed(*args, **kwargs):
            self.record(label)
            return original(*args, **kwargs)
        try:
            setattr(owner, name, observed)
        except (TypeError, AttributeError) as error:
            self.unavailable[label] = str(error)
            return
        self.hooks[label] = (owner, name, observed)
        self.unavailable.pop(label, None)

    def snapshot(self):
        # TorchAir is optional. Only observe modules the actual service loaded;
        # importing an unused compiler can change runtime initialization.
        for module_name in ['torchair.ge._ge_graph', 'torchair._ge_concrete_graph.ge_graph']:
            module = sys.modules.get(module_name)
            if module is not None:
                self.wrap('torchair_ge_compile:'+module_name, getattr(module, 'GeGraph', None), 'compile')
        with self.lock:
            events = list(self.events)
        return {'pid': os.getpid(), 'monotonic_ns': time.monotonic_ns(), 'events': events,
                'installed': {**{name: getattr(owner, attr) is callback for name,(owner,attr,callback) in self.hooks.items()},
                              'dynamo': bool(self.dynamo_handler and self.dynamo_callback in self.dynamo_handler.start_callbacks),
                              'triton': bool(self.triton_knobs and self.triton_knobs.compilation.listener is self.triton_listener)},
                'triton_package_present': importlib.util.find_spec('triton') is not None,
                'unavailable': dict(self.unavailable),
                'scope': 'TorchDynamo callbacks, Triton cache_hit listener where available, device graph capture and loaded TorchAir GeGraph.compile; combined with disk-cache and log audit'}


def snapshot():
    global _watch
    if _watch is None:
        _watch = Watch()
    return _watch.snapshot()
