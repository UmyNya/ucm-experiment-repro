"""Record raw service metrics in a background thread; no worker RPC on the hot path."""
import json
import math
from pathlib import Path
import re
import threading
import time
import urllib.request

from common import save_json

PATTERN = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?$')


def parse(text):
    output = []
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = PATTERN.match(line)
        if match:
            try:
                value = float(match[3])
            except ValueError:
                continue
            if math.isfinite(value):
                output.append({'name': match[1], 'labels': match[2] or '', 'value': value})
    return output


class Sampler:
    def __init__(self, base_url, output, interval=.5):
        self.url, self.output, self.interval = base_url.rstrip('/')+'/metrics', Path(output), interval
        self.done = threading.Event()
        self.rows = []
        self.thread = threading.Thread(target=self.collect, name='bundle-service-metrics', daemon=True)

    def collect(self):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with self.output.open('x') as stream:
            while not self.done.is_set():
                item = {'start_monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
                try:
                    with opener.open(self.url, timeout=2) as response:
                        text = response.read().decode()
                    item.update(raw=text, metrics=parse(text))
                except Exception as error:
                    item.update(error=str(error), metrics=[])
                item['end_monotonic_ns'] = time.monotonic_ns()
                stream.write(json.dumps(item, separators=(',', ':'))+'\n')
                stream.flush()
                self.rows.append({k:v for k,v in item.items() if k != 'raw'})
                self.done.wait(self.interval)

    def start(self):
        self.started_ns = time.monotonic_ns()
        self.thread.start()

    def stop(self):
        self.ended_ns = time.monotonic_ns()
        self.done.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('Metrics sampler did not stop within its HTTP timeout')
        summary = {'samples': len(self.rows), 'successful_samples': sum(bool(x['metrics']) for x in self.rows),
                   'errors': [x['error'] for x in self.rows if 'error' in x], 'interval_s': self.interval,
                   'start_monotonic_ns': self.started_ns, 'end_monotonic_ns': self.ended_ns,
                   'sampling_is_periodic_not_exact_event_trace': True}
        save_json(self.output.with_suffix('.summary.json'), summary)
        return summary
