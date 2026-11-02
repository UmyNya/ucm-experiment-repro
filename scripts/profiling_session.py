"""Capture boundary control, separate from service ownership and offline analysis."""
from pathlib import Path
import uuid

from boundaries import rpc
from common import read_json, save_json, sha256


def metadata(plan, inputs):
    inputs = Path(inputs)
    snap = read_json(inputs/'prompt-snapshots/manifest.json')
    records = snap['formal'] if plan['client'] == 'aisbench' else [
        {'ordinal': r['ordinal'], 'tokens': r['prompt_tokens'], 'sha256_u32le': r['sha256_u32le']}
        for r in snap['requests']]
    if len(records) != plan['request_count'] or sorted(r['ordinal'] for r in records) != list(range(len(records))):
        raise ValueError('Capture metadata must contain every frozen request exactly once')
    return {'client': plan['client'], 'requests': records, 'input_fingerprint': snap['input_fingerprint'],
            'request_id_prefix': ('swe-closed-'+sha256(inputs/'workload/order.jsonl')[:16]+'-'
                                  +str(plan['configuration']['load']['concurrency'])+'-')
                                 if plan['client'] == 'swe' else None}


class Session:
    def __init__(self, plan, attempt, inputs):
        self.plan, self.attempt = plan, Path(attempt)
        self.metadata = metadata(plan, inputs)
        self.started = False
        self.id = 'capture_'+uuid.uuid4().hex

    def start(self):
        # Save the intention before RPC: a partial multiworker failure still has
        # an explicit stop path. Stop only touches this observer's active capture.
        self.started = True
        save_json(self.attempt/'profile-session.json', {'capture_id': self.id, 'metadata': self.metadata,
                  'backend': self.plan['profiling_backend'], 'status': 'starting'})
        try:
            workers = rpc(self.plan, 'bundle_profile_start', [self.id, self.metadata, self.plan['profiling_backend']],
                          timeout=self.plan['configuration']['execution']['drain_timeout_s'])
        except BaseException:
            self.stop()
            raise
        save_json(self.attempt/'profile-session.json', {'capture_id': self.id, 'metadata': self.metadata,
                  'backend': self.plan['profiling_backend'], 'workers': workers, 'status': 'capturing'})

    def stop(self):
        if not self.started:
            return
        try:
            workers = rpc(self.plan, 'bundle_profile_stop', [self.id],
                          timeout=self.plan['configuration']['execution']['benchmark_timeout_s'])
            save_json(self.attempt/'profile-stop.json', {'workers': workers, 'stopped': True})
        except BaseException as error:
            save_json(self.attempt/'profile-stop.json', {'stopped': False, 'error': str(error)})
            raise
        finally:
            self.started = False
