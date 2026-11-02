"""Scheduler request identity and production block IDs; observation does not alter matching."""
import array
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

_seen, _identities, _blocks = {}, {}, set()


def record(request, computed, external, need_load, blocks=None):
    directory = os.environ.get('BUNDLE_PIPELINE_MATCH_DIR')
    if not directory:
        return
    rid = request.request_id
    state = (computed, external, need_load)
    add_blocks = blocks is not None and rid not in _blocks
    if _seen.get(rid) == state and not add_blocks:
        return
    _seen[rid] = state
    if rid not in _identities:
        values = array.array('I', request.prompt_token_ids)
        if sys.byteorder != 'little':
            values.byteswap()
        _identities[rid] = hashlib.sha256(values.tobytes()).hexdigest()
    event = {'request_id': rid, 'prompt_sha256': _identities[rid], 'pid': os.getpid(),
             'arrival_time_wall_s': request.arrival_time, 'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns(),
             'num_computed_tokens': computed, 'external_hit_tokens': external, 'need_load': need_load}
    if add_blocks:
        event['ucm_blocks'] = [bytes(block).hex() for block in blocks]
        _blocks.add(rid)
    with (Path(directory)/f'match_{os.getpid()}.jsonl').open('a') as stream:
        stream.write(json.dumps(event, separators=(',', ':'))+'\n')


def observe_match(fn):
    @functools.wraps(fn)
    def observed(self, request, num_computed_tokens):
        record(request, num_computed_tokens, None, False)
        result = fn(self, request, num_computed_tokens)
        meta = self.requests_meta.get(request.request_id)
        record(request, num_computed_tokens, result[0], result[1],
               blocks=meta.ucm_block_ids if meta is not None else None)
        return result
    return observed
