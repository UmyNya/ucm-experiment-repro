"""Normalize real client results while retaining all native result files."""
import io
import json
import math
from pathlib import Path
import sqlite3
import statistics

from common import read_json, save_json


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def distribution(values):
    if not values:
        return None
    values = sorted(values)
    if any(not math.isfinite(x) for x in values):
        raise ValueError('Non-finite latency in successful request')
    def percentile(q):
        index = (len(values)-1)*q
        lo, hi = math.floor(index), math.ceil(index)
        return values[lo]*(hi-index)+values[hi]*(index-lo) if lo != hi else values[lo]
    return {'count': len(values), 'mean': statistics.fmean(values), 'min': min(values), 'max': max(values),
            'p50': percentile(.5), 'p95': percentile(.95), 'p99': percentile(.99)}


def summarize(samples, duration, requests, checks, *, client):
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid measurement duration')
    return {'client': client, 'requests': requests, 'successful_requests': len(samples),
            'failed_requests': requests-len(samples), 'duration_s': duration,
            'input_tokens': sum(x['input_tokens'] for x in samples),
            'output_tokens': sum(x['output_tokens'] for x in samples),
            'requests_per_s': len(samples)/duration,
            'input_tokens_per_s': sum(x['input_tokens'] for x in samples)/duration,
            'output_tokens_per_s': sum(x['output_tokens'] for x in samples)/duration,
            **{key: distribution([x[key] for x in samples if x.get(key) is not None])
               for key in ['ttft_ms', 'e2e_ms', 'tpot_ms']}, 'checks': checks,
            'passed': all(checks.values()),
            'throughput_definition': 'All successful input tokens including reused prefix tokens / native client measurement wall time'}


def extract_ais(plan, attempt, inputs):
    import numpy as np
    files = list((attempt/'aisbench/outputs').glob('*/performances/*/gsm8k_details.jsonl'))
    if len(files) != 1:
        raise ValueError(f'Expected one AISBench detail file, observed {len(files)}')
    path = files[0]
    details = sorted(rows(path), key=lambda x: int(x['id']))
    manifest = read_json(inputs/'manifest.json')
    expected = rows(inputs/'formal.jsonl')
    arrays = {}
    for name in {item['db_name'] for item in details if item.get('db_name')}:
        db = path.parent/'db_data'/name
        if db.resolve().parent != (path.parent/'db_data').resolve():
            raise ValueError('Unexpected AISBench DB path')
        with sqlite3.connect(db.resolve().as_uri()+'?mode=ro', uri=True) as conn:
            arrays[name] = {key: np.load(io.BytesIO(blob), allow_pickle=False)
                            for key, blob in conn.execute('SELECT id,arr_blob FROM numpy_store')}
    samples = []
    for row in details:
        if not row.get('success'):
            continue
        times = arrays[row['db_name']][row['time_points']['__db_ref__']]
        if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) >= 0).all():
            raise ValueError('Invalid AISBench timestamp sequence')
        streaming = plan['configuration']['load']['stream']
        output = row['output_tokens']
        samples.append({'id': row['id'], 'input_tokens': row['input_tokens'], 'output_tokens': output,
                        'start_s': float(times[0]), 'end_s': float(times[-1]),
                        'ttft_ms': float(times[1]-times[0])*1000 if streaming else None,
                        'e2e_ms': float(times[-1]-times[0])*1000,
                        'tpot_ms': float(times[-1]-times[1])*1000/(output-1) if streaming and output > 1 else None})
    duration = read_json(path.with_name('gsm8k.json'))['Benchmark Duration']['total'].split()
    factors = {'ms': .001, 's': 1.0}
    if len(duration) != 2 or duration[1] not in factors:
        raise ValueError('Unknown AISBench Benchmark Duration unit')
    checks = {'request_count': len(details) == manifest['n'] == plan['request_count'],
              'all_requests_succeeded': bool(details) and len(samples) == len(details),
              'input_identity': len(details) == len(expected) and all(r.get('input') == e['question'] for r,e in zip(details,expected)),
              'input_token_counts': len(details) == len(manifest['formal_input_token_counts']) and all(
                  r.get('input_tokens') == n for r,n in zip(details,manifest['formal_input_token_counts'])),
              'output_token_counts': bool(details) and all(r.get('output_tokens') == plan['configuration']['load']['output_tokens'] for r in details)}
    return summarize(samples, float(duration[0])*factors[duration[1]], len(details), checks, client='aisbench'), samples


def extract_swe(plan, attempt, inputs):
    result = read_json(attempt/'client-result.json')
    order = rows(inputs/'workload/order.jsonl')[:plan['request_count']]
    records = result['records']
    samples = []
    for row in records:
        if row['error'] is not None or row['actual_submit_s'] is None or row['completion_s'] is None:
            continue
        response = row  # Original SWE serialize_record flattens response metrics.
        samples.append({'id': row['ordinal'], 'request_id': row['request_id'], 'trajectory_id': row['trajectory_id'],
                        'input_tokens': response['prompt_tokens'], 'output_tokens': response['output_tokens'],
                        'start_s': row['actual_submit_s'], 'end_s': row['completion_s'],
                        'ttft_ms': response['ttft_ms'], 'e2e_ms': row['total_ms'],
                        'tpot_ms': (row['total_ms']-response['ttft_ms'])/(response['output_tokens']-1) if response['output_tokens'] > 1 else None,
                        'dependency_wait_ms': row['dependency_wait_ms'], 'client_admission_wait_ms': row['client_admission_wait_ms']})
    snapshot = read_json(inputs/'PREPARATION.json')['input_fingerprint']
    checks = {'request_count': len(records) == plan['request_count'],
              'all_requests_succeeded': len(samples) == len(records) and bool(records),
              'input_identity': result['prompt_snapshot']['input_fingerprint'] == snapshot,
              'order_identity': [r['request_id'] for r in records] == [x['request_id'] for x in order],
              'send_order': result['execution_order']['client_send_order_matches_manifest'],
              'measurement_hooks': not result['measurement_errors'],
              'input_token_counts': all(r.get('prompt_tokens') == r.get('expected_salted_prompt_tokens')
                                        for r in records if r['error'] is None),
              'output_token_counts': bool(samples) and all(r['output_tokens'] == plan['configuration']['load']['output_tokens'] for r in samples)}
    metrics = summarize(samples, result['summary']['wall_seconds'], len(records), checks, client='swe')
    metrics.update(native_summary=result['summary'], protocol=result['protocol'])
    return metrics, samples


def extract(plan, attempt, inputs):
    result, samples = (extract_ais if plan['client'] == 'aisbench' else extract_swe)(plan, Path(attempt), Path(inputs))
    save_json(Path(attempt)/'performance.json', result)
    save_json(Path(attempt)/'request-samples.json', samples)
    return result
