"""Validate actual worker capacity and create identity-bound calibration files."""
import math

from block_size import observed_block_size
from common import utc


def validate_workers(plan, rows, *, enforce_targets=True):
    c, expected = plan['configuration'], plan['capacity']
    count = math.prod(c['server'][k] for k in ['tensor_parallel_size', 'pipeline_parallel_size', 'data_parallel_size'])
    errors = []
    if len(rows) != count or len({r['pid'] for r in rows}) != count:
        errors.append(f'Expected {count} unique workers, observed {len(rows)}')
    workers = []
    domains = {}
    try:
        actual_block = observed_block_size(rows)
    except ValueError as error:
        actual_block = None
        errors.append(str(error))
    expected_block = c['server'].get('block_size')
    if expected_block is not None and actual_block != expected_block:
        errors.append('Observed token block size differs from config')
    calculation_block = expected.get('calculation_block_size')
    if calculation_block is not None and actual_block != calculation_block:
        errors.append('Observed token block size differs from the capacity calculation/calibration')
    for row in rows:
        layout = row['capacity_layout']
        for key in ['block_tokens', 'hbm_bytes_per_block', 'l1_node_bytes', 'l1_nodes_per_token_block']:
            if type(layout[key]) is not int or layout[key] <= 0:
                raise ValueError(f'Invalid observed {key}')
        for key in ['hbm_reserved_blocks', 'l1_reserved_nodes']:
            if type(layout[key]) is not int or layout[key] < 0:
                raise ValueError(f'Invalid observed {key}')
        expected_platform = 'cuda' if c['platform']['kind'] == 'cuda' else 'npu'
        if row['platform'] != expected_platform:
            errors.append('Observed device backend differs from config')
        if type(row['hbm_blocks']) is not int or row['hbm_blocks'] <= layout['hbm_reserved_blocks']:
            raise ValueError('Invalid observed HBM block count: no allocatable blocks')
        block = layout['block_tokens']
        hbm = row['hbm_blocks'] * block
        l1 = (row['l1_nodes'] - layout['l1_reserved_nodes']) // layout['l1_nodes_per_token_block'] * block
        if hbm != row['hbm_tokens'] or l1 != row['l1_ordinary_tokens']:
            errors.append('Observer capacity values disagree with its layout')
        if row['kv_tensor_bytes'] != row['hbm_blocks'] * layout['hbm_bytes_per_block']:
            errors.append('Actual KV tensor bytes disagree with block allocation')
        budget = expected['kv_cache_memory_bytes']
        if c['capacity']['hbm']['mode'] != 'auto' and (type(budget) is not int or budget <= 0):
            raise ValueError('Explicit HBM mode requires a positive physical byte budget')
        if budget is not None and row['hbm_blocks'] > budget // layout['hbm_bytes_per_block']:
            errors.append('Actual HBM KV allocation exceeds requested physical budget')
        nodes = expected['cache_buffer_capacity_gb'] * (1 << 30) // layout['l1_node_bytes']
        if row['l1_nodes'] != nodes:
            errors.append(f'L1 nodes differ from physical budget: {row["l1_nodes"]} != {nodes}')
        if enforce_targets:
            target = expected['hbm_target_tokens']
            if target is not None and target % block:
                errors.append('HBM token target is not aligned to the observed block size')
            if target is not None and hbm != target:
                errors.append(f'HBM tokens differ: actual {hbm}, target {target}; calibrate target backend')
            target = expected['l1_target_tokens']
            if target is not None:
                if l1 < target or (c['capacity']['l1']['rounding'] == 'exact' and l1 != target):
                    errors.append(f'L1 ordinary tokens differ: actual {l1}, target {target}')
        domain_shape = (layout['l1_node_bytes'], layout['l1_nodes_per_token_block'], row['l1_nodes'])
        if row['unique_id'] in domains and domains[row['unique_id']] != domain_shape:
            errors.append('Workers sharing an L1 domain report incompatible layouts')
        domains[row['unique_id']] = domain_shape
        workers.append({'pid': row['pid'], 'rank': row['worker_rank'], 'layout': layout,
                        'hbm_physical_tokens': hbm, 'hbm_allocatable_token_upper_bound':
                        (row['hbm_blocks'] - layout['hbm_reserved_blocks']) * block,
                        'kv_tensor_bytes': row['kv_tensor_bytes'], 'l1_ordinary_tokens': l1,
                        'l1_payload_bytes': row['l1_nodes'] * layout['l1_node_bytes'],
                        'shared_memory': row['shared_memory'], 'unique_id': row['unique_id']})
    return {'checked_utc': utc(), 'passed': not errors, 'errors': errors, 'workers': workers,
            'observed_block_size': actual_block,
            'unique_l1_domains': len(domains), 'enforced_token_targets': enforce_targets,
            'scope': 'per-worker KV allocation and per-domain L1 payload; not total device/host RSS'}


def calibration_record(plan, rows):
    validation = validate_workers(plan, rows, enforce_targets=False)
    if not validation['passed']:
        raise ValueError('Invalid observed capacity: ' + '; '.join(validation['errors']))
    identity = {**plan['model_identity'], 'block_size': validation['observed_block_size']}
    return {'schema_version': 1, 'created_utc': utc(), 'identity': identity,
            'workers': [dict(r['capacity_layout'], worker_rank=r['worker_rank'], worker_pid=r['pid'],
                             unique_id=r['unique_id']) for r in rows],
            'validation': validation, 'source_run': plan['run_dir'],
            'basis': 'actual worker KV tensors and native L1 shared-buffer node count'}
