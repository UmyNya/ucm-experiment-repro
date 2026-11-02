"""Native UCM file shape, legacy migration, and experiment-owned capacities."""
from __future__ import annotations

import copy


MANAGED_STORE_FIELDS = {
    'cache_buffer_capacity_gb': 'capacity.l1',
    'posix_capacity_gb': 'capacity.l2.gib',
    'storage_backends': 'paths.l2_root (a private directory is generated per run)',
}
LEGACY_FIELDS = {'enabled', 'source_ref', 'launch', 'store'}


def normalize_ucm(config):
    """Normalize both file formats to native UCM plus separate runtime settings."""
    raw = config.get('ucm')
    if not isinstance(raw, dict):
        raise ValueError('ucm must be an object')
    runtime = config.setdefault('ucm_runtime', {})
    if not isinstance(runtime, dict) or set(runtime) - {'enabled', 'source_ref'}:
        raise ValueError('ucm_runtime supports only enabled and source_ref')
    if LEGACY_FIELDS & raw.keys():
        if set(raw) - LEGACY_FIELDS:
            raise ValueError('Do not mix legacy UCM launch/store metadata with native UCM fields')
        if not isinstance(raw.get('launch'), dict) or not isinstance(raw.get('store'), dict):
            raise ValueError('Legacy UCM configuration requires launch and store objects')
        for name in ['enabled', 'source_ref']:
            if name in raw:
                if name in runtime:
                    raise ValueError(f'Define {name} in legacy ucm or ucm_runtime, never both')
                runtime[name] = raw[name]
        if 'ucm_connectors' in raw['launch'] or LEGACY_FIELDS & raw['launch'].keys():
            raise ValueError('Legacy ucm.launch must contain only native launch parameters')
        raw = {**raw['launch'], 'ucm_connectors': [
            {'ucm_connector_name': 'UcmPipelineStore', 'ucm_connector_config': raw['store']}]}
        config['ucm'] = raw
    runtime.setdefault('enabled', True)
    runtime.setdefault('source_ref', 'HEAD')
    if type(runtime['enabled']) is not bool:
        raise ValueError('ucm_runtime.enabled must be boolean')
    if not isinstance(runtime['source_ref'], str) or not runtime['source_ref'].strip():
        raise ValueError('ucm_runtime.source_ref must be a nonempty Git reference')
    connectors = raw.get('ucm_connectors')
    if not isinstance(connectors, list) or len(connectors) != 1:
        raise ValueError('This experiment protocol requires exactly one UcmPipelineStore connector')
    connector = connectors[0]
    if not isinstance(connector, dict) or connector.get('ucm_connector_name') != 'UcmPipelineStore':
        raise ValueError('This experiment protocol requires ucm_connector_name=UcmPipelineStore')
    store = connector.get('ucm_connector_config')
    if not isinstance(store, dict):
        raise ValueError('ucm_connectors[0].ucm_connector_config must be an object')
    if store.get('store_pipeline') != 'Cache|Posix':
        raise ValueError('This experiment protocol requires store_pipeline=Cache|Posix')
    for name, owner in MANAGED_STORE_FIELDS.items():
        if store.get(name) is not None:
            raise ValueError(f'Configure {name} through {owner}; do not duplicate it in ucm.json')


def store_config(config):
    return config['ucm']['ucm_connectors'][0]['ucm_connector_config']


def apply_ucm_override(config, key, value):
    """Apply native dotted paths (including list indexes) or old launch/store aliases."""
    if not key.startswith(('ucm.', 'ucm_runtime.')):
        return False
    parts = key.split('.')
    if parts[:2] == ['ucm', 'launch']:
        if len(parts) == 2:
            raise ValueError('Override individual ucm.launch fields or use the native ucm object')
        parts = ['ucm', *parts[2:]]
    elif parts[:2] == ['ucm', 'store']:
        parts = ['ucm', 'ucm_connectors', '0', 'ucm_connector_config', *parts[2:]]
    elif parts[:2] in [['ucm', 'enabled'], ['ucm', 'source_ref']]:
        parts[0] = 'ucm_runtime'
    if any(not part for part in parts):
        raise ValueError(f'Invalid UCM override: {key}')
    target = config
    for part in parts[:-1]:
        if isinstance(target, dict):
            target = target.setdefault(part, {})
        elif isinstance(target, list) and part.isdecimal() and int(part) < len(target):
            target = target[int(part)]
        else:
            raise ValueError(f'Invalid UCM override path: {key}')
    last = parts[-1]
    if isinstance(target, dict):
        target[last] = value
    elif isinstance(target, list) and last.isdecimal() and int(last) < len(target):
        target[int(last)] = value
    else:
        raise ValueError(f'Invalid UCM override path: {key}')
    return True


def effective_ucm(config, capacity, l2_path):
    """Preserve native fields and fill only the three experiment-owned store values."""
    native = copy.deepcopy({k: v for k, v in config['ucm'].items() if v is not None})
    connector = native['ucm_connectors'][0]
    store = {k: v for k, v in connector['ucm_connector_config'].items() if v is not None}
    store.update(storage_backends=str(l2_path),
                 cache_buffer_capacity_gb=capacity['cache_buffer_capacity_gb'],
                 posix_capacity_gb=capacity['posix_capacity_gb'])
    connector['ucm_connector_config'] = store
    return native
