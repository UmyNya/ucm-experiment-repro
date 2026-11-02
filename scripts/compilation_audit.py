"""Out-of-window cache inventories and observed compiler activity checks."""
from pathlib import Path
import re

from common import sha256, within

ARTIFACT_SUFFIXES = {'.cubin', '.ptx', '.so', '.o', '.om', '.bin', '.cce', '.bc', '.ll', '.json', '.pkl', '.pickle'}


def inventory(roots):
    files = {}
    roots = sorted({str(Path(p).resolve()) for p in roots})
    for root in roots:
        directory = Path(root)
        if not directory.exists():
            continue
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ARTIFACT_SUFFIXES:
                if not any(within(path, r) for r in roots):
                    raise ValueError('Compiler cache symlink escapes the recorded cache roots')
                st = path.stat()
                files[str(path.resolve())] = {'bytes': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha256(path)}
    return {'roots': roots, 'files': files}


def evaluate(before, after, before_cache, after_cache, log_text, *, graph_enabled, platform):
    errors, changes, workers = [], [], []
    old = {r['pid']: r for r in before}
    new = {r['pid']: r for r in after}
    if old.keys() != new.keys() or not old:
        errors.append('Compiler observer worker set changed or is empty')
    for pid in old.keys() & new.keys():
        a, b = old[pid], new[pid]
        if a['installed'] != b['installed']:
            errors.append(f'Worker {pid}: compiler hook coverage changed during measurement')
        for label in ['dynamo'] + (['device_graph_capture'] if graph_enabled else []):
            if not a['installed'].get(label) or not b['installed'].get(label):
                errors.append(f'Worker {pid}: {label} observation unavailable or replaced')
        triton_present = a.get('triton_package_present') or b.get('triton_package_present')
        if triton_present and (not a['installed'].get('triton') or not b['installed'].get('triton')):
            errors.append(f'Worker {pid}: installed Triton has no verified cache_hit listener; provide a matching runtime adapter')
        if b['events'][:len(a['events'])] != a['events']:
            errors.append(f'Worker {pid}: compilation event history changed')
        events = b['events'][len(a['events']):]
        unexpected = [x for x in events if x['kind'] != 'triton_compile' or x.get('cache_hit') is not True]
        if unexpected:
            errors.append(f'Worker {pid}: actual compilation/capture or unknown cache status in measured window')
        workers.append({'pid': pid, 'events': events, 'triton_disk_cache_hits': sum(x['kind']=='triton_compile' and x.get('cache_hit') is True for x in events),
                        'unexpected_events': unexpected, 'observer_scope': b.get('scope')})
    for name, record in after_cache['files'].items():
        if name not in before_cache['files'] or before_cache['files'][name] != record:
            changes.append(name)
    removed = sorted(set(before_cache['files'])-set(after_cache['files']))
    if changes or removed:
        errors.append('Compiler cache artifacts were added, changed or removed during the measured window')
    stage_patterns = [r'torch\.compile took', r'Autotuning process starts', r'Graph capturing finished',
                      r'(?i)compil(?:ing|ation).*(?:graph|kernel)', r'(?i)(?:graph|kernel).*compil(?:ing|ation)']
    warnings = []
    for line in log_text.splitlines():
        # Historical vLLM Triton post_compile warnings also fire on disk-cache
        # hits; classify these with listener events, not warning counts.
        if 'Triton kernel JIT compilation during inference:' in line:
            continue
        if any(re.search(pattern, line) for pattern in stage_patterns):
            warnings.append(line)
    if warnings:
        errors.append('Compilation/autotuning/capture log evidence requires review')
    return {'passed': not errors, 'errors': errors, 'platform': platform, 'workers': workers,
            'new_or_changed_artifacts': changes, 'removed_artifacts': removed, 'stage_log_lines': warnings,
            'scope': 'Observed compiler APIs, configured local compile caches and measurement log. Does not claim universal coverage of arbitrary third-party in-memory compilers.'}
