"""Strict device-time adapters: CUPTI correlation and Ascend Chrome flow links.

Host submission ranges never stand in for device execution duration. Unsupported
schemas or ambiguous links fail explicitly. All output times use MONOTONIC ns.
"""
import ast
from bisect import bisect_right
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
import sqlite3

from common import read_json


def block_hex(value):
    if len(value) == 32 and all(c in '0123456789abcdefABCDEF' for c in value):
        return value.lower()
    result = bytes(ast.literal_eval(value))
    if len(result) != 16:
        raise ValueError('Unsupported native block ID representation')
    return result.hex()


def attributes(label):
    parts = label.split('|')
    if parts[0] in {'UCM_H2D_ATTR', 'UCM_D2H_ATTR'} and len(parts) == 4:
        return {'stage': 'H2D' if parts[0]=='UCM_H2D_ATTR' else 'D2H',
                'task': int(parts[1]), 'block': block_hex(parts[2]), 'layer': int(parts[3])}
    if parts[0] in {'BUNDLE_EXECUTE', 'BUNDLE_FORWARD', 'BUNDLE_SAMPLE'} and len(parts) == 2:
        return {'batch': int(parts[1])}
    return {}


class ScopeIndex:
    """Per-thread interval search; overlapping different identities are errors."""
    def __init__(self, rows):
        buckets = defaultdict(list)
        for row in rows:
            attrs = attributes(row.get('label', ''))
            if attrs:
                if 'stage' in attrs:
                    attrs.update(submission_start_ns=row['start_ns'], submission_tid=row['tid'])
                buckets[(row['pid'], row['tid'])].append((row['start_ns'], row['end_ns'], attrs))
        self.buckets = {}
        for key, values in buckets.items():
            values.sort(key=lambda x:x[0])
            maxima, previous = [], -1
            for _, end, _ in values:
                previous = max(previous, end)
                maxima.append(previous)
            self.buckets[key] = ([v[0] for v in values], maxima, values)

    def get(self, pid, tid, start, end, tolerance=0):
        item = self.buckets.get((pid, tid))
        if item is None:
            return {}
        starts, maxima, values = item
        cursor = bisect_right(starts, start+tolerance)-1
        found = {}
        while cursor >= 0 and maxima[cursor]+tolerance >= end:
            a, b, attrs = values[cursor]
            if a-tolerance <= start and end <= b+tolerance:
                for name, value in attrs.items():
                    if name in found and found[name] != value:
                        raise ValueError('Ambiguous host submission scope; clock error or incompatible instrumentation')
                    found[name] = value
            cursor -= 1
        return found


def clock_alignment(worker, trace_anchors):
    """Bound a constant trace-minus-monotonic offset using in-scope markers."""
    anchors = {a['monotonic_ns']:a for a in worker['clock_anchors']}
    intervals = []
    for label, stamp in trace_anchors:
        key = int(label.split('|')[1])
        if key in anchors:
            a = anchors[key]
            intervals.append((stamp-a['end_ns'], stamp-a['monotonic_ns']))
    if len(intervals) < 4:
        raise ValueError('Fewer than four clock anchors; cannot align device and native traces')
    low, high = max(a for a,b in intervals), min(b for a,b in intervals)
    if low > high:
        raise ValueError('Clock anchor bounds disagree; refuse a drifting/incorrect clock conversion')
    return {'offset_ns': (low+high)//2, 'uncertainty_ns': (high-low+1)//2,
            'anchors': len(intervals), 'offset_bounds_ns': [low, high]}


def nsight(database, workers, native):
    connection = sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro&immutable=1', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        return _nsight(connection, workers, native)
    finally:
        connection.close()


def _nsight(con, workers, native):
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    required = {'StringIds', 'NVTX_EVENTS', 'CUPTI_ACTIVITY_KIND_KERNEL'}
    if not required <= tables:
        raise ValueError('Unsupported/incomplete Nsight SQLite schema: '+str(required-tables))
    strings = dict(con.execute('select id,value from StringIds'))
    anchors, trace_pids = defaultdict(list), {}
    anchor_workers = {a['monotonic_ns']:w['pid'] for w in workers for a in w['clock_anchors']}
    for r in con.execute('select * from NVTX_EVENTS'):
        label = r['text'] or strings.get(r['textId'], '')
        if label.startswith('BUNDLE_CLOCK|'):
            pid = anchor_workers.get(int(label.split('|')[1]))
            if pid is not None:
                trace_pid = r['globalTid'] & ~((1<<24)-1)
                if trace_pid in trace_pids and trace_pids[trace_pid] != pid:
                    raise ValueError('Nsight process identity changed')
                trace_pids[trace_pid] = pid
                anchors[pid].append((label, r['start']))
    clocks = {w['pid']:clock_alignment(w, anchors[w['pid']]) for w in workers}
    bounds = {w['pid']:(min(a['monotonic_ns'] for a in w['clock_anchors']),
                        max(a['end_ns'] for a in w['clock_anchors'])) for w in workers}
    scopes = ScopeIndex([*native, *[e for w in workers for e in w['events'] if e['event']=='scope']])
    api, api_rows = {}, {}
    for table in ['CUPTI_ACTIVITY_KIND_RUNTIME', 'CUPTI_ACTIVITY_KIND_DRIVER']:
        if table not in tables:
            continue
        for r in con.execute('select rowid,* from '+table):
            trace_pid = r['globalTid'] & ~((1<<24)-1)
            pid = trace_pids.get(trace_pid)
            if pid is None:
                continue
            clock = clocks[pid]
            attrs = scopes.get(pid, r['globalTid'] & ((1<<24)-1),
                               r['start']-clock['offset_ns'], r['end']-clock['offset_ns'], clock['uncertainty_ns'])
            key = (trace_pid, r['correlationId'])
            if attrs:
                for name,value in attrs.items():
                    if name in api.get(key,{}) and api[key][name] != value:
                        raise ValueError('CUPTI runtime/driver correlation disagrees')
                api.setdefault(key, {}).update(attrs)
                api_rows.setdefault(key, []).append({'table':table, 'rowid':r['rowid']})
    events, uncovered = [], []
    for table in ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_KERNEL']:
        if table not in tables:
            continue
        for row in con.execute('select rowid,* from '+table+' order by start'):
            r = dict(row)
            pid = trace_pids.get(r['globalPid'])
            if pid is None:
                continue
            clock = clocks[pid]
            key = (r['globalPid'], r['correlationId'])
            start, end = r['start']-clock['offset_ns'], r['end']-clock['offset_ns']
            if end < bounds[pid][0] or start > bounds[pid][1]:
                continue
            attrs = api.get(key, {})
            if table.endswith('MEMCPY'):
                if 'stage' not in attrs:  # Framework copies remain in raw SQLite.
                    continue
                if r['copyKind'] != (1 if attrs['stage']=='H2D' else 2):
                    raise ValueError('CUPTI copy direction differs from native transfer scope')
                stage = attrs['stage']
            else:
                stage = 'device_compute'
                if 'batch' not in attrs:
                    uncovered.append({'table':table, 'rowid':r['rowid'], 'pid':pid})
                    continue
            events.append({'stage':stage, 'pid':pid, 'start_ns':r['start']-clock['offset_ns'],
                           'end_ns':r['end']-clock['offset_ns'], 'bytes':r.get('bytes',0),
                           **attrs, 'source':{'database':'Nsight SQLite',
                             'table':table, 'rowid':r['rowid'], 'correlation_id':r['correlationId'],
                             'stream':r.get('streamId'), 'host_api_rows':api_rows.get(key,[])}})
    return {'events':events, 'clocks':clocks, 'unattributed_device_events':uncovered,
            'adapter':'Nsight CUPTI process+correlationId; native submission scope; device timestamps'}


def ns(us):
    return int(Decimal(str(us))*1000)


def ascend(trace_path, worker, native):
    document = read_json(trace_path)
    raw = document['traceEvents'] if isinstance(document,dict) else document
    anchors = [(r['name'], ns(r['ts'])) for r in raw if r.get('name','').startswith('BUNDLE_CLOCK|') and r.get('ph')=='X']
    clock = clock_alignment(worker, anchors)
    pid = worker['pid']
    host_pids = {r['pid'] for r in raw if r.get('name','').startswith('BUNDLE_CLOCK|')}
    if len(host_pids) != 1:
        raise ValueError('Ascend trace does not identify a unique framework process')
    host_pid = next(iter(host_pids))
    scopes = ScopeIndex([*native, *[e for e in worker['events'] if e['event']=='scope']])
    complete, endpoints, flows = {}, defaultdict(list), defaultdict(dict)
    for index,r in enumerate(raw):
        if r.get('ph') == 'X' and r.get('dur',0) > 0:
            complete[index] = r
            endpoints[(r['pid'],r['tid'],ns(r['ts']))].append(index)
        elif r.get('ph') in {'s','f'}:
            category = r.get('cat',r.get('name',''))
            if category.lower() not in {'hosttodevice','async_npu','async_task_queue'}:
                continue
            key = (category, str(r.get('id',r.get('id2'))))
            phase = r['ph']
            if phase in flows[key]:
                raise ValueError('Duplicate Ascend flow endpoint; unsupported flow-ID scope')
            flows[key][phase] = r
    # async_npu gives the framework-to-device mapping, HostToDevice gives the
    # ACL-to-device mapping. Merge their evidence by exact device endpoint.
    attributed, origins, device_ids = {}, defaultdict(list), set()
    for key, pair in flows.items():
        if not {'s','f'} <= pair.keys():
            continue
        a,b = pair['s'],pair['f']
        if key[0].lower() == 'async_task_queue':
            continue
        matches = endpoints.get((b['pid'],b['tid'],ns(b['ts'])), [])
        if len(matches) != 1:
            raise ValueError('Ascend device flow must resolve to exactly one complete device event')
        target = matches[0]
        device_ids.add(target)
        if a['pid'] != host_pid:
            continue
        # PID remapping is anchored; Linux TID is required. An incompatible
        # profiler thread mapping is a hard attribution failure, not a guess.
        stamp = ns(a['ts'])-clock['offset_ns']
        attrs = scopes.get(pid, a['tid'], stamp, stamp, clock['uncertainty_ns'])
        if attrs:
            for name,value in attrs.items():
                if name in attributed.get(target,{}) and attributed[target][name] != value:
                    raise ValueError('Ascend flow paths disagree on request/batch/transfer identity')
            attributed.setdefault(target,{}).update(attrs)
            origins[target].append({'category':key[0], 'id':key[1]})
    if not device_ids:
        raise ValueError('No supported Ascend framework/ACL-to-device flow links')
    hardware_pids = {complete[i]['pid'] for i in device_ids}
    # Do not silently omit a kernel merely because its flow was dropped.
    device_ids.update(i for i,r in complete.items() if r['pid'] in hardware_pids)
    events, uncovered = [], []
    for index in sorted(device_ids):
        r, attrs = complete[index], attributed.get(index,{})
        if not attrs or ('stage' not in attrs and 'batch' not in attrs):
            uncovered.append({'trace':str(trace_path), 'event_index':index, 'name':r['name']})
            continue
        stage = attrs.get('stage','device_compute')
        # Copy byte size is reported only when present in the actual trace.
        # Native counter totals are independently retained by boundary snapshots.
        size = r.get('args',{}).get('Bytes') if stage != 'device_compute' else 0
        events.append({'pid':pid,'stage':stage, **attrs, 'start_ns':ns(r['ts'])-clock['offset_ns'],
                       'end_ns':ns(r['ts'])+ns(r['dur'])-clock['offset_ns'], 'bytes':int(size) if size is not None else None,
                       'source':{'trace':str(trace_path),'event_index':index,'name':r['name'],
                                 'flows':origins[index], 'stream':r.get('args',{}).get('Stream Id',r['tid'])}})
    return {'events':events,'clocks':{pid:clock},'unattributed_device_events':uncovered,
            'adapter':'Ascend Chrome async_npu/HostToDevice exact flow endpoint; device timestamps'}
