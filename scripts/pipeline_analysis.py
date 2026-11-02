"""Request/task/layer provenance and five-stage metrics from completed captures."""
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re

from common import read_json, save_json
from device_trace import block_hex, attributes
from client_metrics import distribution

STAGES = ('L2_read','H2D','device_compute','D2H','L2_write')


def merge(intervals):
    result = []
    for a,b in sorted((a,b) for a,b in intervals if b>a):
        if result and a <= result[-1][1]:
            result[-1][1] = max(b,result[-1][1])
        else:
            result.append([a,b])
    return result


def duration(intervals):
    return sum(b-a for a,b in merge(intervals))


def event_statistics(events, start, end):
    relevant=[e for e in events if e['start_ns']<end and e['end_ns']>start]
    complete=[e for e in relevant if e['start_ns']>=start and e['end_ns']<=end]
    crossing=[e for e in relevant if e['start_ns']<start or e['end_ns']>end]
    intervals=[(max(start,e['start_ns']),min(end,e['end_ns'])) for e in relevant]
    union=duration(intervals);total=sum(b-a for a,b in intervals)
    span=max((b for a,b in intervals),default=start)-min((a for a,b in intervals),default=start)
    def byte_sum(rows):return sum(r['bytes'] for r in rows) if all(r['bytes'] is not None for r in rows) else None
    count=curve(intervals,start,end)
    complete_bytes=byte_sum(complete)
    return {'overlapping_events':len(relevant),'fully_contained_events':len(complete),'boundary_crossing_events':len(crossing),
            'sum_ms':total/1e6,'union_ms':union/1e6,'span_ms':span/1e6,'time_coverage':union/(end-start),
            'coverage_pct':100*union/(end-start),'mean_inflight':total/(end-start),'peak_inflight':count['peak'],
            'complete_event_bytes':complete_bytes,'boundary_crossing_event_bytes':byte_sum(crossing),
            'complete_event_latency_ms':distribution([(e['end_ns']-e['start_ns'])/1e6 for e in complete]),
            'effective_GiB_per_s_over_span':complete_bytes/2**30/(span/1e9) if span and complete_bytes is not None else None,
            'bytes_semantics':'Only fully contained events count; crossing bytes are separate and never time-prorated'}


def intersection(left, right):
    left,right = merge(left),merge(right)
    i=j=0
    out=[]
    while i<len(left) and j<len(right):
        a,b = max(left[i][0],right[j][0]),min(left[i][1],right[j][1])
        if b>a:
            out.append([a,b])
        if left[i][1]<right[j][1]: i+=1
        else: j+=1
    return out


def curve(intervals, origin, end):
    changes = defaultdict(int)
    for a,b in intervals:
        a,b = max(a,origin),min(b,end)
        if b>a:
            changes[a]+=1
            changes[b]-=1
    points,current,area,last,peak = [[0,0]],0,0,origin,0
    for stamp,change in sorted(changes.items()):
        area += (stamp-last)*current
        current += change
        peak = max(peak,current)
        last = stamp
        points.append([(stamp-origin)/1e6,current])
    if current != 0:
        raise ValueError('Nonzero terminal count in I/O interval curve')
    return {'points_ms_count':points,'peak':peak,'mean':area/(end-origin) if end>origin else None}


def jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def early_log(text):
    reservations,jobs,observations = {},{},defaultdict(list)
    for line in text.splitlines():
        m = re.search(r'UCM_EARLY (\w+) (.*)',line)
        if not m: continue
        kind,body = m.groups()
        f = dict(re.findall(r'(\w+)=([^\s]+)',body))
        pid = f.get('pid')
        if pid is None:
            process = re.search(r'\[(\d+),\d+\]',line)
            pid = process[1] if process else None
        if kind == 'batch_reserved' and f.get('reservation') not in {None,'None','0'}:
            if pid is None:
                raise ValueError('Early reservation log omitted worker PID')
            key=(int(pid),int(f['reservation']))
            if key in reservations and reservations[key]!=f['key']:
                raise ValueError('Early reservation identity reused')
            reservations[key]=f['key']
        if kind in {'schedule_match','hbm_scheduled','handoff'} and 'request' in f:
            observations[f['request']].append({'kind':kind,**f})
            if f.get('key') not in {None,'None'}:
                if f['key'] in jobs and jobs[f['key']]!=f['request']:
                    raise ValueError('Early job key mapped to multiple requests')
                jobs[f['key']]=f['request']
    return reservations,jobs,observations


def analyze(plan, attempt, workers, native, device, matches, server_text):
    """No fixed N, TP, model namespace, layer count, or physical block size."""
    attempt=Path(attempt)
    checks,errors = {},[]
    def check(name,passed):
        checks[name]=bool(passed)
        if not passed: errors.append(name)
    check('worker_capture_completed', bool(workers) and all(w.get('failure') is None for w in workers))
    check('no_native_trace_write_errors', 'BUNDLE_NATIVE_TRACE_ERROR' not in server_text)
    check('all_device_events_attributed', not device['unattributed_device_events'])
    identities,by_pid,batches,loads,dumps = {},{w['pid']:w for w in workers},{},{},{}
    for w in workers:
        for rid,row in w['requests'].items():
            if rid in identities and any(identities[rid][k]!=row[k] for k in ['ordinal','tokens','sha256_u32le']):
                raise ValueError('Cross-worker request identity mismatch')
            identities[rid]=row
        for event in w['events']:
            kind=event['event']
            if kind=='execute':
                batches[(w['pid'],event['batch'])]=event
            elif kind in {'load_submit','dump_submit'}:
                target=loads if kind=='load_submit' else dumps
                key=(w['pid'],event['task'])
                if key in target:
                    raise ValueError('Native task ID reused within capture')
                target[key]=event
    ordinals={r['ordinal'] for r in identities.values()}
    check('all_frozen_requests_observed', ordinals==set(range(plan['request_count'])))
    check('unique_request_id_per_input', len(identities)==len(ordinals))
    observed_matches=defaultdict(list)
    for row in matches:
        rid=row['request_id']
        if rid in identities:
            check('prompt_hash:'+rid,row['prompt_sha256']==identities[rid]['sha256_u32le'])
            observed_matches[rid].append(row)
    # Logical block provenance is captured from the actual scheduler hasher;
    # physical block IDs are independently recorded at submit_load/dump.
    logical={rid:{b for m in rows for b in m.get('ucm_blocks',[])} for rid,rows in observed_matches.items()}
    for kind,tasks in [('load',loads),('dump',dumps)]:
        for key,event in tasks.items():
            for rid,blocks in event.get('logical_blocks_by_request',{}).items():
                check(f'{kind}_logical_blocks:{key}:{rid}',set(blocks)<=logical.get(rid,set()))
    reservations,jobs,early=early_log(server_text)
    links={'read':{},'write':{}}
    credits=defaultdict(list)
    events=[]
    label_counts=Counter()
    diagnostics=[]
    transfer_scopes=set()
    for index,row in enumerate(native):
        parts=row['label'].split('|');kind=parts[0];pid=row['pid']
        label_counts[kind]+=1
        if kind in {'UCM_READ_LINK','UCM_WRITE_LINK'}:
            target=links['read' if kind=='UCM_READ_LINK' else 'write']
            key=(pid,int(parts[3])); value={'kind':parts[1],'parent':int(parts[2])}
            if len(parts)>4: value['block']=block_hex(parts[4])
            if key in target and target[key]!=value:
                raise ValueError('Conflicting native backend task linkage')
            target[key]=value
        elif kind=='UCM_CREDIT':
            credits[(pid,int(parts[2]))].append((row['start_ns'],int(parts[3])))
        elif kind in {'UCM_L2_READ','UCM_L2_WRITE'}:
            if len(parts)!=7:
                raise ValueError('Unsupported native POSIX span format')
            events.append({'stage':'L2_read' if kind=='UCM_L2_READ' else 'L2_write',
                           'pid':pid,'start_ns':row['start_ns'],'end_ns':row['end_ns'],
                           'bytes':int(parts[1]),'task':int(parts[2]),'block':block_hex(parts[3]),'layer':int(parts[4]),
                           'full_block_read':parts[5]=='true','full_block_write':parts[6]=='true',
                           'source':{'native_event_index':index}})
        elif kind in {'UCM_H2D_ATTR','UCM_D2H_ATTR'}:
            transfer_scopes.add((pid,row['tid'],row['start_ns']))
        elif not row.get('point'):
            diagnostics.append(row)
    def members(task_map,pid,task,block):
        item=task_map.get((pid,task))
        if item is None:
            raise ValueError(f'No request task linkage for PID {pid}, task {task}')
        rids=[rid for rid,blocks in item['blocks_by_request'].items() if block in blocks]
        if not rids or any(rid not in identities for rid in rids):
            raise ValueError('Native block does not belong to observed task requests')
        return sorted(rids)
    for e in events:
        direction='read' if e['stage']=='L2_read' else 'write'
        link=links[direction].get((e['pid'],e['task']))
        if link is None: raise ValueError('Missing native I/O origin linkage')
        if link['kind']=='early':
            rid=jobs[reservations[(e['pid'],link['parent'])]]
            if rid not in identities: raise ValueError('Early read belongs to an unobserved request')
            e['requests']=[rid]
        else:
            e['requests']=members(loads if direction=='read' else dumps,e['pid'],link['parent'],e['block'])
            if link['kind']=='full' and link['block']!=e['block']:
                raise ValueError('Full-block write origin differs from physical write')
        e['provenance']=link
    observed_transfers=set()
    for raw in device['events']:
        e=dict(raw)
        if e['stage'] in {'H2D','D2H'}:
            e['requests']=members(loads if e['stage']=='H2D' else dumps,e['pid'],e['task'],e['block'])
            observed_transfers.add((e['pid'],e['submission_tid'],e['submission_start_ns']))
        else:
            batch=batches.get((e['pid'],e['batch']))
            if batch is None: raise ValueError('Device compute has no observed worker batch')
            e['requests']=sorted(rid for rid,n in batch['scheduled_tokens'].items() if n>0)
            if not e['requests'] or any(rid not in identities for rid in e['requests']):
                raise ValueError('Device batch has unknown request membership')
        events.append(e)
    check('all_native_transfers_have_device_execution',transfer_scopes==observed_transfers)
    check('positive_device_and_io_durations',all(e['end_ns']>e['start_ns'] for e in events))
    statistics_path=attempt/'native-statistics.json'
    if statistics_path.is_file():
        observed=read_json(statistics_path)['counter_totals']
        for stage,counter in [('L2_read','posix_load_read_bytes_total'),('L2_write','posix_dump_write_bytes_total')]:
            check(stage+'_native_byte_conservation',sum(e['bytes'] for e in events if e['stage']==stage)==observed.get(counter,0))
    else:
        check('native_statistics_available',bool(plan.get('validation_only')))
    samples=read_json(attempt/'request-samples.json')
    start=min(round(r['start_s']*1e9) for r in samples)
    end=max(round(r['end_s']*1e9) for r in samples)
    full_end=max([end,*[e['end_ns'] for e in events]])
    grouped=defaultdict(lambda:defaultdict(list))
    for e in events:
        for rid in e['requests']: grouped[rid][e['stage']].append(e)
    sample_by_id={int(s['id']):s for s in samples}
    rows=[]
    for rid,info in sorted(identities.items(),key=lambda item:item[1]['ordinal']):
        sample=sample_by_id[info['ordinal']]
        own=grouped[rid]
        runs=[b for b in batches.values() if b['scheduled_tokens'].get(rid,0)>0]
        first=min((b['start_ns'] for b in runs),default=None)
        # PP/TP repeat scheduled tokens. Per-worker totals are retained and
        # compared, never summed into a fictitious request compute total.
        computed={str(pid):sum(b['scheduled_tokens'].get(rid,0) for (p,_),b in batches.items() if p==pid)
                  for pid in by_pid if rid in by_pid[pid]['requests']}
        if plan.get('configuration',{}).get('load',{}).get('output_tokens',1)==1:
            check('single_token_prompt_compute_conservation:'+rid,
                  bool(computed) and all(value+info['initial_computed_tokens']==info['tokens'] for value in computed.values()))
        prefix_candidates=[m for m in observed_matches[rid] if m['external_hit_tokens'] is not None and m['external_hit_tokens']>=0]
        prefix=next((m for m in prefix_candidates if m['num_computed_tokens']+m['external_hit_tokens']==info['initial_computed_tokens']),None)
        check('prefix_observed:'+rid,prefix is not None)
        schedule=[e for e in early[rid] if e['kind'] in {'schedule_match','hbm_scheduled'}]
        last=schedule[-1] if schedule else {}
        def stamp(name):
            value=last.get(name)
            return int(value) if value not in {None,'None'} else None
        ready,scheduled=stamp('ready_received_ns'),stamp('scheduled_ns')
        if ready is not None:
            check('READY_before_execute:'+rid,first is not None and first>=ready)
        row={'ordinal':info['ordinal'],'request_id':rid,'tokens':info['tokens'],'sha256_u32le':info['sha256_u32le'],
             'client':sample,'computed_tokens_per_worker':computed,'initial_computed_tokens':info['initial_computed_tokens'],
             'hbm_hit_tokens':prefix['num_computed_tokens'] if prefix else None,
             'external_hit_tokens':prefix['external_hit_tokens'] if prefix else None,
             'first_execute_ns':first,'ready_ns':ready,'scheduled_ns':scheduled,
             'ready_to_first_execute_ms':(first-ready)/1e6 if ready is not None and first is not None else None,
             'ready_to_schedule_ms':(scheduled-ready)/1e6 if ready is not None and scheduled is not None else None,
             'stages':{},'h2d_delivery':{},'layer_copy_complete_ns':{},'layer_copy_ranges_ns':{},'matches':observed_matches[rid],'early':early[rid]}
        for stage in STAGES:
            selected=own[stage]
            row['stages'][stage]={'ranges_ns':merge((e['start_ns'],e['end_ns']) for e in selected),
                'union_ms':duration((e['start_ns'],e['end_ns']) for e in selected)/1e6,
                'events':len(selected),'bytes':sum(e['bytes'] for e in selected) if all(e['bytes'] is not None for e in selected) else None}
        check('device_compute_captured:'+rid,bool(own['device_compute']))
        for pid in by_pid:
            copies=[e for e in own['H2D'] if e['pid']==pid]
            changes=defaultdict(int)
            known=all(e['bytes'] is not None for e in copies)
            if known:
                for e in copies: changes[e['end_ns']]+=e['bytes']
            total,points=0,[]
            for stamp_,size in sorted(changes.items()):
                total+=size;points.append([stamp_,total])
            row['h2d_delivery'][str(pid)]={'known_bytes':known,'points_ns_bytes':points}
            layers=defaultdict(list)
            for e in copies: layers[e['layer']].append(e['end_ns'])
            row['layer_copy_complete_ns'][str(pid)]={str(layer):max(times) for layer,times in layers.items()}
            row['layer_copy_ranges_ns'][str(pid)]={str(layer):merge((e['start_ns'],e['end_ns']) for e in copies if e['layer']==layer) for layer in layers}
        ext=row['external_hit_tokens'];hbm=row['hbm_hit_tokens']
        row['source']='unknown' if ext is None else 'cold' if ext+hbm==0 else 'HBM' if ext==0 else 'external_with_own_L2' if own['L2_read'] else 'external_without_own_L2'
        row['h2d_compute_overlap_ms']=duration(intersection(row['stages']['H2D']['ranges_ns'],row['stages']['device_compute']['ranges_ns']))/1e6
        rows.append(row)
    # Credit reclamation is a byte budget. FIFO pairing is used only among
    # equal-sized reads of the same observed backend task, never across requests.
    read_tasks=defaultdict(list)
    for e in events:
        if e['stage']=='L2_read':read_tasks[(e['pid'],e['task'])].append(e)
    credit_wait=[];credit_errors=[]
    for key,credit_rows in credits.items():
        reads=read_tasks.get(key,[])
        units={e['bytes'] for e in reads}
        if len(units)!=1 or not reads:
            credit_errors.append(str(key));continue
        unit=next(iter(units));ends=sorted(e['end_ns'] for e in reads);cursor=0
        for stamp_,size in sorted(credit_rows):
            if size<0 or unit<=0 or size%unit or cursor+size//unit>len(ends):
                credit_errors.append(str(key));break
            chosen=ends[cursor:cursor+size//unit]
            if any(finish>stamp_ for finish in chosen):credit_errors.append(str(key))
            credit_wait += [(finish,stamp_) for finish in chosen]
            cursor+=size//unit
        if cursor!=len(ends):credit_errors.append(str(key))
    check('credit_byte_conservation',not credit_errors)
    metrics={};overlap={}
    for window,(a,b) in {'client':(start,end),'physical_pipeline':(start,full_end)}.items():
        ranges={stage:merge((max(a,e['start_ns']),min(b,e['end_ns'])) for e in events if e['stage']==stage) for stage in STAGES}
        metrics[window]={stage:event_statistics([e for e in events if e['stage']==stage],a,b) for stage in STAGES}
        overlap[window]={left:{right:duration(intersection(ranges[left],ranges[right]))/1e6 for right in STAGES} for left in STAGES}
    totals={stage:{'events':sum(e['stage']==stage for e in events),
             'bytes':sum(e['bytes'] for e in events if e['stage']==stage) if all(e['bytes'] is not None for e in events if e['stage']==stage) else None}
            for stage in STAGES}
    output={'schema_version':1,'passed':not errors,'checks':checks,'failed_checks':errors,
            'request_count':len(rows),'clock':device['clocks'],'adapter':device['adapter'],
            'windows_ns':{'client':[start,end],'physical_pipeline':[start,full_end]},
            'physical_tail_ms':(full_end-end)/1e6,'metrics':metrics,'overlap_ms':overlap,
            'per_worker_metrics':{str(pid):{stage:event_statistics([e for e in events if e['pid']==pid and e['stage']==stage],start,end) for stage in STAGES} for pid in by_pid},
            'capture_totals':totals,'native_label_counts':dict(label_counts),
            'source_counts':dict(Counter(r['source'] for r in rows)),
            'unattributed_device_events':device['unattributed_device_events'],
            'semantics':['Five-stage intervals can overlap; union durations are not additive.',
                         'Device compute is traced kernel activity, not SM/AI Core utilization.',
                         'Shared batch intervals repeat by request membership; never sum request rows.',
                         'HBM delivery means cumulative completed copies, not continuous residency.',
                         'External blocks without own L2 reads may originate from another request.',
                         'Missing device copy bytes stay null; native counters retain separate measured totals.']}
    save_json(attempt/'pipeline-requests.json',rows)
    save_json(attempt/'pipeline-io.json',{'reads':curve([(e['start_ns'],e['end_ns']) for e in events if e['stage']=='L2_read'],start,full_end),
              'writes':curve([(e['start_ns'],e['end_ns']) for e in events if e['stage']=='L2_write'],start,full_end),
              'completed_uncredited_reads':curve(credit_wait,start,full_end),
              'fallback_tasks_without_credits':[list(k) for k in read_tasks if k not in credits], 'credit_errors':credit_errors})
    save_json(attempt/'pipeline-diagnostics.json',{'ranges':diagnostics,'batches':[{'pid':pid,'batch':batch,**event} for (pid,batch),event in batches.items()]})
    with gzip.open(attempt/'pipeline-events.json.gz','wt') as stream:
        json.dump({'clock':'monotonic_ns','events':events},stream,separators=(',',':'))
    save_json(attempt/'pipeline-summary.json',output)
    return output
