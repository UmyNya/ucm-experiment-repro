"""Static, local PNG/SVG/PDF figures from recorded request data."""
from pathlib import Path


def render(records, groups, output, formats=('png', 'svg', 'pdf')):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    charts = []

    def save(fig, name, title):
        for suffix in formats:
            fig.savefig(output/(name+'.'+suffix), dpi=140, bbox_inches='tight')
        plt.close(fig)
        charts.append({'name': name, 'title': title, 'files': [str(output/(name+'.'+x)) for x in formats]})

    # Independent panels prevent accidentally implying comparability between
    # different input fingerprints, devices, models or profiling scopes.
    comparisons = sorted({g['comparability_group'] for g in groups})
    for index, comparison in enumerate(comparisons):
        selected = [g for g in groups if g['comparability_group']==comparison]
        labels = [g['id'] for g in selected]
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for axis, metric, title in zip(axes, ['input_tokens_per_s', 'duration_s', 'requests_per_s'],
                                       ['Input tokens / second', 'Run duration / second', 'Requests / second']):
            values = [g['metrics'][metric]['mean'] for g in selected]
            error = [g['metrics'][metric]['sample_sd'] or 0 for g in selected]
            axis.bar(labels, values, yerr=error, capsize=4, color='#336b87')
            axis.set_title(title)
            axis.grid(axis='y', alpha=.2)
        title = 'SYNTHETIC VALIDATION ONLY' if selected[0]['synthetic'] else 'Recorded formal runs; error bars = sample SD'
        fig.suptitle(title)
        save(fig, f'comparison_{index+1:02d}', '吞吐与耗时；误差条为轮间样本标准差')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for group in selected:
            members = [r for r in records if r['path'] in group['record_paths']]
            for axis, key, label in zip(axes, ['ttft_ms', 'e2e_ms'], ['TTFT / ms', 'End-to-end / ms']):
                values = sorted(s[key] for r in members for s in r['samples'] if s.get(key) is not None)
                if values:
                    axis.plot(values, [(i+1)/len(values) for i in range(len(values))], label=group['id'])
                axis.set_xlabel(label)
                axis.set_ylabel('Empirical CDF')
                axis.grid(alpha=.2)
                if axis.lines:
                    axis.legend()
        fig.suptitle(title)
        save(fig, f'latency_cdf_{index+1:02d}', '成功请求延迟经验分布')
    selected_paths = {path for group in groups for path in group['record_paths']}
    from pipeline_analysis import STAGES,merge
    from pipeline_plots import COLORS
    from common import read_json
    for comparison_index,comparison in enumerate(comparisons):
        members=[r for r in records if r['path'] in selected_paths and r['comparability_group']==comparison and r.get('pipeline')]
        if not members:continue
        fig,axes=plt.subplots(len(members),1,figsize=(14,max(4,3*len(members))),squeeze=False,constrained_layout=True)
        labels=[]
        for index,(axis,record) in enumerate(zip(axes[:,0],members)):
            origin,end=record['pipeline']['windows_ns']['client']
            rows=read_json(Path(record['path'])/'pipeline-requests.json')
            label=record['label']+' / '+Path(record['path']).parent.name+' / '+str(index+1)
            labels.append(label)
            for lane,stage in enumerate(STAGES):
                intervals=merge(iv for row in rows for iv in row['stages'][stage]['ranges_ns'])
                axis.broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in intervals],(lane-.3,.6),facecolors=COLORS[stage])
            axis.axvline((end-origin)/1e6,color='black',ls='--')
            axis.set(yticks=range(5),yticklabels=STAGES,xlabel='ms from this run\'s first submission',title=label)
            axis.grid(axis='x',alpha=.2)
        fig.suptitle(('SYNTHETIC | ' if members[0]['synthetic'] else '')+'Comparable pipeline windows; each run retains its own time origin')
        save(fig,f'pipeline_comparison_{comparison_index+1:02d}','可比配置逐轮五阶段全局时间轴')
        fig,axis=plt.subplots(figsize=(14,5),constrained_layout=True)
        width=.8/len(members)
        for index,record in enumerate(members):
            axis.bar([i+index*width for i in range(5)],
                     [record['pipeline']['metrics']['client'][stage]['union_ms'] for stage in STAGES],
                     width=width,label=labels[index])
        axis.set(xticks=[i+(len(members)-1)*width/2 for i in range(5)],xticklabels=STAGES,ylabel='Client-window union / ms',
                 title='Stage costs; overlapping stage durations must not be added')
        axis.legend(fontsize=7);axis.grid(axis='y',alpha=.2)
        save(fig,f'pipeline_stage_cost_{comparison_index+1:02d}','可比配置逐轮阶段 union 耗时比较')
    for index, record in enumerate(r for r in records if r['path'] in selected_paths):
        samples = record['samples']
        if not samples:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
        x = [int(s['id']) for s in samples]
        for key, name in [('ttft_ms', 'TTFT'), ('e2e_ms', 'E2E')]:
            axes[0].plot(x, [s.get(key) if s.get(key) is not None else float('nan') for s in samples], label=name, lw=.8)
        axes[0].set_ylabel('Latency / ms')
        axes[0].legend()
        axes[1].bar(x, [s['input_tokens'] for s in samples], color='#4b8b64')
        axes[1].set_ylabel('Input tokens')
        for key, name in [('dependency_wait_ms', 'Dependency wait'), ('client_admission_wait_ms', 'Admission wait')]:
            values = [s.get(key) for s in samples]
            if any(v is not None for v in values):
                axes[2].plot(x, [v or 0 for v in values], label=name, lw=.8)
        if axes[2].lines:
            axes[2].legend()
            axes[2].set_ylabel('Client wait / ms')
        else:
            axes[2].text(.5, .5, 'Client dependency/admission timing: N/A for this client', ha='center', transform=axes[2].transAxes)
        for axis in axes:
            axis.set_xlabel('Request ordinal')
            axis.grid(alpha=.2)
        fig.suptitle(('SYNTHETIC | ' if record['synthetic'] else '')+record['label']+' | '+Path(record['path']).parent.name)
        save(fig, f'requests_{index+1:03d}', '逐请求延迟、tokens 与客户端等待')
        events = []
        for sample in samples:
            events += [(sample['start_s'], 1), (sample['end_s'], -1)]
        origin = min(t for t,_ in events)
        current, points, changes = 0, [(0, 0)], {}
        for t, change in events:
            changes[t] = changes.get(t, 0)+change
        for t, change in sorted(changes.items()):
            current += change
            points.append((t-origin, current))
        fig, axis = plt.subplots(figsize=(12, 3.5), constrained_layout=True)
        axis.step([p[0] for p in points], [p[1] for p in points], where='post', label='Successful request in-flight')
        axis.axhline(record['plan']['configuration']['load']['concurrency'], linestyle='--', color='gray', label='Configured C')
        axis.set(xlabel='Seconds from first submission', ylabel='Requests', title='Client in-flight timeline (failed requests excluded)')
        axis.legend()
        axis.grid(alpha=.2)
        save(fig, f'inflight_{index+1:03d}', '成功请求在途数量')
        service_metrics = Path(record['path'])/'service-metrics.jsonl'
        if service_metrics.is_file():
            import json
            rows = [json.loads(line) for line in service_metrics.read_text().splitlines() if line.strip()]
            fig, axes = plt.subplots(2, 1, figsize=(12, 6), constrained_layout=True)
            names = [('num_requests_running', 'Running requests'), ('num_requests_waiting', 'Waiting requests')]
            first = min((r['start_monotonic_ns'] for r in rows), default=0)
            for axis, (suffix, label) in zip(axes, names):
                points = []
                for row in rows:
                    metrics = [m['value'] for m in row['metrics'] if m['name'].endswith(suffix)]
                    if metrics:
                        points.append(((row['start_monotonic_ns']-first)/1e9, sum(metrics)))
                if points:
                    axis.step([x[0] for x in points], [x[1] for x in points], where='post')
                else:
                    axis.text(.5,.5,'N/A: endpoint did not expose this metric',ha='center',transform=axis.transAxes)
                axis.set(xlabel='Seconds from first metrics sample', ylabel=label)
                axis.grid(alpha=.2)
            fig.suptitle('Service queue samples; raw labels retained in service-metrics.jsonl')
            save(fig, f'service_queue_{index+1:03d}', '服务运行与等待队列采样')
        if record['plan']['mode']=='pipeline' and record.get('pipeline',{}).get('passed'):
            from pipeline_plots import render as pipeline_render
            charts += pipeline_render(record, output, formats, f'pipeline_{index+1:03d}')
        native_path = Path(record['path'])/'native-statistics.json'
        if native_path.is_file():
            from common import read_json
            native = read_json(native_path)
            totals = native['counter_totals']
            names = [k for k in totals if k.endswith('bytes_total')]
            names.sort()
            if names:
                fig, axis = plt.subplots(figsize=(13,max(4,len(names)*.3)),constrained_layout=True)
                axis.barh(names,[totals[k]/2**30 for k in names],color='#718096')
                axis.set(xlabel='GiB; summed observed worker counters',title='Native byte counters; metric semantics remain distinct')
                axis.grid(axis='x',alpha=.2)
                save(fig,f'native_bytes_{index+1:03d}','原生字节计数：按指标分别展示，不能跨指标相加')
    return charts
