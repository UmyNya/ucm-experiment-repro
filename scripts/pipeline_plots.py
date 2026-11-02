"""All-request pipeline pages and diagnostics; no external assets or fixed N/layers."""
from collections import Counter
import math
from pathlib import Path

from common import read_json
from pipeline_analysis import STAGES,merge

COLORS={'L2_read':'#805ad5','H2D':'#3182ce','device_compute':'#38a169','D2H':'#dd6b20','L2_write':'#c53030'}


def render(record, output, formats, prefix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    source=Path(record['path']);output=Path(output)
    rows=read_json(source/'pipeline-requests.json');summary=record['pipeline']
    io=read_json(source/'pipeline-io.json');diagnostics=read_json(source/'pipeline-diagnostics.json')
    origin,end=summary['windows_ns']['client'];full_end=summary['windows_ns']['physical_pipeline'][1]
    charts=[]
    title=('SYNTHETIC | ' if record['synthetic'] else '')+record['label']
    def save(fig,name,caption):
        files=[]
        for suffix in formats:
            path=output/(prefix+'_'+name+'.'+suffix)
            fig.savefig(path,dpi=140,bbox_inches='tight');files.append(str(path))
        plt.close(fig)
        charts.append({'name':prefix+'_'+name,'title':caption,'files':files})
    def bars(axis,intervals,y,height=.65,color='gray'):
        axis.broken_barh([((a-origin)/1e6,(b-a)/1e6) for a,b in intervals],(y-height/2,height),facecolors=color)
    fig,axes=plt.subplots(3,1,figsize=(14,11),constrained_layout=True)
    for index,stage in enumerate(STAGES):
        intervals=merge(iv for row in rows for iv in row['stages'][stage]['ranges_ns'])
        bars(axes[0],intervals,index,color=COLORS[stage])
    axes[0].set(yticks=range(5),yticklabels=STAGES,xlabel='Monotonic ms from first client submission',title=title+' | five stages; union across workers')
    axes[0].axvline((end-origin)/1e6,color='black',ls='--',label='Last client response');axes[0].legend()
    for key,label in [('reads','L2 reads in flight'),('writes','L2 writes in flight'),('completed_uncredited_reads','Completed, credit not yet returned')]:
        points=io[key]['points_ms_count']
        axes[1].step([p[0] for p in points],[p[1] for p in points],where='post',label=label)
    axes[1].set(xlabel='ms',ylabel='Operations');axes[1].legend()
    ids=[r['ordinal'] for r in rows]
    axes[2].bar(ids,[r['hbm_hit_tokens'] or 0 for r in rows],label='Initial HBM hit')
    axes[2].bar(ids,[r['external_hit_tokens'] or 0 for r in rows],bottom=[r['hbm_hit_tokens'] or 0 for r in rows],label='External cache hit')
    axes[2].bar(ids,[r['tokens']-r['initial_computed_tokens'] for r in rows],bottom=[r['initial_computed_tokens'] for r in rows],label='Prompt remaining at first execute')
    axes[2].set(xlabel='Request ordinal',ylabel='Prompt tokens');axes[2].legend()
    for axis in axes:axis.grid(alpha=.2)
    save(fig,'overview','五阶段总览、读写并发、完成未归还 credits 与前缀来源')
    fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for offset,(key,label) in enumerate([('mean','Mean'),('p50','P50'),('p95','P95'),('p99','P99')]):
        values=[(summary['metrics']['client'][stage]['complete_event_latency_ms'] or {}).get(key,float('nan')) for stage in STAGES]
        axes[0].bar([i+offset*.18 for i in range(5)],values,width=.18,label=label)
    axes[0].set(xticks=[i+.27 for i in range(5)],xticklabels=STAGES,ylabel='Complete event latency / ms');axes[0].legend()
    axes[1].bar(STAGES,[summary['metrics']['client'][stage]['peak_inflight'] for stage in STAGES],color=[COLORS[s] for s in STAGES])
    axes[1].set(ylabel='Peak concurrent operations/kernels',title='Across observed workers; per-worker statistics retained')
    for axis in axes:axis.tick_params(axis='x',rotation=25);axis.grid(axis='y',alpha=.2)
    fig.suptitle(title);save(fig,'event_statistics','完整事件延迟分布与阶段峰值并发')
    ordered=sorted(rows,key=lambda r:(r['stages']['device_compute']['ranges_ns'][0][0] if r['stages']['device_compute']['ranges_ns'] else float('inf'),r['request_id']))
    fig,axes=plt.subplots(1,2,figsize=(18,min(32,max(7,len(rows)*.10))),constrained_layout=True)
    for index,row in enumerate(rows):
        for j,stage in enumerate(STAGES):bars(axes[0],row['stages'][stage]['ranges_ns'],index+j*.13,height=.11,color=COLORS[stage])
    for index,row in enumerate(ordered):
        bars(axes[1],row['stages']['device_compute']['ranges_ns'],index,color=COLORS['device_compute'])
        if row['ready_ns'] is not None:axes[1].plot((row['ready_ns']-origin)/1e6,index,'o',ms=2,color='#805ad5')
    spacing=max(1,math.ceil(len(rows)/40))
    for axis,values in zip(axes,[rows,ordered]):
        axis.set(yticks=list(range(0,len(values),spacing)),yticklabels=['R'+str(values[i]['ordinal']) for i in range(0,len(values),spacing)],
                 xlabel='Monotonic ms from first client submission')
        axis.invert_yaxis();axis.grid(axis='x',alpha=.15)
    axes[0].set_title('All requests; client ordinal order')
    axes[1].set_title('Device kernels; ordered by first activity, then server request ID')
    axes[0].legend(handles=[Patch(facecolor=COLORS[s],label=s) for s in STAGES],fontsize=7)
    fig.suptitle(title+' | shared batch intervals repeat by request membership')
    save(fig,'all_requests_server_order','全部请求总览与按首次设备活动排序的服务请求时间轴')
    cached=[r for r in rows if r['stages']['H2D']['events']]
    if cached:
        chosen=max(cached,key=lambda r:r['client'].get('ttft_ms') or 0)
        lanes=[(pid,layer,intervals) for pid,layers in chosen.get('layer_copy_ranges_ns',{}).items() for layer,intervals in layers.items()]
        fig,axis=plt.subplots(figsize=(14,min(24,max(4,len(lanes)*.25))),constrained_layout=True)
        for index,(pid,layer,intervals) in enumerate(lanes):bars(axis,intervals,index,color=COLORS['H2D'])
        bars(axis,chosen['stages']['device_compute']['ranges_ns'],len(lanes),color=COLORS['device_compute'])
        axis.set(yticks=list(range(len(lanes)+1)),yticklabels=[f'PID {pid} / index {layer}' for pid,layer,_ in lanes]+['Device compute'],
                 xlabel='Monotonic ms from first client submission',title=title+f" | largest cached-request TTFT: R{chosen['ordinal']}")
        axis.grid(axis='x',alpha=.2)
        save(fig,'layer_intervals','缓存请求中 TTFT 最大者的各层/整块 H2D 区间与设备计算')
    fig,axes=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    keys=list(STAGES)
    matrix=[[summary['overlap_ms']['client'][a][b] for b in keys] for a in keys]
    im=axes[0].imshow(matrix,cmap='Blues');fig.colorbar(im,ax=axes[0],label='Union intersection / ms')
    axes[0].set(xticks=range(5),xticklabels=keys,yticks=range(5),yticklabels=keys,title='Client-window overlap')
    axes[0].tick_params(axis='x',rotation=35)
    names=['UCM_DUMP_CAPACITY_WAIT','UCM_DUMP_SPILL','UCM_DUMP_WAIT_WRITE','UCM_DUMP_IDLE_WAIT']
    axes[1].barh(names,[summary['native_label_counts'].get(k,0) for k in names],color='#718096')
    axes[1].set(xlabel='Observed span count',title='Native wait/spill diagnostics (zero is valid)')
    fig.suptitle(title);save(fig,'overlap_wait','阶段重叠矩阵与容量等待、spill、前次写等待')
    page_size=record['plan']['configuration']['profiling']['request_page_size']
    for page in range(math.ceil(len(rows)/page_size)):
        subset=rows[page*page_size:(page+1)*page_size]
        fig,axes=plt.subplots(len(subset),1,figsize=(16,max(3,len(subset)*2.1)),squeeze=False,constrained_layout=True)
        for axis,row in zip(axes[:,0],subset):
            for index,stage in enumerate(STAGES):bars(axis,row['stages'][stage]['ranges_ns'],index,color=COLORS[stage])
            client=row['client'];a=round(client['start_s']*1e9);b=round(client['end_s']*1e9)
            bars(axis,[[a,b]],5,color='#a0aec0')
            for stamp,color,label in [(row['ready_ns'],'#805ad5','READY'),(row['first_execute_ns'],'black','First execute')]:
                if stamp is not None:axis.axvline((stamp-origin)/1e6,color=color,ls='--',lw=.8)
            axis.set(yticks=range(6),yticklabels=[*STAGES,'Client'],ylabel='R'+str(row['ordinal']),
                     title=f"R{row['ordinal']} | {row['source']} | TTFT={client.get('ttft_ms')} ms")
            axis.tick_params(axis='y',labelsize=7);axis.grid(axis='x',alpha=.2)
        axes[-1,0].set_xlabel('Monotonic ms from first client submission; shared batch compute repeats by membership')
        fig.suptitle(title+f' | requests {subset[0]["ordinal"]}–{subset[-1]["ordinal"]}')
        save(fig,f'requests_{page+1:03d}',f'逐请求五阶段时间轴：{subset[0]["ordinal"]}–{subset[-1]["ordinal"]}')
        fig,axes=plt.subplots(len(subset),2,figsize=(16,max(3,len(subset)*2)),squeeze=False,constrained_layout=True)
        for axes_row,row in zip(axes,subset):
            for pid,data in row['h2d_delivery'].items():
                points=data['points_ns_bytes']
                if points:
                    axes_row[0].step([(p[0]-origin)/1e6 for p in points],[p[1]/2**20 for p in points],where='post',label='PID '+pid)
            if not axes_row[0].lines:
                axes_row[0].text(.5,.5,'No copy / byte size unavailable; see raw counters',ha='center',transform=axes_row[0].transAxes,fontsize=8)
            else:axes_row[0].legend(fontsize=6)
            axes_row[0].set(ylabel=f"R{row['ordinal']} / MiB",title='Completed H2D bytes; not HBM residency')
            for pid,layers in row['layer_copy_complete_ns'].items():
                if layers:axes_row[1].plot([int(x) for x in layers],[(t-origin)/1e6 for t in layers.values()],'.',label='PID '+pid)
            if axes_row[1].lines:axes_row[1].legend(fontsize=6)
            axes_row[1].set(xlabel='Observed shard/layer index',ylabel='Last H2D completion / ms')
            for axis in axes_row:axis.grid(alpha=.2)
        axes[-1,0].set_xlabel('Monotonic ms from first submission')
        fig.suptitle(title+' | per-request delivery and layers')
        save(fig,f'layers_{page+1:03d}',f'逐请求 H2D 完成量与各层完成时刻：{subset[0]["ordinal"]}–{subset[-1]["ordinal"]}')
    fig,axes=plt.subplots(3,1,figsize=(14,10),constrained_layout=True)
    for key,label in [('ready_to_first_execute_ms','READY → first execute'),('ready_to_schedule_ms','Native READY → scheduler timestamp')]:
        valid=[r for r in rows if r.get(key) is not None]
        if valid:axes[0].plot([r['ordinal'] for r in valid],[r[key] for r in valid],'.',label=label)
    if axes[0].lines:axes[0].legend()
    else:axes[0].text(.5,.5,'READY receipt not applicable / unavailable',ha='center',transform=axes[0].transAxes)
    axes[0].axhline(0,color='gray',lw=.5);axes[0].set(xlabel='Request ordinal',ylabel='ms',title='Raw scheduler timestamps may precede READY in one scheduling iteration')
    for stage in ['L2_read','H2D','D2H','L2_write']:
        axes[1].plot(ids,[r['stages'][stage]['union_ms'] for r in rows],label=stage,lw=.8)
    axes[1].set(xlabel='Request ordinal',ylabel='Union duration / ms');axes[1].legend()
    pids=sorted({b['pid'] for b in diagnostics['batches']})
    for pid in pids:
        batches=[b for b in diagnostics['batches'] if b['pid']==pid]
        axes[2].plot([(b['start_ns']-origin)/1e6 for b in batches],
                     [sum(b['scheduled_tokens'].values()) for b in batches],'.',label='PID '+str(pid))
    axes[2].set(xlabel='Batch execute / ms',ylabel='Scheduled tokens (per worker)')
    if pids:axes[2].legend()
    for axis in axes:axis.grid(alpha=.2)
    fig.suptitle(title);save(fig,'ready_batches','READY 与执行间隔、逐请求阶段耗时、逐 worker batch tokens')
    return charts
