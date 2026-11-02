#!/usr/bin/env python3
"""Export and analyze recorded pipeline data; never stops an independent service."""
import argparse
import os
from pathlib import Path

from common import read_json, save_json, sha256
from device_trace import nsight, ascend
from pipeline_analysis import analyze, jsonl
from processes import OwnedProcess


def analyze_capture(plan, attempt):
    attempt=Path(attempt)
    session=read_json(attempt/'profile-session.json')
    if not read_json(attempt/'profile-stop.json')['stopped']:
        raise ValueError('Pipeline capture did not stop cleanly')
    workers=[];native=[];directories=[]
    for row in session['workers']:
        directory=Path(row['directory'])
        directories.append(directory)
        workers.append(read_json(directory/'worker-events.json'))
        for path in sorted(directory.glob('native_*.jsonl')):
            native+=jsonl(path)
    service=Path(plan.get('attached_service_run', plan['run_dir']))
    raw_files=[]
    if session['backend']=='nsys':
        state=service/'resource-cleanup.json'
        if not state.is_file() or not read_json(state).get('released'):
            raise ValueError('Nsight report must be flushed first: stop your standalone service explicitly, then rerun analyze_pipeline.py')
        reports=sorted(service.glob('device-profile*.nsys-rep'))
        if not reports:
            raise ValueError('No Nsight report was flushed; inspect server.log and profiler lifecycle')
        # One service may contain several independent client captures. Locate
        # the database containing all anchors for this capture, never infer
        # request ownership from a filename or run order.
        candidates=[];failures=[]
        for report in reports:
            database=report.with_suffix('.sqlite')
            export_metadata=database.with_suffix('.export.json')
            digest=sha256(report)
            if database.exists():
                if not export_metadata.is_file() or read_json(export_metadata)['report_sha256']!=digest:
                    raise ValueError('Existing Nsight export lacks matching source provenance; use a new output copy')
            else:
                command=[plan['configuration']['profiling']['nsys'],'export','--type=sqlite',
                         '--output='+str(database),str(report)]
                with OwnedProcess(command,cwd=service,environment={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'},
                                  log_path=database.with_suffix('.export.log'),name='nsight-export') as process:
                    if process.wait(plan['configuration']['execution']['benchmark_timeout_s']):
                        raise RuntimeError('Nsight export failed; retained export log explains the error')
                save_json(export_metadata,{'report_sha256':digest,'database_sha256':sha256(database),'command':command})
            try:
                candidates.append((database,nsight(database,workers,native)))
            except ValueError as error:
                failures.append({'database':str(database),'error':str(error)})
        if len(candidates)!=1:
            save_json(attempt/'profile-database-selection.json',{'candidates':[str(p) for p,_ in candidates],'errors':failures})
            raise ValueError('Exactly one Nsight report must contain this capture\'s worker anchors')
        database,device=candidates[0]
        raw_files=[str(database),*[str(p) for p in reports]]
    else:
        device={'events':[],'clocks':{},'unattributed_device_events':[], 'adapter':'Ascend Chrome exact flow correlation'}
        for directory,worker in zip(directories,workers):
            trace=directory/f'device_trace_{worker["pid"]}.json'
            parsed=ascend(trace,worker,[r for r in native if r['pid']==worker['pid']])
            for name in ['events','unattributed_device_events']:device[name]+=parsed[name]
            device['clocks'].update(parsed['clocks'])
            raw_files.append(str(trace))
    matches=[]
    for path in sorted((service/'match-events').glob('*.jsonl')):matches+=jsonl(path)
    log=Path(plan.get('server_log', service/'server.log'))
    result=analyze(plan,attempt,workers,native,device,matches,log.read_text(errors='replace'))
    save_json(attempt/'profile-artifacts.json',{'worker_directories':[str(p) for p in directories],
              'device_traces':raw_files,'server_log':str(log),'match_events':str(service/'match-events')})
    if not result['passed']:
        raise ValueError('Pipeline attribution validation failed: '+', '.join(result['failed_checks']))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True)
    args=p.parse_args()
    try:
        result=analyze_capture(read_json(args.plan),args.attempt)
        print(args.attempt/'pipeline-summary.json')
    except BaseException as error:
        save_json(args.attempt/'pipeline-analysis-status.json',{'status':'failed','error':str(error)})
        raise
    save_json(args.attempt/'pipeline-analysis-status.json',{'status':'completed','passed':result['passed']})


if __name__=='__main__':main()
