#!/usr/bin/env python3
"""Real clients plus real service/campaign controllers, with CPU-only model/RPC fixtures."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'tests'))
from boundaries import LIFECYCLE
from common import read_json,save_json,new_run
from configuration import load
from run_experiment import execute
from service import Service,interrupt_cleanup
from processes import same_process
from test_service_contracts import ServiceContractsTest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--client',choices=['aisbench','swe'],required=True)
    p.add_argument('--python',required=True)
    args=p.parse_args()
    c=load(ROOT/'config/experiment.json',[f'paths.{args.client}_python={args.python}'])
    c['platform']['kind']='cuda'
    c['execution'].update(repetitions=2,health_poll_s=.05,shutdown_timeout_s=3,startup_timeout_s=10)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));c['server']['port']=sock.getsockname()[1]
    profile='aisbench_n40_performance' if args.client=='aisbench' else 'swe_w24_n192_performance'
    run=new_run(c,profile,'managed-mock-validation')
    print(run,flush=True)
    instances=[]
    class FixtureService(Service):
        def __init__(self,config,profile,directory,compile_cache):
            super().__init__(config,profile,directory,compile_cache=compile_cache)
            instances.append(self)
            worker=ServiceContractsTest().worker()
            worker.update(ucm_source=str(run/'fixture-ucm/__init__.py'),lifecycle={k:0 for k in LIFECYCLE},
                          counters={},gauges={},histogram_counts={},pending_dump_task_status=[],pending_dump_tasks=0,metrics_collected=True)
            save_json(self.run/'fixture-worker.json',worker)
            expected={};inputs=run/'inputs'
            if args.client=='aisbench':
                manifest=read_json(inputs/'manifest.json')
                for phase in ['formal','warmup']:
                    texts=[json.loads(line)['question'] for line in (inputs/(phase+'.jsonl')).read_text().splitlines()]
                    for text,count in zip(texts,manifest[phase+'_input_token_counts'],strict=True):
                        expected[hashlib.sha256(text.encode()).hexdigest()]=count
                u=read_json(inputs/'unrelated.json');expected[hashlib.sha256(u['text'].encode()).hexdigest()]=len(u['token_ids'])
            else:
                expected={r['sha256_u32le']:r['prompt_tokens'] for r in read_json(inputs/'prompt-snapshots/manifest.json')['requests']}
                text='Warm up this inference service and answer briefly. This request is excluded from closed-loop metrics.'
                expected[hashlib.sha256(text.encode()).hexdigest()]=18
            save_json(self.run/'fixture-prompts.json',expected)
            self.plan['server_command']=[sys.executable,'-B',str(ROOT/'tests/fake_model_service.py'),str(config['server']['port'])]
            save_json(self.run/'plan.json',self.plan)
    attest={'installed':{'ucm_root':str(run/'fixture-ucm')}}
    passed=False
    os.environ['BUNDLE_VALIDATION_ONLY']='mock_http'
    try:
        with interrupt_cleanup(),patch('service.attest_environment',return_value=attest):
            execute(c,profile,run,service_factory=FixtureService)
        rows=[]
        count=81 if args.client=='aisbench' else 193
        for service in instances:
            records=[json.loads(line) for line in (service.run/'fixture-requests.jsonl').read_text().splitlines()]
            rows.append({'service':str(service.run),'requests':len(records),'expected':count,
                         'exact_payloads_passed':len(records)==count and all(r['matched'] for r in records),
                         'leader_released':same_process(service.process.leader) is None,
                         'resource_cleanup':read_json(service.run/'resource-cleanup.json')})
        passed=len(rows)==3 and all(r['exact_payloads_passed'] and r['leader_released'] and r['resource_cleanup']['released'] for r in rows)
        save_json(run/'managed-mock-audit.json',{'passed':passed,'stages':rows,'synthetic':True,
                  'real_client_real_controller':True,'real_device_or_native_io_tested':False,
                  'calibration_then_two_fresh_formal_services':True,'input_prepared_once':True})
    finally:
        for service in instances:
            if not service.closed:service.stop()
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
