#!/usr/bin/env python3
"""Generate a visibly synthetic nine-request report to verify pagination and local artifacts."""
import hashlib
from pathlib import Path
import re
import sys
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import new_run,save_json,read_json
from configuration import load as load_config,compile_plan
from client_metrics import summarize
from pipeline_analysis import analyze
from generate_report import generate


def main():
    c=load_config(ROOT/'config/experiment.json',['swe.request_limit=9','label=SYNTHETIC-pipeline-validation'])
    run=new_run(c,'swe_w24_n192_pipeline','report-fixture')
    plan=compile_plan(c,'swe_w24_n192_pipeline',run)
    plan.update(validation_only='CPU synthetic trace fixture; not measured inference',execution_scope='external fixture')
    save_json(run/'plan.json',plan)
    save_json(run/'inputs/PREPARATION.json',{'input_fingerprint':'synthetic-nine-request-fixture'})
    out=run/'attempts/001'
    samples=[];native=[];worker={'pid':17,'failure':None,'requests':{},'events':[]};matches=[]
    device={'clocks':{},'adapter':'SYNTHETIC FIXTURE; no GPU/NPU execution','unattributed_device_events':[],'events':[]}
    def span(label,a,b=None):return {'pid':17,'tid':19,'label':label,'start_ns':a,'end_ns':b if b is not None else a,'point':b is None}
    for i in range(9):
        base=i*10000000;rid='fixture-'+str(i);block=hashlib.md5(rid.encode()).hexdigest();sha=hashlib.sha256(rid.encode()).hexdigest()
        load,dump,read,write=4*i+1,4*i+2,4*i+3,4*i+4
        samples.append({'id':i,'input_tokens':128,'output_tokens':1,'start_s':(base+90)/1e9,'end_s':(base+7000000)/1e9,
                        'ttft_ms':6.5,'e2e_ms':6.99991,'tpot_ms':None,'dependency_wait_ms':i*.1,'client_admission_wait_ms':.2})
        worker['requests'][rid]={'ordinal':i,'tokens':128,'sha256_u32le':sha,'initial_computed_tokens':64}
        worker['events'] += [{'event':'execute','batch':i,'start_ns':base+3000000,'end_ns':base+5000000,'scheduled_tokens':{rid:64}},
                            {'event':'load_submit','task':load,'blocks_by_request':{rid:[block]},'logical_blocks_by_request':{rid:[block]}},
                            {'event':'dump_submit','task':dump,'blocks_by_request':{rid:[block]},'logical_blocks_by_request':{rid:[block]}}]
        native += [span(f'UCM_READ_LINK|direct|{load}|{read}',base+100),span(f'UCM_L2_READ|4096|{read}|{block}|0|false|false',base+100,base+2000000),
                   span(f'UCM_H2D_ATTR|{load}|{block}|0',base+2100000,base+2300000),span(f'UCM_D2H_ATTR|{dump}|{block}|0',base+5100000,base+5300000),
                   span(f'UCM_WRITE_LINK|direct|{dump}|{write}',base+6000000),span(f'UCM_L2_WRITE|4096|{write}|{block}|0|false|false',base+6000000,base+9000000)]
        device['events'] += [{'stage':'H2D','pid':17,'start_ns':base+2500000,'end_ns':base+3500000,'task':load,'block':block,'layer':0,'bytes':4096,'submission_tid':19,'submission_start_ns':base+2100000},
                             {'stage':'device_compute','pid':17,'start_ns':base+3200000,'end_ns':base+4800000,'batch':i,'bytes':0},
                             {'stage':'D2H','pid':17,'start_ns':base+5500000,'end_ns':base+6500000,'task':dump,'block':block,'layer':0,'bytes':4096,'submission_tid':19,'submission_start_ns':base+5100000}]
        matches.append({'request_id':rid,'prompt_sha256':sha,'ucm_blocks':[block],'num_computed_tokens':0,'external_hit_tokens':64})
    save_json(out/'request-samples.json',samples)
    result=analyze(plan,out,[worker],native,device,matches,'')
    if not result['passed']:raise RuntimeError(result['failed_checks'])
    save_json(out/'performance.json',summarize(samples,samples[-1]['end_s']-samples[0]['start_s'],9,{'synthetic_fixture':True},client='swe'))
    save_json(out/'status.json',{'status':'completed','synthetic':True})
    save_json(run/'status.json',{'status':'completed','synthetic':True})
    report=generate([run],include_validation=True,output_root=run/'reports')
    charts=read_json(report/'figures.json')
    links=re.findall(r'\]\(([^)]+)\)',(report/'REPORT.md').read_text())
    missing=[link for link in links if not (report/unquote(link)).exists()]
    pages=[chart for chart in charts if re.fullmatch(r'pipeline_\d+_requests_\d+',chart['name'])]
    layers=[chart for chart in charts if re.fullmatch(r'pipeline_\d+_layers_\d+',chart['name'])]
    evidence={'passed':not missing and len(pages)==2 and len(layers)==2,
              'synthetic':True,'report':str(report),'charts':len(charts),'assets':sum(len(c['files']) for c in charts),
              'request_pages':len(pages),'layer_pages':len(layers),'relative_links':len(links),'missing_links':missing}
    save_json(run/'report-validation.json',evidence)
    print(run,flush=True);print(report/'REPORT.md',flush=True)
    if not evidence['passed']:raise SystemExit(1)


if __name__=='__main__':main()
