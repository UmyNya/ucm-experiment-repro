import copy
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import save_json,read_json
from device_trace import nsight,ascend,clock_alignment,ScopeIndex
from pipeline_analysis import analyze,merge,intersection,duration,event_statistics


def clock_worker(pid=17):
    return {'pid':pid,'clock_anchors':[{'monotonic_ns':x,'end_ns':x+20}
             for x in [1000,1100,1900,2000]],'events':[
             {'event':'scope','pid':pid,'tid':19,'label':'BUNDLE_EXECUTE|1','start_ns':1200,'end_ns':1800}],
             'failure':None}


class AdapterTest(unittest.TestCase):
    def test_clock_bounds_reject_inconsistent_or_missing_evidence(self):
        w=clock_worker()
        anchors=[('BUNDLE_CLOCK|'+str(a['monotonic_ns']),a['monotonic_ns']+10010) for a in w['clock_anchors']]
        self.assertEqual(clock_alignment(w,anchors)['offset_ns'],10000)
        with self.assertRaises(ValueError):clock_alignment(w,anchors[:3])
        anchors[-1]=(anchors[-1][0],999999)
        with self.assertRaises(ValueError):clock_alignment(w,anchors)

    def test_nsight_uses_device_duration_and_process_scoped_correlation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite'
            con=sqlite3.connect(path)
            con.executescript('''CREATE TABLE StringIds(id INTEGER,value TEXT);
            CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT,textId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,copyKind INTEGER,bytes INTEGER,streamId INTEGER);
            CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,globalPid INTEGER,correlationId INTEGER,streamId INTEGER);''')
            workers=[clock_worker(17),clock_worker(18)];native=[]
            for worker in workers:
                pid=worker['pid'];global_pid=pid<<24;tid=global_pid+19
                for a in worker['clock_anchors']:
                    con.execute('insert into NVTX_EVENTS values(?,?,?,?,NULL)',(a['monotonic_ns']+10010,a['monotonic_ns']+10011,tid,'BUNDLE_CLOCK|'+str(a['monotonic_ns'])))
                # Use distinct anchor keys across workers as real monotonic_ns does.
                if pid==18:
                    con.execute('update NVTX_EVENTS set text=replace(text,"CLOCK|","CLOCK|9"),start=start+90000,end=end+90000 where globalTid=?',(tid,))
                    for a in worker['clock_anchors']:
                        old=a['monotonic_ns'];a['monotonic_ns']=int('9'+str(old));a['end_ns']=a['monotonic_ns']+20
                native.append({'pid':pid,'tid':19,'start_ns':1300,'end_ns':1320,'label':'UCM_H2D_ATTR|7|'+'ab'*16+'|0'})
                con.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)',(11301,11305,tid,8))
                con.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?)',(11400,11405,tid,9))
                con.execute('insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?,?,?)',(11350,11450,global_pid,8,1,pid*10,3))
                con.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?)',(11500,11700,global_pid,9,4))
            # Worker 18 uses shifted native+device times with identical IDs.
            # Keep anchor drift exactly zero under that shift.
            for row in con.execute('select rowid,text from NVTX_EVENTS where globalTid=?',(18<<24|19,)).fetchall():
                stamp=int(row[1].split('|')[1]);con.execute('update NVTX_EVENTS set start=?,end=? where rowid=?',(stamp+10010,stamp+10011,row[0]))
            for table in ['CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_KERNEL']:
                field='globalTid' if table.endswith('RUNTIME') else 'globalPid'
                value=(18<<24|19) if field=='globalTid' else 18<<24
                con.execute('update '+table+' set start=start+90000,end=end+90000 where '+field+'=?',(value,))
            workers[1]['events'][0]['start_ns']+=90000;workers[1]['events'][0]['end_ns']+=90000
            native[1]['start_ns']+=90000;native[1]['end_ns']+=90000
            con.commit();con.close()
            result=nsight(path,workers,native)
            self.assertEqual(len(result['events']),4)
            self.assertFalse(result['unattributed_device_events'])
            copies=[r for r in result['events'] if r['stage']=='H2D']
            self.assertEqual([r['bytes'] for r in copies],[170,180])
            self.assertTrue(all(r['end_ns']-r['start_ns']==100 for r in copies))

    def test_ascend_exact_flows_and_unlinked_kernel_are_visible(self):
        w=clock_worker();events=[]
        for a in w['clock_anchors']:
            events.append({'ph':'X','name':'BUNDLE_CLOCK|'+str(a['monotonic_ns']),'pid':17,'tid':19,
                           'ts':(a['monotonic_ns']+10010)/1000,'dur':.001})
        events += [{'ph':'s','cat':'async_npu','id':1,'pid':17,'tid':19,'ts':11.4},
                   {'ph':'f','cat':'async_npu','id':1,'pid':99,'tid':3,'ts':11.5},
                   {'ph':'X','name':'MatMul','pid':99,'tid':3,'ts':11.5,'dur':.2},
                   {'ph':'X','name':'DroppedFlowKernel','pid':99,'tid':3,'ts':11.8,'dur':.1}]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.json';save_json(path,{'traceEvents':events})
            parsed=ascend(path,w,[])
            self.assertEqual(parsed['events'][0]['start_ns'],1500)
            self.assertEqual(parsed['events'][0]['end_ns'],1700)
            self.assertEqual(parsed['events'][0]['batch'],1)
            self.assertEqual(len(parsed['unattributed_device_events']),1)
            events[1]['ts']+=10
            save_json(path,{'traceEvents':events})
            with self.assertRaises(ValueError):ascend(path,w,[])

    def test_scope_overlap_never_guesses_a_task(self):
        native=[{'pid':1,'tid':2,'start_ns':10,'end_ns':30,'label':'UCM_H2D_ATTR|'+str(task)+'|'+'ab'*16+'|0'} for task in [1,2]]
        with self.assertRaises(ValueError):ScopeIndex(native).get(1,2,15,20)


class AnalysisTest(unittest.TestCase):
    def test_boundary_crossing_bytes_are_not_time_prorated(self):
        events=[{'start_ns':-1,'end_ns':1,'bytes':10},{'start_ns':1,'end_ns':3,'bytes':20},
                {'start_ns':2,'end_ns':4,'bytes':30}]
        stats=event_statistics(events,0,3)
        self.assertEqual(stats['complete_event_bytes'],20)
        self.assertEqual(stats['boundary_crossing_event_bytes'],40)
        self.assertEqual(stats['boundary_crossing_events'],2)
        self.assertEqual(stats['peak_inflight'],2)
        self.assertEqual(stats['time_coverage'],1)
        events[1]['bytes']=None
        self.assertIsNone(event_statistics(events,0,3)['complete_event_bytes'])

    def test_interval_unions_and_overlap_do_not_add_double_counted_work(self):
        self.assertEqual(merge([[0,4],[2,6],[7,9]]),[[0,6],[7,9]])
        self.assertEqual(duration(intersection([[0,6],[7,9]],[[3,8]])),4)

    def test_complete_five_stage_provenance_and_physical_tail(self):
        block='ab'*16;rid='request0';pid=17
        worker={'pid':pid,'failure':None,'requests':{rid:{'ordinal':0,'tokens':128,'sha256_u32le':'exact','initial_computed_tokens':64}},
                'events':[{'event':'execute','batch':1,'start_ns':300,'end_ns':500,'scheduled_tokens':{rid:64}},
                          {'event':'load_submit','task':1,'blocks_by_request':{rid:[block]},'logical_blocks_by_request':{rid:[block]}},
                          {'event':'dump_submit','task':2,'blocks_by_request':{rid:[block]},'logical_blocks_by_request':{rid:[block]}}]}
        def native(label,a,b=None):return {'pid':pid,'tid':19,'label':label,'start_ns':a,'end_ns':b if b is not None else a,'point':b is None}
        rows=[native('UCM_READ_LINK|direct|1|10',100),native('UCM_L2_READ|64|10|'+block+'|0|false|false',100,200),
              native('UCM_H2D_ATTR|1|'+block+'|0',210,230),native('UCM_D2H_ATTR|2|'+block+'|0',510,530),
              native('UCM_WRITE_LINK|direct|2|20',600),native('UCM_L2_WRITE|64|20|'+block+'|0|false|false',600,900)]
        device={'clocks':{},'adapter':'SYNTHETIC','unattributed_device_events':[], 'events':[
            {'stage':'H2D','pid':pid,'start_ns':250,'end_ns':350,'task':1,'block':block,'layer':0,'bytes':64,'submission_tid':19,'submission_start_ns':210},
            {'stage':'device_compute','pid':pid,'start_ns':320,'end_ns':480,'batch':1,'bytes':0},
            {'stage':'D2H','pid':pid,'start_ns':550,'end_ns':650,'task':2,'block':block,'layer':0,'bytes':64,'submission_tid':19,'submission_start_ns':510}]}
        matches=[{'request_id':rid,'prompt_sha256':'exact','ucm_blocks':[block],'num_computed_tokens':0,'external_hit_tokens':64}]
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            save_json(out/'request-samples.json',[{'id':0,'start_s':.00000009,'end_s':.0000007,'input_tokens':128,'ttft_ms':1}])
            result=analyze({'request_count':1,'validation_only':'unit fixture'},out,[worker],rows,device,matches,'')
            self.assertTrue(result['passed'],result['failed_checks'])
            self.assertAlmostEqual(result['physical_tail_ms'],.0002)
            self.assertEqual(result['capture_totals']['L2_write']['bytes'],64)
            self.assertAlmostEqual(result['overlap_ms']['client']['H2D']['device_compute'],.00003)
            detailed=read_json(out/'pipeline-requests.json')[0]
            self.assertEqual(detailed['source'],'external_with_own_L2')
            bad=copy.deepcopy(device);bad['events']=bad['events'][1:]
            result=analyze({'request_count':1,'validation_only':'unit fixture'},out,[worker],rows,bad,matches,'')
            self.assertFalse(result['passed'])
            self.assertIn('all_native_transfers_have_device_execution',result['failed_checks'])


if __name__=='__main__':unittest.main()
