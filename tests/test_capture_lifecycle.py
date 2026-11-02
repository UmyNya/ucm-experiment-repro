import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import read_json
from pipeline_capture import Capture,token_hash
from runtime_observer import BundleObserver


class CaptureLifecycleTest(unittest.TestCase):
    def test_observer_restores_methods_and_preserves_real_return_values(self):
        actions=[]
        class Rank:
            @staticmethod
            def submit_load(manager,store,block_ids_by_request,block_ids,shard_indices,ptrs):
                return SimpleNamespace(task_id=7)
            @staticmethod
            def submit_dump(manager,store,block_ids_by_request,block_ids,shard_indices,ptrs,event_handle,finalize=False):
                return SimpleNamespace(task_id=8)
        class Runner:
            def execute_model(self,scheduler_output):return 'original-result'
            def sample_tokens(self):return 99
        runtime=SimpleNamespace(cudaProfilerStart=lambda:actions.append('start') or 0,
                                cudaProfilerStop=lambda:actions.append('stop') or 0)
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:None,cudart=lambda:runtime,
                       nvtx=SimpleNamespace(range_push=lambda _:None,range_pop=lambda:None)))
        module=ModuleType('ucm.integration.vllm.rank_consistency');module.RankConsistencyManager=Rank
        modules={'torch':torch,'ucm.integration.vllm.rank_consistency':module}
        tokens=[1,2,3,4];digest=token_hash(tokens)
        meta={'client':'aisbench','requests':[{'ordinal':0,'tokens':4,'sha256_u32le':digest}]}
        runner=Runner();original=runner.execute_model;load=Rank.submit_load
        connector=SimpleNamespace(tp_rank=0,is_mla=True)
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BUNDLE_RUN_DIR':tmp}),patch.dict(sys.modules,modules),patch('runtime_observer._context',return_value=(None,connector)),patch('pipeline_capture.cuda_profiler_api',return_value=runtime):
            capture=Capture(SimpleNamespace(model_runner=runner),'capture_test',meta,'nsys')
            capture.start()
            request=SimpleNamespace(req_id='rid',prompt_token_ids=tokens,num_computed_tokens=0)
            output=SimpleNamespace(scheduled_new_reqs=[request],num_scheduled_tokens={'rid':4})
            self.assertEqual(runner.execute_model(output),'original-result')
            self.assertEqual(runner.sample_tokens(),99)
            block=b'a'*16
            self.assertEqual(Rank.submit_load(None,None,{'rid':[block]},[block],[0],None).task_id,7)
            capture.stop()
            self.assertEqual(runner.execute_model,original)
            self.assertIs(Rank.submit_load,load)
            self.assertNotIn('UCM_PIPELINE_CAPTURE',os.environ)
            recorded=read_json(capture.directory/'worker-events.json')
            self.assertEqual(recorded['requests']['rid']['sha256_u32le'],digest)
            self.assertEqual(len(recorded['clock_anchors']),10)
            self.assertEqual(actions,['start','stop'])

    def test_foreign_capture_cannot_be_stopped(self):
        observer=BundleObserver()
        observer._bundle_capture=SimpleNamespace(capture_id='other')
        with self.assertRaisesRegex(RuntimeError,'another client'):
            observer.bundle_profile_stop('mine')
        observer._bundle_capture=None
        self.assertTrue(observer.bundle_profile_stop('mine')['already_inactive'])

    def test_prompt_identity_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'BUNDLE_RUN_DIR':tmp}):
            meta={'client':'swe','request_id_prefix':'swe-prefix-','requests':[{'ordinal':0,'tokens':2,'sha256_u32le':token_hash([1,2])}]}
            capture=Capture(None,'capture_test',meta,'nsys')
            request=SimpleNamespace(req_id='cmpl-swe-prefix-0-abcd',prompt_token_ids=[1,3],num_computed_tokens=0)
            with self.assertRaisesRegex(ValueError,'disagree'):capture.identify(request)


if __name__=='__main__':unittest.main()
