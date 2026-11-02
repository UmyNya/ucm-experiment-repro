import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from configuration import load
from prepare_environment import adapt_metrics_dispatcher
from native_statistics import summarize


class NativeStatisticsTest(unittest.TestCase):
    def test_default_disabled_multiproc_and_consumed_connector_do_not_erase_observation(self):
        root=os.environ.get('UCM_SOURCE_ROOT') or load(ROOT/'config/experiment.json')['paths']['ucm_repository']
        source=Path(root)/'ucm/metrics_dispatcher.py'
        if os.environ.get('UCM_SOURCE_ROOT') and not source.is_file():
            self.fail('UCM_SOURCE_ROOT does not contain ucm/metrics_dispatcher.py')
        if not source.is_file():self.skipTest('Actual UCM source is required for this adapter test')
        code=adapt_metrics_dispatcher(source.read_text())
        definitions=[SimpleNamespace(name='bytes',metric_type='counter',vllm_connector_enabled=True),
                     SimpleNamespace(name='latency',metric_type='histogram',vllm_connector_enabled=True)]
        config=ModuleType('ucm.metrics_config')
        config.MULTIPROC_CONSUMER='multiproc';config.VLLM_CONNECTOR_CONSUMER='vllm_connector'
        config.MetricDefinition=object
        config.consumer_enabled=lambda c,k:k=='vllm_connector'
        config.get_metric_definitions=lambda c:definitions
        stats=iter([({'bytes':5},{},{'latency':([1,0],2.)}),({'bytes':3},{},{'latency':([0,1],4.)}),({},{},{})])
        native=ModuleType('ucm.shared.metrics');native.ucmmetrics=SimpleNamespace(get_all_stats_and_clear=lambda:next(stats))
        with patch.dict(sys.modules,{'ucm.metrics_config':config,'ucm.shared.metrics':native}):
            scope={};exec(compile(code,'private-metrics-copy','exec'),scope)
            dispatcher=scope['MetricsDispatcher']({})
            dispatcher.drain_to_consumers()
            self.assertEqual(dispatcher.get_stats_and_clear('multiproc'),({},{},{}))
            self.assertEqual(dispatcher.get_stats_and_clear('vllm_connector')[0],{'bytes':5.})
            self.assertEqual(dispatcher._bundle_observer_buffer[0],{'bytes':5.})
            dispatcher.drain_to_consumers()
            self.assertEqual(dispatcher.get_stats_and_clear('vllm_connector')[0],{'bytes':3.})
            self.assertEqual(dispatcher._bundle_observer_buffer[0],{'bytes':8.})
            self.assertEqual(dispatcher._bundle_observer_buffer[2]['latency'],([1,1],6.))
            dispatcher.drain_to_consumers()
            self.assertEqual(dispatcher._bundle_observer_buffer[0],{'bytes':8.})

    def test_worker_delta_preserves_weighted_histogram_mean_and_rejects_resets(self):
        before=[{'pid':1,'counters':{'bytes':10},'histogram_counts':{'latency':1},'histogram_sums':{'latency':3}}]
        after=[{'pid':1,'counters':{'bytes':30},'histogram_counts':{'latency':3},'histogram_sums':{'latency':11}}]
        result=summarize(before,after)
        self.assertEqual(result['counter_totals']['bytes'],20)
        self.assertEqual(result['histogram_means']['latency'],4)
        with self.assertRaisesRegex(ValueError,'reset'):summarize(after,before)


if __name__=='__main__':unittest.main()
