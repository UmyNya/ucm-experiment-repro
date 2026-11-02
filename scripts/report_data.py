"""Select valid recorded attempts; never infer measured settings from a label."""
import copy
from pathlib import Path
import statistics

from client_metrics import distribution
from common import json_hash, read_json


def maybe(path, default=None):
    return read_json(path) if Path(path).is_file() else default


def collect(run):
    run = Path(run).resolve()
    plan = read_json(run/'plan.json')
    config = plan['configuration']
    state = maybe(run/'campaign-state.json')
    prepared = maybe(run/'inputs/PREPARATION.json', {})
    fingerprint = prepared.get('input_fingerprint') or plan.get('input_fingerprint')
    if state:
        paths = [(run/s['path']/'attempt', s['phase'], s) for s in state['stages']]
        scope = 'managed'
    else:
        paths = [(p, 'external', maybe(p/'status.json', {})) for p in sorted((run/'attempts').glob('*')) if p.is_dir()]
        scope = 'external'
    if not paths:
        paths = [(run, 'startup_or_preparation', {'status':'failed'})]
    output = []
    for path, phase, stage in paths:
        active_plan = maybe(path.parent/'plan.json', plan) if scope == 'managed' and path != run else plan
        actual_service = active_plan if scope == 'managed' else maybe(run/'actual-service-plan.json')
        environment = maybe(path.parent/'service/service-environment.json') if scope == 'managed' else maybe(run/'service-environment.json')
        performance = maybe(path/'performance.json')
        status = maybe(path/'status.json', {})
        reasons = []
        if phase == 'calibration':
            reasons.append('编译校准，不计正式成绩')
        if not performance or not performance.get('passed'):
            reasons.append('没有通过客户端结果校验')
        if status.get('status') != 'completed' or stage.get('status') != 'completed':
            reasons.append('本轮未完成或失败')
        if not fingerprint:
            reasons.append('缺少精确输入指纹')
        compilation = maybe(path/'compilation-check.json')
        cleanup = maybe(path.parent/'service/resource-cleanup.json') if scope == 'managed' else None
        if scope == 'managed':
            if not cleanup or not cleanup.get('released'):
                reasons.append('没有资源释放完成证据')
            if config['validation']['reject_compilation_during_measurement'] and (not compilation or not compilation.get('passed')):
                reasons.append('正式窗口编译检查未通过')
        pipeline = maybe(path/'pipeline-summary.json')
        if active_plan['mode'] == 'pipeline' and (not pipeline or not pipeline.get('passed')):
            reasons.append('流水线采集/关联校验未通过')
        synthetic = bool(active_plan.get('validation_only') or plan.get('validation_only'))
        # Model, workload and observation scopes must match before aggregation.
        compare = {'client': active_plan['client'], 'mode': active_plan['mode'], 'scope': scope,
                   'input_fingerprint': fingerprint, 'output_tokens': config['load']['output_tokens'],
                   'stream': config['load']['stream'], 'platform': config['platform']['kind'],
                   'model_path': active_plan['model_identity']['model_path'],
                   'model_config_sha256': active_plan['model_identity']['model_config_sha256'],
                   'hf_overrides': config['server']['hf_overrides'], 'synthetic': synthetic,
                   'unobserved_external_service_run': str(run) if scope == 'external' and actual_service is None else None}
        parameters = {k: copy.deepcopy(config[k]) for k in ['server', 'ucm', 'load', 'capacity', 'platform', 'aisbench', 'swe']}
        parameters['server'].pop('port', None)
        parameters['server'].pop('host', None)
        parameters['capacity'].pop('calibration_file', None)
        parameters['actual_service_configuration'] = actual_service['configuration'] if actual_service else None
        if parameters['actual_service_configuration']:
            parameters['actual_service_configuration'] = {k:v for k,v in parameters['actual_service_configuration'].items()
                                                         if k not in {'configuration_sources', 'paths', 'execution', 'label'}}
            parameters['actual_service_configuration']['server'] = {k:v for k,v in parameters['actual_service_configuration']['server'].items() if k not in {'host','port'}}
        parameters['built_ucm_identity'] = {'commit': environment['source']['exported_commit'], 'wheel_sha256': environment['wheel_sha256']} if environment else None
        identity = json_hash(parameters)
        output.append({'run': str(run), 'path': str(path), 'phase': phase, 'scope': scope,
                       'label': config['label'], 'configuration_group': identity, 'comparability': compare,
                       'comparability_group': json_hash(compare), 'input_fingerprint': fingerprint,
                       'valid': not reasons, 'excluded_reasons': reasons, 'synthetic': synthetic,
                       'performance': performance, 'samples': maybe(path/'request-samples.json', []),
                       'compilation': compilation, 'cleanup': cleanup, 'pipeline': pipeline,
                       'plan': active_plan, 'root_status': maybe(run/'status.json', {})})
    return output


def aggregate(records, include_validation=False):
    groups = {}
    for row in records:
        if not row['valid'] or (row['synthetic'] and not include_validation):
            continue
        key = (row['comparability_group'], row['configuration_group'])
        groups.setdefault(key, []).append(row)
    output = []
    for index, ((comparable, configuration), rows) in enumerate(sorted(groups.items())):
        samples = [sample for r in rows for sample in r['samples']]
        metrics = {}
        for name in ['duration_s', 'requests_per_s', 'input_tokens_per_s', 'output_tokens_per_s']:
            values = [r['performance'][name] for r in rows]
            metrics[name] = {'mean': statistics.fmean(values), 'sample_sd': statistics.stdev(values) if len(values)>1 else None,
                             'per_run': values}
        output.append({'id': f'G{index+1:02d}', 'labels': sorted({r['label'] for r in rows}),
                       'comparability_group': comparable, 'configuration_group': configuration,
                       'comparability': rows[0]['comparability'], 'runs': len(rows),
                       'requests': sum(r['performance']['requests'] for r in rows),
                       'successful_requests': sum(r['performance']['successful_requests'] for r in rows),
                       'metrics': metrics, 'pooled': {name: distribution([s[name] for s in samples if s.get(name) is not None])
                          for name in ['ttft_ms', 'e2e_ms', 'tpot_ms']},
                       'record_paths': [r['path'] for r in rows], 'synthetic': rows[0]['synthetic']})
    return output
