#!/usr/bin/env python3
"""Run AISBench or SWE against an already running service; never manages that service."""
import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
from urllib.parse import urlsplit

from common import BUNDLE, new_run, read_json, save_json, json_hash
from configuration import load, compile_plan
from ucm_configuration import normalize_ucm
from processes import OwnedProcess, same_process
from service import interrupt_cleanup
from boundaries import rpc
from block_size import input_block_size, observed_block_size


def client_environment(run):
    return {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'HF_HUB_OFFLINE': '1',
            'TRANSFORMERS_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
            'HF_HOME': str(run/'scratch/hf'), 'XDG_CACHE_HOME': str(run/'scratch/xdg'),
            'ENABLE_UCM_PATCH': '0', 'CUDA_VISIBLE_DEVICES': '', 'ASCEND_RT_VISIBLE_DEVICES': ''}


def client_python(plan):
    return plan['configuration']['paths'][plan['client']+'_python' if plan['client'] == 'swe' else 'aisbench_python']


def prepare(plan, run, *, frozen_without_tokenizer=False):
    if plan['client'] == 'aisbench':
        input_block_size(plan)
    command = [client_python(plan), '-B', str(BUNDLE/'scripts/prepare_inputs.py'),
               '--worker-plan', str(run/'plan.json'), '--worker-output', str(run/'inputs')]
    if frozen_without_tokenizer:
        command.append('--frozen-without-tokenizer')
    with OwnedProcess(command, cwd=run, environment=client_environment(run), log_path=run/'preparation.log',
                      name='input-preparation') as process:
        if process.wait(plan['configuration']['execution']['benchmark_timeout_s']):
            raise RuntimeError('Input preparation failed; see preparation.log')


def attempt(plan, run, output, inputs, *, observe):
    command = [client_python(plan), '-B', str(BUNDLE/'scripts/client_worker.py'), '--plan', str(run/'plan.json'),
               '--attempt', str(output), '--inputs', str(inputs)]
    if observe:
        command.append('--observe')
    with OwnedProcess(command, cwd=run, environment=client_environment(run), log_path=run/(output.name+'-client.log'),
                      name=output.name+'-client', shutdown_timeout=plan['configuration']['execution']['shutdown_timeout_s']) as process:
        if process.wait(plan['configuration']['execution']['benchmark_timeout_s']):
            raise RuntimeError(f'Client attempt failed: {output}')


def main(client=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=BUNDLE/'config/experiment.json')
    p.add_argument('--profile', default='aisbench_n40_performance' if client == 'aisbench' else 'swe_w24_n192_performance')
    p.add_argument('--set', action='append', default=[])
    endpoint = p.add_mutually_exclusive_group()
    endpoint.add_argument('--base-url', help='Existing OpenAI-compatible service root, without /v1; defaults to configured host/port')
    endpoint.add_argument('--service-run', type=Path, help='Attach to a ready serve.py run, preserving its actual service configuration')
    p.add_argument('--observe', action='store_true', help='Require BundleObserver native capacity/drain RPC; automatic with --service-run')
    p.add_argument('--repetitions', type=int)
    args = p.parse_args()
    overrides = args.set + ([f'execution.repetitions={args.repetitions}'] if args.repetitions is not None else [])
    c = load(args.config, overrides)
    profile = c['profiles'].get(args.profile)
    if not profile or (client and profile['client'] != client):
        p.error('Selected profile does not belong to this client')
    os.umask(0o077)
    run = new_run(c, args.profile, 'external-client')
    plan = compile_plan(c, args.profile, run)
    if os.environ.get('BUNDLE_VALIDATION_ONLY') == 'mock_http':
        plan['validation_only'] = 'CPU mock HTTP service; synthetic performance must be excluded from experiment comparisons'
    observation = args.observe or args.service_run is not None
    if args.service_run:
        service_run = args.service_run.resolve()
        state = read_json(service_run/'service-status.json')
        if state['status'] != 'ready' or not same_process(state['supervisor']):
            raise ValueError('Selected service has no live ready supervisor')
        plan['base_url'] = state['base_url']
        server_plan = read_json(service_run/'plan.json')
        save_json(run/'actual-service-plan.json', server_plan)
        if c['paths']['tokenizer'] != server_plan['configuration']['paths']['tokenizer']:
            raise ValueError('Client tokenizer path differs from the attached service; set the intended tokenizer explicitly')
        for name in ['server','ucm','capacity','platform']:
            plan['configuration'][name] = copy.deepcopy(server_plan['configuration'][name])
        plan['configuration'].pop('ucm_runtime', None)
        if 'ucm_runtime' in server_plan['configuration']:
            plan['configuration']['ucm_runtime'] = copy.deepcopy(server_plan['configuration']['ucm_runtime'])
        normalize_ucm(plan['configuration'])
        plan['configuration']['paths']['model'] = server_plan['configuration']['paths']['model']
        for name in ['capacity','model_identity','max_model_len','ucm_configuration','profiling_backend',
                     'server_command','server_environment','l2_path']:
            plan[name] = copy.deepcopy(server_plan[name])
        for name in ['vllm_configuration', 'vllm_config_file', 'observed_block_size']:
            plan.pop(name, None)
            if name in server_plan:
                plan[name] = copy.deepcopy(server_plan[name])
        if plan['mode']=='pipeline' and server_plan['mode']!='pipeline':
            raise ValueError('Start serve.py with a pipeline profile before attaching a pipeline client')
        for name in ['service-environment.json', 'startup-workers.json', 'capacity-validation.json', 'vllm-effective.yaml']:
            if (service_run/name).is_file():
                shutil.copyfile(service_run/name, run/name)
        plan['attached_service_run'] = str(service_run)
        plan['server_log'] = server_plan.get('server_log', str(service_run/'server.log'))
        plan['compile_cache_roots'] = server_plan.get('compile_cache_roots', [])
    elif args.base_url:
        parsed = urlsplit(args.base_url)
        if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip('/'):
            p.error('--base-url must be an HTTP(S) service root without credentials, query or path')
        plan['base_url'] = args.base_url.rstrip('/')
    plan['execution_scope'] = 'external client only; service is not started, restarted, reset, or stopped'
    plan['historical_cold_start_protocol_proven'] = False
    plan['protocol'].update(cache_start='existing service and caches; state depends on its previous use',
                            calibration=False,cold_restart_after_calibration=False,
                            reuse_only_this_campaign_compile_cache=False)
    plan['configuration_sha256'] = json_hash(plan['configuration'])
    if plan['mode']=='pipeline':
        plan['pipeline_analysis'] = 'Run analyze_pipeline.py after capture; Nsight requires the service owner to stop/flush the service first'
    save_json(run/'plan.json', plan)
    print(run, flush=True)
    try:
        with interrupt_cleanup():
            if (plan['client'] == 'aisbench' and plan['configuration']['server'].get('block_size') is None
                    and plan.get('observed_block_size') is None and observation):
                rows = rpc(plan, 'bundle_snapshot', [False])
                save_json(run/'input-layout-workers.json', rows)
                plan['observed_block_size'] = observed_block_size(rows)
                plan['model_identity']['block_size'] = plan['observed_block_size']
                save_json(run/'plan.json', plan)
            prepare(plan, run)
            for index in range(c['execution']['repetitions']):
                output = run/'attempts'/f'{index+1:03d}'
                attempt(plan, run, output, run/'inputs', observe=observation)
        save_json(run/'status.json', {'status': 'completed', 'repetitions': c['execution']['repetitions'],
                  'external_service_preserved': True, 'fresh_service_between_repetitions': False,
                  'cold_start_protocol_proven': False, 'physical_drain_observed': observation,
                  'pipeline_analysis_pending': plan['mode']=='pipeline',
                  'scope': 'standalone client measurement; use managed experiment entry for the complete historical cold-start protocol'})
    except BaseException as error:
        save_json(run/'status.json', {'status': 'failed', 'error': str(error), 'external_service_preserved': True})
        raise


if __name__ == '__main__':
    main()
