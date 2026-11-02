#!/usr/bin/env python3
"""One configuration: full calibration, fresh services, formal repetitions, retained evidence."""
import argparse
import copy
import os
from pathlib import Path

from common import BUNDLE, new_run, read_json, save_json, utc
from configuration import load, compile_plan
from run_client import prepare, attempt
from service import Service, interrupt_cleanup
from analyze_pipeline import analyze_capture
from block_size import input_block_size


def execute(config, profile, run, *, service_factory=Service, prepare_inputs=prepare, run_client=attempt,
            analyze_pipeline=analyze_capture):
    plan = compile_plan(config, profile, run)
    plan['execution_scope'] = 'managed single-configuration campaign'
    if os.environ.get('BUNDLE_VALIDATION_ONLY') == 'mock_http':
        plan['validation_only'] = 'CPU fixture; synthetic metrics excluded from real comparisons'
    save_json(run/'plan.json', plan)
    state = {'status': 'preparing', 'profile': profile, 'started_utc': utc(), 'stages': [],
             'single_configuration': True, 'input_fingerprint': None}
    save_json(run/'campaign-state.json', state)
    def prepare_once():
        prepare_inputs(plan, run)
        prepared = read_json(run/'inputs/PREPARATION.json')
        state['input_fingerprint'] = prepared['input_fingerprint']

    try:
        delayed_inputs = plan['client'] == 'aisbench' and config['server'].get('block_size') is None
        if not delayed_inputs:
            prepare_once()
        stages = []
        if config['execution']['compile_calibration']:
            stages += [('calibration', index+1) for index in range(config['execution']['calibration_repetitions'])]
        stages += [('formal', index+1) for index in range(config['execution']['repetitions'])]
        for phase, repetition in stages:
            directory = run/'stages'/f'{phase}-{repetition:03d}'
            directory.mkdir(parents=True, exist_ok=False)
            c = copy.deepcopy(config)
            if phase == 'calibration':
                c['profiles'][profile]['mode'] = 'performance'
            row = {'phase': phase, 'repetition': repetition, 'path': str(directory.relative_to(run)),
                   'status': 'starting', 'request_count': plan['request_count'], 'started_utc': utc()}
            state['stages'].append(row)
            state['status'] = phase
            save_json(run/'campaign-state.json', state)
            with service_factory(c, profile, directory/'service', compile_cache=run/'compile-cache') as service:
                if plan['client'] == 'aisbench':
                    block = input_block_size(service.plan)
                    if delayed_inputs and state['input_fingerprint'] is None:
                        plan['observed_block_size'] = block
                        plan['model_identity']['block_size'] = block
                        save_json(run/'plan.json', plan)
                        prepare_once()
                    elif block != input_block_size(plan):
                        raise ValueError('Service block_size changed from the prepared AISBench input alignment')
                selected = copy.deepcopy(service.plan)
                selected.update(measurement_phase=phase, repetition=repetition,
                                input_fingerprint=state['input_fingerprint'], campaign=str(run),
                                execution_scope=plan['execution_scope'])
                if 'validation_only' in plan:
                    selected['validation_only'] = plan['validation_only']
                save_json(directory/'plan.json', selected)
                row['status'] = 'running'
                save_json(run/'campaign-state.json', state)
                run_client(selected, directory, directory/'attempt', run/'inputs', observe=True)
                metrics = read_json(directory/'attempt/performance.json')
                if not metrics['passed'] or metrics['requests'] != plan['request_count']:
                    raise RuntimeError('Client result did not satisfy the selected single workload')
                if phase == 'formal' and config['validation']['reject_compilation_during_measurement']:
                    if not read_json(directory/'attempt/compilation-check.json')['passed']:
                        raise RuntimeError('Formal compilation checks did not pass')
                row['client_passed'] = True
            cleanup = read_json(directory/'service/resource-cleanup.json')
            if not cleanup['released']:
                raise RuntimeError('Owned service resources were not released')
            if selected['mode']=='pipeline':
                analyze_pipeline(selected,directory/'attempt')
            row.update(status='completed', resources_released=True, finished_utc=utc())
            save_json(run/'campaign-state.json', state)
        state.update(status='completed', finished_utc=utc(), expected_formal_repetitions=config['execution']['repetitions'],
                     formal_repetitions_completed=sum(x['phase']=='formal' and x['status']=='completed' for x in state['stages']),
                     complete_calibration_before_fresh_formal_service=config['execution']['compile_calibration'])
        save_json(run/'status.json', state)
        return state
    except BaseException as error:
        if state['stages'] and state['stages'][-1]['status'] != 'completed':
            state['stages'][-1].update(status='failed', error=str(error))
        state.update(status='failed', error=str(error), finished_utc=utc())
        save_json(run/'status.json', state)
        raise
    finally:
        save_json(run/'campaign-state.json', state)


def main(default_profile=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=BUNDLE/'config/experiment.json')
    p.add_argument('--profile', default=default_profile, required=default_profile is None)
    p.add_argument('--set', action='append', default=[])
    p.add_argument('--repetitions', type=int)
    args = p.parse_args()
    overrides = args.set + ([f'execution.repetitions={args.repetitions}'] if args.repetitions is not None else [])
    c = load(args.config, overrides)
    if args.profile not in c['profiles']:
        p.error('Unknown profile')
    os.umask(0o077)
    run = new_run(c, args.profile, 'campaign')
    print(run, flush=True)
    with interrupt_cleanup():
        execute(c, args.profile, run)


if __name__ == '__main__':
    main()
