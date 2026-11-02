#!/usr/bin/env python3
"""One external-client attempt. Never starts, restarts, or stops the model service."""
import argparse
import asyncio
import os
from pathlib import Path
import sys
import time

from boundaries import drain, rpc
from capacity_validation import validate_workers
from common import http, read_json, save_json
from client_metrics import extract, rows
from compilation_audit import inventory, evaluate
from metrics_sampler import Sampler
from profiling_session import Session
from native_statistics import summarize as native_statistics
from service import interrupt_cleanup
from workloads import activate_source, ais_benchmark_command


class Boundary:
    def __init__(self, plan, attempt, observe, inputs=None):
        self.plan, self.attempt, self.observe = plan, attempt, observe
        self.compile_before = None
        self.cache_before = None
        self.log_offset = None
        self.audit = bool(observe and plan.get('server_log') and plan.get('compile_cache_roots'))
        self.sampler = None
        self.profile = Session(plan, attempt, inputs) if plan['mode'] == 'pipeline' else None

    def close(self):
        if self.sampler is not None:
            self.sampler.stop()
            self.sampler = None
        if self.profile is not None:
            self.profile.stop()

    def __call__(self, stage):
        boundary_time = {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}
        if stage == 'after' and self.sampler is not None:
            self.sampler.stop()
            self.sampler = None
        log = Path(self.plan['server_log']) if self.audit else None
        end_offset = log.stat().st_size if self.audit and stage == 'after' else None
        if self.observe:
            snapshot = drain(self.plan, self.attempt, stage)
            check = validate_workers(self.plan, snapshot)
            save_json(self.attempt/(stage+'-capacity-validation.json'), check)
            if not check['passed'] and self.plan['configuration']['validation']['require_exact_capacity']:
                raise ValueError('; '.join(check['errors']))
            if stage=='after':
                save_json(self.attempt/'native-statistics.json',
                          native_statistics(read_json(self.attempt/'before-workers.json'),snapshot))
        if self.profile is not None and stage == 'before':
            self.profile.start()
        if self.profile is not None and stage == 'after':
            self.profile.stop()
        if self.audit:
            observed = rpc(self.plan, 'bundle_compilation_snapshot')
            cache = inventory(self.plan['compile_cache_roots'])
            save_json(self.attempt/(stage+'-compilation-workers.json'), observed)
            save_json(self.attempt/(stage+'-compile-cache.json'), cache)
            if stage == 'before':
                self.compile_before, self.cache_before = observed, cache
                self.log_offset = log.stat().st_size
            else:
                with log.open('rb') as stream:
                    stream.seek(self.log_offset)
                    text = stream.read(max(0, end_offset-self.log_offset)).decode(errors='replace')
                (self.attempt/'measurement-server.log').write_text(text)
                c = self.plan['configuration']
                result = evaluate(self.compile_before, observed, self.cache_before, cache, text,
                                  graph_enabled=not c['server']['enforce_eager'], platform=c['platform']['kind'])
                save_json(self.attempt/'compilation-check.json', result)
                if not result['passed'] and self.plan.get('measurement_phase') == 'formal' and c['validation']['reject_compilation_during_measurement']:
                    raise ValueError('Formal compilation audit failed: ' + '; '.join(result['errors']))
        if stage == 'before' and self.observe:
            self.sampler = Sampler(self.plan['base_url'], self.attempt/'service-metrics.jsonl',
                                   self.plan['configuration']['execution']['metrics_interval_s'])
            self.sampler.start()
        save_json(self.attempt/(stage+'-boundary.json'), {**(boundary_time if stage == 'after' else
                  {'monotonic_ns': time.monotonic_ns(), 'wall_ns': time.time_ns()}),
                  'physical_drain_observed': self.observe, 'compilation_observed': self.audit})


def ais_warmup(plan, attempt, inputs):
    c = plan['configuration']
    manifest = read_json(inputs/'manifest.json')
    unrelated = read_json(inputs/'unrelated.json')
    prompts = [('unrelated', unrelated['text'], len(unrelated['token_ids']), c['aisbench']['unrelated_cache_salt'])]
    prompts += [(f'seed-{index:04d}', row['question'], manifest['warmup_input_token_counts'][index], c['aisbench']['cache_salt'])
                for index,row in enumerate(rows(inputs/'warmup.jsonl'))]
    evidence = []
    for name, text, tokens, salt in prompts:
        payload = {'model': c['server']['served_model_name'], 'prompt': text, 'max_tokens': 1,
                   'temperature': 0, 'ignore_eos': True, 'add_special_tokens': False,
                   'seed': c['aisbench']['seed'], 'stream': False, 'logprobs': 1}
        if c['aisbench']['cache_salt']:
            payload['cache_salt'] = salt
        start = time.monotonic_ns()
        response = http(plan['base_url'], '/v1/completions', payload, c['execution']['request_timeout_s'])
        item = {'name': name, 'elapsed_s': (time.monotonic_ns()-start)/1e9, 'response': response,
                'expected_prompt_tokens': tokens, 'included_in_measurement': False}
        evidence.append(item)
        save_json(attempt/'warmup-responses.json', evidence)
        usage = response.get('usage', {})
        if usage.get('prompt_tokens') != tokens or usage.get('completion_tokens') != 1:
            raise ValueError(f'Warmup {name} token counts do not match frozen input')
    save_json(attempt/'cache-preparation.json', {'warmup_requests': len(prompts),
              'target_prompts': len(prompts)-1, 'HBM': 'natural residual; never flushed after warmup',
              'L1_L2': 'target-prefix warmup; physical drain checked separately when observation enabled'})


def run(plan, attempt, inputs, observe):
    attempt.mkdir(parents=True, exist_ok=False)
    c = plan['configuration']
    if plan['mode'] == 'pipeline' and not observe:
        raise ValueError('Pipeline requires --service-run or --observe and a compatible instrumented service')
    boundary = Boundary(plan, attempt, observe, inputs)
    try:
        return run_with_boundary(plan, attempt, inputs, observe, boundary)
    finally:
        boundary.close()


def run_with_boundary(plan, attempt, inputs, observe, boundary):
    c = plan['configuration']
    activate_source(plan)
    if plan['client'] == 'aisbench':
        ais_warmup(plan, attempt, inputs)
        command = ais_benchmark_command(plan, attempt, inputs)
        boundary('before')
        try:
            # This worker is already supervised in its own session. Calling
            # AISBench here keeps its children in that session for cleanup.
            from ais_bench.benchmark.cli.main import main as ais_main
            save_json(attempt/'client-command.json', command)
            previous = sys.argv
            sys.argv = ['ais_bench', *command[4:]]
            try:
                try:
                    result = ais_main()
                    code = int(result) if isinstance(result, int) else 0
                except SystemExit as error:
                    code = int(error.code or 0)
            finally:
                sys.argv = previous
        finally:
            boundary('after')
    else:
        from swe_bench.config import make_args, run_settings
        from swe_bench.replay import replay
        args = make_args(inputs/'workload', attempt, base_url=plan['base_url'],
                         model=c['server']['served_model_name'], concurrency=c['load']['concurrency'],
                         output_tokens=c['load']['output_tokens'], max_model_len=plan['max_model_len'],
                         request_timeout=c['execution']['request_timeout_s'], request_limit=plan['request_count'], label=c['label'])
        args.measurement_hook = boundary
        save_json(attempt/'client-settings.json', run_settings(args))
        _, code = asyncio.run(replay(args))
    metrics = extract(plan, attempt, inputs)
    save_json(attempt/'status.json', {'status': 'completed' if code == 0 and metrics['passed'] else 'failed',
              'client_returncode': code, 'physical_drain_observed': observe,
              'managed_service': False, 'service_stopped_by_client': False})
    if code or not metrics['passed']:
        raise RuntimeError('Client execution or result validation failed; retained native results show the cause')
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--attempt', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--observe', action='store_true')
    args = p.parse_args()
    try:
        with interrupt_cleanup():
            run(read_json(args.plan), args.attempt, args.inputs, args.observe)
    except BaseException as error:
        save_json(args.attempt/'status.json', {'status': 'failed', 'error': str(error), 'service_stopped_by_client': False})
        raise


if __name__ == '__main__':
    main()
