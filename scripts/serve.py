#!/usr/bin/env python3
"""Start/inspect/stop a standalone owned vLLM+UCM service inside an existing container."""
import argparse
import os
from pathlib import Path
import time

from common import BUNDLE, http, new_run, read_json, save_json, utc
from configuration import load
from processes import same_process
from service import Service, interrupt_cleanup


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    start = sub.add_parser('start', help='Run in foreground; Ctrl-C releases only this service and its caches')
    start.add_argument('--config', type=Path, default=BUNDLE/'config/experiment.json')
    start.add_argument('--profile', default='swe_w24_n192_performance', help='Select max-model-length/profile defaults; no benchmark is started')
    start.add_argument('--set', action='append', default=[])
    for name in ['status', 'stop']:
        sub.add_parser(name).add_argument('run_dir', type=Path)
    args = parser.parse_args()
    os.umask(0o077)
    if args.action != 'start':
        run = args.run_dir.resolve()
        status = read_json(run/'service-status.json')
        live = same_process(status['supervisor'])
        if args.action == 'status':
            result = dict(status, supervisor_alive=bool(live))
            if status['status'] == 'ready' and live:
                try:
                    http(status['base_url'], '/health', timeout=5)
                    result['healthy'] = True
                except Exception as error:
                    result.update(healthy=False, health_error=str(error))
            import json
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        if status['status'] != 'ready' or not live:
            raise ValueError('This run has no live ready supervisor; no PID will be signalled')
        save_json(run/'stop-request.json', {'supervisor': status['supervisor'], 'requested_utc': utc()})
        deadline = time.monotonic()+read_json(run/'configuration.json')['execution']['shutdown_timeout_s']+30
        while time.monotonic() < deadline:
            state = read_json(run/'service-status.json')
            if state['status'] in {'stopped', 'failed'}:
                print(run/'resource-cleanup.json')
                if state['status'] == 'failed':
                    raise RuntimeError('Supervisor cleanup failed; inspect retained resources')
                return
            time.sleep(.25)
        raise TimeoutError('Stop request recorded; supervisor has not confirmed cleanup. Inspect logs before taking further action.')
    c = load(args.config, args.set)
    run = new_run(c, args.profile, 'standalone-service')
    print(run, flush=True)
    server = Service(c, args.profile, run)
    state = {'status': 'starting', 'base_url': server.plan['base_url'],
             'supervisor': read_json(run/'resource-namespace.json')['supervisor']}
    save_json(run/'service-status.json', state)
    error = None
    try:
        with interrupt_cleanup(), server:
            print(f"READY {server.plan['base_url']}  run={run}", flush=True)
            while True:
                if server.process.poll() is not None:
                    raise RuntimeError('Service process exited unexpectedly; inspect server.log')
                request = run/'stop-request.json'
                if request.is_file() and read_json(request)['supervisor'] == state['supervisor']:
                    break
                server.capture_resources()
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    except BaseException as failure:
        error = failure
    finally:
        state.update(status='failed' if error else 'stopped', stopped_utc=utc(), error=str(error) if error else None)
        save_json(run/'service-status.json', state)
    if error:
        raise error


if __name__ == '__main__':
    main()
