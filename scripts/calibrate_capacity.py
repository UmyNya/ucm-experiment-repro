#!/usr/bin/env python3
"""Measure worker layouts with physical L1 and an explicit or automatic HBM budget."""
import argparse
from pathlib import Path

from common import BUNDLE, new_run, save_json
from configuration import load
from capacity_validation import calibration_record
from service import Service, interrupt_cleanup


def write_calibration(run, value):
    """Retain the identity-bound aggregate and export capacity.py-compatible layouts."""
    run = Path(run)
    layouts = run/'layouts'
    layouts.mkdir(exist_ok=False)
    files = []
    for index, worker in enumerate(value['workers']):
        path = layouts/f'worker-{index:03d}.json'
        save_json(path, {**worker, 'identity': value['identity'], 'source_run': value['source_run'],
                        'scope': 'One observed worker/cache domain; do not sum shared domains',
                        'basis': value['basis']})
        files.append({'worker_index': index, 'worker_rank': worker['worker_rank'],
                      'worker_pid': worker['worker_pid'], 'path': str(path.relative_to(run))})
    save_json(run/'capacity-calibration.json', {**value, 'layout_files': files})
    save_json(run/'next-config.json', {'capacity.calibration_file': str(run/'capacity-calibration.json'),
              'capacity.hbm.calculation.method': 'calibration', 'capacity.l1.calculation.method': 'calibration',
              'note': 'Set mode=tokens and desired token targets after reviewing this measurement; original configuration is never rewritten'})


def calibration_config(path, overrides=()):
    # Preserve engine-selected HBM sizing; explicit/token configurations use
    # their physical bootstrap budget without relying on a stale calibration.
    config = load(path, [*overrides, 'capacity.l1.mode="physical"', 'capacity.calibration_file=null'])
    if config['capacity']['hbm']['mode'] != 'auto':
        config['capacity']['hbm']['mode'] = 'physical'
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=BUNDLE/'config/experiment.json')
    parser.add_argument('--profile', default='swe_w24_n192_performance')
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    c = calibration_config(args.config, args.set)
    run = new_run(c, args.profile, 'capacity-calibration')
    print(run, flush=True)
    try:
        with interrupt_cleanup(), Service(c, args.profile, run, enforce_capacity=False) as service:
            value = calibration_record(service.plan, service.rows)
            write_calibration(run, value)
        save_json(run/'status.json', {'status': 'calibrated', 'service_requests_sent': 0, 'resources_released': True})
    except BaseException as error:
        save_json(run/'status.json', {'status': 'failed', 'error': str(error)})
        raise
    print(run/'capacity-calibration.json')


if __name__ == '__main__':
    main()
