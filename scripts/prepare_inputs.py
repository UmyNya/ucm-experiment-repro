#!/usr/bin/env python3
"""Freeze exact inputs under a new directory, without starting a model or sending requests."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from common import BUNDLE, new_run, read_json, save_json
from configuration import load, compile_plan
from processes import OwnedProcess


def worker(plan_file, output, frozen_only):
    from workloads import prepare_ais, prepare_swe
    plan = read_json(plan_file)
    result = (prepare_ais(plan, output, frozen_without_tokenizer=frozen_only)
              if plan["client"] == "aisbench" else prepare_swe(plan, output))
    save_json(output / "PREPARATION.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config/experiment.json")
    parser.add_argument("--profile")
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--frozen-without-tokenizer", action="store_true",
                        help="AISBench frozen fixture only; explicitly does not verify target tokenizer")
    parser.add_argument("--worker-plan", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.umask(0o077)
    if args.worker_plan:
        worker(args.worker_plan, args.worker_output, args.frozen_without_tokenizer)
        return
    if not args.profile:
        parser.error("--profile is required")
    c = load(args.config, args.set)
    run = new_run(c, args.profile, "inputs")
    plan = compile_plan(c, args.profile, run)
    save_json(run / "plan.json", plan)
    python = c["paths"]["aisbench_python" if plan["client"] == "aisbench" else "swe_python"]
    command = [python, "-B", str(Path(__file__).resolve()), "--worker-plan", str(run / "plan.json"),
               "--worker-output", str(run / "inputs")]
    if args.frozen_without_tokenizer:
        if plan["client"] != "aisbench":
            parser.error("--frozen-without-tokenizer applies only to AISBench")
        command.append("--frozen-without-tokenizer")
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "HF_HOME": str(run / "scratch/hf"), "XDG_CACHE_HOME": str(run / "scratch/xdg")}
    save_json(run / "command.json", command)
    try:
        with OwnedProcess(command, cwd=run, environment=env, log_path=run/'preparation.log', name='prepare-inputs') as process:
            code = process.wait(c['execution']['benchmark_timeout_s'])
    except BaseException as error:
        save_json(run/'status.json', {'status': 'failed', 'error': str(error), 'service_requests_sent': 0})
        print(run)
        raise
    save_json(run / "status.json", {"status": "prepared" if code == 0 else "failed",
                                    "returncode": code, "service_requests_sent": 0})
    print(run)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
