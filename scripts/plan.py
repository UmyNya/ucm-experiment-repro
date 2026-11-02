#!/usr/bin/env python3
"""Inspect one configuration without preparing inputs or starting a service."""
import argparse
import json
from pathlib import Path

from common import BUNDLE, save_json
from configuration import load, compile_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=BUNDLE / "config/experiment.json")
    parser.add_argument("--profile", required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        c = load(args.config, args.set)
        # This path is a rendering example, not a reservation or actual run.
        result = compile_plan(c, args.profile, Path(c["paths"]["output_root"]) / "PLAN_ONLY")
        result["plan_only"] = True
        if args.output:
            if args.output.exists():
                raise FileExistsError("Plan output must not exist")
            from common import safe_output
            save_json(safe_output(args.output, c), result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
