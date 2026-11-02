#!/usr/bin/env python3
"""Read-only host/interpreter probe. Does not start containers or allocate device memory."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        reply = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        return {"available": True, "returncode": reply.returncode,
                "stdout": reply.stdout[-24000:], "stderr": reply.stderr[-2000:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": True, "error_type": type(error).__name__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file")
    parser.add_argument("--image", default="quay.io/ascend/vllm-ascend:v0.30.0-a5")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--disk-path", type=Path, default=Path.cwd())
    parser.add_argument("--skip-device-query", action="store_true")
    args = parser.parse_args()
    os.umask(0o077)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing probe")
    packages = {}
    for name in ["vllm", "vllm-ascend", "torch", "torch-npu", "uc-manager",
                 "transformers", "ais-bench", "aiohttp", "numpy"]:
        try:
            distribution = metadata.distribution(name)
            packages[name] = {"version": distribution.version,
                              "location": str(distribution.locate_file(""))}
        except metadata.PackageNotFoundError:
            packages[name] = {"installed": False}
    result = {
        "captured_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Current host/container and current Python interpreter only; no inference validation",
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": {"machine": platform.machine(), "system": platform.system(),
                     "release": platform.release(), "libc": platform.libc_ver()},
        "cpu_count": os.cpu_count(), "packages": packages,
        "devices": [str(p) for p in sorted(Path("/dev").glob("davinci*"))],
        "device_query_skipped": args.skip_device_query,
        "image": command(["docker", "image", "inspect", args.image, "--format",
                          "{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RepoDigests}}"]),
        "profiler": {name: shutil.which(name) for name in ["msprof", "nsys"]},
        "disk": {}, "model": None,
    }
    for key, path in [("output_volume", args.disk_path), ("shared_memory", Path("/dev/shm"))]:
        if path.exists():
            usage = shutil.disk_usage(path)
            result["disk"][key] = {"path": str(path.resolve()), "total": usage.total,
                                   "used": usage.used, "free": usage.free}
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        result["memory_kib"] = {line.split(":")[0]: int(line.split()[1])
                                for line in meminfo.read_text().splitlines()
                                if line.startswith(("MemTotal:", "MemAvailable:", "SwapTotal:"))}
    if not args.skip_device_query:
        result["npu_smi"] = command(["npu-smi", "info"])
        result["nvidia_smi"] = command(["nvidia-smi",
                                       "--query-gpu=name,uuid,memory.total,memory.used",
                                       "--format=csv,noheader"])
    if args.model:
        model = {"path": str(args.model.resolve()), "exists": args.model.is_dir(),
                 "weight_contents_read": False, "metadata": {}}
        for name in ["config.json", "tokenizer_config.json", "generation_config.json"]:
            path = args.model / name
            if path.is_file() and path.stat().st_size < 4 * 1024 * 1024:
                raw = path.read_bytes()
                model["metadata"][name] = {"sha256": hashlib.sha256(raw).hexdigest(),
                                           "size_bytes": len(raw)}
                if name == "config.json":
                    config = json.loads(raw)
                    model["config_summary"] = {key: config.get(key) for key in [
                        "model_type", "architectures", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "hidden_size", "head_dim", "kv_lora_rank",
                        "qk_rope_head_dim", "torch_dtype", "dtype", "quantization_config"]}
        result["model"] = model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
