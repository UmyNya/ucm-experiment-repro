"""Compile one editable JSON config into an explicit single-configuration run plan."""
from __future__ import annotations

import copy
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re

from capacity import resolve as resolve_layout, GIB, ceildiv
from common import BUNDLE, json_hash, read_json, safe_output, sha256

SECTIONS = {
    "build": "jobs cmake_args wheelhouse install_dependencies image_label",
    "paths": "ucm_repository aisbench_repository swe_repository model tokenizer server_python aisbench_python swe_python output_root l2_root",
    "platform": "kind visible_devices",
    "server": "host port served_model_name seed tensor_parallel_size data_parallel_size pipeline_parallel_size hf_overrides max_model_len max_num_seqs max_num_batched_tokens block_size gpu_memory_utilization kv_cache_dtype enable_prefix_caching enable_chunked_prefill enable_expert_parallel enforce_eager async_scheduling extra_args environment platform_environment",
    "capacity": "calibration_file hbm l1 l2",
    "ucm": "enabled source_ref launch store",
    "load": "concurrency request_rate stream output_tokens",
    "aisbench": "source n input_tokens prefix_ratio seed shuffle dataset warmup_dataset cache_salt unrelated_cache_salt",
    "swe": "session_pool request_limit full_request_limit seed workload_dir regenerate_for_tokenizer trajectory_limit",
    "execution": "repetitions compile_calibration calibration_repetitions startup_timeout_s request_timeout_s benchmark_timeout_s drain_timeout_s shutdown_timeout_s health_poll_s metrics_interval_s",
    "validation": "require_input_identity require_exact_capacity require_cache_drain reject_compilation_during_measurement require_profile_attribution strict_source_clean",
    "profiling": "instrument_native backend patch_files extra_python_paths worker_extension nsys required_stages request_page_size output_formats keep_raw_trace",
}


def positive(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")


def keys(value, allowed, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    unknown = set(value) - set(allowed.split())
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")


def path_value(value, base):
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Path must be nonempty string or null")
    expanded = os.path.expandvars(os.path.expanduser(value))
    if "$" in expanded:
        raise ValueError(f"Unresolved environment variable in path: {value}")
    path = Path(expanded)
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load(path, overrides=()):
    path = Path(path).resolve()
    config = read_json(path)
    keys(config, "schema_version label profiles config_files configuration_sources " + " ".join(SECTIONS), "top-level")
    if config.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1")
    origins = {name: path.parent for name in SECTIONS}
    sources = [{"section": "main", "path": str(path), "sha256": sha256(path)}]
    references = config.pop("config_files", {})
    keys(references, "server ucm aisbench swe", "config_files")
    for section, filename in references.items():
        if section in config:
            raise ValueError(f"Define {section} inline or via config_files, never both")
        source = Path(path_value(filename, path.parent))
        config[section] = read_json(source)
        origins[section] = source.parent
        sources.append({"section": section, "path": str(source), "sha256": sha256(source)})
    config["configuration_sources"] = sources
    for section, allowed in SECTIONS.items():
        keys(config.get(section), allowed, section)
    for override in overrides:
        key, sep, raw = override.partition("=")
        if not sep:
            raise ValueError("Overrides use dotted.key=JSON_value")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        parts = key.split(".")
        target = config
        for part in parts[:-1]:
            if not isinstance(target, dict) or part not in target:
                raise ValueError(f"Unknown override: {key}")
            target = target[part]
        dynamic = key.startswith(("ucm.launch.", "ucm.store.", "server.environment.",
                                  "server.hf_overrides.", "server.platform_environment."))
        if not isinstance(target, dict) or (parts[-1] not in target and not dynamic):
            raise ValueError(f"Unknown override: {key}")
        target[parts[-1]] = value
    base = path.parent
    for key, value in config["paths"].items():
        if key.endswith("_python"):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"paths.{key} must name a Python executable")
            if "/" not in value:
                continue
            # Preserve venv/bin/python: resolving its final symlink loses the venv.
            expanded = os.path.expandvars(os.path.expanduser(value))
            if "$" in expanded:
                raise ValueError(f"Unresolved environment variable in path: {value}")
            config["paths"][key] = os.path.abspath(base / expanded)
            continue
        config["paths"][key] = path_value(value, base)
    for section, name in [("capacity", "calibration_file"), ("aisbench", "dataset"),
                          ("aisbench", "warmup_dataset"), ("swe", "workload_dir")]:
        config[section][name] = path_value(config[section][name], origins[section])
    config["build"]["wheelhouse"] = path_value(config["build"]["wheelhouse"], base)
    for name in ["patch_files", "extra_python_paths"]:
        config["profiling"][name] = [path_value(p, base) for p in config["profiling"][name]]
    validate(config)
    safe_output(config["paths"]["output_root"], config)
    if config["paths"]["l2_root"]:
        safe_output(config["paths"]["l2_root"], config)
    return config


def validate(c):
    if not isinstance(c["label"], str) or not c["label"].strip():
        raise ValueError("label must be a nonempty string")
    if c["platform"]["kind"] not in {"cuda", "ascend"}:
        raise ValueError("platform.kind must be cuda or ascend")
    devices = c["platform"]["visible_devices"]
    if not isinstance(devices, list) or not devices or any(not isinstance(x, str) or not re.fullmatch(r"[A-Za-z0-9_./-]+", x) for x in devices):
        raise ValueError("visible_devices must be a nonempty list of device identifiers")
    if len(devices) != len(set(devices)):
        raise ValueError("visible_devices contains duplicates")
    s = c["server"]
    for key in ["port", "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
                "max_num_seqs", "max_num_batched_tokens", "block_size"]:
        positive(s[key], "server." + key)
    if s["port"] > 65535 or s["host"] not in {"127.0.0.1", "localhost", "0.0.0.0"}:
        raise ValueError("Use a local IPv4 bind host and a port <=65535")
    if math.prod(s[k] for k in ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"]) > len(devices):
        raise ValueError("Visible device count is smaller than TP × PP × DP")
    if type(s["gpu_memory_utilization"]) not in (float, int) or not 0 < s["gpu_memory_utilization"] <= 1:
        raise ValueError("gpu_memory_utilization must be in (0,1]")
    if not isinstance(s["extra_args"], list) or any(not isinstance(x, str) for x in s["extra_args"]):
        raise ValueError("server.extra_args must be an argv array, not a shell command")
    managed = {"model", "host", "port", "served-model-name", "kv-transfer-config", "worker-extension-cls",
               "hf-overrides", "kv-cache-memory-bytes", "seed"} | {k.replace("_", "-") for k in s}
    for token in s["extra_args"]:
        if token.startswith("--") and token.split("=", 1)[0][2:] in managed:
            raise ValueError(f"Set the structured config field instead of duplicate extra_args: {token}")
    for mapping in [s["environment"], *s["platform_environment"].values()]:
        if not isinstance(mapping, dict) or any(not isinstance(v, str) for v in mapping.values()):
            raise ValueError("Configured environment values must be strings")
        for key in mapping:
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
                raise ValueError(f"Invalid environment variable name: {key}")
            if re.search(r"PASSWORD|SECRET|API_KEY|ACCESS_TOKEN|AUTHORIZATION", key, re.I):
                raise ValueError("Do not store credentials in configuration snapshots; use inherited environment")
    for group, names in [("server", "enable_prefix_caching enable_chunked_prefill enable_expert_parallel enforce_eager async_scheduling"),
                         ("execution", "compile_calibration"), ("load", "stream"), ("ucm", "enabled"),
                         ("validation", " ".join(c["validation"]))]:
        for name in names.split():
            if type(c[group][name]) is not bool:
                raise ValueError(f"{group}.{name} must be boolean")
    for key in ["concurrency", "output_tokens"]:
        positive(c["load"][key], "load." + key)
    rate = c["load"]["request_rate"]
    if type(rate) not in (int, float) or not math.isfinite(rate) or rate < 0:
        raise ValueError("request_rate must be finite and nonnegative")
    for name, value in c["execution"].items():
        if name != "compile_calibration":
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"execution.{name} must be finite and positive")
    positive(c["execution"]["repetitions"], "execution.repetitions")
    positive(c["execution"]["calibration_repetitions"], "execution.calibration_repetitions")
    if c["swe"]["session_pool"] not in {24, 32} and not c["swe"]["workload_dir"]:
        raise ValueError("Other session pools require a prepared swe.workload_dir")
    for name in ["request_limit", "full_request_limit"]:
        positive(c["swe"][name], "swe." + name)
    for name in ["n", "input_tokens"]:
        positive(c["aisbench"][name], "aisbench." + name)
    positive(c["profiling"]["request_page_size"], "profiling.request_page_size")
    profiling = c['profiling']
    if profiling['backend'] not in {'auto','nsys','torch_npu'}:
        raise ValueError('profiling.backend must be auto, nsys or torch_npu')
    if profiling['backend'] not in {'auto', 'nsys' if c['platform']['kind']=='cuda' else 'torch_npu'}:
        raise ValueError('Profiling backend does not match the configured device platform')
    if type(profiling['instrument_native']) is not bool:
        raise ValueError('profiling.instrument_native must be boolean')
    if not profiling['output_formats'] or not set(profiling['output_formats']) <= {'png','svg','pdf'}:
        raise ValueError('profiling.output_formats supports png/svg/pdf')
    if set(profiling['required_stages']) != {'L2_read','H2D','device_compute','D2H','L2_write'}:
        raise ValueError('All five stage categories are required; stages with zero actual events remain zero')
    if profiling['keep_raw_trace'] is not True:
        raise ValueError('Raw traces are required for reproducible reports; keep_raw_trace must be true')
    for field in ['require_input_identity','require_cache_drain','require_profile_attribution']:
        if c['validation'][field] is not True:
            raise ValueError(f'validation.{field} is mandatory for the requested reproduction protocol')
    positive(c["build"]["jobs"], "build.jobs")
    if type(c["build"]["install_dependencies"]) is not bool:
        raise ValueError("build.install_dependencies must be boolean")
    if not isinstance(c["build"]["cmake_args"], list) or any(not isinstance(x, str) for x in c["build"]["cmake_args"]):
        raise ValueError("build.cmake_args must be an argv array")
    keys(c["capacity"]["hbm"], "mode tokens bytes calculation", "capacity.hbm")
    keys(c["capacity"]["l1"], "mode gib tokens rounding calculation", "capacity.l1")
    keys(c["capacity"]["l2"], "gib enforcement filesystem_limit_gib", "capacity.l2")
    for name in ["hbm", "l1"]:
        if c["capacity"][name]["mode"] not in {"tokens", "physical"}:
            raise ValueError(f"capacity.{name}.mode must be tokens or physical")
    positive(c["capacity"]["l2"]["gib"], "capacity.l2.gib")
    if c["capacity"]["l2"]["enforcement"] not in {"ucm_gc_budget", "filesystem", "loopback"}:
        raise ValueError("Unknown L2 enforcement mode")
    for profile, value in c["profiles"].items():
        keys(value, "client mode max_model_len subset", "profiles." + profile)
        if value["client"] not in {"aisbench", "swe"} or value["mode"] not in {"performance", "pipeline"}:
            raise ValueError(f"Invalid profile: {profile}")


def capacity_plan(c):
    cap, block = c["capacity"], c["server"]["block_size"]
    hbm, l1, l2 = cap["hbm"], cap["l1"], cap["l2"]
    calibration = read_json(cap["calibration_file"]) if cap["calibration_file"] else None
    workers = calibration["workers"] if calibration else []
    if calibration and calibration.get("identity") != model_identity(c):
        raise ValueError("Capacity calibration identity differs from model/backend/parallel/cache configuration")
    if hbm["mode"] == "physical":
        positive(hbm["bytes"], "capacity.hbm.bytes")
        hbm_bytes, target = hbm["bytes"], None
    else:
        positive(hbm["tokens"], "capacity.hbm.tokens")
        target = hbm["tokens"]
        if target % block:
            raise ValueError("HBM tokens must be a multiple of server.block_size")
        if hbm["calculation"]["method"] == "calibration":
            if not workers:
                raise ValueError("Token HBM calibration method requires capacity.calibration_file")
            sizes = {w["hbm_bytes_per_block"] for w in workers}
            if len(sizes) != 1:
                raise ValueError("Unequal per-worker block bytes cannot use one exact HBM byte budget; use physical mode")
            hbm_bytes = target // block * next(iter(sizes))
        elif hbm["calculation"]["method"] == "bytes_per_token":
            factor = Fraction(str(hbm["calculation"]["bytes_per_token"]))
            if factor <= 0:
                raise ValueError("bytes_per_token must be positive")
            byte_value = target * factor
            if byte_value.denominator != 1:
                raise ValueError("HBM conversion must produce an integer number of bytes")
            hbm_bytes = int(byte_value)
        else:
            raise ValueError("HBM calculation.method must be bytes_per_token or calibration")
    l1_derived = []
    if l1["mode"] == "physical":
        positive(l1["gib"], "capacity.l1.gib")
        l1_gib = l1["gib"]
    else:
        calc = l1["calculation"]
        if calc["method"] == "calibration":
            if not workers:
                raise ValueError("Token L1 calibration method requires capacity.calibration_file")
            layouts = workers
        elif calc["method"] == "layout":
            layouts = [{"block_tokens": block, "hbm_bytes_per_block": 1,
                        "l1_node_bytes": calc["node_bytes"], "l1_nodes_per_token_block": calc["nodes_per_block"],
                        "l1_reserved_nodes": calc["reserved_nodes"]}]
        else:
            raise ValueError("L1 calculation.method must be layout or calibration")
        results = [resolve_layout(w, block * 2, l1_tokens=l1["tokens"], rounding=l1["rounding"]) for w in layouts]
        l1_gib = max(r["cache_buffer_capacity_gb"] for r in results)
        for w in layouts:
            r = resolve_layout(w, block * 2, l1_gib=l1_gib)
            if l1["rounding"] == "exact" and r["l1_derived_ordinary_tokens"] != l1["tokens"]:
                raise ValueError("Exact L1 target is not simultaneously representable on all workers")
            l1_derived.append(r)
    return {"kv_cache_memory_bytes": hbm_bytes, "hbm_target_tokens": target,
            "cache_buffer_capacity_gb": l1_gib, "l1_target_tokens": l1["tokens"] if l1["mode"] == "tokens" else None,
            "l1_per_worker_derivation": l1_derived, "posix_capacity_gb": l2["gib"],
            "l2_enforcement": l2["enforcement"], "l2_filesystem_limit_gib": l2["filesystem_limit_gib"],
            "calibration_sha256": sha256(cap["calibration_file"]) if calibration else None,
            "actual_capacity_verified": False}


def model_identity(c):
    path = Path(c["paths"]["model"]) / "config.json"
    s = c["server"]
    return {"model_path": c["paths"]["model"], "model_config_sha256": sha256(path) if path.is_file() else None,
            "hf_overrides": s["hf_overrides"], "platform": c["platform"]["kind"],
            "tp": s["tensor_parallel_size"], "pp": s["pipeline_parallel_size"], "dp": s["data_parallel_size"],
            "block_size": s["block_size"], "kv_dtype": s["kv_cache_dtype"],
            "ucm_layout_config": {k: c["ucm"]["store"].get(k) for k in ["cache_full_block_l1", "cache_full_block_load", "posix_packed_layer_bytes", "cache_load_exclusive_buffer_number"]},
            "use_layerwise": c["ucm"]["launch"].get("use_layerwise")}


def compile_plan(config, profile, run_dir):
    c = copy.deepcopy(config)
    if profile not in c["profiles"]:
        raise ValueError(f"Unknown profile: {profile}")
    p = c["profiles"][profile]
    run = Path(run_dir).resolve()
    s = c["server"]
    max_len = s["max_model_len"] or p["max_model_len"]
    positive(max_len, "max_model_len")
    if p["client"] == "aisbench" and c["aisbench"]["input_tokens"] + c["load"]["output_tokens"] > max_len:
        raise ValueError("AISBench input plus output exceeds max_model_len")
    if p["client"] == "swe" and (c["load"]["request_rate"] != 0 or not c["load"]["stream"]):
        raise ValueError("SWE protocol requires request_rate=0 and stream=true")
    capacity = capacity_plan(c)
    l2_base = Path(c["paths"]["l2_root"]) if c["paths"]["l2_root"] else run / "scratch"
    l2_path = l2_base / ("l2-" + json_hash(str(run))[:20])
    store = {k: v for k, v in c["ucm"]["store"].items() if v is not None}
    store.update(storage_backends=str(l2_path), cache_buffer_capacity_gb=capacity["cache_buffer_capacity_gb"],
                 posix_capacity_gb=capacity["posix_capacity_gb"])
    ucm = {**c["ucm"]["launch"], "ucm_connectors": [{"ucm_connector_name": "UcmPipelineStore", "ucm_connector_config": store}]}
    args = [c["paths"]["server_python"], "-B", "-m", "vllm.entrypoints.openai.api_server",
            "--model", c["paths"]["model"], "--host", s["host"], "--port", str(s["port"]),
            "--served-model-name", s["served_model_name"], "--max-model-len", str(max_len),
            "--kv-cache-memory-bytes", str(capacity["kv_cache_memory_bytes"])]
    for key in ["seed", "tensor_parallel_size", "data_parallel_size", "pipeline_parallel_size",
                "max_num_seqs", "max_num_batched_tokens", "block_size", "gpu_memory_utilization", "kv_cache_dtype"]:
        args += ["--" + key.replace("_", "-"), str(s[key])]
    if s["hf_overrides"]:
        args += ["--hf-overrides", json.dumps(s["hf_overrides"], separators=(",", ":"))]
    for key in ["enable_prefix_caching", "enable_chunked_prefill", "enable_expert_parallel", "enforce_eager", "async_scheduling"]:
        args += ["--" + ("" if s[key] else "no-") + key.replace("_", "-")]
    if c["ucm"]["enabled"]:
        args += ["--kv-transfer-config", json.dumps({"kv_connector": "UCMConnector", "kv_connector_module_path": "ucm.integration.vllm.ucm_connector", "kv_role": "kv_both", "engine_id": "bundle-" + json_hash(str(run))[:20], "kv_connector_extra_config": {"UCM_CONFIG_FILE": str(run / "ucm.json")}}, separators=(",", ":"))]
    args += ["--worker-extension-cls", c["profiling"]["worker_extension"] or "runtime_observer.BundleObserver"]
    args += s["extra_args"]
    kind = c["platform"]["kind"]
    env = {**s["environment"], **s["platform_environment"].get(kind, {}),
           "VLLM_SERVER_DEV_MODE": "1", "BUNDLE_PLATFORM": kind,
           "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "HF_HOME": str(run / "scratch/hf"), "XDG_CACHE_HOME": str(run / "compile-cache/xdg"),
           "TRITON_CACHE_DIR": str(run / "compile-cache/triton"), "TORCHINDUCTOR_CACHE_DIR": str(run / "compile-cache/inductor"),
           "VLLM_CACHE_ROOT": str(run / "compile-cache/vllm"), "TORCH_EXTENSIONS_DIR": str(run / "compile-cache/torch-extensions"),
           "CUDA_CACHE_PATH": str(run / "compile-cache/cuda-driver"),
           "ASCEND_CACHE_PATH": str(run / "compile-cache/ascend"), "ASCEND_WORK_PATH": str(run / "scratch/ascend-work"),
           "UCM_LOG_PATH": str(run / "ucm-logs"), "BUNDLE_RUN_DIR": str(run), "UCM_CONFIG_FILE": str(run / "ucm.json")}
    env["CUDA_VISIBLE_DEVICES" if kind == "cuda" else "ASCEND_RT_VISIBLE_DEVICES"] = ",".join(c["platform"]["visible_devices"])
    request_count = c["aisbench"]["n"] if p["client"] == "aisbench" else c["swe"]["full_request_limit" if p.get("subset") == "full" else "request_limit"]
    return {"schema_version": 1, "profile": profile, "client": p["client"], "mode": p["mode"],
            "label": c["label"], "run_dir": str(run), "configuration": c, "configuration_sha256": json_hash(c),
            "server_command": args, "server_environment": env, "ucm_configuration": ucm,
            "capacity": capacity, "model_identity": model_identity(c), "max_model_len": max_len,
            "request_count": request_count, "base_url": f"http://127.0.0.1:{s['port']}", "l2_path": str(l2_path),
            "profiling_backend": ("nsys" if kind == "cuda" else "torch_npu") if c["profiling"]["backend"] == "auto" else c["profiling"]["backend"],
            "protocol": {"cache_start": "empty private L1/L2 and fresh server per measured repetition",
                         "warmup": "unrelated + N serial target prompts, then drain; retain natural HBM" if p["client"] == "aisbench" else "one unrelated prompt after a separate full workload compilation calibration",
                         "calibration": c["execution"]["compile_calibration"], "measurements": c["execution"]["repetitions"],
                         "cold_restart_after_calibration": True, "reuse_only_this_campaign_compile_cache": True},
            "warnings": ["Historical CUDA conversion factors are examples; verify actual worker layout before measurement.",
                         "posix_capacity_gb is a cache/GC budget, not a hard filesystem quota."]}
