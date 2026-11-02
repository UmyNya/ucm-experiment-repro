"""Client preparation and direct AISBench configuration, independent of device backend."""
from __future__ import annotations

import array
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from common import read_json, save_json, sha256, json_hash


def activate_source(plan):
    paths = plan["configuration"]["paths"]
    if plan["client"] == "aisbench":
        os.environ["AISBENCH_UCM_ROOT"] = paths["aisbench_repository"]
        sys.path.insert(0, str(Path(paths["aisbench_repository"]) / "src"))
    else:
        sys.path.insert(0, paths["swe_repository"])
    sys.dont_write_bytecode = True


def ais_config(plan):
    activate_source(plan)
    from aisbench_ucm.config import DEFAULTS, validate
    c = plan["configuration"]
    config = copy.deepcopy(DEFAULTS)
    config["mode"] = "external"
    config["experiment"] = {"repetitions": 1, "early_prefetch": [bool(c["ucm"]["launch"].get("enable_early_prefetch"))], "alternate_order": False}
    for key in ["source", "n", "input_tokens", "prefix_ratio", "seed", "shuffle", "dataset", "warmup_dataset"]:
        config["workload"][key] = c["aisbench"][key]
    config["workload"]["output_tokens"] = c["load"]["output_tokens"]
    config["workload"]["block_size"] = c["server"]["block_size"]
    config["endpoint"] = {"base_url": plan["base_url"], "model": c["server"]["served_model_name"],
                          "tokenizer": c["paths"]["tokenizer"], "api_key_env": "AISBENCH_API_KEY"}
    executable = c["paths"]["aisbench_python"]
    config["runtime"]["bench_python"] = shutil.which(executable) if "/" not in executable else executable
    if not config["runtime"]["bench_python"]:
        raise FileNotFoundError(f"Client Python not found: {executable}")
    config["server"]["max_model_len"] = plan["max_model_len"]
    config["server"]["port"] = c["server"]["port"]
    config["load"] = {k: c["load"][k] for k in ["concurrency", "request_rate", "stream"]}
    config["observation"]["enabled"] = False  # Service observer is owned by this bundle.
    config["preparation"]["wait_dump"] = False  # Bundle brackets the measurement with native drain checks.
    config["preparation"]["cache_salt"] = c["aisbench"]["cache_salt"]
    config["preparation"]["unrelated_cache_salt"] = c["aisbench"]["unrelated_cache_salt"]
    validate(config)
    return config


def write_tokens(path, token_lists):
    manifest = []
    with Path(path).open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
            for index, tokens in enumerate(token_lists):
                values = array.array("I", tokens)
                if sys.byteorder != "little":
                    values.byteswap()
                blob = values.tobytes()
                compressed.write(blob)
                manifest.append({"ordinal": index, "tokens": len(tokens),
                                 "sha256_u32le": hashlib.sha256(blob).hexdigest()})
    return manifest


def prepare_ais(plan, output, *, frozen_without_tokenizer=False):
    config = ais_config(plan)
    from aisbench_ucm.data import prepare, read_pool
    from aisbench_ucm.util import resolve_path
    manifest = prepare(config, output, offline=frozen_without_tokenizer)
    if "pending" in manifest:
        raise ValueError("This workload requires a local tokenizer; frozen-only offline preparation is insufficient")
    save_json(output / "client-config.json", config)
    if manifest["source"] == "frozen":
        items = read_pool(resolve_path(config["workload"]["pool"]))
        formal = [items[i]["formal"]["token_ids"] for i in manifest["formal_order"]]
        warmup = [items[i]["seed"]["token_ids"] for i in manifest["warmup_order"]]
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["endpoint"]["tokenizer"], local_files_only=True)
        def encoded(name):
            return [tokenizer.encode(json.loads(line)["question"], add_special_tokens=False)
                    for line in (output / name).read_text().splitlines() if line]
        formal, warmup = encoded("formal.jsonl"), encoded("warmup.jsonl")
    snapshots = output / "prompt-snapshots"
    snapshots.mkdir(mode=0o700)
    formal_rows = write_tokens(snapshots / "formal.u32le.gz", formal)
    warmup_rows = write_tokens(snapshots / "warmup.u32le.gz", warmup)
    fingerprint = hashlib.sha256(b"".join(bytes.fromhex(x["sha256_u32le"]) for x in formal_rows)).hexdigest()
    snapshot = {"format": "concatenated uint32 little-endian; per-request token counts give slice boundaries",
                "formal": formal_rows, "warmup": warmup_rows, "input_fingerprint": fingerprint,
                "tokenizer_checked": not frozen_without_tokenizer,
                "texts": {"formal": "../formal.jsonl", "warmup": "../warmup.jsonl", "unrelated": "../unrelated.json"}}
    save_json(snapshots / "manifest.json", snapshot)
    return {"client": "aisbench", "manifest": manifest, "prompt_snapshot": snapshot,
            "service_requests_sent": 0, "input_fingerprint": fingerprint}


def prepare_swe(plan, output):
    activate_source(plan)
    from swe_bench.config import ROOT, make_args
    from swe_bench.workload import validate_closed_loop_workload
    from swe_bench.prompts import load_prompt_sources
    from swe_bench.snapshots import save_prompt_snapshot
    from swe_bench.token_cache import load_token_cache
    from swe_bench.verify import verify_inputs
    from swe_bench.snapshots import little_endian_bytes
    c, swe = plan["configuration"], plan["configuration"]["swe"]
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source_check = verify_inputs(tokens=False)
    save_json(output / "source-data-verification.json", source_check)
    if swe["workload_dir"]:
        source = Path(swe["workload_dir"])
    elif swe["regenerate_for_tokenizer"] or swe["seed"] != 20260911 or swe["trajectory_limit"] != 100:
        from swe_bench.prepare import prepare_workload
        source = output / "generated-workload"
        prepare_workload(source, session_pool=swe["session_pool"], seed=swe["seed"],
                         trajectory_limit=swe["trajectory_limit"],
                         tokenizer=Path(c["paths"]["tokenizer"]) if swe["regenerate_for_tokenizer"] else None)
    else:
        source = ROOT / f"workloads/W{swe['session_pool']}-N3379-seed20260911"
    manifest = read_json(source / "manifest.json")
    args = make_args(source, output / "unused-client-output", base_url=plan["base_url"],
                     model=c["server"]["served_model_name"], concurrency=c["load"]["concurrency"],
                     output_tokens=c["load"]["output_tokens"], max_model_len=plan["max_model_len"],
                     request_limit=plan["request_count"])
    order, _ = validate_closed_loop_workload(args.order, args.manifest, args.profile,
                                             seed=args.seed, session_pool=args.session_pool,
                                             prompt_salt_prefix=args.prompt_salt_prefix)
    if args.session_pool != swe["session_pool"] or args.seed != swe["seed"]:
        raise ValueError("Prepared SWE workload W/seed differs from the central config")
    order = order[:plan["request_count"]]
    if "token_storage" in manifest:
        sources, summary = load_token_cache(source, manifest["token_storage"], order)
    else:
        reference_tokenizer = ROOT / "inputs/tokenizer/tokenizer.json"
        target_tokenizer = Path(c["paths"]["tokenizer"]) / "tokenizer.json"
        if not target_tokenizer.is_file() or sha256(target_tokenizer) != sha256(reference_tokenizer):
            raise ValueError("Frozen SWE token IDs require the original tokenizer; set swe.regenerate_for_tokenizer=true for another model")
        sources, summary = load_prompt_sources(order, parquet_path=args.parquet, tools_path=args.tools,
                                               tokenizer_path=args.tokenizer,
                                               prompt_salt_prefix=args.prompt_salt_prefix)
    snapshot = save_prompt_snapshot(output / "prompt-snapshots", order, sources,
                                   max_model_len=plan["max_model_len"], output_tokens=c["load"]["output_tokens"])
    token_hash_check = {"scope": "new prepared input", "reference_requests_checked": None}
    if "token_storage" not in manifest:
        expected = {r["request_id"]: r for r in read_json(ROOT / "reference/payload-hashes.json")["rows"]}
        for event in order:
            vector = sources[str(event["trajectory_id"])].prompt_for(int(event["request_index"]))
            actual = hashlib.sha256(little_endian_bytes(vector)).hexdigest()
            ref = expected[event["request_id"]]
            if actual != ref["token_ids_sha256"] or len(vector) != ref["tokens"]:
                raise ValueError(f"Frozen reference token identity mismatch: {event['request_id']}")
        token_hash_check = {"scope": "frozen reference payload hashes", "reference_requests_checked": len(order),
                            "reference_file_sha256": sha256(ROOT / "reference/payload-hashes.json")}
    save_json(output / "token-identity-verification.json", token_hash_check)
    # Prepare a self-contained token-cache workload for repetitions. No re-tokenization during replay.
    workload = output / "workload"
    workload.mkdir()
    for name in ["order.jsonl", "profiles.jsonl", "manifest.json"]:
        shutil.copyfile(source / name, workload / name)
    shutil.copyfile(source / "manifest.json", output / "source-manifest.json")
    storage = {"format": "prepared-uint32le-v1", "directory": "token-cache",
               "chat_template_sha256": manifest.get("token_storage", {}).get("chat_template_sha256", "frozen-token-vectors"),
               "trajectories": {}}
    (workload / "token-cache").mkdir()
    sm = read_json(output / "prompt-snapshots/manifest.json")
    for trajectory, record in sm["trajectories"].items():
        shutil.copyfile(output / "prompt-snapshots" / record["file"], workload / "token-cache" / record["file"])
        storage["trajectories"][trajectory] = {"file": record["file"], "sha256_u32le": record["uncompressed_sha256"],
                                              "ends": record["ends"], "salt_ids": record["salt_ids"],
                                              "starts": record["starts"] or {key: 0 for key in record["ends"]}}
    manifest["token_storage"] = storage
    manifest["bundle_snapshot"] = {"source_manifest_sha256": sha256(source / "manifest.json"),
                                    "available_request_prefix": plan["request_count"],
                                    "prompt_fingerprint": snapshot["input_fingerprint"]}
    save_json(workload / "manifest.json", manifest)
    # Re-open the generated cache through the original reader before accepting the snapshot.
    copied_sources, _ = load_token_cache(workload, storage, order)
    for event in order:
        key, index = str(event["trajectory_id"]), int(event["request_index"])
        if copied_sources[key].prompt_for(index) != sources[key].prompt_for(index):
            raise ValueError("SWE snapshot round-trip changed a request")
    return {"client": "swe", "source": str(source), "source_manifest_sha256": sha256(source / "manifest.json"),
            "input_fingerprint": snapshot["input_fingerprint"], "prompt_snapshot": snapshot,
            "prompt_preparation": summary, "request_count": len(order), "workload": str(workload),
            "token_identity_verification": token_hash_check, "service_requests_sent": 0}


def ais_benchmark_command(plan, attempt, inputs):
    """Write a direct AISBench config; no writes/symlinks into its installed package."""
    c = plan["configuration"]
    work = Path(attempt) / "aisbench"
    dataset = work / "dataset"
    dataset.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(inputs) / "formal.jsonl", dataset / "test.jsonl")
    with (Path(inputs) / "formal.jsonl").open() as source:
        first = source.readline()
    (dataset / "train.jsonl").write_text(first)  # ZeroRetriever never sends a training example.
    generation = {"temperature": 0, "ignore_eos": True, "add_special_tokens": False,
                  "seed": c["aisbench"]["seed"]}
    if c["aisbench"]["cache_salt"]:
        generation["cache_salt"] = c["aisbench"]["cache_salt"]
    model = {"attr": "service", "abbr": "bundle-model", "path": c["paths"]["tokenizer"],
             "model": c["server"]["served_model_name"], "api_key": "", "stream": c["load"]["stream"],
             "request_rate": c["load"]["request_rate"], "retry": 1,
             "host_ip": "127.0.0.1", "host_port": c["server"]["port"], "url": plan["base_url"],
             "max_out_len": c["load"]["output_tokens"], "batch_size": c["load"]["concurrency"],
             "generation_kwargs": generation}
    text = "\n".join([
        "from ais_bench.benchmark.models import VLLMCustomAPI",
        "from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate",
        "from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever",
        "from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer",
        "from ais_bench.benchmark.datasets import GSM8KDataset, Gsm8kEvaluator",
        "from ais_bench.benchmark.summarizers import DefaultPerfSummarizer",
        "from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator",
        f"models = [dict(type=VLLMCustomAPI, **{model!r})]",
        "datasets = [dict(abbr='gsm8k', type=GSM8KDataset, path=" + repr(str(dataset)) + ",",
        " reader_cfg=dict(input_columns=['question'], output_column='answer'),",
        " infer_cfg=dict(prompt_template=dict(type=PromptTemplate, template='{question}'),",
        " retriever=dict(type=ZeroRetriever), inferencer=dict(type=GenInferencer)),",
        " eval_cfg=dict(evaluator=dict(type=Gsm8kEvaluator)))]",
        "summarizer = dict(attr='performance', type=DefaultPerfSummarizer,",
        " calculator=dict(type=DefaultPerfMetricCalculator, stats_list=['Average','Min','Max','Median','P75','P90','P95','P99']))",
        "",
    ])
    (work / "config.py").write_text(text)
    return [c["paths"]["aisbench_python"], "-B", "-c", "from ais_bench.benchmark.cli.main import main; main()",
            str(work / "config.py"), "--mode", "perf", "--work-dir", str(work / "outputs"),
            "--debug", "--num-warmups", "0"]
