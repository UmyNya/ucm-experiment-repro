import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from configuration import load, compile_plan, model_identity
from common import safe_output


class ConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.c = load(ROOT / "config/experiment.json")

    def test_one_config_applies_to_all_five_profiles(self):
        self.c["load"]["concurrency"] = 7
        self.c["server"]["max_num_seqs"] = 3
        self.c["ucm"]["ucm_connectors"][0]["ucm_connector_config"]["cache_dump_capacity_policy"] = "spill"
        self.c["capacity"]["hbm"]["tokens"] = 160000
        self.c["capacity"]["l2"]["gib"] = 17
        for profile in self.c["profiles"]:
            with self.subTest(profile=profile):
                p = compile_plan(self.c, profile, ROOT / "runs/test-plan" / profile)
                self.assertEqual(p["vllm_configuration"]["max-num-seqs"], 3)
                self.assertEqual(p["capacity"]["kv_cache_memory_bytes"], 2686080000)
                store = p["ucm_configuration"]["ucm_connectors"][0]["ucm_connector_config"]
                self.assertEqual(store["cache_dump_capacity_policy"], "spill")
                self.assertEqual(store["posix_capacity_gb"], 17)
                self.assertEqual(p["configuration"]["load"]["concurrency"], 7)

    def test_gpu_and_npu_visibility_and_profile_dispatch(self):
        for kind, env, profiler in [("cuda", "CUDA_VISIBLE_DEVICES", "nsys"),
                                    ("ascend", "ASCEND_RT_VISIBLE_DEVICES", "torch_npu")]:
            c = copy.deepcopy(self.c)
            c["platform"] = {"kind": kind, "visible_devices": ["2", "5"]}
            c["server"]["tensor_parallel_size"] = 2
            plan = compile_plan(c, "swe_w24_n192_pipeline", ROOT / "runs/dispatch")
            self.assertEqual(plan["server_environment"][env], "2,5")
            self.assertEqual(plan["profiling_backend"], profiler)

    def test_separate_config_files_have_recorded_sources(self):
        sources = {x['section']: x for x in self.c['configuration_sources']}
        self.assertEqual(set(sources), {'main', 'server', 'server_env', 'ucm', 'aisbench', 'swe'})
        self.assertEqual(Path(sources['server']['path']).name, 'vllm.json')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'main.json'
            c = copy.deepcopy(self.c)
            c['config_files'] = {'server': sources['server']['path']}
            path.write_text(json.dumps(c))
            with self.assertRaisesRegex(ValueError, 'never both'):
                load(path)

    def test_l2_and_engine_identity_are_per_run(self):
        plans = [compile_plan(self.c, "aisbench_n40_performance", ROOT / "runs" / name) for name in ["one", "two"]]
        self.assertNotEqual(plans[0]["l2_path"], plans[1]["l2_path"])
        self.assertNotEqual(plans[0]["server_command"], plans[1]["server_command"])

    def test_overrides_and_venv_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "venv/bin").mkdir(parents=True)
            (base / "venv/bin/python").symlink_to(sys.executable)
            c = copy.deepcopy(self.c)
            c["paths"]["server_python"] = "venv/bin/python"
            file = base / "config.json"
            file.write_text(json.dumps(c))
            actual = load(file, ["execution.repetitions=3"])
            self.assertEqual(actual["paths"]["server_python"], str(base / "venv/bin/python"))
            self.assertEqual(actual["execution"]["repetitions"], 3)
            with self.assertRaisesRegex(ValueError, "Unknown override"):
                load(file, ["load.concurency=3"])

    def test_calibration_cannot_cross_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / "layout.json"
            file.write_text(json.dumps({"identity": model_identity(self.c), "workers": [{
                "block_tokens": 64, "hbm_bytes_per_block": 1074432, "l1_node_bytes": 1093632,
                "l1_nodes_per_token_block": 1, "l1_reserved_nodes": 1024}]}))
            self.c["capacity"]["calibration_file"] = str(file)
            self.c["capacity"]["hbm"]["calculation"]["method"] = "calibration"
            plan = compile_plan(self.c, "aisbench_n40_performance", ROOT / "runs/cal-plan")
            self.assertEqual(plan["capacity"]["kv_cache_memory_bytes"], 5372160000)
            self.c["platform"]["kind"] = "cuda"
            with self.assertRaisesRegex(ValueError, "identity differs"):
                compile_plan(self.c, "aisbench_n40_performance", ROOT / "runs/cal-plan")

    def test_no_protected_output_and_no_conflicting_cli_override(self):
        with self.assertRaisesRegex(ValueError, "protected"):
            safe_output(Path(self.c["paths"]["ucm_repository"]) / "experiment", self.c)
        with self.assertRaisesRegex(ValueError, "duplicate extra_args"):
            load(ROOT / "config/experiment.json", ['server.extra_args=["--port=1"]'])

    def test_swe_count_is_explicit_and_w32_supported(self):
        self.c["swe"]["session_pool"] = 32
        self.c["swe"]["request_limit"] = 256
        small = compile_plan(self.c, "swe_w24_n192_performance", ROOT / "runs/small")
        full = compile_plan(self.c, "swe_w24_n3379_performance", ROOT / "runs/full")
        self.assertEqual(small["request_count"], 256)
        self.assertEqual(full["request_count"], 3379)
        self.assertEqual(small["configuration"]["swe"]["session_pool"], 32)


if __name__ == "__main__":
    unittest.main()
