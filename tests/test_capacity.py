"""Check capacity rounding against captured physical layouts, not mocked model formulas."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from capacity import resolve


class CapacityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT / "tests/fixtures/capacity-layouts.json"
        cls.history = json.loads(path.read_text())

    def layout(self, name):
        evidence = self.history[name]["capacity-derivation.json"]
        return {
            "block_tokens": evidence["block_size_tokens"],
            "hbm_bytes_per_block": evidence["real_tensor_bytes_per_block"],
            "hbm_reserved_blocks": evidence["null_reserved_blocks"],
            "l1_node_bytes": evidence["physical_shard_bytes"],
            "l1_nodes_per_token_block": evidence["shards_per_token_block"],
            "l1_reserved_nodes": evidence["reserved_shards_excluded"],
        }

    def test_three_measured_layouts(self):
        for name in ["A-R1", "OFF-R1", "D21-R1"]:
            with self.subTest(name=name):
                evidence = self.history[name]["capacity-derivation.json"]
                result = resolve(self.layout(name), 320000, l1_gib=evidence["derived_l1_gib"])
                self.assertEqual(result["l1_nodes"], evidence["actual_nodes"])
                self.assertEqual(result["l1_derived_ordinary_tokens"], evidence["actual_l1_ordinary_tokens"])
                self.assertEqual(result["kv_cache_memory_bytes"], 5372160000)
                self.assertEqual(result["hbm_allocatable_upper_bound_tokens"], 319936)

    def test_round_up_is_minimal_and_reports_difference(self):
        for name in ["A-R1", "OFF-R1", "D21-R1"]:
            with self.subTest(name=name):
                result = resolve(self.layout(name), 320000, l1_tokens=1000000)
                self.assertGreaterEqual(result["l1_derived_ordinary_tokens"], 1000000)
                previous = resolve(self.layout(name), 320000,
                                   l1_gib=result["cache_buffer_capacity_gb"] - 1)
                self.assertLess(previous["l1_derived_ordinary_tokens"], 1000000)
                self.assertEqual(result["l1_target_difference_tokens"],
                                 result["l1_derived_ordinary_tokens"] - 1000000)

    def test_exact_rejects_unrepresentable_target(self):
        with self.assertRaisesRegex(ValueError, "not representable"):
            resolve(self.layout("D21-R1"), 320000, l1_tokens=1000000, rounding="exact")

    def test_invalid_hbm_alignment(self):
        with self.assertRaisesRegex(ValueError, "multiple"):
            resolve(self.layout("D21-R1"), 320001, l1_gib=18)

    def test_insufficient_l1_and_bool(self):
        layout = self.layout("D21-R1")
        layout["l1_reserved_nodes"] = 10000000
        with self.assertRaisesRegex(ValueError, "no ordinary"):
            resolve(layout, 320000, l1_gib=1)
        with self.assertRaises(ValueError):
            resolve(self.layout("D21-R1"), True, l1_gib=18)


if __name__ == "__main__":
    unittest.main()
