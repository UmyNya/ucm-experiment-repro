#!/usr/bin/env python3
"""Convert token targets using measured per-worker layout; no generic model-size guess."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

GIB = 1 << 30


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def resolve(layout, hbm_tokens, l1_tokens=None, l1_gib=None, rounding="at_least"):
    """Capacity is per worker/cache domain; this function does not aggregate TP/DP."""
    required = ["block_tokens", "hbm_bytes_per_block", "l1_node_bytes",
                "l1_nodes_per_token_block", "l1_reserved_nodes"]
    for name in required:
        integer(layout[name], name, 0 if name == "l1_reserved_nodes" else 1)
    integer(hbm_tokens, "hbm_tokens")
    block = layout["block_tokens"]
    if hbm_tokens % block:
        raise ValueError("HBM token target must be a multiple of block_tokens")
    if (l1_tokens is None) == (l1_gib is None):
        raise ValueError("Specify exactly one of l1_tokens and l1_gib")
    if rounding not in {"at_least", "exact"}:
        raise ValueError("rounding must be at_least or exact")
    if l1_tokens is not None:
        integer(l1_tokens, "l1_tokens")
        nodes = (ceildiv(l1_tokens, block) * layout["l1_nodes_per_token_block"]
                 + layout["l1_reserved_nodes"])
        l1_gib = ceildiv(nodes * layout["l1_node_bytes"], GIB)
    else:
        integer(l1_gib, "l1_gib")
    nodes = l1_gib * GIB // layout["l1_node_bytes"]
    ordinary_nodes = nodes - layout["l1_reserved_nodes"]
    if ordinary_nodes < layout["l1_nodes_per_token_block"]:
        raise ValueError("L1 budget leaves no ordinary complete token block")
    actual = ordinary_nodes // layout["l1_nodes_per_token_block"] * block
    if l1_tokens is not None and rounding == "exact" and actual != l1_tokens:
        raise ValueError(
            f"Exact L1 target is not representable by integer GiB: target={l1_tokens}, "
            f"at_least={actual} at {l1_gib} GiB")
    null_blocks = integer(layout.get("hbm_reserved_blocks", 1), "hbm_reserved_blocks", 0)
    hbm_blocks = hbm_tokens // block
    if hbm_blocks <= null_blocks:
        raise ValueError("HBM target leaves no allocatable token block")
    return {
        "layout": layout,
        "scope": "One worker/cache domain; derived plan, not runtime verification",
        "hbm_target_tokens": hbm_tokens,
        "hbm_blocks": hbm_blocks,
        "kv_cache_memory_bytes": hbm_blocks * layout["hbm_bytes_per_block"],
        "hbm_allocatable_upper_bound_tokens": (hbm_blocks - null_blocks) * block,
        "l1_target_ordinary_tokens": l1_tokens,
        "cache_buffer_capacity_gb": l1_gib,
        "l1_payload_budget_bytes": l1_gib * GIB,
        "l1_shared_memory_metadata_bytes": None,
        "l1_nodes": nodes,
        "l1_derived_ordinary_tokens": actual,
        "l1_rounding_policy": rounding,
        "l1_target_difference_tokens": actual - l1_tokens if l1_tokens is not None else None,
        "runtime_verification": "required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--hbm-tokens", type=int, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--l1-tokens", type=int)
    group.add_argument("--l1-gib", type=int)
    parser.add_argument("--rounding", choices=["at_least", "exact"], default="at_least")
    args = parser.parse_args()
    try:
        value = resolve(json.loads(args.layout.read_text()), args.hbm_tokens,
                        args.l1_tokens, args.l1_gib, args.rounding)
    except (KeyError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
