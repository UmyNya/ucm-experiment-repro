"""vLLM worker extension: device-neutral capacity and physical I/O lifecycle snapshots.

Imports torch/vLLM/UCM only inside worker RPC methods. Unsupported layouts fail
explicitly so a new backend cannot silently pass a historical invariant.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import struct
import time


def _context():
    from vllm.distributed.kv_transfer import get_kv_transfer_group
    group = get_kv_transfer_group()
    connector = group.connector
    for field in ["kv_cache_layout", "kv_caches", "unique_id", "connector_configs", "store"]:
        if not hasattr(connector, field):
            raise RuntimeError(f"UCM observer adapter lacks required field {field}; provide a compatible worker extension")
    return group, connector


def _plain(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _tensors(value, prefix=""):
    if isinstance(value, dict):
        for name, item in value.items():
            yield from _tensors(item, prefix + str(name) + "/")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensors(item, prefix + str(index) + "/")
    else:
        yield prefix.rstrip("/"), value


class BundleObserver:
    def bundle_profile_start(self, capture_id, metadata, backend):
        from pipeline_capture import Capture
        if getattr(self, '_bundle_capture', None) is not None:
            raise RuntimeError('A pipeline capture is already active')
        capture = Capture(self, capture_id, metadata, backend)
        result = capture.start()
        self._bundle_capture = capture
        return result

    def bundle_profile_stop(self, capture_id):
        capture = getattr(self, '_bundle_capture', None)
        if capture is None:
            return {'pid': os.getpid(), 'stopped': True, 'already_inactive': True}
        if capture.capture_id != capture_id:
            raise RuntimeError('Refusing to stop another client capture')
        try:
            return capture.stop()
        finally:
            self._bundle_capture = None

    def bundle_compilation_snapshot(self):
        from compilation_watch import snapshot
        return snapshot()

    def bundle_snapshot(self, collect_metrics=True):
        import torch
        import vllm
        import ucm
        from ucm.integration.vllm.ucm_connector import _get_store_io_sizes
        from vllm.platforms import current_platform

        group, connector = _context()
        layout = connector.kv_cache_layout
        store_config = connector.connector_configs[0]["ucm_connector_config"]
        actual = getattr(connector.store, "config", store_config)
        shard_bytes, block_bytes = _get_store_io_sizes(layout.shard_size, layout.block_size)
        domain = connector.store.domain_info() if hasattr(connector.store, "domain_info") else None
        whole = bool(actual.get("cache_full_block_l1", False))
        if whole and not domain:
            raise RuntimeError("Whole-block L1 requires native domain_info for accurate capacity calibration")
        node_bytes = int(domain["shard_bytes"] if domain else shard_bytes)
        physical_block = int(domain["block_bytes"] if domain else block_bytes)
        if node_bytes <= 0 or physical_block % node_bytes:
            raise RuntimeError("Unsupported nonintegral L1 block/node layout")
        nodes_per_block = physical_block // node_bytes
        reserved = int(store_config.get("cache_load_exclusive_buffer_number", 1024))
        uid = str(connector.unique_id)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", uid):
            raise RuntimeError("Unsafe UCM unique_id in shared-memory provenance")
        shm = Path("/dev/shm") / ("uc_shm_cache_" + uid)
        shm_record = None
        if store_config.get("share_buffer_enable", True):
            with shm.open("rb") as stream:
                header = stream.read(16)
            magic, nodes = struct.unpack("=QQ", header)
            if magic != (ord("S") << 16 | ord("b") << 8 | (4 if whole else 3)):
                raise RuntimeError(f"Unknown shared L1 header format: {magic:#x}")
            st = shm.stat()
            shm_record = {"path": str(shm), "device": st.st_dev, "inode": st.st_ino,
                          "size_bytes": st.st_size, "unique_id": uid}
        elif domain and "nodes" in domain:
            nodes = int(domain["nodes"])
        else:
            raise RuntimeError("Nonshared L1 adapter requires observed native node count")
        if nodes <= reserved:
            raise RuntimeError("Observed L1 has no ordinary cache nodes")
        tensors, seen = {}, set()
        for name, tensor in _tensors(connector.kv_caches):
            identity = (int(tensor.data_ptr()), int(tensor.numel()), str(tensor.dtype))
            if identity in seen:
                continue
            seen.add(identity)
            tensors[name] = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                             "bytes": int(tensor.numel() * tensor.element_size()), "device": str(tensor.device)}
        tensor_bytes = sum(x["bytes"] for x in tensors.values())
        blocks = int(layout.num_blocks)
        if blocks <= 0 or tensor_bytes <= 0 or tensor_bytes % blocks:
            raise RuntimeError("Cannot derive uniform per-block KV bytes from observed tensors")
        pending = []
        for item in connector._pending_dump_tasks:
            pending.append({"task_id": int(item.task.task_id), "complete": bool(connector.store.check(item.task)),
                            "sources_released": bool(item.sources_released)})
        dispatcher = group._metrics_dispatcher
        required_metrics = {'posix_dump_queue_wait_duration_ms','posix_dump_task_duration_ms',
                            'posix_h2s_bytes_total','posix_dump_write_bytes_total'}
        if dispatcher is None or not hasattr(dispatcher, '_bundle_observer_buffer'):
            raise RuntimeError('UCM requires the bundle cumulative-metrics patch; disabled/consumed buffers are not evidence of zero I/O')
        if not required_metrics <= set(dispatcher._definitions_by_name):
            raise RuntimeError('Metrics configuration omits required physical-drain counters/histograms')
        if collect_metrics:
            dispatcher.drain_to_consumers()
        with dispatcher._lock:
            counters, gauges, histograms = copy.deepcopy(dispatcher._bundle_observer_buffer)
        # Every pending dimension is explicit; missing fields never become a passing zero.
        lifecycle = {"unfinished_dump_tasks": sum(not row["complete"] for row in pending),
                     "pending_load_tasks": len(connector._pending_load_tasks),
                     "block_dump_batch_pending": int(bool(getattr(connector, "_block_dump_enabled", False)
                         and getattr(connector, "_dump_ready_layers", set())
                         and not getattr(connector, "_block_dump_finalized", False))),
                     "early_jobs": 0, "early_request_mappings": 0, "early_consumer_tasks": 0,
                     "early_active_consumers": 0, "store_load_receipts": 0, "consistency_load_contexts": 0}
        early = getattr(connector, "_early", None)
        if connector.launch_config.get("enable_early_prefetch") and early is None:
            raise RuntimeError("Early prefetch configured but no runtime Early owner was observed")
        if early is not None:
            with early.condition:
                lifecycle.update(early_jobs=len(early.jobs), early_request_mappings=len(early.request_keys))
            consumer = early.owner_protocol.consumer
            with consumer.lock:
                lifecycle.update(early_consumer_tasks=len(consumer.tasks),
                                 early_active_consumers=sum(x["active"] for x in consumer.generations.values()))
            lifecycle["store_load_receipts"] = len(connector.store.store_.receipts)
            lifecycle["consistency_load_contexts"] = len(connector._rank_consistency._load_task_contexts)
            if early.owner_protocol.fatal:
                raise RuntimeError("Early owner protocol is fenced after an async failure")
        cfg = self.vllm_config
        result = {"schema_version": 1, "pid": os.getpid(), "ppid": os.getppid(),
                  "monotonic_ns": time.monotonic_ns(), "platform": current_platform.device_type,
                  "worker_rank": int(getattr(self, "rank", getattr(self, "local_rank", 0))),
                  "connector_class": type(connector).__name__, "unique_id": uid,
                  "shared_memory": shm_record, "native_domain": domain,
                  "capacity_layout": {"block_tokens": int(connector.block_size),
                      "hbm_bytes_per_block": tensor_bytes // blocks, "hbm_reserved_blocks": 1,
                      "l1_node_bytes": node_bytes, "l1_nodes_per_token_block": nodes_per_block,
                      "l1_reserved_nodes": reserved},
                  "hbm_blocks": blocks, "hbm_tokens": blocks * int(connector.block_size),
                  "l1_nodes": nodes, "l1_ordinary_tokens": (nodes - reserved) // nodes_per_block * int(connector.block_size),
                  "tensors": tensors, "kv_tensor_bytes": tensor_bytes,
                  "pending_dump_task_status": pending, "pending_dump_tasks": len(pending), "lifecycle": lifecycle,
                  "counters": counters, "gauges": gauges,
                  "histogram_counts": {k: sum(v[0]) for k, v in histograms.items()},
                  "histogram_sums": {k: v[1] for k, v in histograms.items()},
                  "histogram_bucket_counts": {k: v[0] for k, v in histograms.items()},
                  "metrics_snapshot_contract": "bundle-cumulative-v1",
                  "metrics_collected": bool(collect_metrics),
                  "ucm_launch_config": connector.launch_config, "store_config": actual,
                  "vllm_config": str(cfg), "vllm_version": str(vllm.__version__),
                  "torch_version": str(torch.__version__), "ucm_source": str(Path(ucm.__file__).resolve()),
                  "loaded_native_libraries": sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                                    if '/ucm/' in line and '.so' in line}),
                  "run_dir": os.environ.get("BUNDLE_RUN_DIR")}
        return _plain(result)

    def bundle_reap_completed(self):
        _, connector = _context()
        for item in connector._pending_dump_tasks:
            if not connector.store.check(item.task) or not item.sources_released:
                raise RuntimeError("Cannot reap unfinished dump or unreleased source")
        before = len(connector._pending_dump_tasks)
        connector.get_finished(set())
        return {"pending_before": before, "pending_after": len(connector._pending_dump_tasks),
                "retired_requests_after": len(getattr(connector, "_block_dump_finished_req_ids", set()))}
